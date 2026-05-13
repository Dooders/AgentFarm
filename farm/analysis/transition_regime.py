"""Transition-regime analysis for intrinsic-evolution seed sweeps.

This module turns a collection of intrinsic-evolution run artifacts into the
evidence bundle needed for a transition-regime claim:

``in profile P, with parameter Q in range R, the system transitions between
mode A and mode B with probability p, controlled by mechanism M``.

The functions are intentionally conservative.  They produce an exit paragraph
only after mode support, probability, and mechanism gates are satisfied; when
the evidence is incomplete they return explicit refusal reasons instead of
silently overclaiming from a noisy seed sweep.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - exercised by fallback tests indirectly
    GaussianMixture = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]


DEFAULT_MODE_FEATURES: Tuple[str, ...] = (
    "late_speciation_mean",
    "late_speciation_slope",
    "population_early_overshoot",
)

HIGH_SPECIATION_MODE = "high_speciation"
LOW_SPECIATION_MODE = "low_speciation"
BASELINE_INTERVENTION = "baseline"
CROSSOVER_INTERVENTION = "crossover_on"


@dataclass(frozen=True)
class TransitionRunMetrics:
    """Run-level features used to detect transition-regime modes."""

    run_dir: str
    seed: Optional[int]
    profile: str
    parameter_name: str
    parameter_value: float
    intervention: str = BASELINE_INTERVENTION
    status: str = "ok"
    final_speciation: float = float("nan")
    late_speciation_mean: float = float("nan")
    late_speciation_slope: float = float("nan")
    mean_population: float = float("nan")
    final_population: float = float("nan")
    peak_population: float = float("nan")
    population_early_overshoot: float = float("nan")
    early_birth_rate_mean: float = float("nan")
    early_death_rate_mean: float = float("nan")
    late_selection_strength_mean: float = float("nan")
    late_reproduction_cost_mean: float = float("nan")
    cluster_split_rate: float = float("nan")
    cluster_merge_rate: float = float("nan")
    gene_pct_shift: Dict[str, float] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)

    def feature_value(self, name: str) -> float:
        """Return a numeric feature by name, defaulting to NaN when absent."""

        value = getattr(self, name, float("nan"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModeAssignment:
    """Assigned mode for one run."""

    run_dir: str
    seed: Optional[int]
    parameter_value: float
    intervention: str
    mode: str
    confidence: float
    classifier: str
    score: float
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionProbability:
    """Binomial estimate for a mode in a parameter range."""

    parameter_name: str
    parameter_range: str
    mode: str
    n: int
    k: int
    p: float
    ci95: Tuple[float, float]
    intervention: str = BASELINE_INTERVENTION
    range_min: Optional[float] = None
    range_max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["ci95"] = list(self.ci95)
        return payload


@dataclass(frozen=True)
class MechanismEvidence:
    """Evidence for one candidate mechanism controlling the mode transition."""

    mechanism: str
    description: str
    supported: bool
    effect_size: float
    baseline_value: float
    comparison_value: float
    comparison: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionRegimeSummary:
    """Complete evidence bundle for a transition-regime analysis."""

    profile: str
    parameter_name: str
    metrics: List[TransitionRunMetrics]
    mode_assignments: List[ModeAssignment]
    probabilities: List[TransitionProbability]
    mechanisms: List[MechanismEvidence]
    exit_paragraph: Optional[str]
    evidence_gate_reasons: List[str]
    mode_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "parameter_name": self.parameter_name,
            "metrics": [m.to_dict() for m in self.metrics],
            "mode_assignments": [a.to_dict() for a in self.mode_assignments],
            "probabilities": [p.to_dict() for p in self.probabilities],
            "mechanisms": [m.to_dict() for m in self.mechanisms],
            "exit_paragraph": self.exit_paragraph,
            "evidence_gate_reasons": list(self.evidence_gate_reasons),
            "mode_counts": dict(self.mode_counts),
        }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_values(values: Iterable[Any]) -> List[float]:
    return [float(v) for v in values if _finite(v)]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mu = _mean(values)
    return float(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _slope_per_100(pairs: Sequence[Tuple[float, float]]) -> float:
    clean = [(x, y) for x, y in pairs if _finite(x) and _finite(y)]
    if len(clean) < 2:
        return float("nan")
    xs = np.array([p[0] for p in clean], dtype=float)
    ys = np.array([p[1] for p in clean], dtype=float)
    try:
        return float(np.polyfit(xs, ys, 1)[0]) * 100.0
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")


def _pct_shift(initial: float, final: float) -> float:
    if not _finite(initial) or not _finite(final) or abs(float(initial)) < 1e-12:
        return float("nan")
    return 100.0 * (float(final) - float(initial)) / abs(float(initial))


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mode_entropy(assignments: Sequence[ModeAssignment]) -> float:
    if not assignments:
        return float("nan")
    counts = Counter(a.mode for a in assignments)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy


def _standardized_mean_difference(high_values: Sequence[float], low_values: Sequence[float]) -> float:
    high = _finite_values(high_values)
    low = _finite_values(low_values)
    if len(high) < 2 or len(low) < 2:
        return float("nan")
    pooled_var = (_variance(high) + _variance(low)) / 2.0
    if not _finite(pooled_var) or pooled_var <= 0.0:
        return float("nan")
    return (_mean(high) - _mean(low)) / math.sqrt(pooled_var)


def extract_transition_run_metrics(
    run_dir: Path,
    factor_metadata: Optional[Dict[str, Any]] = None,
) -> TransitionRunMetrics:
    """Extract transition-regime features from one intrinsic-evolution run."""

    factor_metadata = dict(factor_metadata or {})
    trajectory = _read_jsonl(run_dir / "intrinsic_gene_trajectory.jsonl")
    snapshots = _read_jsonl(run_dir / "intrinsic_gene_snapshots.jsonl")
    cluster_rows = _read_jsonl(run_dir / "cluster_lineage.jsonl")
    metadata = _read_json(run_dir / "intrinsic_evolution_metadata.json")

    missing: List[str] = []
    if not trajectory:
        missing.append("intrinsic_gene_trajectory.jsonl")
    if not snapshots:
        missing.append("intrinsic_gene_snapshots.jsonl")

    spec_pairs = [
        (float(row.get("step", 0.0)), float(row.get("speciation_index")))
        for row in trajectory
        if _finite(row.get("speciation_index"))
    ]
    spec_values = [value for _, value in spec_pairs]
    late_start = max(0, int(len(spec_pairs) * 0.75))
    late_pairs = spec_pairs[late_start:] if spec_pairs else []

    n_alive = _finite_values(row.get("n_alive") for row in trajectory)
    early_window_size = min(50, max(1, int(len(trajectory) * 0.1))) if trajectory else 0
    early_rows = trajectory[:early_window_size]
    late_rows = trajectory[late_start:] if trajectory else []
    early_alive = _finite_values(row.get("n_alive") for row in early_rows)

    first_population = n_alive[0] if n_alive else float("nan")
    population_early_overshoot = (
        max(early_alive) - first_population
        if early_alive and _finite(first_population)
        else float("nan")
    )

    split_count = sum(1 for row in cluster_rows if row.get("transition_type") == "split")
    merge_count = sum(1 for row in cluster_rows if row.get("transition_type") == "merge")
    cluster_denominator = len(cluster_rows) if cluster_rows else 0

    gene_pct_shift: Dict[str, float] = {}
    non_empty_snapshots = [snap for snap in snapshots if snap.get("agents")]
    if non_empty_snapshots:
        first = non_empty_snapshots[0]
        last = non_empty_snapshots[-1]
        genes = sorted(
            {
                gene
                for agent in first.get("agents", []) + last.get("agents", [])
                for gene in agent.get("chromosome", {})
            }
        )
        for gene in genes:
            initial_vals = _finite_values(
                agent.get("chromosome", {}).get(gene)
                for agent in first.get("agents", [])
            )
            final_vals = _finite_values(
                agent.get("chromosome", {}).get(gene)
                for agent in last.get("agents", [])
            )
            if initial_vals and final_vals:
                shift = _pct_shift(_mean(initial_vals), _mean(final_vals))
                if _finite(shift):
                    gene_pct_shift[gene] = shift

    seed = factor_metadata.get("seed", metadata.get("seed"))
    parameter_name = str(factor_metadata.get("parameter_name", "initial_agent_resource_level"))
    raw_parameter_value = factor_metadata.get("parameter_value")
    if raw_parameter_value is None:
        resolved_ic = metadata.get("resolved_initial_conditions") or {}
        raw_parameter_value = resolved_ic.get(parameter_name, float("nan"))
    parameter_value = float(raw_parameter_value) if _finite(raw_parameter_value) else float("nan")

    profile = str(
        factor_metadata.get(
            "profile",
            (metadata.get("initial_conditions") or {}).get("profile", "stable"),
        )
    )

    return TransitionRunMetrics(
        run_dir=str(run_dir),
        seed=int(seed) if _finite(seed) else None,
        profile=profile,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        intervention=str(factor_metadata.get("intervention", BASELINE_INTERVENTION)),
        status=str(factor_metadata.get("status", "ok")),
        final_speciation=spec_values[-1] if spec_values else float("nan"),
        late_speciation_mean=_mean([value for _, value in late_pairs]),
        late_speciation_slope=_slope_per_100(late_pairs),
        mean_population=_mean(n_alive),
        final_population=n_alive[-1] if n_alive else float("nan"),
        peak_population=max(n_alive) if n_alive else float("nan"),
        population_early_overshoot=population_early_overshoot,
        early_birth_rate_mean=_mean(_finite_values(row.get("realized_birth_rate") for row in early_rows)),
        early_death_rate_mean=_mean(_finite_values(row.get("realized_death_rate") for row in early_rows)),
        late_selection_strength_mean=_mean(
            _finite_values(row.get("effective_selection_strength") for row in late_rows)
        ),
        late_reproduction_cost_mean=_mean(
            _finite_values(row.get("mean_reproduction_cost") for row in late_rows)
        ),
        cluster_split_rate=(split_count / cluster_denominator if cluster_denominator else float("nan")),
        cluster_merge_rate=(merge_count / cluster_denominator if cluster_denominator else float("nan")),
        gene_pct_shift=gene_pct_shift,
        missing_fields=missing,
    )


def _threshold_assignments(
    metrics: Sequence[TransitionRunMetrics],
    mode_features: Sequence[str],
    threshold: Optional[float],
    classifier_name: str,
) -> List[ModeAssignment]:
    primary = mode_features[0] if mode_features else "late_speciation_mean"
    values = [metric.feature_value(primary) for metric in metrics]
    finite = _finite_values(values)
    resolved_threshold = float(threshold) if threshold is not None else _mean(sorted(finite)[len(finite) // 2:len(finite) // 2 + 1])
    if not _finite(resolved_threshold):
        resolved_threshold = 0.0
    spread = max(finite) - min(finite) if len(finite) > 1 else 1.0
    if spread <= 0.0:
        spread = 1.0

    assignments: List[ModeAssignment] = []
    for metric in metrics:
        score = metric.feature_value(primary)
        if not _finite(score):
            mode = LOW_SPECIATION_MODE
            confidence = 0.0
        elif score >= resolved_threshold:
            mode = HIGH_SPECIATION_MODE
            confidence = min(1.0, abs(score - resolved_threshold) / spread + 0.5)
        else:
            mode = LOW_SPECIATION_MODE
            confidence = min(1.0, abs(score - resolved_threshold) / spread + 0.5)
        assignments.append(
            ModeAssignment(
                run_dir=metric.run_dir,
                seed=metric.seed,
                parameter_value=metric.parameter_value,
                intervention=metric.intervention,
                mode=mode,
                confidence=confidence,
                classifier=classifier_name,
                score=score,
                features={feature: metric.feature_value(feature) for feature in mode_features},
            )
        )
    return assignments


def classify_modes(
    metrics: Sequence[TransitionRunMetrics],
    *,
    mode_features: Sequence[str] = DEFAULT_MODE_FEATURES,
    n_modes: int = 2,
    threshold: Optional[float] = None,
) -> List[ModeAssignment]:
    """Classify run-level outcomes into low/high speciation modes."""

    if n_modes != 2:
        raise ValueError("transition-regime mode classification currently supports exactly two modes.")
    if threshold is not None or GaussianMixture is None or StandardScaler is None:
        return _threshold_assignments(metrics, mode_features, threshold, "threshold")

    complete: List[Tuple[int, TransitionRunMetrics, List[float]]] = []
    for idx, metric in enumerate(metrics):
        row = [metric.feature_value(feature) for feature in mode_features]
        if all(_finite(value) for value in row):
            complete.append((idx, metric, row))

    if len(complete) < max(4, n_modes * 2):
        return _threshold_assignments(metrics, mode_features, threshold, "threshold_fallback")

    X = np.array([row for _, _, row in complete], dtype=float)
    try:
        X_scaled = StandardScaler().fit_transform(X)
        model = GaussianMixture(n_components=n_modes, random_state=0, n_init=10)
        labels = model.fit_predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
    except (ValueError, np.linalg.LinAlgError):
        return _threshold_assignments(metrics, mode_features, threshold, "threshold_fallback")

    component_spec_means: Dict[int, float] = {}
    for label in range(n_modes):
        component_values = [
            metric.late_speciation_mean
            for row_label, (_, metric, _) in zip(labels, complete)
            if int(row_label) == label and _finite(metric.late_speciation_mean)
        ]
        component_spec_means[label] = _mean(component_values)
    high_component = max(component_spec_means, key=lambda key: component_spec_means[key])

    assignments_by_index: Dict[int, ModeAssignment] = {}
    for row_index, ((metric_index, metric, _row), label, proba) in enumerate(zip(complete, labels, probabilities)):
        label_int = int(label)
        mode = HIGH_SPECIATION_MODE if label_int == high_component else LOW_SPECIATION_MODE
        assignments_by_index[metric_index] = ModeAssignment(
            run_dir=metric.run_dir,
            seed=metric.seed,
            parameter_value=metric.parameter_value,
            intervention=metric.intervention,
            mode=mode,
            confidence=float(proba[row_index, label_int]),
            classifier="gmm",
            score=metric.late_speciation_mean,
            features={feature: metric.feature_value(feature) for feature in mode_features},
        )

    fallback_assignments = _threshold_assignments(metrics, mode_features, threshold, "threshold_fallback")
    return [
        assignments_by_index.get(idx, fallback_assignment)
        for idx, fallback_assignment in enumerate(fallback_assignments)
    ]


def estimate_transition_probabilities(
    assignments: Sequence[ModeAssignment],
    metrics: Sequence[TransitionRunMetrics],
    *,
    parameter_name: str,
    range_bins: Optional[Sequence[Tuple[float, float]]] = None,
    target_mode: str = HIGH_SPECIATION_MODE,
    intervention: Optional[str] = BASELINE_INTERVENTION,
) -> List[TransitionProbability]:
    """Estimate mode probabilities by parameter value or range."""

    assignment_by_run_dir = {assignment.run_dir: assignment for assignment in assignments}
    paired = [
        (metric, assignment_by_run_dir[metric.run_dir])
        for metric in metrics
        if metric.run_dir in assignment_by_run_dir
        and _finite(metric.parameter_value)
        and (intervention is None or metric.intervention == intervention)
    ]

    if range_bins is None:
        values = sorted({metric.parameter_value for metric, _ in paired})
        range_bins = [(value, value) for value in values]

    probabilities: List[TransitionProbability] = []
    for range_min, range_max in range_bins:
        if math.isclose(range_min, range_max):
            in_range = [
                (metric, assignment)
                for metric, assignment in paired
                if math.isclose(metric.parameter_value, range_min)
            ]
            label = f"{parameter_name}={range_min:g}"
        else:
            in_range = [
                (metric, assignment)
                for metric, assignment in paired
                if range_min <= metric.parameter_value <= range_max
            ]
            label = f"{range_min:g} ≤ {parameter_name} ≤ {range_max:g}"

        n = len(in_range)
        k = sum(1 for _, assignment in in_range if assignment.mode == target_mode)
        p = k / n if n else float("nan")
        probabilities.append(
            TransitionProbability(
                parameter_name=parameter_name,
                parameter_range=label,
                mode=target_mode,
                n=n,
                k=k,
                p=p,
                ci95=_wilson_ci(k, n),
                intervention=intervention or "all",
                range_min=range_min,
                range_max=range_max,
            )
        )
    return probabilities


def evaluate_mechanisms(
    metrics: Sequence[TransitionRunMetrics],
    assignments: Sequence[ModeAssignment],
    *,
    baseline_intervention: str = BASELINE_INTERVENTION,
    effect_threshold: float = 0.5,
) -> List[MechanismEvidence]:
    """Evaluate candidate controls over the low/high mode transition."""

    assignment_by_run_dir = {assignment.run_dir: assignment for assignment in assignments}
    paired = [
        (metric, assignment_by_run_dir[metric.run_dir])
        for metric in metrics
        if metric.run_dir in assignment_by_run_dir
    ]
    baseline_pairs = [(m, a) for m, a in paired if m.intervention == baseline_intervention]
    evidence: List[MechanismEvidence] = []

    crossover_pairs = [(m, a) for m, a in paired if m.intervention == CROSSOVER_INTERVENTION]
    if baseline_pairs and crossover_pairs:
        baseline_entropy = _mode_entropy([assignment for _, assignment in baseline_pairs])
        crossover_entropy = _mode_entropy([assignment for _, assignment in crossover_pairs])
        baseline_var = _variance(_finite_values(metric.final_speciation for metric, _ in baseline_pairs))
        crossover_var = _variance(_finite_values(metric.final_speciation for metric, _ in crossover_pairs))
        entropy_drop = baseline_entropy - crossover_entropy if _finite(baseline_entropy) and _finite(crossover_entropy) else 0.0
        variance_drop = baseline_var - crossover_var if _finite(baseline_var) and _finite(crossover_var) else 0.0
        effect = entropy_drop if abs(entropy_drop) >= abs(variance_drop) else variance_drop
        evidence.append(
            MechanismEvidence(
                mechanism="gene_flow",
                description=(
                    "Crossover-enabled gene flow changes mode entropy or final-speciation variance "
                    "relative to the mutation-only baseline."
                ),
                supported=effect >= effect_threshold,
                effect_size=effect,
                baseline_value=baseline_entropy,
                comparison_value=crossover_entropy,
                comparison=f"{baseline_intervention} vs {CROSSOVER_INTERVENTION}",
                details={
                    "baseline_entropy": baseline_entropy,
                    "crossover_entropy": crossover_entropy,
                    "baseline_final_speciation_variance": baseline_var,
                    "crossover_final_speciation_variance": crossover_var,
                    "entropy_drop": entropy_drop,
                    "variance_drop": variance_drop,
                },
            )
        )

    def _association_evidence(feature: str, mechanism: str, description: str) -> MechanismEvidence:
        high_values = [
            metric.feature_value(feature)
            for metric, assignment in baseline_pairs
            if assignment.mode == HIGH_SPECIATION_MODE
        ]
        low_values = [
            metric.feature_value(feature)
            for metric, assignment in baseline_pairs
            if assignment.mode == LOW_SPECIATION_MODE
        ]
        effect = _standardized_mean_difference(high_values, low_values)
        high_mean = _mean(_finite_values(high_values))
        low_mean = _mean(_finite_values(low_values))
        return MechanismEvidence(
            mechanism=mechanism,
            description=description,
            supported=_finite(effect) and abs(effect) >= effect_threshold,
            effect_size=effect,
            baseline_value=low_mean,
            comparison_value=high_mean,
            comparison=f"{LOW_SPECIATION_MODE} vs {HIGH_SPECIATION_MODE}",
            details={
                "feature": feature,
                "low_mode_mean": low_mean,
                "high_mode_mean": high_mean,
                "standardized_mean_difference": effect,
            },
        )

    if baseline_pairs:
        evidence.append(
            _association_evidence(
                "late_selection_strength_mean",
                "selection_strength",
                "Mode assignment tracks density-dependent reproduction-cost variation.",
            )
        )
        evidence.append(
            _association_evidence(
                "population_early_overshoot",
                "startup_transient",
                "Mode assignment tracks early population overshoot after warmup/startup.",
            )
        )
    return evidence


def _dominant_profile(metrics: Sequence[TransitionRunMetrics]) -> str:
    if not metrics:
        return "unknown"
    return Counter(metric.profile for metric in metrics).most_common(1)[0][0]


def _best_probability(
    probabilities: Sequence[TransitionProbability],
    *,
    min_runs_per_range: int,
    min_probability: float,
    max_probability: float,
) -> Optional[TransitionProbability]:
    candidates = [
        probability
        for probability in probabilities
        if probability.n >= min_runs_per_range
        and _finite(probability.p)
        and min_probability <= probability.p <= max_probability
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (abs(0.5 - item.p), -item.n))[0]


def build_exit_paragraph(
    summary: TransitionRegimeSummary,
    *,
    min_runs_per_range: int = 6,
    min_mode_count: int = 2,
    min_probability: float = 0.2,
    max_probability: float = 0.8,
) -> Optional[str]:
    """Return the requested one-paragraph claim when evidence gates pass."""

    mode_counts = Counter(assignment.mode for assignment in summary.mode_assignments)
    if len([count for count in mode_counts.values() if count >= min_mode_count]) < 2:
        return None
    supported_mechanisms = [mechanism for mechanism in summary.mechanisms if mechanism.supported]
    if not supported_mechanisms:
        return None
    probability = _best_probability(
        summary.probabilities,
        min_runs_per_range=min_runs_per_range,
        min_probability=min_probability,
        max_probability=max_probability,
    )
    if probability is None:
        return None

    mechanism = max(supported_mechanisms, key=lambda item: abs(item.effect_size) if _finite(item.effect_size) else 0.0)
    ci_low, ci_high = probability.ci95
    mechanism_phrase = mechanism.description[0].lower() + mechanism.description[1:]
    return (
        f"In profile {summary.profile}, with parameter {summary.parameter_name} in range "
        f"{probability.parameter_range}, the system transitions between mode {LOW_SPECIATION_MODE} "
        f"and mode {HIGH_SPECIATION_MODE} with probability p={probability.p:.2f} "
        f"(Wilson 95% CI [{ci_low:.2f}, {ci_high:.2f}]), controlled by mechanism "
        f"{mechanism.mechanism}: {mechanism_phrase}"
    )


def _evidence_gate_reasons(
    assignments: Sequence[ModeAssignment],
    probabilities: Sequence[TransitionProbability],
    mechanisms: Sequence[MechanismEvidence],
    *,
    min_runs_per_range: int,
    min_mode_count: int,
    min_probability: float,
    max_probability: float,
) -> List[str]:
    reasons: List[str] = []
    mode_counts = Counter(assignment.mode for assignment in assignments)
    if len([count for count in mode_counts.values() if count >= min_mode_count]) < 2:
        reasons.append(
            f"need at least two modes with ≥{min_mode_count} runs; observed {dict(mode_counts)}"
        )
    if not any(mechanism.supported for mechanism in mechanisms):
        reasons.append("no candidate mechanism met the support threshold")
    if _best_probability(
        probabilities,
        min_runs_per_range=min_runs_per_range,
        min_probability=min_probability,
        max_probability=max_probability,
    ) is None:
        reasons.append(
            f"no parameter range has n≥{min_runs_per_range} and "
            f"{min_probability:.2f}≤p≤{max_probability:.2f}"
        )
    return reasons


def summarize_transition_regime(
    metrics: Sequence[TransitionRunMetrics],
    *,
    parameter_name: str = "initial_agent_resource_level",
    profile: Optional[str] = None,
    mode_features: Sequence[str] = DEFAULT_MODE_FEATURES,
    range_bins: Optional[Sequence[Tuple[float, float]]] = None,
    min_runs_per_range: int = 6,
    min_mode_count: int = 2,
    min_probability: float = 0.2,
    max_probability: float = 0.8,
    mechanism_effect_threshold: float = 0.5,
) -> TransitionRegimeSummary:
    """Classify modes, estimate probabilities, and produce an evidence summary."""

    metrics_list = list(metrics)
    assignments = classify_modes(metrics_list, mode_features=mode_features)
    probabilities = estimate_transition_probabilities(
        assignments,
        metrics_list,
        parameter_name=parameter_name,
        range_bins=range_bins,
    )
    mechanisms = evaluate_mechanisms(
        metrics_list,
        assignments,
        effect_threshold=mechanism_effect_threshold,
    )
    provisional = TransitionRegimeSummary(
        profile=profile or _dominant_profile(metrics_list),
        parameter_name=parameter_name,
        metrics=metrics_list,
        mode_assignments=assignments,
        probabilities=probabilities,
        mechanisms=mechanisms,
        exit_paragraph=None,
        evidence_gate_reasons=[],
        mode_counts=dict(Counter(assignment.mode for assignment in assignments)),
    )
    exit_paragraph = build_exit_paragraph(
        provisional,
        min_runs_per_range=min_runs_per_range,
        min_mode_count=min_mode_count,
        min_probability=min_probability,
        max_probability=max_probability,
    )
    reasons = _evidence_gate_reasons(
        assignments,
        probabilities,
        mechanisms,
        min_runs_per_range=min_runs_per_range,
        min_mode_count=min_mode_count,
        min_probability=min_probability,
        max_probability=max_probability,
    )
    return TransitionRegimeSummary(
        profile=provisional.profile,
        parameter_name=provisional.parameter_name,
        metrics=provisional.metrics,
        mode_assignments=provisional.mode_assignments,
        probabilities=provisional.probabilities,
        mechanisms=provisional.mechanisms,
        exit_paragraph=exit_paragraph,
        evidence_gate_reasons=reasons,
        mode_counts=provisional.mode_counts,
    )
