#!/usr/bin/env python3
"""Compare Baldwinian baseline against inheritance-mode treatment arm(s).

Outputs
-------
- ``inheritance_ab_summary.json`` — machine-readable paired comparisons
- ``inheritance_ab_summary.md`` — verdict-focused markdown report
- ``paired_delta_heatmap.png`` — mean paired deltas by profile/arm
- ``speciation_trajectories_with_arms.png`` — speciation traces by profile/arm
- ``startup_transient_comparison.png`` — startup-transient paired deltas
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from farm.analysis.lineage_metrics import (  # noqa: E402
    lineage_metrics as _lineage_metrics,
)
from scripts.analyze_stable_profile_seed_sweep import (  # noqa: E402
    PROFILE_ORDER,
    SIGN_AGREEMENT_THRESHOLD,
    _discover_runs,
    _extract_run_metrics,
    _mean,
    _t_ci,
    _variance,
)

METRIC_KEYS: List[str] = [
    "speciation_final",
    "speciation_slope",
    "speciation_mean",
    "population_mean",
    "population_final",
]
LINEAGE_KEYS: List[str] = ["mean_k", "churn_rate"]
STABILITY_KEYS: List[str] = [
    "startup_transient.peak_birth_rate",
    "startup_transient.peak_death_rate",
    "startup_transient.oscillation_amplitude",
]
INFO_KEYS: List[str] = ["lamarckian_warmstart_rate", "decide_action_failures"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Baldwinian baseline against one or more inheritance-mode "
            "treatment sweeps, paired by seed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        required=True,
        help="Sweep dir used as baseline (typically the baldwinian arm).",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="baldwinian",
        help="Label for the baseline arm (used in verdicts and plots).",
    )
    parser.add_argument(
        "--treatment-dir",
        action="append",
        required=True,
        help="Treatment sweep dir(s) to compare against the baseline.",
    )
    parser.add_argument(
        "--arm-labels",
        nargs="+",
        default=None,
        help="Optional labels for treatment arms (same order as --treatment-dir).",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=PROFILE_ORDER,
        choices=PROFILE_ORDER,
        metavar="PROFILE",
        help="Profiles to include in the comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/inheritance_ab/aggregate",
        help="Output directory for summary + plots.",
    )
    return parser


def _paired_delta_summary(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(v) for v in values if not math.isnan(v)]
    if not clean:
        # Empty summaries use ``None`` rather than ``float("nan")`` so that
        # ``json.dump`` emits valid JSON (``"null"``) instead of the
        # JavaScript-only ``NaN`` token; downstream tools that consume the
        # summary file should treat ``None`` as "no data".
        return {
            "mean_delta": None,
            "variance": None,
            "ci95": [None, None],
            "sign_agreement": None,
            "n": 0,
        }

    positive = sum(1 for v in clean if v > 0.0)
    negative = sum(1 for v in clean if v < 0.0)
    sign_agreement = max(positive, negative) / len(clean)
    lo, hi = _t_ci(clean)
    return {
        "mean_delta": _mean(clean),
        "variance": _variance(clean),
        "ci95": [lo, hi],
        "sign_agreement": sign_agreement,
        "n": len(clean),
    }


def _ci_excludes_zero(ci: Sequence[float]) -> bool:
    if len(ci) != 2:
        return False
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return False
    return (lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)


def _extract_metadata_metrics(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "intrinsic_evolution_metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    startup = meta.get("startup_transient_metrics", {})
    inheritance = meta.get("policy_inheritance_metrics", {})
    applied = int(inheritance.get("lamarckian_warmstart_applied", 0))
    skipped = int(inheritance.get("lamarckian_warmstart_skipped", 0))
    skipped_reasons = dict(
        inheritance.get("lamarckian_warmstart_skipped_reasons", {}) or {}
    )
    denom = applied + skipped
    warmstart_rate = float(applied / denom) if denom > 0 else float("nan")
    # ``decide_action_failures`` was added so paired A/B comparisons can flag
    # arms where the new fail-loud decision path is degrading silently
    # (see ``InheritanceTelemetry`` and the 2026-05-22 dev-log entry).
    decide_failures = float(inheritance.get("decide_action_failures", 0))
    decide_failure_reasons = dict(
        inheritance.get("decide_action_failure_reasons", {}) or {}
    )
    return {
        "startup_transient": {
            "peak_birth_rate": float(startup.get("peak_birth_rate", float("nan"))),
            "peak_death_rate": float(startup.get("peak_death_rate", float("nan"))),
            "oscillation_amplitude": float(startup.get("oscillation_amplitude", float("nan"))),
        },
        "lamarckian_warmstart_applied": applied,
        "lamarckian_warmstart_skipped": skipped,
        "lamarckian_warmstart_rate": warmstart_rate,
        "lamarckian_warmstart_skipped_reasons": skipped_reasons,
        "decide_action_failures": decide_failures,
        "decide_action_failure_reasons": decide_failure_reasons,
    }


def _extract_full_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    metrics = _extract_run_metrics(run_dir)
    if metrics is None:
        return None
    metrics.update({"lineage": _lineage_metrics(run_dir)})
    metrics.update(_extract_metadata_metrics(run_dir))
    return metrics


def _load_arm(
    sweep_dir: Path, profiles: Sequence[str]
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    run_map = _discover_runs(sweep_dir, list(profiles))
    for profile, seed_dirs in run_map.items():
        out[profile] = {}
        for seed, run_dir in seed_dirs:
            metrics = _extract_full_run(run_dir)
            if metrics is None:
                continue
            out[profile][seed] = metrics
    return out


def _resolve_metric_value(run: Dict[str, Any], key: str) -> Optional[float]:
    """Resolve a possibly-nested metric reference such as ``"lineage.churn_rate"``.

    Returns ``None`` when any segment is missing or the leaf value is not
    numeric, so callers can uniformly skip seeds with incomplete data.
    """
    parts = key.split(".")
    value: Any = run
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    if not isinstance(value, (int, float)):
        return None
    return float(value)


# Flat list of metric keys (dotted for nested lookups). Centralised so the
# delta computation, verdict logic, and markdown table all operate on the
# same canonical set.
ALL_METRIC_KEYS: List[str] = (
    METRIC_KEYS
    + [f"lineage.{k}" for k in LINEAGE_KEYS]
    + STABILITY_KEYS
    + INFO_KEYS
)


def _paired_metric_deltas(
    baseline_runs: Dict[int, Dict[str, Any]],
    treatment_runs: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[int]]:
    """Compute per-metric paired-by-seed deltas.

    Returns a ``(deltas_by_key, paired_seeds)`` tuple. The deltas dict maps
    each metric key in :data:`ALL_METRIC_KEYS` to its summary stats. Seeds
    where either side is missing or NaN for a given metric are dropped from
    that metric's pool but still contribute to other metrics.
    """
    paired_seeds = sorted(set(baseline_runs.keys()) & set(treatment_runs.keys()))
    deltas: Dict[str, Dict[str, Any]] = {}

    for key in ALL_METRIC_KEYS:
        values: List[float] = []
        for seed in paired_seeds:
            base = _resolve_metric_value(baseline_runs[seed], key)
            treat = _resolve_metric_value(treatment_runs[seed], key)
            if base is None or treat is None:
                continue
            if math.isnan(base) or math.isnan(treat):
                continue
            values.append(treat - base)
        deltas[key] = _paired_delta_summary(values)

    return deltas, paired_seeds


def _is_robust(summary: Dict[str, Any]) -> bool:
    if summary.get("n", 0) < 2:
        return False
    return (
        summary.get("sign_agreement", 0.0) >= SIGN_AGREEMENT_THRESHOLD
        and _ci_excludes_zero(summary.get("ci95", []))
    )


# Metric groups consulted by the verdict classifier. Ordered for stable
# tie-breaking and aligned with the protocol in
# ``docs/experiments/intrinsic_evolution/inheritance_mode_ab.md``.
PERFORMANCE_METRICS: List[str] = ["population_mean", "population_final"]
STABILITY_METRICS: List[str] = [
    "startup_transient.peak_death_rate",
    "startup_transient.oscillation_amplitude",
    "lineage.churn_rate",
]
COLLAPSE_METRIC: str = "speciation_slope"


def _robust_signed_delta(summary: Dict[str, Any], expected_sign: int) -> bool:
    """``True`` when the summary is robust and the mean delta has ``expected_sign``.

    ``expected_sign`` is ``+1`` for "treatment > baseline" and ``-1`` for
    "treatment < baseline". Robustness gates: 95% CI excludes zero AND
    sign-agreement >= :data:`SIGN_AGREEMENT_THRESHOLD`.
    """
    if not _is_robust(summary):
        return False
    mean = summary.get("mean_delta", 0.0)
    if expected_sign > 0:
        return mean > 0.0
    return mean < 0.0


def _classify_regime_verdict(
    profile_deltas: Dict[str, Dict[str, Any]],
    treatment_label: str,
    baseline_label: str,
) -> str:
    """Classify the regime for one (profile, arm) pair.

    Performance: any metric in :data:`PERFORMANCE_METRICS` shows a robust
    positive delta (treatment > baseline).
    Stability loss: any metric in :data:`STABILITY_METRICS` shows a robust
    positive delta (treatment makes startup more violent / lineages churn
    faster than baseline).
    Speciation collapse: ``speciation_slope`` shows a robust negative delta
    (treatment compresses speciation faster than baseline).
    """
    has_perf_win = any(
        _robust_signed_delta(profile_deltas.get(m, {}), expected_sign=+1)
        for m in PERFORMANCE_METRICS
    )
    has_stability_loss = any(
        _robust_signed_delta(profile_deltas.get(m, {}), expected_sign=+1)
        for m in STABILITY_METRICS
    )
    has_collapse = _robust_signed_delta(
        profile_deltas.get(COLLAPSE_METRIC, {}), expected_sign=-1
    )

    if has_perf_win and not has_stability_loss and not has_collapse:
        return f"net recommend {treatment_label}"
    if has_stability_loss and not has_perf_win:
        return f"net recommend {baseline_label}"
    if has_perf_win and has_stability_loss:
        return "performance win + stability loss"
    if has_collapse:
        return "speciation collapse risk"
    return "no robust effect"


def _extract_speciation_trace(run: Dict[str, Any]) -> Tuple[List[int], List[float]]:
    trajectory = run.get("trajectory", [])
    steps: List[int] = []
    values: List[float] = []
    for row in trajectory:
        value = row.get("speciation_index")
        step = row.get("step")
        if value is None or step is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        steps.append(int(step))
        values.append(value)
    return steps, values


def _plot_speciation_trajectories(
    baseline: Dict[str, Dict[int, Dict[str, Any]]],
    treatments: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    profiles: Sequence[str],
    output_path: Path,
    baseline_label: str = "baldwinian",
) -> None:
    fig, axes = plt.subplots(len(profiles), 1, figsize=(11, 3.6 * len(profiles)), sharex=False)
    if len(profiles) == 1:
        axes = [axes]

    for ax, profile in zip(axes, profiles):
        arms = [(baseline_label, baseline.get(profile, {}))]
        arms.extend((label, runs.get(profile, {})) for label, runs in treatments.items())
        for label, runs in arms:
            all_steps = sorted(
                {
                    step
                    for run in runs.values()
                    for step in _extract_speciation_trace(run)[0]
                }
            )
            if not all_steps:
                continue
            step_to_vals: Dict[int, List[float]] = {s: [] for s in all_steps}
            for run in runs.values():
                steps, values = _extract_speciation_trace(run)
                for s, v in zip(steps, values):
                    step_to_vals[s].append(v)
            means = [float(np.mean(step_to_vals[s])) if step_to_vals[s] else float("nan") for s in all_steps]
            ax.plot(all_steps, means, label=label)

        ax.set_title(f"{profile} - speciation index trajectory")
        ax.set_xlabel("step")
        ax.set_ylabel("speciation_index")
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_paired_delta_heatmap(
    deltas: Dict[str, Dict[str, Dict[str, Any]]],
    profiles: Sequence[str],
    arm_labels: Sequence[str],
    output_path: Path,
) -> None:
    columns = [
        "population_mean",
        "speciation_slope",
        "startup_transient.peak_death_rate",
        "startup_transient.oscillation_amplitude",
        "lineage.churn_rate",
    ]
    row_labels: List[str] = []
    matrix: List[List[float]] = []
    for profile in profiles:
        for arm in arm_labels:
            row_labels.append(f"{profile}:{arm}")
            row = []
            arm_d = deltas.get(profile, {}).get(arm, {})
            for col in columns:
                raw = arm_d.get(col, {}).get("mean_delta", float("nan"))
                row.append(float("nan") if raw is None else float(raw))
            matrix.append(row)
    if not matrix:
        return

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.8 * len(columns) + 4, 0.45 * len(row_labels) + 3))
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            text = "nan" if math.isnan(v) else f"{v:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean paired delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_startup_transient(
    deltas: Dict[str, Dict[str, Dict[str, Any]]],
    profiles: Sequence[str],
    arm_labels: Sequence[str],
    output_path: Path,
) -> None:
    metrics = [
        "startup_transient.peak_birth_rate",
        "startup_transient.peak_death_rate",
        "startup_transient.oscillation_amplitude",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.2), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        width = 0.8 / max(1, len(arm_labels))
        x = np.arange(len(profiles), dtype=float)
        for idx, arm in enumerate(arm_labels):
            values = [
                float(
                    deltas.get(profile, {})
                    .get(arm, {})
                    .get(metric, {})
                    .get("mean_delta", float("nan"))
                )
                for profile in profiles
            ]
            offset = (idx - (len(arm_labels) - 1) / 2.0) * width
            ax.bar(x + offset, values, width=width, label=arm)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=20, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _warmstart_rate_from_run(run: Dict[str, Any]) -> Optional[float]:
    applied = int(run.get("lamarckian_warmstart_applied", 0))
    skipped = int(run.get("lamarckian_warmstart_skipped", 0))
    denom = applied + skipped
    if denom <= 0:
        return None
    return float(applied / denom)


def _summarize_warmstart_coverage(
    treatment_runs: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate Lamarckian warm-start coverage for one (profile, arm) pair."""
    per_seed: Dict[str, Dict[str, Any]] = {}
    rates: List[float] = []
    total_applied = 0
    total_skipped = 0
    skip_reasons: Counter = Counter()

    for seed in sorted(treatment_runs):
        run = treatment_runs[seed]
        applied = int(run.get("lamarckian_warmstart_applied", 0))
        skipped = int(run.get("lamarckian_warmstart_skipped", 0))
        rate = _warmstart_rate_from_run(run)
        per_seed[str(seed)] = {
            "applied": applied,
            "skipped": skipped,
            "rate": rate,
            "skip_reasons": dict(run.get("lamarckian_warmstart_skipped_reasons", {}) or {}),
        }
        total_applied += applied
        total_skipped += skipped
        for reason, count in (run.get("lamarckian_warmstart_skipped_reasons", {}) or {}).items():
            skip_reasons[str(reason)] += int(count)
        if rate is not None:
            rates.append(rate)

    lo, hi = _t_ci(rates) if rates else (None, None)
    return {
        "n": len(rates),
        "mean_rate": _mean(rates) if rates else None,
        "rate_ci95": [lo, hi],
        "total_applied": total_applied,
        "total_skipped": total_skipped,
        "skip_reasons": dict(skip_reasons),
        "per_seed": per_seed,
    }


def _compute_mechanism_coverage(
    treatments: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    profiles: Sequence[str],
    arm_labels: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Treatment-only mechanism stats (not paired against baseline)."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for profile in profiles:
        out[profile] = {}
        for arm in arm_labels:
            treatment_runs = treatments.get(arm, {}).get(profile, {})
            out[profile][arm] = {
                "lamarckian_warmstart": _summarize_warmstart_coverage(treatment_runs),
            }
    return out


def _fmt_skip_reasons(reasons: Dict[str, int]) -> str:
    if not reasons:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))


def _fmt(value: Any, places: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "n/a"
        return f"{value:.{places}f}"
    if isinstance(value, list) and len(value) == 2:
        return f"[{_fmt(value[0], places)}, {_fmt(value[1], places)}]"
    return str(value)


def _build_markdown(
    deltas: Dict[str, Dict[str, Dict[str, Any]]],
    verdicts: Dict[str, Dict[str, str]],
    arm_labels: Sequence[str],
    baseline_label: str,
    mechanism_coverage: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> str:
    lines: List[str] = [
        "# Inheritance-mode A/B comparison",
        "",
        (
            f"Compares inheritance-mode treatment arms against the `{baseline_label}` "
            "baseline using paired-by-seed deltas per profile."
        ),
        "",
        "## Regime verdicts",
        "",
        "| Profile | " + " | ".join(arm_labels) + " |",
        "| --- |" + " --- |" * len(arm_labels),
    ]
    for profile in PROFILE_ORDER:
        if profile not in deltas:
            continue
        cells = [verdicts.get(profile, {}).get(arm, "n/a") for arm in arm_labels]
        lines.append(f"| {profile} | " + " | ".join(cells) + " |")

    lines += [
        "",
        (
            "*Robustness gate: paired 95% CI excludes zero and sign agreement "
            f"is at least {SIGN_AGREEMENT_THRESHOLD:.0%}.*"
        ),
        "",
    ]

    if mechanism_coverage:
        lines += [
            "## Mechanism coverage (treatment only)",
            "",
            (
                "Lamarckian warm-start rate is an absolute treatment-arm statistic. "
                f"The `{baseline_label}` baseline performs no warm-start attempts, "
                "so paired deltas are undefined."
            ),
            "",
            "| Profile | Arm | Mean rate | 95% CI | Applied | Skipped | Skip reasons | n |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for profile in PROFILE_ORDER:
            if profile not in mechanism_coverage:
                continue
            for arm in arm_labels:
                warmstart = mechanism_coverage.get(profile, {}).get(arm, {}).get(
                    "lamarckian_warmstart", {}
                )
                if not warmstart:
                    continue
                lines.append(
                    f"| {profile} | {arm} | {_fmt(warmstart.get('mean_rate'))} "
                    f"| {_fmt(warmstart.get('rate_ci95'))} "
                    f"| {warmstart.get('total_applied', 0)} "
                    f"| {warmstart.get('total_skipped', 0)} "
                    f"| {_fmt_skip_reasons(warmstart.get('skip_reasons', {}))} "
                    f"| {warmstart.get('n', 0)} |"
                )
        lines.append("")

    headline_keys = [
        ("population_mean", "population mean"),
        ("population_final", "population final"),
        ("speciation_slope", "speciation slope/100"),
        ("startup_transient.peak_death_rate", "startup peak death rate"),
        ("startup_transient.oscillation_amplitude", "startup oscillation amplitude"),
        ("decide_action_failures", "decide_action failures delta"),
    ]
    lines += ["## Paired deltas (treatment - baseline)", ""]
    for profile in PROFILE_ORDER:
        if profile not in deltas:
            continue
        lines += [f"### {profile}", ""]
        lines += [
            "| Arm | Metric | Mean delta | 95% CI | Sign agreement | n |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for arm in arm_labels:
            arm_data = deltas.get(profile, {}).get(arm, {})
            for key, label in headline_keys:
                summary = arm_data.get(key, {})
                if not summary:
                    continue
                lines.append(
                    f"| {arm} | {label} | {_fmt(summary.get('mean_delta'))} "
                    f"| {_fmt(summary.get('ci95'))} | {_fmt(summary.get('sign_agreement'), 2)} "
                    f"| {summary.get('n', 0)} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _build_parser().parse_args()
    baseline_dir = Path(args.baseline_dir)
    baseline_label = str(args.baseline_label)
    treatment_dirs = [Path(p) for p in args.treatment_dir]
    if args.arm_labels is None:
        arm_labels = [p.name for p in treatment_dirs]
    else:
        arm_labels = list(args.arm_labels)
    if len(arm_labels) != len(treatment_dirs):
        raise ValueError("--arm-labels must match --treatment-dir count.")

    profiles = [p for p in PROFILE_ORDER if p in args.profiles]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_arm(baseline_dir, profiles)
    treatments = {label: _load_arm(path, profiles) for label, path in zip(arm_labels, treatment_dirs)}

    deltas: Dict[str, Dict[str, Dict[str, Any]]] = {}
    verdicts: Dict[str, Dict[str, str]] = {}
    paired_seeds: Dict[str, Dict[str, List[int]]] = {}
    for profile in profiles:
        deltas[profile] = {}
        verdicts[profile] = {}
        paired_seeds[profile] = {}
        baseline_runs = baseline.get(profile, {})
        for label in arm_labels:
            treatment_runs = treatments[label].get(profile, {})
            profile_deltas, profile_paired_seeds = _paired_metric_deltas(
                baseline_runs, treatment_runs
            )
            deltas[profile][label] = profile_deltas
            paired_seeds[profile][label] = profile_paired_seeds
            verdicts[profile][label] = _classify_regime_verdict(
                profile_deltas,
                treatment_label=label,
                baseline_label=baseline_label,
            )

    mechanism_coverage = _compute_mechanism_coverage(treatments, profiles, arm_labels)

    summary = {
        "comparison_type": "inheritance_mode_ab",
        "baseline_dir": str(baseline_dir),
        "baseline_label": baseline_label,
        "treatments": {label: str(path) for label, path in zip(arm_labels, treatment_dirs)},
        "profiles": profiles,
        "deltas": deltas,
        "verdicts": verdicts,
        "paired_seeds": paired_seeds,
        "mechanism_coverage": mechanism_coverage,
    }

    json_path = output_dir / "inheritance_ab_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md_path = output_dir / "inheritance_ab_summary.md"
    md_path.write_text(
        _build_markdown(
            deltas,
            verdicts,
            arm_labels,
            baseline_label,
            mechanism_coverage=mechanism_coverage,
        ),
        encoding="utf-8",
    )

    _plot_speciation_trajectories(
        baseline=baseline,
        treatments=treatments,
        profiles=profiles,
        output_path=output_dir / "speciation_trajectories_with_arms.png",
        baseline_label=baseline_label,
    )
    _plot_paired_delta_heatmap(
        deltas=deltas,
        profiles=profiles,
        arm_labels=arm_labels,
        output_path=output_dir / "paired_delta_heatmap.png",
    )
    _plot_startup_transient(
        deltas=deltas,
        profiles=profiles,
        arm_labels=arm_labels,
        output_path=output_dir / "startup_transient_comparison.png",
    )

    print(f"Wrote summary JSON: {json_path}")
    print(f"Wrote summary markdown: {md_path}")
    print(f"Wrote plots under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
