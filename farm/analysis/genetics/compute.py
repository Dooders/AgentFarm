"""
Genetics Analysis Computation

Core computation functions for the genetics analysis module, including the
shared ``parse_parent_ids`` helper, population-level accessors for both
simulation-database and evolution-experiment data sources, genotypic
diversity metrics (heterozygosity, Shannon entropy, per-locus stats), and
allele-frequency trajectory tracking with selection-pressure detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from farm.analysis.genetics.utils import parse_parent_ids
from farm.utils.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from farm.runners.evolution_experiment import EvolutionExperimentResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Diversity metric result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousLocusDiversity:
    """Diversity metrics for a single continuous locus (hyperparameter gene).

    Computed from N individuals' values for one gene.

    Attributes
    ----------
    locus_name:
        Identifier for this locus/gene.
    n_individuals:
        Number of individuals contributing data.
    mean:
        Population mean: ``μ = (1/N) Σ xᵢ``.
    std:
        Population standard deviation (ddof=0): ``σ = sqrt((1/N) Σ (xᵢ-μ)²)``.
    normalized_variance:
        Variance normalized by squared range span:
        ``V̂ = Var(x) / span²`` where ``span = max_bound - min_bound``.
        ``nan`` when bounds are not provided or span is zero.
    coefficient_of_variation:
        Relative spread: ``CV = σ / |μ|``.
        ``nan`` when mean is zero or ``n_individuals`` is 0.
    range_occupancy:
        Fraction of the allowed range actually used:
        ``R = (max(x) - min(x)) / span``.
        ``nan`` when bounds are not provided; ``0.0`` when span is zero.
    shannon_entropy:
        Histogram-based Shannon entropy:
        ``H = -Σᵢ pᵢ ln(pᵢ)`` where ``pᵢ`` are bin-frequency probabilities.
        ``None`` when entropy computation was not requested
        (``compute_entropy=False``).
    """

    locus_name: str
    n_individuals: int
    mean: float
    std: float
    normalized_variance: float
    coefficient_of_variation: float
    range_occupancy: float
    shannon_entropy: Optional[float]


@dataclass(frozen=True)
class CategoricalLocusDiversity:
    """Diversity metrics for a categorical/weighted locus (action weights).

    Uses the population-mean allele frequencies
    ``p_i = mean_j(w_{ij})`` (renormalized to sum to 1) as inputs to all
    formulas.

    Attributes
    ----------
    locus_name:
        Identifier for this locus (e.g. ``"action_weights"``).
    n_individuals:
        Number of individuals contributing data.
    allele_frequencies:
        Mapping from category (action) name to population-mean frequency.
    shannon_entropy:
        ``H = -Σᵢ pᵢ ln(pᵢ)`` (nats).
        Zero for a monomorphic locus (single non-zero category).
    simpson_index:
        ``D = Σᵢ pᵢ²``.
        One for a monomorphic locus; minimum of ``1/k`` for ``k`` uniform
        categories.
    expected_heterozygosity:
        ``He = 1 - Σᵢ pᵢ²``.
        Zero for a monomorphic locus; maximum of ``1 - 1/k`` for ``k``
        equally-weighted categories.
    """

    locus_name: str
    n_individuals: int
    allele_frequencies: Dict[str, float]
    shannon_entropy: float
    simpson_index: float
    expected_heterozygosity: float


@dataclass(frozen=True)
class PopulationDiversitySummary:
    """Aggregated diversity summary across all loci for a population snapshot.

    Attributes
    ----------
    n_individuals:
        Number of individuals in the snapshot.
    continuous_loci:
        Per-locus diversity for each continuous (hyperparameter) gene,
        keyed by gene name.
    categorical_loci:
        Per-locus diversity for each categorical (action-weight) group,
        keyed by locus name.
    mean_normalized_variance:
        Mean ``normalized_variance`` across all continuous loci, excluding
        ``nan`` values.  ``nan`` when no finite values are available.
    mean_coefficient_of_variation:
        Mean ``coefficient_of_variation`` across all continuous loci,
        excluding ``nan`` values.  ``nan`` when none are available.
    mean_shannon_entropy:
        Mean Shannon entropy across *all* loci (continuous histogram entropy
        when requested, plus all categorical loci).  ``nan`` when none are
        available.
    mean_heterozygosity:
        Mean ``expected_heterozygosity`` across categorical loci.
        ``nan`` when no categorical loci exist.
    """

    n_individuals: int
    continuous_loci: Dict[str, ContinuousLocusDiversity] = field(default_factory=dict)
    categorical_loci: Dict[str, CategoricalLocusDiversity] = field(default_factory=dict)
    mean_normalized_variance: float = float("nan")
    mean_coefficient_of_variation: float = float("nan")
    mean_shannon_entropy: float = float("nan")
    mean_heterozygosity: float = float("nan")


# ---------------------------------------------------------------------------
# DB-backed population accessor
# ---------------------------------------------------------------------------

#: Columns produced by :func:`build_agent_genetics_dataframe`.
AGENT_GENETICS_COLUMNS = [
    "agent_id",
    "agent_type",
    "generation",
    "birth_time",
    "death_time",
    "genome_id",
    "parent_ids",
    "action_weights",
]


def build_agent_genetics_dataframe(session: "Session") -> pd.DataFrame:
    """Build a normalized genetics DataFrame from a simulation database session.

    Each row represents one agent.  The ``parent_ids`` column contains a
    Python list of parent agent-ID strings (empty list for genesis agents).
    Action weights are stored as a dict in the ``action_weights`` column.

    Parameters
    ----------
    session:
        An active SQLAlchemy session connected to a simulation database.

    Returns
    -------
    pd.DataFrame
        One row per agent with columns defined in :data:`AGENT_GENETICS_COLUMNS`.
        Returns an empty DataFrame (with correct columns) when no agents are
        found.
    """
    from farm.database.models import AgentModel

    agents: List[AgentModel] = session.query(AgentModel).all()

    if not agents:
        logger.info("build_agent_genetics_dataframe: no agents found in session")
        return pd.DataFrame(columns=AGENT_GENETICS_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for agent in agents:
        rows.append(
            {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "generation": agent.generation,
                "birth_time": agent.birth_time,
                "death_time": agent.death_time,
                "genome_id": agent.genome_id,
                "parent_ids": parse_parent_ids(agent.genome_id),
                "action_weights": agent.action_weights or {},
            }
        )

    df = pd.DataFrame(rows, columns=AGENT_GENETICS_COLUMNS)
    logger.info("build_agent_genetics_dataframe: built frame with %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Evolution-experiment-backed population accessor
# ---------------------------------------------------------------------------

#: Columns produced by :func:`build_evolution_experiment_dataframe`.
EVOLUTION_GENETICS_COLUMNS = [
    "candidate_id",
    "generation",
    "fitness",
    "parent_ids",
    "chromosome_values",
]


def build_evolution_experiment_dataframe(result: "EvolutionExperimentResult") -> pd.DataFrame:
    """Build a normalized genetics DataFrame from an
    :class:`~farm.runners.evolution_experiment.EvolutionExperimentResult`.

    Each row represents one evaluated candidate across all generations.

    Parameters
    ----------
    result:
        A completed evolution-experiment result object.

    Returns
    -------
    pd.DataFrame
        One row per evaluated candidate with columns defined in
        :data:`EVOLUTION_GENETICS_COLUMNS`.  Returns an empty DataFrame when
        *result* contains no evaluations.
    """
    evaluations = result.evaluations
    if not evaluations:
        logger.info("build_evolution_experiment_dataframe: no evaluations in result")
        return pd.DataFrame(columns=EVOLUTION_GENETICS_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for ev in evaluations:
        rows.append(
            {
                "candidate_id": ev.candidate_id,
                "generation": ev.generation,
                "fitness": ev.fitness,
                "parent_ids": list(ev.parent_ids),
                "chromosome_values": dict(ev.chromosome_values),
            }
        )

    df = pd.DataFrame(rows, columns=EVOLUTION_GENETICS_COLUMNS)
    logger.info(
        "build_evolution_experiment_dataframe: built frame with %d rows across %d generation(s)",
        len(df),
        df["generation"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Diversity metric computation
# ---------------------------------------------------------------------------


def compute_continuous_locus_diversity(
    values: Sequence[float],
    locus_name: str,
    bounds: Optional[Tuple[float, float]] = None,
    entropy_bins: int = 10,
    compute_entropy: bool = False,
) -> ContinuousLocusDiversity:
    """Compute diversity metrics for a single continuous locus.

    Parameters
    ----------
    values:
        Per-individual values for this locus.  Must be non-empty.
    locus_name:
        Name used in the returned :class:`ContinuousLocusDiversity` object.
    bounds:
        Optional ``(min_value, max_value)`` defining the full allowed range.
        Required for :attr:`~ContinuousLocusDiversity.normalized_variance`
        and :attr:`~ContinuousLocusDiversity.range_occupancy` to be finite.
    entropy_bins:
        Number of equal-width histogram bins used when computing optional
        Shannon entropy.  Ignored when ``compute_entropy=False``.
    compute_entropy:
        When ``True``, compute the histogram-based Shannon entropy and
        populate :attr:`~ContinuousLocusDiversity.shannon_entropy`.

    Returns
    -------
    ContinuousLocusDiversity

    Raises
    ------
    ValueError
        When ``values`` is empty.

    Notes
    -----
    * Population variance (``ddof=0``) is used throughout.
    * Histogram bins span ``[bounds[0], bounds[1]]`` when bounds are given,
      otherwise span the observed data range.
    * When bounds are provided and entropy is requested, out-of-range values
      are clipped to the nearest bound before histogramming so all individuals
      contribute to entropy.
    * Zero-count bins are excluded from entropy (convention: ``0 ln 0 = 0``).
    """
    if bounds is not None and bounds[0] >= bounds[1]:
        raise ValueError(
            f"compute_continuous_locus_diversity: bounds must satisfy bounds[0] < bounds[1], "
            f"got {bounds!r} for locus {locus_name!r}"
        )
    if compute_entropy and entropy_bins < 1:
        raise ValueError(
            f"compute_continuous_locus_diversity: entropy_bins must be >= 1, "
            f"got {entropy_bins!r} for locus {locus_name!r}"
        )

    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        raise ValueError(
            f"compute_continuous_locus_diversity: no values for locus {locus_name!r}"
        )

    n = int(len(arr))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))
    var_val = float(np.var(arr, ddof=0))

    # Normalized variance: V̂ = Var(x) / span²
    if bounds is not None:
        span = bounds[1] - bounds[0]
        normalized_variance = var_val / (span ** 2) if span > 0.0 else float("nan")
    else:
        normalized_variance = float("nan")

    # Coefficient of variation: CV = σ / |μ|
    if abs(mean_val) > 0.0:
        coefficient_of_variation = std_val / abs(mean_val)
    else:
        coefficient_of_variation = float("nan")

    # Range occupancy: R = (max(x) - min(x)) / span
    observed_range = float(np.max(arr) - np.min(arr))
    if bounds is not None:
        span = bounds[1] - bounds[0]
        range_occupancy = min(observed_range / span, 1.0) if span > 0.0 else 0.0
    else:
        range_occupancy = float("nan")

    # Optional histogram-based Shannon entropy: H = -Σ pᵢ ln(pᵢ)
    shannon_entropy: Optional[float] = None
    if compute_entropy:
        if bounds is not None:
            lower, upper = bounds
            below_bounds = int(np.sum(arr < lower))
            above_bounds = int(np.sum(arr > upper))
            if below_bounds or above_bounds:
                logger.warning(
                    "compute_continuous_locus_diversity: clipped %d value(s) for locus %r to fit bounds %r "
                    "(below=%d, above=%d)",
                    below_bounds + above_bounds,
                    locus_name,
                    bounds,
                    below_bounds,
                    above_bounds,
                )
            arr_for_entropy = np.clip(arr, lower, upper)
            bin_edges = np.linspace(bounds[0], bounds[1], entropy_bins + 1)
            counts, _ = np.histogram(arr_for_entropy, bins=bin_edges)
        else:
            counts, _ = np.histogram(arr, bins=entropy_bins)
        total = int(counts.sum())
        if total > 0:
            probs = counts[counts > 0] / total
            shannon_entropy = float(-np.sum(probs * np.log(probs)))
        else:
            shannon_entropy = 0.0

    return ContinuousLocusDiversity(
        locus_name=locus_name,
        n_individuals=n,
        mean=mean_val,
        std=std_val,
        normalized_variance=normalized_variance,
        coefficient_of_variation=coefficient_of_variation,
        range_occupancy=range_occupancy,
        shannon_entropy=shannon_entropy,
    )


def compute_categorical_locus_diversity(
    weight_vectors: Sequence[Mapping[str, float]],
    locus_name: str = "action_weights",
) -> CategoricalLocusDiversity:
    """Compute diversity metrics for a categorical/weighted locus.

    Treats the population-mean action weights as allele frequencies and
    applies standard population-genetics diversity formulas.

    Parameters
    ----------
    weight_vectors:
        Per-individual weight dicts ``{action_name: weight}``.
        Must be non-empty; individuals with all-zero weights are included
        (their zero contributions do not affect population-mean frequencies).
    locus_name:
        Name used in the returned :class:`CategoricalLocusDiversity` object.

    Returns
    -------
    CategoricalLocusDiversity

    Raises
    ------
    ValueError
        When ``weight_vectors`` is empty.

    Notes
    -----
    * Population-mean frequencies are renormalized to sum to 1 to guard
      against floating-point rounding errors.
    * When all mean weights are zero (degenerate input), a uniform
      distribution over the observed categories is assumed.
    * Zero-frequency categories contribute 0 to entropy (convention:
      ``0 ln 0 = 0``).
    """
    if not weight_vectors:
        raise ValueError(
            f"compute_categorical_locus_diversity: no weight vectors for locus {locus_name!r}"
        )

    n = len(weight_vectors)

    # Collect union of all action names
    all_actions: List[str] = sorted(
        {action for wv in weight_vectors for action in wv}
    )

    if not all_actions:
        raise ValueError(
            f"compute_categorical_locus_diversity: weight_vectors contain no categories "
            f"for locus {locus_name!r} (all dicts are empty)"
        )

    # Compute mean weight per action
    mean_weights: Dict[str, float] = {
        action: sum(float(wv.get(action, 0.0)) for wv in weight_vectors) / n
        for action in all_actions
    }

    # Renormalize to guard against rounding
    total_weight = sum(mean_weights.values())
    if total_weight > 0.0:
        allele_frequencies: Dict[str, float] = {
            a: w / total_weight for a, w in mean_weights.items()
        }
    else:
        # Degenerate: all zeros – fall back to uniform distribution
        k = len(all_actions)
        allele_frequencies = {a: 1.0 / k for a in all_actions} if k > 0 else {}

    freqs = list(allele_frequencies.values())

    # Shannon entropy: H = -Σᵢ pᵢ ln(pᵢ)
    shannon_entropy = -sum(p * math.log(p) for p in freqs if p > 0.0)

    # Simpson index: D = Σᵢ pᵢ²
    simpson_index = sum(p ** 2 for p in freqs)

    # Expected heterozygosity: He = 1 - Σᵢ pᵢ²
    expected_heterozygosity = 1.0 - simpson_index

    return CategoricalLocusDiversity(
        locus_name=locus_name,
        n_individuals=n,
        allele_frequencies=allele_frequencies,
        shannon_entropy=shannon_entropy,
        simpson_index=simpson_index,
        expected_heterozygosity=expected_heterozygosity,
    )


def compute_population_diversity(
    df: pd.DataFrame,
    gene_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    entropy_bins: int = 10,
    compute_continuous_entropy: bool = False,
) -> PopulationDiversitySummary:
    """Compute genotypic diversity metrics across a population snapshot.

    Accepts both DB-backed DataFrames (with an ``action_weights`` column
    produced by :func:`build_agent_genetics_dataframe`) and
    evolution-experiment DataFrames (with a ``chromosome_values`` column
    produced by :func:`build_evolution_experiment_dataframe`).

    Parameters
    ----------
    df:
        Population snapshot DataFrame.  Recognised column sets:

        - ``action_weights``: column of ``{action: weight}`` dicts →
          categorical diversity via :func:`compute_categorical_locus_diversity`.
        - ``chromosome_values``: column of ``{gene: value}`` dicts →
          continuous diversity via :func:`compute_continuous_locus_diversity`.

        Both columns can be present simultaneously.
    gene_bounds:
        Optional mapping from gene name to ``(min_value, max_value)`` used
        to compute :attr:`~ContinuousLocusDiversity.normalized_variance` and
        :attr:`~ContinuousLocusDiversity.range_occupancy`.
    entropy_bins:
        Number of histogram bins for optional continuous locus entropy.
    compute_continuous_entropy:
        When ``True``, compute histogram Shannon entropy for each continuous
        locus and include it in :attr:`~PopulationDiversitySummary.mean_shannon_entropy`.

    Returns
    -------
    PopulationDiversitySummary
        Empty (zero individuals, all ``nan`` summaries) when *df* is empty.
    """
    n = len(df)

    if n == 0:
        return PopulationDiversitySummary(n_individuals=0)

    bounds_map: Mapping[str, Tuple[float, float]] = gene_bounds or {}
    continuous_loci: Dict[str, ContinuousLocusDiversity] = {}
    categorical_loci: Dict[str, CategoricalLocusDiversity] = {}

    # --- Continuous loci: chromosome_values column ---
    if "chromosome_values" in df.columns:
        all_genes: set = set()
        for row in df["chromosome_values"]:
            if isinstance(row, dict):
                all_genes.update(row.keys())
        for gene in sorted(all_genes):
            values: List[float] = []
            for row in df["chromosome_values"]:
                if not isinstance(row, dict) or gene not in row:
                    continue
                raw_value = row[gene]
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric_value):
                    values.append(numeric_value)
            if values:
                locus = compute_continuous_locus_diversity(
                    values,
                    locus_name=gene,
                    bounds=bounds_map.get(gene),
                    entropy_bins=entropy_bins,
                    compute_entropy=compute_continuous_entropy,
                )
                continuous_loci[gene] = locus

    # --- Categorical loci: action_weights column ---
    if "action_weights" in df.columns:
        weight_vectors = [
            row
            for row in df["action_weights"]
            if isinstance(row, dict) and len(row) > 0
        ]
        if weight_vectors:
            cat_locus = compute_categorical_locus_diversity(
                weight_vectors, locus_name="action_weights"
            )
            categorical_loci["action_weights"] = cat_locus

    # --- Population-level summaries ---
    norm_vars = [
        locus.normalized_variance
        for locus in continuous_loci.values()
        if not math.isnan(locus.normalized_variance)
    ]
    mean_normalized_variance = float(np.mean(norm_vars)) if norm_vars else float("nan")

    cvs = [
        locus.coefficient_of_variation
        for locus in continuous_loci.values()
        if not math.isnan(locus.coefficient_of_variation)
    ]
    mean_coefficient_of_variation = float(np.mean(cvs)) if cvs else float("nan")

    # Mean Shannon entropy aggregates continuous (when requested) + categorical
    entropies: List[float] = []
    for locus in continuous_loci.values():
        if locus.shannon_entropy is not None:
            entropies.append(locus.shannon_entropy)
    for locus in categorical_loci.values():
        entropies.append(locus.shannon_entropy)
    mean_shannon_entropy = float(np.mean(entropies)) if entropies else float("nan")

    hets = [locus.expected_heterozygosity for locus in categorical_loci.values()]
    mean_heterozygosity = float(np.mean(hets)) if hets else float("nan")

    return PopulationDiversitySummary(
        n_individuals=n,
        continuous_loci=continuous_loci,
        categorical_loci=categorical_loci,
        mean_normalized_variance=mean_normalized_variance,
        mean_coefficient_of_variation=mean_coefficient_of_variation,
        mean_shannon_entropy=mean_shannon_entropy,
        mean_heterozygosity=mean_heterozygosity,
    )


def compute_evolution_diversity_timeseries(
    result: "EvolutionExperimentResult",
    gene_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    entropy_bins: int = 10,
    compute_continuous_entropy: bool = False,
) -> List[Dict[str, Any]]:
    """Compute per-generation diversity metrics from an EvolutionExperimentResult.

    Groups candidate evaluations by generation and applies
    :func:`compute_population_diversity` to each group.
    When per-candidate evaluations are not available, falls back to
    ``result.generation_summaries`` and derives a reduced
    :class:`PopulationDiversitySummary` from each summary's
    ``gene_statistics``.

    Parameters
    ----------
    result:
        A completed :class:`~farm.runners.evolution_experiment.EvolutionExperimentResult`.
    gene_bounds:
        Optional mapping from gene name to ``(min_value, max_value)`` for
        normalizing continuous diversity metrics.
    entropy_bins:
        Number of histogram bins for optional continuous locus entropy.
    compute_continuous_entropy:
        When ``True``, compute histogram entropy for continuous loci.

    Notes
    -----
    The fallback ``generation_summaries`` path is intentionally limited because
    generation summaries store aggregate gene statistics, not per-candidate
    gene values:

    * Continuous locus metrics available from summary aggregates:
      ``mean``, ``std``, ``normalized_variance`` (when bounds are provided),
      and ``coefficient_of_variation``.
    * Metrics that require per-candidate distributions are unavailable in the
      fallback path and returned as ``nan``/empty:
      ``range_occupancy``, Shannon entropy, categorical diversity, and
      heterozygosity.
    * ``n_individuals`` is unknown in this path and set to ``0``.

    Returns
    -------
    list of dict
        Each element is ``{"generation": int, "diversity": PopulationDiversitySummary}``,
        sorted by generation in ascending order.
        Returns an empty list when *result* contains neither evaluations nor
        generation summaries.
    """
    if result.evaluations:
        df = build_evolution_experiment_dataframe(result)
        rows: List[Dict[str, Any]] = []
        for gen, group in df.groupby("generation"):
            summary = compute_population_diversity(
                group.reset_index(drop=True),
                gene_bounds=gene_bounds,
                entropy_bins=entropy_bins,
                compute_continuous_entropy=compute_continuous_entropy,
            )
            rows.append({"generation": int(gen), "diversity": summary})

        rows.sort(key=lambda r: r["generation"])
        logger.info(
            "compute_evolution_diversity_timeseries: computed diversity for %d generation(s) from evaluations",
            len(rows),
        )
        return rows

    if not result.generation_summaries:
        return []

    rows = []
    for generation_summary in result.generation_summaries:
        diversity = _build_summary_diversity_from_generation_summary(
            generation_summary,
            gene_bounds=gene_bounds,
        )
        rows.append({"generation": int(generation_summary.generation), "diversity": diversity})

    rows.sort(key=lambda r: r["generation"])
    logger.info(
        "compute_evolution_diversity_timeseries: computed diversity for %d generation(s) from generation_summaries",
        len(rows),
    )
    return rows


# ---------------------------------------------------------------------------
# Allele-frequency tracking and selection-pressure detection
# ---------------------------------------------------------------------------

#: Sentinel allele name for the mean of a continuous locus.
ALLELE_MEAN = "__mean__"

#: Sentinel allele name for the variance of a continuous locus.
ALLELE_VARIANCE = "__variance__"

#: Columns produced by :func:`compute_allele_frequency_timeseries`.
ALLELE_FREQUENCY_COLUMNS = [
    "generation",
    "locus",
    "locus_type",
    "allele",
    "frequency",
    "n_individuals",
]

#: Columns produced by :func:`compute_selection_pressure_summary`.
SELECTION_PRESSURE_COLUMNS = [
    "locus",
    "allele",
    "locus_type",
    "n_generations",
    "mean_delta_frequency",
    "cumulative_shift",
    "z_score",
    "regression_slope",
    "effect_size",
    "is_under_selection",
    "collapse_detected",
]


def compute_allele_frequency_timeseries(
    df: pd.DataFrame,
    gene_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,  # noqa: ARG001
) -> pd.DataFrame:
    """Compute per-(generation, locus) allele-frequency / moment trajectories.

    For **categorical loci** (``action_weights`` column): tracks the
    population-mean normalized weight of each action per generation as an
    allele frequency.

    For **continuous loci** (``chromosome_values`` column): tracks two
    distribution moments per generation as named synthetic alleles:

    * :data:`ALLELE_MEAN` (``"__mean__"``) – population mean of the gene value.
    * :data:`ALLELE_VARIANCE` (``"__variance__"``) – population variance
      (ddof=0) of the gene value.

    Parameters
    ----------
    df:
        DataFrame with a ``generation`` column and at least one of:

        * ``chromosome_values``: column of ``{gene_name: float}`` dicts
          (e.g. from :func:`build_evolution_experiment_dataframe`).
        * ``action_weights``: column of ``{action_name: weight}`` dicts
          (e.g. from :func:`build_agent_genetics_dataframe`).

        Rows with a non-numeric or ``NaN`` ``generation`` value are silently
        skipped.
    gene_bounds:
        Accepted for API symmetry with other ``compute_*`` helpers; not
        used in the current implementation.

    Returns
    -------
    pd.DataFrame
        Tidy frame with columns defined in :data:`ALLELE_FREQUENCY_COLUMNS`:

        * ``generation`` (int)
        * ``locus`` (str) – gene name, or ``"action_weights"``
        * ``locus_type`` (str) – ``"continuous"`` or ``"categorical"``
        * ``allele`` (str) – action name, ``"__mean__"``, or
          ``"__variance__"``
        * ``frequency`` (float) – allele frequency in ``[0, 1]`` for
          categorical; population mean / variance for continuous
        * ``n_individuals`` (int) – number of valid observations in this
          generation for this locus

        Returns an empty DataFrame (with correct columns) when *df* is
        empty or contains no recognised locus columns.

    Notes
    -----
    * Rows are sorted by ``(locus, allele, generation)`` in ascending order.
    * Rows with non-finite gene values are excluded from continuous-locus
      statistics (``NaN``, ``±inf``).
    * For categorical loci, allele frequencies are renormalized to sum to 1
      per generation (guarding against floating-point rounding errors).
    """
    if df.empty or "generation" not in df.columns:
        return pd.DataFrame(columns=ALLELE_FREQUENCY_COLUMNS)

    has_continuous = "chromosome_values" in df.columns
    has_categorical = "action_weights" in df.columns

    if not has_continuous and not has_categorical:
        return pd.DataFrame(columns=ALLELE_FREQUENCY_COLUMNS)

    rows: List[Dict[str, Any]] = []

    for gen, group in df.groupby("generation"):
        try:
            gen_int = int(gen)
        except (TypeError, ValueError):
            continue

        # --- Continuous loci: chromosome_values ---
        if has_continuous:
            all_genes: set = set()
            for cv in group["chromosome_values"]:
                if isinstance(cv, dict):
                    all_genes.update(cv.keys())
            for gene in sorted(all_genes):
                values: List[float] = []
                for cv in group["chromosome_values"]:
                    if not isinstance(cv, dict) or gene not in cv:
                        continue
                    raw = cv[gene]
                    try:
                        v = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(v):
                        values.append(v)
                if not values:
                    continue
                arr = np.array(values, dtype=float)
                mean_val = float(np.mean(arr))
                var_val = float(np.var(arr, ddof=0))
                n = len(values)
                rows.append(
                    {
                        "generation": gen_int,
                        "locus": gene,
                        "locus_type": "continuous",
                        "allele": ALLELE_MEAN,
                        "frequency": mean_val,
                        "n_individuals": n,
                    }
                )
                rows.append(
                    {
                        "generation": gen_int,
                        "locus": gene,
                        "locus_type": "continuous",
                        "allele": ALLELE_VARIANCE,
                        "frequency": var_val,
                        "n_individuals": n,
                    }
                )

        # --- Categorical loci: action_weights ---
        if has_categorical:
            weight_vectors = [
                wv
                for wv in group["action_weights"]
                if isinstance(wv, dict) and len(wv) > 0
            ]
            if weight_vectors:
                n_cat = len(weight_vectors)
                all_actions = sorted({a for wv in weight_vectors for a in wv})
                mean_weights = {
                    a: sum(float(wv.get(a, 0.0)) for wv in weight_vectors) / n_cat
                    for a in all_actions
                }
                total = sum(mean_weights.values())
                if total > 0:
                    freqs: Dict[str, float] = {a: w / total for a, w in mean_weights.items()}
                else:
                    k = len(all_actions)
                    freqs = {a: 1.0 / k for a in all_actions} if k > 0 else {}
                for action, freq in freqs.items():
                    rows.append(
                        {
                            "generation": gen_int,
                            "locus": "action_weights",
                            "locus_type": "categorical",
                            "allele": action,
                            "frequency": freq,
                            "n_individuals": n_cat,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=ALLELE_FREQUENCY_COLUMNS)

    result_df = pd.DataFrame(rows, columns=ALLELE_FREQUENCY_COLUMNS)
    result_df = result_df.sort_values(["locus", "allele", "generation"]).reset_index(drop=True)
    logger.info(
        "compute_allele_frequency_timeseries: %d rows across %d generation(s)",
        len(result_df),
        result_df["generation"].nunique(),
    )
    return result_df


def compute_selection_pressure_summary(
    freq_df: pd.DataFrame,
    pop_size: Optional[int] = None,
    significance_threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute per-(locus, allele) selection-pressure metrics from allele-frequency trajectories.

    Detects (locus, allele) pairs with statistically significant directional
    changes in allele frequency (or gene-value moment) across generations,
    relative to a neutral-drift baseline.

    Parameters
    ----------
    freq_df:
        Output of :func:`compute_allele_frequency_timeseries`.  The DataFrame
        must contain ``generation``, ``locus``, ``allele``, ``frequency``, and
        ``locus_type`` columns.  Empty DataFrames are handled gracefully.
    pop_size:
        Effective population size *N_e*.  When provided for **categorical**
        loci, the Wright-Fisher binomial variance ``p̄(1−p̄)/N_e`` is used as
        the per-generation drift variance.  For **continuous** loci (and for
        categorical loci when *pop_size* is ``None``), the empirical standard
        deviation of generation-to-generation Δ values is used as the drift
        baseline.
    significance_threshold:
        Absolute z-score above which a (locus, allele) pair is flagged with
        ``is_under_selection = True``.  Default ``2.0`` (≈ 95th percentile
        under a standard normal).

    Returns
    -------
    pd.DataFrame
        One row per (locus, allele) with columns defined in
        :data:`SELECTION_PRESSURE_COLUMNS`:

        * ``locus`` (str)
        * ``allele`` (str)
        * ``locus_type`` (str) – ``"continuous"`` or ``"categorical"``
        * ``n_generations`` (int) – number of generation observations
        * ``mean_delta_frequency`` (float) – mean Δ between consecutive
          generation pairs; ``nan`` when fewer than 2 observations
        * ``cumulative_shift`` (float) – frequency at last generation minus
          frequency at first generation; ``nan`` when fewer than 2
          observations
        * ``z_score`` (float) – mean Δ divided by its standard error under
          the drift baseline; ``nan`` when fewer than 2 generation pairs are
          available or the drift variance is zero
        * ``regression_slope`` (float) – OLS slope of frequency vs.
          generation index; ``nan`` when fewer than 2 observations or when
          all generation indices are identical
        * ``effect_size`` (float) – ``|cumulative_shift|``; ``nan`` when
          ``cumulative_shift`` is ``nan``
        * ``is_under_selection`` (bool) – ``True`` when
          ``|z_score| ≥ significance_threshold``
        * ``collapse_detected`` (bool) – ``True`` when the (locus, allele)
          pair shows fixation signs: for the ``"__variance__"`` allele of
          a continuous locus, the final-generation variance is near zero
          (< 1e-12); for a categorical allele, the final-generation
          frequency is ≥ 0.99

        Returns an empty DataFrame (with correct columns) when *freq_df* is
        empty or missing required columns.

    Notes
    -----
    **Drift baseline assumptions**

    For categorical loci with *pop_size* provided::

        σ_drift = sqrt(p̄(1−p̄) / N_e)

    where *p̄* is the time-mean allele frequency.  This is the standard
    Wright-Fisher binomial model for frequency change under neutral drift in a
    finite population of size *N_e*.

    For continuous loci or when *pop_size* is ``None``::

        σ_drift = std(Δ_t)   (empirical, ddof=1)

    The z-score is then::

        z = mean(Δ_t) / (σ_drift / sqrt(T))

    where *T* is the number of consecutive generation pairs.  Under neutral
    drift this statistic is approximately standard-normal for large *T*.

    **Boundary-collapse detection**

    A locus-allele pair is flagged (``collapse_detected = True``) when the
    final generation shows evidence of fixation:

    * Continuous ``"__variance__"`` allele: variance < 1e-12 (all individuals
      share the same gene value, consistent with boundary collapse).
    * Categorical allele: frequency ≥ 0.99 (near-fixation of that allele).
    """
    required = {"generation", "locus", "allele", "frequency", "locus_type"}
    if freq_df.empty or not required.issubset(freq_df.columns):
        return pd.DataFrame(columns=SELECTION_PRESSURE_COLUMNS)

    summary_rows: List[Dict[str, Any]] = []

    for (locus, allele), group in freq_df.groupby(["locus", "allele"], sort=False):
        group_sorted = group.sort_values("generation")
        freqs = group_sorted["frequency"].tolist()
        gens = group_sorted["generation"].tolist()
        locus_type = str(group_sorted["locus_type"].iloc[0])
        n_gen = len(freqs)

        cumulative_shift: float = freqs[-1] - freqs[0] if n_gen >= 2 else float("nan")

        deltas = [freqs[i + 1] - freqs[i] for i in range(n_gen - 1)]
        T = len(deltas)
        mean_delta: float = float(np.mean(deltas)) if T > 0 else float("nan")

        # --- Drift baseline and z-score ---
        z_score = float("nan")
        if T >= 2:
            if locus_type == "categorical" and pop_size is not None and pop_size > 0:
                # Wright-Fisher: σ_drift = sqrt(p̄(1−p̄) / N_e)
                p_mean = float(np.mean(freqs))
                drift_var = p_mean * (1.0 - p_mean) / pop_size
                drift_std = math.sqrt(drift_var) if drift_var > 0 else 0.0
            else:
                # Empirical drift baseline: std of Δ_t
                drift_std = float(np.std(deltas, ddof=1))

            if drift_std > 0.0:
                se = drift_std / math.sqrt(T)
                z_score = mean_delta / se
            # When drift_std == 0 (no variation in Δ) the z-score is indeterminate.

        # --- OLS regression slope ---
        regression_slope = float("nan")
        if n_gen >= 2:
            gens_arr = np.array(gens, dtype=float)
            freqs_arr = np.array(freqs, dtype=float)
            gen_mean = float(np.mean(gens_arr))
            freq_mean = float(np.mean(freqs_arr))
            cov = float(np.mean((gens_arr - gen_mean) * (freqs_arr - freq_mean)))
            var_gen = float(np.var(gens_arr, ddof=0))
            if var_gen > 0.0:
                regression_slope = cov / var_gen

        effect_size = abs(cumulative_shift) if not math.isnan(cumulative_shift) else float("nan")
        is_under_selection = (not math.isnan(z_score)) and abs(z_score) >= significance_threshold

        # --- Boundary/collapse detection ---
        collapse_detected = False
        if locus_type == "continuous" and allele == ALLELE_VARIANCE:
            if n_gen >= 1 and freqs[-1] < 1e-12:
                collapse_detected = True
        elif locus_type == "categorical":
            if n_gen >= 1 and freqs[-1] >= 0.99:
                collapse_detected = True

        summary_rows.append(
            {
                "locus": locus,
                "allele": allele,
                "locus_type": locus_type,
                "n_generations": n_gen,
                "mean_delta_frequency": mean_delta,
                "cumulative_shift": cumulative_shift,
                "z_score": z_score,
                "regression_slope": regression_slope,
                "effect_size": effect_size,
                "is_under_selection": is_under_selection,
                "collapse_detected": collapse_detected,
            }
        )

    if not summary_rows:
        return pd.DataFrame(columns=SELECTION_PRESSURE_COLUMNS)

    result_df = pd.DataFrame(summary_rows, columns=SELECTION_PRESSURE_COLUMNS)
    result_df = result_df.sort_values(["locus", "allele"]).reset_index(drop=True)

    n_selected = int(result_df["is_under_selection"].sum())
    n_collapsed = int(result_df["collapse_detected"].sum())
    logger.info(
        "compute_selection_pressure_summary: %d locus-allele pairs; "
        "%d under selection; %d boundary collapse(s)",
        len(result_df),
        n_selected,
        n_collapsed,
    )
    return result_df


def _build_summary_diversity_from_generation_summary(
    generation_summary: Any,
    gene_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> PopulationDiversitySummary:
    """Build reduced diversity metrics from an EvolutionGenerationSummary.

    This helper is used as a fallback when an evolution result has no
    per-candidate evaluations. It relies on aggregate per-gene statistics
    provided by ``generation_summary.gene_statistics``.
    """
    bounds_map: Mapping[str, Tuple[float, float]] = gene_bounds or {}
    continuous_loci: Dict[str, ContinuousLocusDiversity] = {}

    gene_statistics = generation_summary.gene_statistics or {}
    for gene, stats in sorted(gene_statistics.items()):
        if not isinstance(stats, dict):
            continue
        raw_mean = stats.get("mean")
        raw_std = stats.get("std")
        try:
            mean_val = float(raw_mean)
            std_val = float(raw_std)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(mean_val) and math.isfinite(std_val)):
            continue

        var_val = std_val ** 2
        bounds = bounds_map.get(gene)
        if bounds is not None and bounds[0] < bounds[1]:
            span = bounds[1] - bounds[0]
            normalized_variance = var_val / (span ** 2) if span > 0.0 else float("nan")
        else:
            normalized_variance = float("nan")

        coefficient_of_variation = std_val / abs(mean_val) if abs(mean_val) > 0.0 else float("nan")

        continuous_loci[gene] = ContinuousLocusDiversity(
            locus_name=gene,
            n_individuals=0,
            mean=mean_val,
            std=std_val,
            normalized_variance=normalized_variance,
            coefficient_of_variation=coefficient_of_variation,
            range_occupancy=float("nan"),
            shannon_entropy=None,
        )

    norm_vars = [
        locus.normalized_variance
        for locus in continuous_loci.values()
        if not math.isnan(locus.normalized_variance)
    ]
    mean_normalized_variance = float(np.mean(norm_vars)) if norm_vars else float("nan")

    cvs = [
        locus.coefficient_of_variation
        for locus in continuous_loci.values()
        if not math.isnan(locus.coefficient_of_variation)
    ]
    mean_coefficient_of_variation = float(np.mean(cvs)) if cvs else float("nan")

    return PopulationDiversitySummary(
        n_individuals=0,
        continuous_loci=continuous_loci,
        categorical_loci={},
        mean_normalized_variance=mean_normalized_variance,
        mean_coefficient_of_variation=mean_coefficient_of_variation,
        mean_shannon_entropy=float("nan"),
        mean_heterozygosity=float("nan"),
    )
