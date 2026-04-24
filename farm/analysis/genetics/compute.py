"""
Genetics Analysis Computation

Core computation functions for the genetics analysis module, including the
shared ``parse_parent_ids`` helper, population-level accessors for both
simulation-database and evolution-experiment data sources, genotypic
diversity metrics (heterozygosity, Shannon entropy, per-locus stats),
allele-frequency trajectory tracking with selection-pressure detection,
fitness landscape analysis (single-locus correlations, pairwise epistasis),
Wright-Fisher neutral-drift simulation, and gene-flow / F_ST differentiation
across configurable subpopulations.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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

        Rows with a non-numeric, non-finite, or non-integer ``generation``
        value are skipped.
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

    skipped_generation_rows = 0

    for gen, group in df.groupby("generation"):
        try:
            gen_float = float(gen)
        except (TypeError, ValueError):
            skipped_generation_rows += len(group)
            continue
        if not math.isfinite(gen_float) or not gen_float.is_integer():
            skipped_generation_rows += len(group)
            continue
        gen_int = int(gen_float)

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
            weight_vectors: List[Dict[str, float]] = []
            skipped_weight_values = 0
            for raw_wv in group["action_weights"]:
                if not isinstance(raw_wv, dict) or len(raw_wv) == 0:
                    continue
                sanitized: Dict[str, float] = {}
                for action, raw_weight in raw_wv.items():
                    try:
                        numeric_weight = float(raw_weight)
                    except (TypeError, ValueError):
                        skipped_weight_values += 1
                        continue
                    if not math.isfinite(numeric_weight):
                        skipped_weight_values += 1
                        continue
                    sanitized[action] = numeric_weight
                if sanitized:
                    weight_vectors.append(sanitized)
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
            if skipped_weight_values:
                logger.warning(
                    "compute_allele_frequency_timeseries: skipped %d non-finite/non-numeric action_weights value(s) in generation %d",
                    skipped_weight_values,
                    gen_int,
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
    if skipped_generation_rows:
        logger.warning(
            "compute_allele_frequency_timeseries: skipped %d row(s) with non-integer generation values",
            skipped_generation_rows,
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
        Must be finite and non-negative.

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

    Raises
    ------
    ValueError
        If ``significance_threshold`` is non-finite or negative.

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
    if not math.isfinite(significance_threshold) or significance_threshold < 0.0:
        raise ValueError(
            "compute_selection_pressure_summary: significance_threshold must be finite and >= 0"
        )

    summary_rows: List[Dict[str, Any]] = []

    for (locus, allele, locus_type), group in freq_df.groupby(
        ["locus", "allele", "locus_type"], sort=False
    ):
        group_sorted = group.sort_values("generation")
        freqs = group_sorted["frequency"].tolist()
        gens = group_sorted["generation"].tolist()
        locus_type = str(locus_type)
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


# ---------------------------------------------------------------------------
# Fitness landscape: single-locus correlations and pairwise epistasis
# ---------------------------------------------------------------------------

#: Columns produced by :func:`compute_fitness_gene_correlations`.
FITNESS_GENE_CORRELATION_COLUMNS = [
    "gene",
    "n_samples",
    "pearson_r",
    "pearson_p",
    "spearman_r",
    "spearman_p",
    "ols_slope",
    "ols_intercept",
    "ols_r2",
    "ols_p",
    "ci_lower",
    "ci_upper",
    "effect_size",
    "bh_rejected",
]

#: Columns produced by :func:`compute_pairwise_epistasis`.
PAIRWISE_EPISTASIS_COLUMNS = [
    "gene_i",
    "gene_j",
    "n_samples",
    "interaction_coef",
    "interaction_p",
    "main_i_coef",
    "main_j_coef",
    "model_r2",
    "bh_rejected",
]


def _extract_gene_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a (candidate × gene) numeric matrix from an evolution DataFrame.

    Reads the ``chromosome_values`` dict column and flattens it into
    individual gene columns.  Rows with missing fitness are dropped.
    Only columns with non-zero variance are kept.

    Parameters
    ----------
    df:
        DataFrame with at least ``fitness`` and ``chromosome_values`` columns.

    Returns
    -------
    pd.DataFrame
        Index-reset numeric frame with ``fitness`` plus one column per gene.
        May be empty if there is insufficient data.
    """
    required = {"fitness", "chromosome_values"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        fitness = row["fitness"]
        chrom = row["chromosome_values"]
        if not isinstance(chrom, dict):
            continue
        try:
            fitness_val = float(fitness)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fitness_val):
            continue
        entry: Dict[str, Any] = {"fitness": fitness_val}
        for gene, val in chrom.items():
            try:
                entry[gene] = float(val)
            except (TypeError, ValueError):
                # Ignore non-numeric gene values; retain other valid loci for this row.
                continue
        rows.append(entry)

    if not rows:
        return pd.DataFrame()

    matrix = pd.DataFrame(rows).dropna(subset=["fitness"]).reset_index(drop=True)

    # Drop zero-variance gene columns (constant across all candidates)
    gene_cols = [c for c in matrix.columns if c != "fitness"]
    stds = matrix[gene_cols].std(ddof=0)
    varying = stds[stds > 0.0].index.tolist()
    return matrix[["fitness"] + varying]


def _bh_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Benjamini–Hochberg multiple-testing correction.

    Parameters
    ----------
    p_values:
        Raw p-values, one per hypothesis.
    alpha:
        False discovery rate threshold.  Default ``0.05``.

    Returns
    -------
    list[bool]
        Boolean mask: ``True`` where the null hypothesis is rejected after
        BH correction (i.e. the finding survives multiple-testing correction).
    """
    m = len(p_values)
    if m == 0:
        return []

    # Pair (p_value, original_index) and sort by p
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # BH step-up procedure: find largest k such that p_(k) <= (k/m) * alpha
    last_rejected_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * alpha:
            last_rejected_rank = rank

    rejected = [False] * m
    for rank, (orig_idx, _) in enumerate(indexed, start=1):
        if rank <= last_rejected_rank:
            rejected[orig_idx] = True

    return rejected


def _validate_fitness_landscape_params(
    alpha: float,
    min_samples: int,
    confidence_level: Optional[float] = None,
    min_gene_variance: Optional[float] = None,
) -> None:
    """Validate common parameters for fitness-landscape computations."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    if min_samples < 2:
        raise ValueError(f"min_samples must be >= 2, got {min_samples}")

    if confidence_level is not None and not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    if min_gene_variance is not None and min_gene_variance < 0.0:
        raise ValueError(
            f"min_gene_variance must be >= 0, got {min_gene_variance}"
        )


def compute_fitness_gene_correlations(
    df: pd.DataFrame,
    alpha: float = 0.05,
    min_samples: int = 10,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute single-locus fitness–gene associations for evolution data.

    For each evolvable gene found in ``chromosome_values``, computes:

    * Pearson and Spearman correlation with fitness (+ p-values)
    * OLS simple linear regression slope, intercept, R², and p-value
    * 95 % (or ``confidence_level``) bootstrap-free CI on OLS slope
      using the *t*-distribution
    * Benjamini–Hochberg multiple-testing correction across all genes

    Results are sorted descending by absolute effect size (``|pearson_r|``).

    Parameters
    ----------
    df:
        DataFrame containing at least ``fitness`` (float) and
        ``chromosome_values`` (dict) columns.  Typically produced by
        :func:`build_evolution_experiment_dataframe`.
    alpha:
        FDR level for Benjamini–Hochberg correction.  Default ``0.05``.
    min_samples:
        Minimum number of valid (fitness, gene) pairs required to compute
        statistics for a gene.  Genes with fewer observations are skipped.
    confidence_level:
        Confidence level for the OLS slope confidence interval (e.g. 0.95
        gives a 95 % CI).  Must be in (0, 1).

    Returns
    -------
    pd.DataFrame
        One row per gene, columns defined in
        :data:`FITNESS_GENE_CORRELATION_COLUMNS`, sorted by descending
        ``effect_size``.  Returns an empty DataFrame (with correct columns)
        when the input is insufficient.

    Notes
    -----
    **Statistical caveats**

    * Candidates within a generation are *not independent* – fitness is
      assigned by the same evaluator and selection acts on the whole cohort.
    * Candidates across generations share ancestry; effect estimates may be
      inflated by drift-selection correlation rather than pure causal gene
      effects.
    * The BH correction operates over genes (columns) and ignores
      the generational non-independence.
    * With small ``n`` (< ~30) the *t*-distribution CI is approximate.
    """
    _validate_fitness_landscape_params(
        alpha=alpha,
        min_samples=min_samples,
        confidence_level=confidence_level,
    )

    matrix = _extract_gene_matrix(df)
    if matrix.empty:
        return pd.DataFrame(columns=FITNESS_GENE_CORRELATION_COLUMNS)

    fitness = matrix["fitness"].values.astype(float)
    gene_cols = [c for c in matrix.columns if c != "fitness"]

    if not gene_cols:
        return pd.DataFrame(columns=FITNESS_GENE_CORRELATION_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for gene in gene_cols:
        gene_vals = matrix[gene].values.astype(float)
        valid_mask = np.isfinite(fitness) & np.isfinite(gene_vals)
        n = int(valid_mask.sum())
        if n < min_samples:
            continue

        gene_valid = gene_vals[valid_mask]
        fitness_valid = fitness[valid_mask]

        # Per-gene masking can reduce variance to zero; skip these degenerate fits.
        if float(np.std(gene_valid, ddof=0)) <= 0.0:
            continue

        # --- Pearson ---
        pr, pp = scipy_stats.pearsonr(gene_valid, fitness_valid)

        # --- Spearman ---
        sr, sp = scipy_stats.spearmanr(gene_valid, fitness_valid)

        # --- OLS simple regression ---
        slope, intercept, r_value, ols_p, std_err = scipy_stats.linregress(
            gene_valid, fitness_valid
        )
        r2 = float(r_value ** 2)

        # CI on slope using t-distribution (df = n - 2)
        if n > 2:
            t_crit = scipy_stats.t.ppf((1 + confidence_level) / 2, df=n - 2)
            ci_lower = float(slope - t_crit * std_err)
            ci_upper = float(slope + t_crit * std_err)
        else:
            ci_lower = float("nan")
            ci_upper = float("nan")

        rows.append(
            {
                "gene": gene,
                "n_samples": int(n),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
                "ols_slope": float(slope),
                "ols_intercept": float(intercept),
                "ols_r2": r2,
                "ols_p": float(ols_p),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "effect_size": abs(float(pr)),
                "bh_rejected": False,  # placeholder – filled below
            }
        )

    if not rows:
        return pd.DataFrame(columns=FITNESS_GENE_CORRELATION_COLUMNS)

    result_df = pd.DataFrame(rows, columns=FITNESS_GENE_CORRELATION_COLUMNS)

    # Benjamini–Hochberg correction on finite OLS p-values
    valid_ols_mask = result_df["ols_p"].notna() & np.isfinite(result_df["ols_p"])
    if valid_ols_mask.any():
        bh_mask = _bh_correction(
            result_df.loc[valid_ols_mask, "ols_p"].tolist(), alpha=alpha
        )
        result_df.loc[valid_ols_mask, "bh_rejected"] = bh_mask

    # Sort by descending effect size
    result_df = result_df.sort_values(
        "effect_size", ascending=False, na_position="last"
    ).reset_index(drop=True)

    logger.info(
        "compute_fitness_gene_correlations: %d gene(s) tested; %d significant after BH correction",
        len(result_df),
        int(result_df["bh_rejected"].sum()),
    )
    return result_df


def compute_pairwise_epistasis(
    df: pd.DataFrame,
    alpha: float = 0.05,
    min_samples: int = 20,
    min_gene_variance: float = 1e-8,
) -> pd.DataFrame:
    """Estimate pairwise epistasis between all evolvable gene pairs.

    For each pair (g_i, g_j) with sufficient variance, fits the linear
    interaction model::

        fitness ~ β₀ + β₁·g_i + β₂·g_j + β₃·(g_i × g_j) + ε

    and reports the interaction coefficient β₃ and its p-value.
    Benjamini–Hochberg correction is applied across all tested pairs.
    Results are sorted by descending absolute interaction coefficient.

    Parameters
    ----------
    df:
        DataFrame with ``fitness`` and ``chromosome_values`` columns.
    alpha:
        FDR level for Benjamini–Hochberg correction.  Default ``0.05``.
    min_samples:
        Minimum number of rows required to attempt the interaction fit for a
        pair.  Default ``20`` (4 parameters need headroom).
    min_gene_variance:
        Pairs where either gene column has variance below this threshold are
        skipped (near-constant genes make the model ill-conditioned).

    Returns
    -------
    pd.DataFrame
        One row per (gene_i, gene_j) pair, columns defined in
        :data:`PAIRWISE_EPISTASIS_COLUMNS`, sorted by descending absolute
        ``interaction_coef``.  Returns an empty DataFrame (with correct
        columns) when the input is insufficient.

    Notes
    -----
    **Statistical caveats**

    * Linear interaction in gene values is a first-order approximation to
      epistasis; non-linear or sign-epistasis is not captured.
    * Generational non-independence inflates effective sample size; treat
      p-values as descriptive rather than strictly inferential.
    * The number of pairs grows as O(k²) in the number of genes; with many
      genes the BH correction becomes conservative.  Consider Holm or other
      corrections for very large gene sets.
    """
    _validate_fitness_landscape_params(
        alpha=alpha,
        min_samples=min_samples,
        min_gene_variance=min_gene_variance,
    )

    matrix = _extract_gene_matrix(df)
    if matrix.empty:
        return pd.DataFrame(columns=PAIRWISE_EPISTASIS_COLUMNS)

    fitness = matrix["fitness"].values.astype(float)
    gene_cols = [c for c in matrix.columns if c != "fitness"]

    if len(gene_cols) < 2:
        return pd.DataFrame(columns=PAIRWISE_EPISTASIS_COLUMNS)

    # Further filter genes with sufficient variance
    varying = [
        g
        for g in gene_cols
        if (
            matrix[g].notna().any()
            and float(matrix[g].dropna().var(ddof=0)) >= min_gene_variance
        )
    ]
    if len(varying) < 2:
        return pd.DataFrame(columns=PAIRWISE_EPISTASIS_COLUMNS)

    rows: List[Dict[str, Any]] = []

    for gene_i, gene_j in combinations(varying, 2):
        gi = matrix[gene_i].values.astype(float)
        gj = matrix[gene_j].values.astype(float)
        valid_mask = np.isfinite(fitness) & np.isfinite(gi) & np.isfinite(gj)
        n = int(valid_mask.sum())

        if n < min_samples:
            continue

        gi = gi[valid_mask]
        gj = gj[valid_mask]
        fitness_valid = fitness[valid_mask]

        interaction = gi * gj

        # Design matrix: intercept, g_i, g_j, g_i*g_j
        X = np.column_stack([np.ones(n), gi, gj, interaction])

        # OLS via numpy least-squares
        try:
            coeffs, residuals_ss, rank, _ = np.linalg.lstsq(X, fitness_valid, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if rank < 4:
            # Collinear – skip this pair
            continue

        beta0, beta_i, beta_j, beta_ij = float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])

        # Compute residual variance and standard errors
        predicted = X @ coeffs
        resid = fitness_valid - predicted
        rss = float(np.dot(resid, resid))
        df_resid = n - 4  # n observations - 4 parameters

        if df_resid < 1:
            continue

        sigma2 = rss / df_resid
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue

        var_coeffs = sigma2 * np.diag(XtX_inv)
        if var_coeffs[3] < 0.0:
            interaction_p = float("nan")
        else:
            se_ij = math.sqrt(float(var_coeffs[3]))
            if se_ij > 0.0:
                t_stat = beta_ij / se_ij
                interaction_p = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=df_resid))
            else:
                interaction_p = float("nan")

        # R² of the full model
        fitness_mean = float(np.mean(fitness_valid))
        fitness_centered = fitness_valid - fitness_mean
        tss = float(np.dot(fitness_centered, fitness_centered))
        r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")

        rows.append(
            {
                "gene_i": gene_i,
                "gene_j": gene_j,
                "n_samples": int(n),
                "interaction_coef": beta_ij,
                "interaction_p": interaction_p,
                "main_i_coef": beta_i,
                "main_j_coef": beta_j,
                "model_r2": r2,
                "bh_rejected": False,  # placeholder
            }
        )

    if not rows:
        return pd.DataFrame(columns=PAIRWISE_EPISTASIS_COLUMNS)

    result_df = pd.DataFrame(rows, columns=PAIRWISE_EPISTASIS_COLUMNS)

    # BH correction on interaction p-values (drop NaN for correction)
    valid_mask = result_df["interaction_p"].notna() & np.isfinite(result_df["interaction_p"])
    valid_ps = result_df.loc[valid_mask, "interaction_p"].tolist()
    bh_valid = _bh_correction(valid_ps, alpha=alpha)
    result_df.loc[valid_mask, "bh_rejected"] = bh_valid

    # Sort primarily by scale-invariant statistical evidence (BH rejection, then
    # raw interaction p-value), using absolute interaction coefficient only as a
    # tie-breaker.  Sorting by raw coefficient magnitude is misleading because
    # coefficients are scale-dependent (e.g. learning_rate spans orders of
    # magnitude while gamma is in [0, 1]).
    result_df = (
        result_df.assign(_abs_interaction_coef=result_df["interaction_coef"].abs())
        .sort_values(
            by=["bh_rejected", "interaction_p", "_abs_interaction_coef"],
            ascending=[False, True, False],
            na_position="last",
        )
        .drop(columns=["_abs_interaction_coef"])
        .reset_index(drop=True)
    )

    logger.info(
        "compute_pairwise_epistasis: %d pair(s) tested; %d significant after BH correction",
        len(result_df),
        int(result_df["bh_rejected"].sum()),
    )
    return result_df


# ---------------------------------------------------------------------------
# Wright-Fisher neutral drift simulator
# ---------------------------------------------------------------------------

#: Columns produced by :func:`simulate_wright_fisher`.
WRIGHT_FISHER_COLUMNS = ["generation", "allele", "frequency"]


def simulate_wright_fisher(
    initial_frequencies: Mapping[str, float],
    n_effective: int,
    n_generations: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate allele-frequency drift under the Wright-Fisher model.

    At each generation the allele counts are drawn from a multinomial
    distribution parameterised by the current frequencies and the effective
    population size ``N_e``.  The simulation is fully neutral — no selection,
    mutation, or migration.  It is intended as a neutral baseline against
    which observed allele-frequency trajectories can be compared.

    Parameters
    ----------
    initial_frequencies:
        Mapping from allele name to its initial frequency.  Values must be
        non-negative and must sum to ``1.0`` within a tolerance of ``1e-6``.
        At least one allele is required.
    n_effective:
        Effective population size *N_e*.  Must be ``>= 1``.
    n_generations:
        Number of generations to simulate beyond generation 0.  Must be
        ``>= 0``.  Passing ``0`` returns only the initial frequencies.
    seed:
        Optional integer seed for the random number generator.  When
        provided, the simulation is fully deterministic.

    Returns
    -------
    pd.DataFrame
        Tidy frame with columns defined in :data:`WRIGHT_FISHER_COLUMNS`:

        * ``generation`` (int) – ``0`` to ``n_generations`` inclusive
        * ``allele`` (str) – allele name from *initial_frequencies*
        * ``frequency`` (float) – allele frequency in ``[0, 1]``

        The returned frame contains
        ``(n_generations + 1) * len(initial_frequencies)`` rows, sorted by
        ``(allele, generation)``.

    Raises
    ------
    ValueError
        If *initial_frequencies* is empty, contains negative values, or
        does not sum to approximately ``1.0``.
        If *n_effective* ``< 1`` or *n_generations* ``< 0``.

    Notes
    -----
    * The expected allele frequency is conserved across generations:
      ``E[p(t)] = p(0)`` for all *t*.
    * The variance of the allele frequency after *t* generations is
      approximately ``p(0)·(1−p(0))/N_e`` per generation under the
      standard diffusion approximation.
    * The simulation uses ``numpy.random.Generator.multinomial``, which
      guarantees reproducibility with the same *seed* using NumPy's
      new-style Generator API.
    """
    if not initial_frequencies:
        raise ValueError("simulate_wright_fisher: initial_frequencies must not be empty")

    alleles = list(initial_frequencies.keys())
    freqs = np.array([float(initial_frequencies[a]) for a in alleles], dtype=float)

    if np.any(freqs < 0.0):
        raise ValueError(
            "simulate_wright_fisher: all initial frequencies must be >= 0"
        )

    total = float(np.sum(freqs))
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"simulate_wright_fisher: initial frequencies must sum to 1.0, got {total}"
        )

    if n_effective < 1:
        raise ValueError(
            f"simulate_wright_fisher: n_effective must be >= 1, got {n_effective}"
        )

    if n_generations < 0:
        raise ValueError(
            f"simulate_wright_fisher: n_generations must be >= 0, got {n_generations}"
        )

    # Normalise to sum exactly to 1.0 (guard against accumulated floating-point error)
    freqs = freqs / freqs.sum()

    rng = np.random.default_rng(seed)

    rows: List[Dict[str, Any]] = []

    # Generation 0: record the initial frequencies
    for allele, freq in zip(alleles, freqs):
        rows.append({"generation": 0, "allele": allele, "frequency": float(freq)})

    current_freqs = freqs.copy()

    # Simulate generations 1 … n_generations
    for gen in range(1, n_generations + 1):
        counts = rng.multinomial(n_effective, current_freqs)
        current_freqs = counts.astype(float) / float(n_effective)
        for allele, freq in zip(alleles, current_freqs):
            rows.append({"generation": gen, "allele": allele, "frequency": float(freq)})

    result_df = (
        pd.DataFrame(rows, columns=WRIGHT_FISHER_COLUMNS)
        .sort_values(["allele", "generation"])
        .reset_index(drop=True)
    )
    logger.info(
        "simulate_wright_fisher: %d generation(s) simulated; N_e=%d; %d allele(s)",
        n_generations,
        n_effective,
        len(alleles),
    )
    return result_df


# ---------------------------------------------------------------------------
# Gene-flow / F_ST differentiation across subpopulations
# ---------------------------------------------------------------------------

#: Columns produced by :func:`compute_fst_pairwise`.
FST_COLUMNS = [
    "locus",
    "locus_type",
    "subpopulation_a",
    "subpopulation_b",
    "n_a",
    "n_b",
    "fst",
]

#: Columns produced by :func:`compute_gene_flow_timeseries`.
GENE_FLOW_COLUMNS = [
    "generation",
    "subpop_definition",
    "locus",
    "locus_type",
    "subpopulation_a",
    "subpopulation_b",
    "fst",
    "n_migrants",
    "migration_data_available",
]

#: Columns produced by :func:`compute_migration_counts`.
MIGRATION_COLUMNS = [
    "agent_id",
    "generation",
    "subpop_definition",
    "offspring_subpopulation",
    "parent_subpopulation",
    "is_migrant",
]


def _fst_continuous_pairwise(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
) -> float:
    """Compute F_ST analog (η²) for a continuous trait between two groups.

    Uses the variance-partitioning formula::

        F_ST = V_B / V_T

    where ``V_T`` is the total (pooled) variance and
    ``V_B = V_T - V_W`` is the between-group variance component
    (ANOVA decomposition: ``SST = SSW + SSB``).

    Returns ``0.0`` when total variance is zero (monomorphic locus).
    Result is clamped to ``[0.0, 1.0]``.
    """
    n_a, n_b = len(vals_a), len(vals_b)
    all_vals = np.concatenate([vals_a, vals_b])
    v_t = float(np.var(all_vals, ddof=0))
    if v_t <= 0.0:
        return 0.0
    var_a = float(np.var(vals_a, ddof=0))
    var_b = float(np.var(vals_b, ddof=0))
    v_w = (n_a * var_a + n_b * var_b) / (n_a + n_b)
    fst = float((v_t - v_w) / v_t)
    return float(np.clip(fst, 0.0, 1.0))


def _fst_categorical_pairwise(
    freq_a: Mapping[str, float],
    freq_b: Mapping[str, float],
    n_a: int,
    n_b: int,
) -> float:
    """Compute classical multi-allele F_ST between two subpopulations.

    Uses the expected-heterozygosity formula::

        F_ST = (H_T - H_S) / H_T

    where ``H_T`` is the expected heterozygosity of the merged population
    and ``H_S`` is the size-weighted mean within-subpopulation expected
    heterozygosity.

    Returns ``0.0`` when ``H_T == 0`` (all alleles fixed identically in both
    subpopulations).  Result is clamped to ``[0.0, 1.0]``.
    """
    all_alleles = sorted(set(freq_a.keys()) | set(freq_b.keys()))
    # Total allele frequencies (size-weighted)
    freq_t = {
        a: (n_a * freq_a.get(a, 0.0) + n_b * freq_b.get(a, 0.0)) / (n_a + n_b)
        for a in all_alleles
    }
    h_t = 1.0 - sum(p * p for p in freq_t.values())
    if h_t <= 0.0:
        return 0.0
    h_a = 1.0 - sum(p * p for p in freq_a.values())
    h_b = 1.0 - sum(p * p for p in freq_b.values())
    h_s = (n_a * h_a + n_b * h_b) / (n_a + n_b)
    fst = float((h_t - h_s) / h_t)
    return float(np.clip(fst, 0.0, 1.0))


def _mean_allele_freqs(
    weight_vectors: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute normalized mean allele frequencies from a list of weight dicts."""
    if not weight_vectors:
        return {}
    all_actions = sorted({a for wv in weight_vectors for a in wv})
    n = len(weight_vectors)
    means: Dict[str, float] = {
        a: sum(float(wv.get(a, 0.0)) for wv in weight_vectors) / n
        for a in all_actions
    }
    total = sum(means.values())
    if total > 0.0:
        return {a: w / total for a, w in means.items()}
    k = len(all_actions)
    return {a: 1.0 / k for a in all_actions} if k > 0 else {}


def compute_fst_pairwise(
    df: pd.DataFrame,
    subpop_col: str = "agent_type",
) -> pd.DataFrame:
    """Compute pairwise F_ST differentiation per locus between subpopulation pairs.

    Analyzes both continuous loci (``chromosome_values`` column) and
    categorical loci (``action_weights`` column) when present.

    **Continuous loci** (hyperparameter genes from ``chromosome_values``):
    Uses the variance-partitioning F_ST analog (η²)::

        F_ST = V_B / V_T

    where ``V_T`` is the total variance and ``V_B = V_T − V_W`` is the
    between-group variance component.

    **Categorical loci** (action weights):
    Uses the heterozygosity-based G_ST estimator (Nei 1973)::

        G_ST = (H_T − H_S) / H_T

    where ``H_T`` and ``H_S`` are the expected heterozygosities of the total
    and sub-populations respectively.

    Parameters
    ----------
    df:
        DataFrame with a *subpop_col* column and at least one of
        ``chromosome_values`` (dict) or ``action_weights`` (dict).
        Typically produced by :func:`build_agent_genetics_dataframe` or
        :func:`build_evolution_experiment_dataframe`.
    subpop_col:
        Name of the column whose distinct values define subpopulations.
        Default ``"agent_type"``.

    Returns
    -------
    pd.DataFrame
        One row per *(locus, subpopulation_a, subpopulation_b)* combination,
        with columns defined in :data:`FST_COLUMNS`:

        * ``locus`` (str)
        * ``locus_type`` (str) – ``"continuous"`` or ``"categorical"``
        * ``subpopulation_a`` (str)
        * ``subpopulation_b`` (str)
        * ``n_a`` (int) – individuals in subpopulation A
        * ``n_b`` (int) – individuals in subpopulation B
        * ``fst`` (float) – F_ST value in ``[0, 1]``

        Returns an empty DataFrame (with correct columns) when the input is
        insufficient (fewer than two non-null subpopulations, or no recognised
        locus column).
    """
    has_continuous = "chromosome_values" in df.columns
    has_categorical = "action_weights" in df.columns

    if df.empty or subpop_col not in df.columns:
        return pd.DataFrame(columns=FST_COLUMNS)

    if not has_continuous and not has_categorical:
        return pd.DataFrame(columns=FST_COLUMNS)

    subpops = sorted(df[subpop_col].dropna().unique())
    if len(subpops) < 2:
        return pd.DataFrame(columns=FST_COLUMNS)

    rows: List[Dict[str, Any]] = []

    # ---- Continuous loci (chromosome_values) ----
    if has_continuous:
        gene_subpop_values: Dict[str, Dict[Any, List[float]]] = {}
        for _, row in df.iterrows():
            subpop = row[subpop_col]
            if pd.isna(subpop):
                continue
            cv = row.get("chromosome_values")
            if not isinstance(cv, dict):
                continue
            for gene, val in cv.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue
                gene_subpop_values.setdefault(gene, {}).setdefault(subpop, []).append(v)

        for gene, subpop_vals in sorted(gene_subpop_values.items()):
            eligible = [sp for sp in subpops if sp in subpop_vals and len(subpop_vals[sp]) >= 1]
            for sp_a, sp_b in combinations(eligible, 2):
                vals_a = np.array(subpop_vals[sp_a], dtype=float)
                vals_b = np.array(subpop_vals[sp_b], dtype=float)
                rows.append(
                    {
                        "locus": gene,
                        "locus_type": "continuous",
                        "subpopulation_a": str(sp_a),
                        "subpopulation_b": str(sp_b),
                        "n_a": int(len(vals_a)),
                        "n_b": int(len(vals_b)),
                        "fst": _fst_continuous_pairwise(vals_a, vals_b),
                    }
                )

    # ---- Categorical loci (action_weights) ----
    if has_categorical:
        subpop_weight_vectors: Dict[Any, List[Dict[str, float]]] = {}
        for _, row in df.iterrows():
            subpop = row[subpop_col]
            if pd.isna(subpop):
                continue
            raw_wv = row.get("action_weights")
            if not isinstance(raw_wv, dict) or len(raw_wv) == 0:
                continue
            sanitized: Dict[str, float] = {}
            for action, raw_weight in raw_wv.items():
                try:
                    w = float(raw_weight)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(w):
                    sanitized[action] = w
            if sanitized:
                subpop_weight_vectors.setdefault(subpop, []).append(sanitized)

        eligible_cat = [sp for sp in subpops if sp in subpop_weight_vectors]
        if len(eligible_cat) >= 2:
            subpop_freq: Dict[Any, Dict[str, float]] = {
                sp: _mean_allele_freqs(subpop_weight_vectors[sp])
                for sp in eligible_cat
            }
            for sp_a, sp_b in combinations(eligible_cat, 2):
                n_a = len(subpop_weight_vectors[sp_a])
                n_b = len(subpop_weight_vectors[sp_b])
                rows.append(
                    {
                        "locus": "action_weights",
                        "locus_type": "categorical",
                        "subpopulation_a": str(sp_a),
                        "subpopulation_b": str(sp_b),
                        "n_a": n_a,
                        "n_b": n_b,
                        "fst": _fst_categorical_pairwise(
                            subpop_freq[sp_a], subpop_freq[sp_b], n_a, n_b
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=FST_COLUMNS)

    result_df = pd.DataFrame(rows, columns=FST_COLUMNS)
    logger.info(
        "compute_fst_pairwise: %d locus–pair row(s) for %d subpopulation(s)",
        len(result_df),
        len(subpops),
    )
    return result_df


def compute_migration_counts(
    df: pd.DataFrame,
    subpop_col: str = "agent_type",
) -> pd.DataFrame:
    """Track gene-flow events (migrations) between subpopulations.

    An agent is classified as a *migrant* when at least one of its parents
    belongs to a different subpopulation than the agent itself.  The parent's
    subpopulation is the majority subpopulation among all recorded parents;
    ties are broken by the first occurrence in ``parent_ids``.

    Parameters
    ----------
    df:
        DataFrame containing a *subpop_col* column, a ``parent_ids`` column,
        and either an ``agent_id`` or ``candidate_id`` column.  Typically
        produced by :func:`build_agent_genetics_dataframe` or
        :func:`build_evolution_experiment_dataframe`.
    subpop_col:
        Column name whose distinct values define subpopulations.
        Default ``"agent_type"``.

    Returns
    -------
    pd.DataFrame
        One row per agent that has at least one traceable parent, with
        columns defined in :data:`MIGRATION_COLUMNS`:

        * ``agent_id`` (str)
        * ``generation`` (int or ``NaN``)
        * ``subpop_definition`` (str) – the value of *subpop_col*
        * ``offspring_subpopulation`` (str)
        * ``parent_subpopulation`` (str) – majority subpopulation of parents
        * ``is_migrant`` (bool) – ``True`` when
          ``offspring_subpopulation != parent_subpopulation``

        Returns an empty DataFrame (with correct columns) when required
        columns are absent or no traceable parent relationships exist.
    """
    required = {subpop_col, "parent_ids"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=MIGRATION_COLUMNS)

    # Determine the individual-identifier column
    id_col: Optional[str] = None
    for candidate in ("agent_id", "candidate_id"):
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        return pd.DataFrame(columns=MIGRATION_COLUMNS)

    # Build id -> subpopulation lookup
    agent_subpop: Dict[str, str] = {}
    for _, row in df.iterrows():
        sp = row[subpop_col]
        if not pd.isna(sp):
            agent_subpop[str(row[id_col])] = str(sp)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        agent_id = str(row[id_col])
        sp_val = row[subpop_col]
        if pd.isna(sp_val):
            continue
        offspring_subpop = str(sp_val)

        parent_ids = row.get("parent_ids", [])
        if not isinstance(parent_ids, (list, tuple)) or len(parent_ids) == 0:
            continue  # genesis agent – no parents to compare against

        parent_subpops = [agent_subpop[pid] for pid in parent_ids if pid in agent_subpop]
        if not parent_subpops:
            continue  # parents not present in this DataFrame

        counts = Counter(parent_subpops)
        max_count = max(counts.values())
        parent_subpop = next(sp for sp in parent_subpops if counts[sp] == max_count)

        rows.append(
            {
                "agent_id": agent_id,
                "generation": row.get("generation"),
                "subpop_definition": subpop_col,
                "offspring_subpopulation": offspring_subpop,
                "parent_subpopulation": parent_subpop,
                "is_migrant": offspring_subpop != parent_subpop,
            }
        )

    if not rows:
        return pd.DataFrame(columns=MIGRATION_COLUMNS)

    result_df = pd.DataFrame(rows, columns=MIGRATION_COLUMNS)
    n_migrants = int(result_df["is_migrant"].sum())
    logger.info(
        "compute_migration_counts: %d agent(s) tracked; %d migrant(s) (%s definition)",
        len(result_df),
        n_migrants,
        subpop_col,
    )
    return result_df


def compute_gene_flow_timeseries(
    df: pd.DataFrame,
    subpop_col: str = "agent_type",
) -> pd.DataFrame:
    """Compute per-generation F_ST differentiation and migration counts.

    For each generation, computes pairwise F_ST between all subpopulations
    (using :func:`compute_fst_pairwise` on the generation's subset) and
    counts boundary-crossing migration events between each subpopulation pair
    (using :func:`compute_migration_counts` on the full DataFrame).  Migration
    counts are undirected: agents moving A→B and B→A are both included in the
    A↔B count.

    Parameters
    ----------
    df:
        DataFrame with a ``generation`` column, a *subpop_col* column, and
        at least one of ``chromosome_values`` or ``action_weights``.
    subpop_col:
        Column name whose distinct values define subpopulations.
        Default ``"agent_type"``.

    Returns
    -------
    pd.DataFrame
        One row per *(generation, subpopulation_a, subpopulation_b, locus)*,
        with columns defined in :data:`GENE_FLOW_COLUMNS`:

        * ``generation`` (int)
        * ``subpop_definition`` (str) – the value of *subpop_col*
        * ``locus`` (str)
        * ``locus_type`` (str)
        * ``subpopulation_a`` (str)
        * ``subpopulation_b`` (str)
        * ``fst`` (float) – F_ST for this generation
        * ``n_migrants`` (int or ``NaN``) – number of migration events this
          generation where offspring crossed the A↔B boundary.  ``NaN`` when
          migration cannot be computed because lineage columns are unavailable.
        * ``migration_data_available`` (bool) – whether lineage columns were
          present and migration counts are therefore computable.

        Returns an empty DataFrame (with correct columns) when the input is
        insufficient (missing required columns, fewer than two subpopulations,
        or no recognised locus column).
    """
    if df.empty or "generation" not in df.columns or subpop_col not in df.columns:
        return pd.DataFrame(columns=GENE_FLOW_COLUMNS)

    has_loci = "chromosome_values" in df.columns or "action_weights" in df.columns
    if not has_loci:
        return pd.DataFrame(columns=GENE_FLOW_COLUMNS)

    has_migration_columns = "parent_ids" in df.columns and (
        "agent_id" in df.columns or "candidate_id" in df.columns
    )
    if has_migration_columns:
        # Pre-compute migration events across all generations at once
        migration_df = compute_migration_counts(df, subpop_col=subpop_col)
    else:
        migration_df = pd.DataFrame(columns=MIGRATION_COLUMNS)
        logger.warning(
            "compute_gene_flow_timeseries: migration counts unavailable because required "
            "lineage columns are missing (need parent_ids plus agent_id/candidate_id)"
        )

    rows: List[Dict[str, Any]] = []

    for gen, gen_group in df.groupby("generation"):
        try:
            gen_int = int(float(gen))
        except (TypeError, ValueError):
            continue

        fst_df = compute_fst_pairwise(gen_group, subpop_col=subpop_col)
        if fst_df.empty:
            continue

        # Subset migration events for this generation
        if not migration_df.empty and "generation" in migration_df.columns:
            gen_migrants = migration_df[migration_df["generation"] == gen]
        else:
            gen_migrants = pd.DataFrame()

        for _, fst_row in fst_df.iterrows():
            sp_a = str(fst_row["subpopulation_a"])
            sp_b = str(fst_row["subpopulation_b"])

            # Count agents that crossed the A↔B boundary this generation
            if has_migration_columns and not gen_migrants.empty:
                n_migrants = int(
                    gen_migrants[
                        gen_migrants["is_migrant"]
                        & gen_migrants["offspring_subpopulation"].isin([sp_a, sp_b])
                        & gen_migrants["parent_subpopulation"].isin([sp_a, sp_b])
                    ].shape[0]
                )
            elif has_migration_columns:
                n_migrants = 0
            else:
                n_migrants = float("nan")

            rows.append(
                {
                    "generation": gen_int,
                    "subpop_definition": subpop_col,
                    "locus": fst_row["locus"],
                    "locus_type": fst_row["locus_type"],
                    "subpopulation_a": sp_a,
                    "subpopulation_b": sp_b,
                    "fst": float(fst_row["fst"]),
                    "n_migrants": n_migrants,
                    "migration_data_available": has_migration_columns,
                }
            )

    if not rows:
        return pd.DataFrame(columns=GENE_FLOW_COLUMNS)

    result_df = (
        pd.DataFrame(rows, columns=GENE_FLOW_COLUMNS)
        .sort_values(["generation", "locus", "subpopulation_a", "subpopulation_b"])
        .reset_index(drop=True)
    )
    logger.info(
        "compute_gene_flow_timeseries: %d row(s) across %d generation(s)",
        len(result_df),
        result_df["generation"].nunique(),
    )
    return result_df
