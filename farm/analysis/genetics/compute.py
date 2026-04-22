"""
Genetics Analysis Computation

Core computation functions for the genetics analysis module, including the
shared ``parse_parent_ids`` helper, population-level accessors for both
simulation-database and evolution-experiment data sources, and genotypic
diversity metrics (heterozygosity, Shannon entropy, per-locus stats).
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
    * Zero-count bins are excluded from entropy (convention: ``0 ln 0 = 0``).
    """
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
        range_occupancy = observed_range / span if span > 0.0 else 0.0
    else:
        range_occupancy = float("nan")

    # Optional histogram-based Shannon entropy: H = -Σ pᵢ ln(pᵢ)
    shannon_entropy: Optional[float] = None
    if compute_entropy:
        if bounds is not None:
            bin_edges = np.linspace(bounds[0], bounds[1], entropy_bins + 1)
            counts, _ = np.histogram(arr, bins=bin_edges)
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
            values = [
                float(row[gene])
                for row in df["chromosome_values"]
                if isinstance(row, dict) and gene in row
            ]
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

    Returns
    -------
    list of dict
        Each element is ``{"generation": int, "diversity": PopulationDiversitySummary}``,
        sorted by generation in ascending order.
        Returns an empty list when *result* contains no evaluations.
    """
    df = build_evolution_experiment_dataframe(result)
    if df.empty:
        return []

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
        "compute_evolution_diversity_timeseries: computed diversity for %d generation(s)",
        len(rows),
    )
    return rows
