"""
Genetics Analysis Module

Registers the genetics analysis module with the analysis framework so it is
discoverable via :class:`~farm.analysis.service.AnalysisService` and the
module registry.
"""

from typing import Any, Dict, Optional

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.genetics.data import process_genetics_data
from farm.analysis.genetics.analyze import analyze_genetics
from farm.analysis.genetics.plot import (
    plot_generation_distribution,
    plot_fitness_over_generations,
    plot_marginal_fitness_effect,
    plot_fitness_landscape_2d,
)
from farm.analysis.genetics.compute import (
    compute_allele_frequency_timeseries,
    compute_selection_pressure_summary,
    compute_fitness_gene_correlations,
    compute_pairwise_epistasis,
    simulate_wright_fisher,
    compute_fst_pairwise,
    compute_migration_counts,
    compute_gene_flow_timeseries,
    compute_realized_mutation_rate,
    compute_realized_mutation_rate_per_locus,
    compute_conserved_runs,
    compute_conserved_run_fitness_correlation,
    compute_sweep_candidates,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def _simulate_wright_fisher_for_analysis(
    df: Any,
    initial_frequencies: Optional[Dict[str, float]] = None,
    n_effective: Optional[int] = None,
    n_generations: Optional[int] = None,
    seed: Optional[int] = None,
) -> Any:
    """Adapter for module-run execution of Wright-Fisher simulation.

    The analysis runner always calls registered functions with DataFrame-oriented
    signatures. This adapter allows the Wright-Fisher simulator to be included
    in groups safely while still requiring explicit simulation parameters.
    """
    _ = df  # Unused by design; simulation is parameter-driven.
    if initial_frequencies is None or n_effective is None or n_generations is None:
        logger.warning(
            "simulate_wright_fisher skipped: provide initial_frequencies, "
            "n_effective, and n_generations via analysis kwargs"
        )
        return None
    return simulate_wright_fisher(
        initial_frequencies=initial_frequencies,
        n_effective=n_effective,
        n_generations=n_generations,
        seed=seed,
    )


def _compute_conserved_run_fitness_correlation_for_analysis(
    df: Any,
    conserved_df: Optional[Any] = None,
    epsilon: float = 1e-4,
    min_run_length: int = 2,
) -> Any:
    """Adapter for module-run execution of conserved-run / fitness correlation.

    The analysis runner provides one DataFrame by default. This adapter derives
    ``conserved_df`` via :func:`compute_conserved_runs` unless an explicit
    ``conserved_df`` kwarg is provided.
    """
    if conserved_df is None:
        conserved_df = compute_conserved_runs(
            df,
            epsilon=epsilon,
            min_run_length=min_run_length,
        )
    return compute_conserved_run_fitness_correlation(conserved_df, df)


def _compute_sweep_candidates_for_analysis(
    df: Any,
    conserved_df: Optional[Any] = None,
    pressure_df: Optional[Any] = None,
    epsilon: float = 1e-4,
    min_run_length: int = 2,
    pop_size: Optional[int] = None,
    significance_threshold: float = 2.0,
) -> Any:
    """Adapter for module-run execution of sweep-candidate detection.

    The analysis runner provides one DataFrame by default. This adapter derives
    required intermediate inputs unless they are explicitly provided via
    analysis kwargs.
    """
    if conserved_df is None:
        conserved_df = compute_conserved_runs(
            df,
            epsilon=epsilon,
            min_run_length=min_run_length,
        )

    if pressure_df is None:
        allele_freq_df = compute_allele_frequency_timeseries(df)
        pressure_df = compute_selection_pressure_summary(
            allele_freq_df,
            pop_size=pop_size,
            significance_threshold=significance_threshold,
        )

    return compute_sweep_candidates(conserved_df, pressure_df)


class GeneticsModule(BaseAnalysisModule):
    """Analysis module for agent-genome and chromosome statistics.

    Supports two data sources:

    * **Simulation database** – loads per-agent action weights, generation,
      and lineage via
      :func:`~farm.analysis.genetics.compute.build_agent_genetics_dataframe`.
    * **Evolution-experiment result** – loads per-candidate chromosome values,
      fitness, and parent IDs via
      :func:`~farm.analysis.genetics.compute.build_evolution_experiment_dataframe`.
    """

    def __init__(self) -> None:
        super().__init__(
            name="genetics",
            description=(
                "Analysis of agent genomes, chromosomes, lineage, and population-level "
                "genetic statistics for simulation databases and evolution experiments"
            ),
        )

        validator = CompositeValidator(
            [
                ColumnValidator(required_columns=[], column_types={}),
                DataQualityValidator(min_rows=0),
            ]
        )
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all genetics analysis functions."""

        self._functions = {
            "analyze_genetics": make_analysis_function(analyze_genetics),
            "plot_generation_distribution": make_analysis_function(plot_generation_distribution),
            "plot_fitness_over_generations": make_analysis_function(plot_fitness_over_generations),
            "compute_fitness_gene_correlations": make_analysis_function(
                compute_fitness_gene_correlations
            ),
            "compute_pairwise_epistasis": make_analysis_function(compute_pairwise_epistasis),
            "plot_marginal_fitness_effect": make_analysis_function(plot_marginal_fitness_effect),
            "plot_fitness_landscape_2d": make_analysis_function(plot_fitness_landscape_2d),
            "simulate_wright_fisher": make_analysis_function(_simulate_wright_fisher_for_analysis),
            "compute_fst_pairwise": make_analysis_function(compute_fst_pairwise),
            "compute_migration_counts": make_analysis_function(compute_migration_counts),
            "compute_gene_flow_timeseries": make_analysis_function(compute_gene_flow_timeseries),
            "compute_realized_mutation_rate": make_analysis_function(compute_realized_mutation_rate),
            "compute_realized_mutation_rate_per_locus": make_analysis_function(
                compute_realized_mutation_rate_per_locus
            ),
            "compute_conserved_runs": make_analysis_function(compute_conserved_runs),
            "compute_conserved_run_fitness_correlation": make_analysis_function(
                _compute_conserved_run_fitness_correlation_for_analysis
            ),
            "compute_sweep_candidates": make_analysis_function(_compute_sweep_candidates_for_analysis),
        }

        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_genetics"],
                self._functions["compute_fitness_gene_correlations"],
                self._functions["compute_pairwise_epistasis"],
            ],
            "plots": [
                self._functions["plot_generation_distribution"],
                self._functions["plot_fitness_over_generations"],
                self._functions["plot_marginal_fitness_effect"],
                self._functions["plot_fitness_landscape_2d"],
            ],
            "basic": [
                self._functions["analyze_genetics"],
                self._functions["plot_generation_distribution"],
            ],
            "fitness_landscape": [
                self._functions["compute_fitness_gene_correlations"],
                self._functions["compute_pairwise_epistasis"],
                self._functions["plot_marginal_fitness_effect"],
                self._functions["plot_fitness_landscape_2d"],
            ],
            "population_genetics": [
                self._functions["simulate_wright_fisher"],
                self._functions["compute_fst_pairwise"],
                self._functions["compute_migration_counts"],
                self._functions["compute_gene_flow_timeseries"],
            ],
            "adaptation_signatures": [
                self._functions["compute_realized_mutation_rate"],
                self._functions["compute_realized_mutation_rate_per_locus"],
                self._functions["compute_conserved_runs"],
                self._functions["compute_conserved_run_fitness_correlation"],
                self._functions["compute_sweep_candidates"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Return the data processor for genetics analysis."""
        return SimpleDataProcessor(process_genetics_data)

    def supports_database(self) -> bool:
        """This module can use a simulation database."""
        return True

    def get_db_filename(self) -> str:
        """Database filename used by this module."""
        return "simulation.db"


# Singleton instance consumed by the module registry
genetics_module = GeneticsModule()
