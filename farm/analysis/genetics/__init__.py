"""
Genetics Analysis Package

Provides tools for analyzing agent genomes, chromosomes, lineage, and
population-level genetic statistics from simulation databases and
evolution-experiment artifacts.

Features:
- Shared ``parse_parent_ids`` helper centralising ``GenomeId.from_string`` usage
- DB-backed population accessor (action weights, generation, lineage)
- Evolution-experiment-backed population accessor (chromosome values, fitness)
- Normalized DataFrame output for both sources
- Genotypic diversity metrics: heterozygosity, Shannon entropy, per-locus stats
- Allele-frequency trajectory tracking and selection-pressure detection
- Fitness landscape: single-locus correlations and pairwise epistasis analysis
- Wright-Fisher neutral drift simulator (seeded, per-allele trajectories)
- Gene-flow / F_ST differentiation across configurable subpopulations
- Adaptation signatures: realized mutation rate (per generation and per
  locus), conserved-run detector with categorical fixation support,
  conserved-run / fitness-improvement correlation, and sweep-candidate
  identification

Quick Start::

    >>> from farm.analysis.genetics.compute import (
    ...     parse_parent_ids,
    ...     build_agent_genetics_dataframe,
    ...     build_evolution_experiment_dataframe,
    ...     compute_continuous_locus_diversity,
    ...     compute_categorical_locus_diversity,
    ...     compute_population_diversity,
    ...     compute_evolution_diversity_timeseries,
    ...     compute_allele_frequency_timeseries,
    ...     compute_selection_pressure_summary,
    ...     compute_fitness_gene_correlations,
    ...     compute_pairwise_epistasis,
    ...     simulate_wright_fisher,
    ...     compute_fst_pairwise,
    ...     compute_migration_counts,
    ...     compute_gene_flow_timeseries,
    ...     compute_realized_mutation_rate,
    ...     compute_realized_mutation_rate_per_locus,
    ...     compute_conserved_runs,
    ...     compute_conserved_run_fitness_correlation,
    ...     compute_sweep_candidates,
    ... )
"""

from farm.analysis.genetics.utils import parse_parent_ids
from farm.analysis.genetics.compute import (
    ContinuousLocusDiversity,
    CategoricalLocusDiversity,
    PopulationDiversitySummary,
    build_agent_genetics_dataframe,
    build_evolution_experiment_dataframe,
    compute_continuous_locus_diversity,
    compute_categorical_locus_diversity,
    compute_population_diversity,
    compute_evolution_diversity_timeseries,
    compute_allele_frequency_timeseries,
    compute_selection_pressure_summary,
    ALLELE_MEAN,
    ALLELE_VARIANCE,
    ALLELE_FREQUENCY_COLUMNS,
    SELECTION_PRESSURE_COLUMNS,
    compute_fitness_gene_correlations,
    compute_pairwise_epistasis,
    FITNESS_GENE_CORRELATION_COLUMNS,
    PAIRWISE_EPISTASIS_COLUMNS,
    simulate_wright_fisher,
    WRIGHT_FISHER_COLUMNS,
    compute_fst_pairwise,
    FST_COLUMNS,
    compute_migration_counts,
    MIGRATION_COLUMNS,
    compute_gene_flow_timeseries,
    GENE_FLOW_COLUMNS,
    compute_realized_mutation_rate,
    REALIZED_MUTATION_COLUMNS,
    compute_realized_mutation_rate_per_locus,
    REALIZED_MUTATION_PER_LOCUS_COLUMNS,
    compute_conserved_runs,
    CONSERVED_RUNS_COLUMNS,
    CATEGORICAL_LOCUS_PREFIX,
    compute_conserved_run_fitness_correlation,
    CONSERVED_RUN_FITNESS_CORRELATION_COLUMNS,
    compute_sweep_candidates,
    SWEEP_CANDIDATE_COLUMNS,
)
from farm.analysis.genetics.analyze import analyze_genetics, generate_genetics_report
from farm.analysis.genetics.module import genetics_module, GeneticsModule

__all__ = [
    "parse_parent_ids",
    # DataFrame accessors
    "build_agent_genetics_dataframe",
    "build_evolution_experiment_dataframe",
    # Diversity result types
    "ContinuousLocusDiversity",
    "CategoricalLocusDiversity",
    "PopulationDiversitySummary",
    # Diversity computation functions
    "compute_continuous_locus_diversity",
    "compute_categorical_locus_diversity",
    "compute_population_diversity",
    "compute_evolution_diversity_timeseries",
    # Allele-frequency tracking and selection-pressure detection
    "compute_allele_frequency_timeseries",
    "compute_selection_pressure_summary",
    "ALLELE_MEAN",
    "ALLELE_VARIANCE",
    "ALLELE_FREQUENCY_COLUMNS",
    "SELECTION_PRESSURE_COLUMNS",
    # Fitness landscape correlations and epistasis
    "compute_fitness_gene_correlations",
    "compute_pairwise_epistasis",
    "FITNESS_GENE_CORRELATION_COLUMNS",
    "PAIRWISE_EPISTASIS_COLUMNS",
    # Wright-Fisher neutral drift simulator
    "simulate_wright_fisher",
    "WRIGHT_FISHER_COLUMNS",
    # Gene-flow / F_ST differentiation
    "compute_fst_pairwise",
    "FST_COLUMNS",
    "compute_migration_counts",
    "MIGRATION_COLUMNS",
    "compute_gene_flow_timeseries",
    "GENE_FLOW_COLUMNS",
    # Adaptation signatures
    "compute_realized_mutation_rate",
    "REALIZED_MUTATION_COLUMNS",
    "compute_realized_mutation_rate_per_locus",
    "REALIZED_MUTATION_PER_LOCUS_COLUMNS",
    "compute_conserved_runs",
    "CONSERVED_RUNS_COLUMNS",
    "CATEGORICAL_LOCUS_PREFIX",
    "compute_conserved_run_fitness_correlation",
    "CONSERVED_RUN_FITNESS_CORRELATION_COLUMNS",
    "compute_sweep_candidates",
    "SWEEP_CANDIDATE_COLUMNS",
    # High-level analysis
    "analyze_genetics",
    "generate_genetics_report",
    "genetics_module",
    "GeneticsModule",
]
