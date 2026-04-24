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
)
from farm.analysis.genetics.analyze import analyze_genetics
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
    # High-level analysis
    "analyze_genetics",
    "genetics_module",
    "GeneticsModule",
]
