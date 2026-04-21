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

Quick Start::

    >>> from farm.analysis.genetics.compute import (
    ...     parse_parent_ids,
    ...     build_agent_genetics_dataframe,
    ...     build_evolution_experiment_dataframe,
    ... )
"""

from farm.analysis.genetics.compute import (
    parse_parent_ids,
    build_agent_genetics_dataframe,
    build_evolution_experiment_dataframe,
)
from farm.analysis.genetics.analyze import analyze_genetics
from farm.analysis.genetics.module import genetics_module, GeneticsModule

__all__ = [
    "parse_parent_ids",
    "build_agent_genetics_dataframe",
    "build_evolution_experiment_dataframe",
    "analyze_genetics",
    "genetics_module",
    "GeneticsModule",
]
