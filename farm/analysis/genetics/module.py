"""
Genetics Analysis Module

Registers the genetics analysis module with the analysis framework so it is
discoverable via :class:`~farm.analysis.service.AnalysisService` and the
module registry.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.genetics.data import process_genetics_data
from farm.analysis.genetics.analyze import analyze_genetics
from farm.analysis.genetics.plot import (
    plot_generation_distribution,
    plot_fitness_over_generations,
)


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
        }

        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [self._functions["analyze_genetics"]],
            "plots": [
                self._functions["plot_generation_distribution"],
                self._functions["plot_fitness_over_generations"],
            ],
            "basic": [
                self._functions["analyze_genetics"],
                self._functions["plot_generation_distribution"],
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
