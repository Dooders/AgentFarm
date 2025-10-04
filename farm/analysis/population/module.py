"""
Population analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.population.data import process_population_data
from farm.analysis.population.analyze import (
    analyze_population_dynamics,
    analyze_agent_composition,
)
from farm.analysis.population.plot import (
    plot_population_over_time,
    plot_birth_death_rates,
    plot_agent_composition,
)


class PopulationModule(BaseAnalysisModule):
    """Module for analyzing population dynamics in simulations."""

    def __init__(self):
        super().__init__(
            name="population",
            description="Analysis of population dynamics, births, deaths, and agent composition"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['step', 'total_agents'],
                column_types={'step': int, 'total_agents': int}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all population analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_dynamics": make_analysis_function(analyze_population_dynamics),
            "analyze_composition": make_analysis_function(analyze_agent_composition),
            "plot_population": make_analysis_function(plot_population_over_time),
            "plot_births_deaths": make_analysis_function(plot_birth_death_rates),
            "plot_composition": make_analysis_function(plot_agent_composition),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_dynamics"],
                self._functions["analyze_composition"],
            ],
            "plots": [
                self._functions["plot_population"],
                self._functions["plot_births_deaths"],
                self._functions["plot_composition"],
            ],
            "basic": [
                self._functions["analyze_dynamics"],
                self._functions["plot_population"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for population analysis."""
        return SimpleDataProcessor(process_population_data)


# Create singleton instance
population_module = PopulationModule()
