"""
Genesis analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.genesis.data import process_genesis_data
from farm.analysis.genesis.analyze import (
    analyze_genesis_factors,
    analyze_genesis_across_simulations,
    analyze_critical_period,
    analyze_genesis_patterns,
)
from farm.analysis.genesis.plot import (
    plot_genesis_analysis_results,
    plot_initial_state_comparison,
    plot_critical_period_analysis,
    plot_genesis_patterns,
    plot_genesis_timeline,
)


class GenesisModule(BaseAnalysisModule):
    """Module for analyzing how initial conditions impact simulation outcomes."""

    def __init__(self):
        super().__init__(
            name="genesis",
            description="Analysis of initial conditions and their impact on dominance patterns and simulation outcomes"
        )

        # Set up validation - flexible since we work with database sessions and various data types
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=[],  # Flexible validation for different data types
                column_types={}
            ),
            DataQualityValidator(min_rows=0)  # Allow empty DataFrames for database-based analysis
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all genesis analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_factors": make_analysis_function(analyze_genesis_factors),
            "analyze_across_simulations": make_analysis_function(analyze_genesis_across_simulations),
            "analyze_critical_period": make_analysis_function(analyze_critical_period),
            "analyze_genesis_patterns": make_analysis_function(analyze_genesis_patterns),
            "plot_results": make_analysis_function(plot_genesis_analysis_results),
            "plot_initial_comparison": make_analysis_function(plot_initial_state_comparison),
            "plot_critical_period": make_analysis_function(plot_critical_period_analysis),
            "plot_genesis_patterns": make_analysis_function(plot_genesis_patterns),
            "plot_genesis_timeline": make_analysis_function(plot_genesis_timeline),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_factors"],
                self._functions["analyze_across_simulations"],
                self._functions["analyze_critical_period"],
                self._functions["analyze_genesis_patterns"],
            ],
            "plots": [
                self._functions["plot_results"],
                self._functions["plot_initial_comparison"],
                self._functions["plot_critical_period"],
                self._functions["plot_genesis_patterns"],
                self._functions["plot_genesis_timeline"],
            ],
            "basic": [
                self._functions["analyze_factors"],
                self._functions["plot_results"],
            ],
            "critical_period": [
                self._functions["analyze_critical_period"],
                self._functions["plot_critical_period"],
            ],
            "comparative": [
                self._functions["analyze_across_simulations"],
                self._functions["plot_initial_comparison"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for genesis analysis."""
        return SimpleDataProcessor(process_genesis_data)

    def supports_database(self) -> bool:
        """This module uses database access for analysis."""
        return True

    def get_db_filename(self) -> str:
        """Get database filename for genesis analysis."""
        return "simulation.db"


# Create singleton instance
genesis_module = GenesisModule()
