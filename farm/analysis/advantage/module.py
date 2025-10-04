"""
Advantage analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.advantage.data import process_advantage_data
from farm.analysis.advantage.analyze import (
    analyze_advantage_patterns,
)
from farm.analysis.advantage.plot import (
    plot_advantage_results,
)


class AdvantageModule(BaseAnalysisModule):
    """Module for analyzing relative advantages between agent types."""

    def __init__(self):
        super().__init__(
            name="advantage",
            description="Analysis of relative advantages between agent types and their impact on dominance patterns"
        )

        # Set up validation - flexible since we work with multiple data types
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=[],  # Flexible validation for different data types
                column_types={}
            ),
            DataQualityValidator(min_rows=0)  # Allow empty DataFrames for database-based analysis
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all advantage analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_patterns": make_analysis_function(analyze_advantage_patterns),
            "plot_results": make_analysis_function(plot_advantage_results),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_patterns"],
            ],
            "plots": [
                self._functions["plot_results"],
            ],
            "basic": [
                self._functions["analyze_patterns"],
                self._functions["plot_results"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for advantage analysis."""
        return SimpleDataProcessor(process_advantage_data)

    def supports_database(self) -> bool:
        """This module primarily uses database access."""
        return True

    def get_db_filename(self) -> str:
        """Get database filename for advantage analysis."""
        return "simulation.db"


# Create singleton instance
advantage_module = AdvantageModule()
