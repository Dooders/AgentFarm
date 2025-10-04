"""
Temporal analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.temporal.data import process_temporal_data
from farm.analysis.temporal.analyze import (
    analyze_temporal_patterns,
    analyze_event_segmentation,
    analyze_time_series_overview,
    analyze_temporal_efficiency,
)
from farm.analysis.temporal.plot import (
    plot_temporal_patterns,
    plot_rolling_averages,
    plot_event_segmentation,
    plot_temporal_efficiency,
    plot_action_type_evolution,
    plot_reward_trends,
)


class TemporalModule(BaseAnalysisModule):
    """Module for analyzing temporal patterns in simulations."""

    def __init__(self):
        super().__init__(
            name="temporal",
            description="Analysis of temporal patterns, time series analysis, and event segmentation in agent behavior"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['step'],  # Core required columns
                column_types={'step': int}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all temporal analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_patterns": make_analysis_function(analyze_temporal_patterns),
            "analyze_segmentation": make_analysis_function(analyze_event_segmentation),
            "analyze_overview": make_analysis_function(analyze_time_series_overview),
            "analyze_efficiency": make_analysis_function(analyze_temporal_efficiency),
            "plot_patterns": make_analysis_function(plot_temporal_patterns),
            "plot_rolling": make_analysis_function(plot_rolling_averages),
            "plot_segmentation": make_analysis_function(plot_event_segmentation),
            "plot_efficiency": make_analysis_function(plot_temporal_efficiency),
            "plot_evolution": make_analysis_function(plot_action_type_evolution),
            "plot_trends": make_analysis_function(plot_reward_trends),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_patterns"],
                self._functions["analyze_segmentation"],
                self._functions["analyze_overview"],
                self._functions["analyze_efficiency"],
            ],
            "plots": [
                self._functions["plot_patterns"],
                self._functions["plot_rolling"],
                self._functions["plot_segmentation"],
                self._functions["plot_efficiency"],
                self._functions["plot_evolution"],
                self._functions["plot_trends"],
            ],
            "patterns": [
                self._functions["analyze_patterns"],
                self._functions["plot_patterns"],
                self._functions["plot_rolling"],
            ],
            "events": [
                self._functions["analyze_segmentation"],
                self._functions["plot_segmentation"],
            ],
            "basic": [
                self._functions["analyze_overview"],
                self._functions["plot_patterns"],
                self._functions["plot_trends"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for temporal analysis."""
        return SimpleDataProcessor(process_temporal_data)

    def supports_database(self) -> bool:
        """Whether this module uses database storage."""
        return True

    def get_db_filename(self) -> str:
        """Get database filename if using database."""
        return "simulation.db"


# Create singleton instance
temporal_module = TemporalModule()
