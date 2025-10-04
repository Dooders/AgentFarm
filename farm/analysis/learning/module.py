"""
Learning analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.learning.data import process_learning_data
from farm.analysis.learning.analyze import (
    analyze_learning_performance,
    analyze_agent_learning_curves,
    analyze_module_performance,
    analyze_learning_progress,
)
from farm.analysis.learning.plot import (
    plot_learning_curves,
    plot_reward_distribution,
    plot_module_performance,
    plot_action_frequencies,
    plot_learning_efficiency,
    plot_reward_vs_step,
)


class LearningModule(BaseAnalysisModule):
    """Module for analyzing learning dynamics in simulations."""

    def __init__(self):
        super().__init__(
            name="learning",
            description="Analysis of learning performance, agent learning curves, and module efficiency"
        )

        # Set up validation - DataQualityValidator first to catch empty data before type checking
        validator = CompositeValidator([
            DataQualityValidator(min_rows=1),
            ColumnValidator(
                required_columns=['step', 'reward'],  # Core required columns
                column_types={'step': int, 'reward': float}
            )
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all learning analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_performance": make_analysis_function(analyze_learning_performance),
            "analyze_curves": make_analysis_function(analyze_agent_learning_curves),
            "analyze_modules": make_analysis_function(analyze_module_performance),
            "analyze_progress": make_analysis_function(analyze_learning_progress),
            "plot_curves": make_analysis_function(plot_learning_curves),
            "plot_distribution": make_analysis_function(plot_reward_distribution),
            "plot_modules": make_analysis_function(plot_module_performance),
            "plot_actions": make_analysis_function(plot_action_frequencies),
            "plot_efficiency": make_analysis_function(plot_learning_efficiency),
            "plot_progression": make_analysis_function(plot_reward_vs_step),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_performance"],
                self._functions["analyze_curves"],
                self._functions["analyze_modules"],
                self._functions["analyze_progress"],
            ],
            "plots": [
                self._functions["plot_curves"],
                self._functions["plot_distribution"],
                self._functions["plot_modules"],
                self._functions["plot_actions"],
                self._functions["plot_efficiency"],
                self._functions["plot_progression"],
            ],
            "performance": [
                self._functions["analyze_performance"],
                self._functions["plot_curves"],
                self._functions["plot_efficiency"],
            ],
            "comparison": [
                self._functions["analyze_modules"],
                self._functions["plot_modules"],
            ],
            "basic": [
                self._functions["analyze_performance"],
                self._functions["plot_curves"],
                self._functions["plot_distribution"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for learning analysis."""
        return SimpleDataProcessor(process_learning_data)

    def supports_database(self) -> bool:
        """Whether this module uses database storage."""
        return True

    def get_db_filename(self) -> str:
        """Get database filename if using database."""
        return "simulation.db"


# Create singleton instance
learning_module = LearningModule()
