"""
Actions analysis module implementation.
"""

import numpy as np

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.actions.data import process_action_data
from farm.analysis.actions.analyze import (
    analyze_action_patterns,
    analyze_sequence_patterns,
    analyze_decision_patterns,
    analyze_reward_analysis,
)
from farm.analysis.actions.plot import (
    plot_action_frequencies,
    plot_sequence_patterns,
    plot_decision_patterns,
    plot_reward_distributions,
)


class ActionsModule(BaseAnalysisModule):
    """Module for analyzing agent actions in simulations."""

    def __init__(self):
        super().__init__(
            name="actions",
            description="Analysis of action patterns, sequences, decision-making, and performance metrics"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['step', 'action_type', 'frequency'],
                # Use np.number to accept both int and float for frequency
                column_types={'step': int, 'frequency': np.number}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all action analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_patterns": make_analysis_function(analyze_action_patterns),
            "analyze_sequences": make_analysis_function(analyze_sequence_patterns),
            "analyze_decisions": make_analysis_function(analyze_decision_patterns),
            "analyze_rewards": make_analysis_function(analyze_reward_analysis),
            "plot_frequencies": make_analysis_function(plot_action_frequencies),
            "plot_sequences": make_analysis_function(plot_sequence_patterns),
            "plot_decisions": make_analysis_function(plot_decision_patterns),
            "plot_rewards": make_analysis_function(plot_reward_distributions),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_patterns"],
                self._functions["analyze_sequences"],
                self._functions["analyze_decisions"],
                self._functions["analyze_rewards"],
            ],
            "plots": [
                self._functions["plot_frequencies"],
                self._functions["plot_sequences"],
                self._functions["plot_decisions"],
                self._functions["plot_rewards"],
            ],
            "basic": [
                self._functions["analyze_patterns"],
                self._functions["plot_frequencies"],
            ],
            "sequences": [
                self._functions["analyze_sequences"],
                self._functions["plot_sequences"],
            ],
            "performance": [
                self._functions["analyze_decisions"],
                self._functions["analyze_rewards"],
                self._functions["plot_rewards"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for action analysis."""
        return SimpleDataProcessor(process_action_data)


# Create singleton instance
actions_module = ActionsModule()
