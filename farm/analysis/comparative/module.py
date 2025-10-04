"""
Comparative analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, make_analysis_function

from farm.analysis.comparative.analyze import (
    analyze_simulation_comparison,
    analyze_parameter_differences,
    analyze_performance_comparison,
)
from farm.analysis.comparative.plot import (
    plot_comparison_metrics,
    plot_parameter_differences,
    plot_performance_comparison,
)


class ComparativeModule(BaseAnalysisModule):
    """Module for comparative analysis of simulations."""

    def __init__(self):
        super().__init__(
            name="comparative",
            description="Analysis for comparing multiple simulations and their differences"
        )

    def register_functions(self) -> None:
        """Register all comparative analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_comparison": make_analysis_function(analyze_simulation_comparison),
            "analyze_parameters": make_analysis_function(analyze_parameter_differences),
            "analyze_performance": make_analysis_function(analyze_performance_comparison),
            "plot_metrics": make_analysis_function(plot_comparison_metrics),
            "plot_parameters": make_analysis_function(plot_parameter_differences),
            "plot_performance": make_analysis_function(plot_performance_comparison),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_comparison"],
                self._functions["analyze_parameters"],
                self._functions["analyze_performance"],
            ],
            "plots": [
                self._functions["plot_metrics"],
                self._functions["plot_parameters"],
                self._functions["plot_performance"],
            ],
            "basic": [
                self._functions["analyze_comparison"],
                self._functions["plot_metrics"],
            ],
        }

    def get_data_processor(self):
        """Comparative analysis doesn't use a standard data processor."""
        return None


# Create singleton instance
comparative_module = ComparativeModule()
