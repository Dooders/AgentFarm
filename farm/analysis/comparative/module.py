"""
Comparative analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, make_analysis_function, SimpleDataProcessor

from farm.analysis.comparative.analyze import (
    analyze_simulation_comparison,
    analyze_parameter_differences,
    analyze_performance_comparison,
    compare_experiments,
    process_comparative_data,
)
from farm.analysis.comparative.plot import (
    plot_comparison_metrics,
    plot_parameter_differences,
    plot_performance_comparison,
    plot_comparative_analysis,
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

        # Analysis functions - register with expected names
        self._functions = {
            "plot_comparative_analysis": make_analysis_function(plot_comparative_analysis),
            "plot_performance_comparison": make_analysis_function(plot_performance_comparison),
            "plot_metric_correlations": make_analysis_function(plot_comparison_metrics),  # Alias
            "plot_experiment_differences": make_analysis_function(plot_parameter_differences),  # Alias
            "plot_statistical_comparison": make_analysis_function(plot_performance_comparison),  # Alias
            "compare_experiments": make_analysis_function(compare_experiments),
            "analyze_experiment_variability": make_analysis_function(analyze_simulation_comparison),  # Alias
            # Keep existing names for backwards compatibility
            "analyze_comparison": make_analysis_function(analyze_simulation_comparison),
            "analyze_parameters": make_analysis_function(analyze_parameter_differences),
            "analyze_performance": make_analysis_function(analyze_performance_comparison),
            "plot_metrics": make_analysis_function(plot_comparison_metrics),
            "plot_parameters": make_analysis_function(plot_parameter_differences),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_comparison"],
                self._functions["analyze_parameters"],
                self._functions["analyze_performance"],
                self._functions["compare_experiments"],
                self._functions["analyze_experiment_variability"],
            ],
            "plots": [
                self._functions["plot_comparative_analysis"],
                self._functions["plot_performance_comparison"],
                self._functions["plot_metric_correlations"],
                self._functions["plot_experiment_differences"],
                self._functions["plot_statistical_comparison"],
                self._functions["plot_metrics"],
                self._functions["plot_parameters"],
            ],
            "basic": [
                self._functions["analyze_comparison"],
                self._functions["plot_comparative_analysis"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for comparative analysis."""
        return SimpleDataProcessor(process_comparative_data)


# Create singleton instance
comparative_module = ComparativeModule()
