"""
Resources analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.resources.data import process_resource_data
from farm.analysis.resources.analyze import (
    analyze_resource_patterns,
    analyze_consumption_analysis,
    analyze_resource_efficiency,
    analyze_hotspots,
)
from farm.analysis.resources.plot import (
    plot_resource_distribution,
    plot_consumption_over_time,
    plot_efficiency_metrics,
    plot_resource_hotspots,
)


class ResourcesModule(BaseAnalysisModule):
    """Module for analyzing resource dynamics in simulations."""

    def __init__(self):
        super().__init__(
            name="resources",
            description="Analysis of resource distribution, consumption, efficiency, and hotspot patterns"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['step', 'total_resources'],
                column_types={'step': int, 'total_resources': (int, float)}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all resource analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_patterns": make_analysis_function(analyze_resource_patterns),
            "analyze_consumption": make_analysis_function(analyze_consumption_analysis),
            "analyze_efficiency": make_analysis_function(analyze_resource_efficiency),
            "analyze_hotspots": make_analysis_function(analyze_hotspots),
            "plot_distribution": make_analysis_function(plot_resource_distribution),
            "plot_consumption": make_analysis_function(plot_consumption_over_time),
            "plot_efficiency": make_analysis_function(plot_efficiency_metrics),
            "plot_hotspots": make_analysis_function(plot_resource_hotspots),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_patterns"],
                self._functions["analyze_consumption"],
                self._functions["analyze_efficiency"],
                self._functions["analyze_hotspots"],
            ],
            "plots": [
                self._functions["plot_distribution"],
                self._functions["plot_consumption"],
                self._functions["plot_efficiency"],
                self._functions["plot_hotspots"],
            ],
            "basic": [
                self._functions["analyze_patterns"],
                self._functions["plot_distribution"],
            ],
            "efficiency": [
                self._functions["analyze_efficiency"],
                self._functions["plot_efficiency"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for resource analysis."""
        return SimpleDataProcessor(process_resource_data)


# Create singleton instance
resources_module = ResourcesModule()
