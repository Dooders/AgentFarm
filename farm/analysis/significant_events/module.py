"""
Significant events analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.significant_events.analyze import (
    analyze_significant_events,
    analyze_event_patterns,
    analyze_event_impact,
)
from farm.analysis.significant_events.plot import (
    plot_event_timeline,
    plot_event_severity_distribution,
    plot_event_impact_analysis,
)


class SignificantEventsModule(BaseAnalysisModule):
    """Module for analyzing significant events in simulations."""

    def __init__(self):
        super().__init__(
            name="significant_events",
            description="Analysis of significant events, their severity, patterns, and impact",
        )

        # Set up validation for significant events data
        validator = CompositeValidator(
            [
                ColumnValidator(
                    required_columns=["type", "step", "impact_scale"], column_types={"step": int, "impact_scale": float}
                ),
                DataQualityValidator(min_rows=1),
            ]
        )
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all significant events analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_events": make_analysis_function(analyze_significant_events),
            "analyze_patterns": make_analysis_function(analyze_event_patterns),
            "analyze_impact": make_analysis_function(analyze_event_impact),
            "plot_timeline": make_analysis_function(plot_event_timeline),
            "plot_severity": make_analysis_function(plot_event_severity_distribution),
            "plot_impact": make_analysis_function(plot_event_impact_analysis),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_events"],
                self._functions["analyze_patterns"],
                self._functions["analyze_impact"],
            ],
            "plots": [
                self._functions["plot_timeline"],
                self._functions["plot_severity"],
                self._functions["plot_impact"],
            ],
            "basic": [
                self._functions["analyze_events"],
                self._functions["plot_timeline"],
            ],
            "patterns": [
                self._functions["analyze_patterns"],
                self._functions["plot_timeline"],
            ],
            "impact": [
                self._functions["analyze_impact"],
                self._functions["plot_impact"],
            ],
        }

    def get_data_processor(self):
        """Significant events analysis uses database queries, not standard data processing."""
        return None


# Create singleton instance
significant_events_module = SignificantEventsModule()
