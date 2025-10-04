"""
Significant events analysis module.

Provides comprehensive analysis of significant events in simulations including:
- Event detection and classification
- Severity scoring and filtering
- Temporal event patterns
- Event impact analysis
- Event visualization and reporting
"""

from farm.analysis.significant_events.module import significant_events_module, SignificantEventsModule
from farm.analysis.significant_events.compute import (
    compute_event_severity,
    compute_event_patterns,
    compute_event_impact,
    detect_significant_events,
)
from farm.analysis.significant_events.analyze import (
    analyze_significant_events,
    analyze_event_patterns,
    analyze_event_impact,
)
from farm.analysis.significant_events.plot import (
    plot_event_timeline,
    plot_event_severity_distribution,
    plot_event_impact_analysis,
    plot_significant_events,
)

__all__ = [
    "significant_events_module",
    "SignificantEventsModule",
    "compute_event_severity",
    "compute_event_patterns",
    "compute_event_impact",
    "detect_significant_events",
    "analyze_significant_events",
    "analyze_event_patterns",
    "analyze_event_impact",
    "plot_event_timeline",
    "plot_event_severity_distribution",
    "plot_event_impact_analysis",
    "plot_significant_events",
]
