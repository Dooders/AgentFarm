"""
Resources analysis module.

Provides comprehensive analysis of resource dynamics including:
- Resource distribution patterns
- Consumption rates and efficiency
- Resource hotspots and competition
- Resource scarcity effects
- Agent resource strategies
"""

from farm.analysis.resources.module import resources_module, ResourcesModule
from farm.analysis.resources.compute import (
    compute_resource_statistics,
    compute_consumption_patterns,
    compute_resource_efficiency,
    compute_resource_hotspots,
)
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

__all__ = [
    "resources_module",
    "ResourcesModule",
    "compute_resource_statistics",
    "compute_consumption_patterns",
    "compute_resource_efficiency",
    "compute_resource_hotspots",
    "analyze_resource_patterns",
    "analyze_consumption_analysis",
    "analyze_resource_efficiency",
    "analyze_hotspots",
    "plot_resource_distribution",
    "plot_consumption_over_time",
    "plot_efficiency_metrics",
    "plot_resource_hotspots",
]
