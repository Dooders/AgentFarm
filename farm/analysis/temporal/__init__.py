"""
Temporal analysis module.

Provides comprehensive analysis of temporal patterns including:
- Time series analysis of agent actions
- Rolling averages and trend analysis
- Event segmentation and impact analysis
- Temporal efficiency metrics
- Action type evolution over time
"""

from farm.analysis.temporal.module import temporal_module, TemporalModule
from farm.analysis.temporal.compute import (
    compute_temporal_statistics,
    compute_event_segmentation_metrics,
    compute_temporal_patterns,
    compute_temporal_efficiency_metrics,
)
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
from farm.analysis.temporal.data import (
    process_temporal_data,
    process_time_series_data,
    process_event_segmentation_data,
    extract_temporal_patterns,
)

__all__ = [
    "temporal_module",
    "TemporalModule",
    "compute_temporal_statistics",
    "compute_event_segmentation_metrics",
    "compute_temporal_patterns",
    "compute_temporal_efficiency_metrics",
    "analyze_temporal_patterns",
    "analyze_event_segmentation",
    "analyze_time_series_overview",
    "analyze_temporal_efficiency",
    "plot_temporal_patterns",
    "plot_rolling_averages",
    "plot_event_segmentation",
    "plot_temporal_efficiency",
    "plot_action_type_evolution",
    "plot_reward_trends",
    "process_temporal_data",
    "process_time_series_data",
    "process_event_segmentation_data",
    "extract_temporal_patterns",
]
