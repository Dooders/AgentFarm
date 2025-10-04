"""
Spatial analysis module.

Provides comprehensive analysis of spatial patterns including:
- Movement trajectories and patterns
- Location effects and hotspots
- Spatial distribution and clustering
- Agent-resource spatial interactions
"""

from farm.analysis.spatial.module import spatial_module, SpatialModule
from farm.analysis.spatial.compute import (
    compute_spatial_statistics,
    compute_movement_patterns,
    compute_location_hotspots,
    compute_spatial_distribution_metrics,
)
from farm.analysis.spatial.analyze import (
    analyze_spatial_overview,
    analyze_movement_patterns,
    analyze_location_hotspots,
    analyze_spatial_distribution,
)
from farm.analysis.spatial.plot import (
    plot_spatial_overview,
    plot_movement_trajectories,
    plot_location_hotspots,
    plot_spatial_density,
    plot_movement_directions,
    plot_clustering_analysis,
)
from farm.analysis.spatial.data import (
    process_spatial_data,
    process_movement_data,
    process_location_analysis_data,
)
from farm.analysis.spatial.movement import (
    analyze_movement_trajectories,
    analyze_movement_patterns_detailed,
    calculate_euclidean_distance,
)
from farm.analysis.spatial.location import (
    analyze_location_effects,
    analyze_clustering_patterns,
    analyze_resource_location_patterns,
)

__all__ = [
    "spatial_module",
    "SpatialModule",
    "compute_spatial_statistics",
    "compute_movement_patterns",
    "compute_location_hotspots",
    "compute_spatial_distribution_metrics",
    "analyze_spatial_overview",
    "analyze_movement_patterns",
    "analyze_location_hotspots",
    "analyze_spatial_distribution",
    "plot_spatial_overview",
    "plot_movement_trajectories",
    "plot_location_hotspots",
    "plot_spatial_density",
    "plot_movement_directions",
    "plot_clustering_analysis",
    "process_spatial_data",
    "process_movement_data",
    "process_location_analysis_data",
    "analyze_movement_trajectories",
    "analyze_movement_patterns_detailed",
    "calculate_euclidean_distance",
    "analyze_location_effects",
    "analyze_clustering_patterns",
    "analyze_resource_location_patterns",
]
