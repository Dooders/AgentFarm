"""
Social Behavior Analysis Module

Provides comprehensive analysis of social behaviors in agent simulations,
including cooperation, competition, social networks, and spatial clustering.

Features:
- Social network analysis (connections, interactions, centrality)
- Cooperation vs competition metrics
- Resource sharing patterns
- Spatial clustering and group formation
- Agent type comparisons
- Temporal evolution of social dynamics

Quick Start:
    >>> from farm.analysis.service import AnalysisService, AnalysisRequest
    >>> from pathlib import Path
    >>>
    >>> service = AnalysisService()
    >>> request = AnalysisRequest(
    ...     module_name="social_behavior",
    ...     experiment_path=Path("data/experiment"),
    ...     output_path=Path("results/social")
    ... )
    >>> result = service.run(request)
"""

from farm.analysis.social_behavior.module import social_behavior_module, SocialBehaviorModule
from farm.analysis.social_behavior.compute import (
    compute_all_social_metrics,
    compute_social_network_metrics,
    compute_resource_sharing_metrics,
    compute_spatial_clustering,
    compute_cooperation_competition_metrics,
    compute_reproduction_social_patterns,
)
from farm.analysis.social_behavior.analyze import (
    analyze_social_behaviors,
    analyze_social_behaviors_across_simulations,
    extract_social_behavior_insights,
)
from farm.analysis.social_behavior.plot import (
    plot_social_network_overview,
    plot_cooperation_competition_balance,
    plot_resource_sharing_patterns,
    plot_spatial_clustering,
)
from farm.analysis.social_behavior.data import (
    process_social_behavior_data,
    load_social_behavior_data_from_db,
)

__all__ = [
    "social_behavior_module",
    "SocialBehaviorModule",
    "compute_all_social_metrics",
    "compute_social_network_metrics",
    "compute_resource_sharing_metrics",
    "compute_spatial_clustering",
    "compute_cooperation_competition_metrics",
    "compute_reproduction_social_patterns",
    "analyze_social_behaviors",
    "analyze_social_behaviors_across_simulations",
    "extract_social_behavior_insights",
    "plot_social_network_overview",
    "plot_cooperation_competition_balance",
    "plot_resource_sharing_patterns",
    "plot_spatial_clustering",
    "process_social_behavior_data",
    "load_social_behavior_data_from_db",
]
