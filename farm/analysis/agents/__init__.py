"""
Agents analysis module.

Provides comprehensive analysis of individual agents including:
- Lifespan and survival analysis
- Behavior clustering and patterns
- Agent performance metrics
- Learning and adaptation curves
- Risk-reward analysis
- Environmental interactions
"""

from farm.analysis.agents.module import agents_module, AgentsModule
from farm.analysis.agents.compute import (
    compute_lifespan_statistics,
    compute_behavior_patterns,
    compute_performance_metrics,
    compute_learning_curves,
)
from farm.analysis.agents.analyze import (
    analyze_lifespan_patterns,
    analyze_behavior_clustering,
    analyze_performance_analysis,
    analyze_learning_curves,
)
from farm.analysis.agents.plot import (
    plot_lifespan_distributions,
    plot_behavior_clusters,
    plot_performance_metrics,
    plot_learning_curves,
)
from farm.analysis.agents.lifespan import (
    analyze_agent_lifespans,
    plot_lifespan_histogram,
)
from farm.analysis.agents.behavior import (
    cluster_agent_behaviors,
    plot_behavior_clusters,
)

__all__ = [
    "agents_module",
    "AgentsModule",
    "compute_lifespan_statistics",
    "compute_behavior_patterns",
    "compute_performance_metrics",
    "compute_learning_curves",
    "analyze_lifespan_patterns",
    "analyze_behavior_clustering",
    "analyze_performance_analysis",
    "analyze_learning_curves",
    "plot_lifespan_distributions",
    "plot_behavior_clusters",
    "plot_performance_metrics",
    "plot_learning_curves",
    "analyze_agent_lifespans",
    "plot_lifespan_histogram",
    "cluster_agent_behaviors",
]
