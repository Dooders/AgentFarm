"""
Learning analysis module.

Provides comprehensive analysis of learning dynamics including:
- Learning performance metrics
- Agent learning curves
- Module performance comparison
- Learning efficiency analysis
- Reward distributions and progression
"""

from farm.analysis.learning.module import learning_module, LearningModule
from farm.analysis.learning.compute import (
    compute_learning_statistics,
    compute_agent_learning_curves,
    compute_learning_efficiency_metrics,
    compute_module_performance_comparison,
)
from farm.analysis.learning.analyze import (
    analyze_learning_performance,
    analyze_agent_learning_curves,
    analyze_module_performance,
    analyze_learning_progress,
)
from farm.analysis.learning.plot import (
    plot_learning_curves,
    plot_reward_distribution,
    plot_module_performance,
    plot_action_frequencies,
    plot_learning_efficiency,
    plot_reward_vs_step,
)
from farm.analysis.learning.data import (
    process_learning_data,
    process_learning_progress_data,
)

__all__ = [
    "learning_module",
    "LearningModule",
    "compute_learning_statistics",
    "compute_agent_learning_curves",
    "compute_learning_efficiency_metrics",
    "compute_module_performance_comparison",
    "analyze_learning_performance",
    "analyze_agent_learning_curves",
    "analyze_module_performance",
    "analyze_learning_progress",
    "plot_learning_curves",
    "plot_reward_distribution",
    "plot_module_performance",
    "plot_action_frequencies",
    "plot_learning_efficiency",
    "plot_reward_vs_step",
    "process_learning_data",
    "process_learning_progress_data",
]
