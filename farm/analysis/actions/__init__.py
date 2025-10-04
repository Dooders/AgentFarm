"""
Actions analysis module.

Provides comprehensive analysis of agent actions including:
- Action frequency and success rates
- Action sequences and patterns
- Decision-making analysis
- Reward and performance metrics
- Temporal action patterns
"""

from farm.analysis.actions.module import actions_module, ActionsModule
from farm.analysis.actions.compute import (
    compute_action_statistics,
    compute_sequence_patterns,
    compute_decision_patterns,
    compute_reward_metrics,
)
from farm.analysis.actions.analyze import (
    analyze_action_patterns,
    analyze_sequence_patterns,
    analyze_decision_patterns,
    analyze_reward_analysis,
)
from farm.analysis.actions.plot import (
    plot_action_frequencies,
    plot_sequence_patterns,
    plot_decision_patterns,
    plot_reward_distributions,
)

__all__ = [
    "actions_module",
    "ActionsModule",
    "compute_action_statistics",
    "compute_sequence_patterns",
    "compute_decision_patterns",
    "compute_reward_metrics",
    "analyze_action_patterns",
    "analyze_sequence_patterns",
    "analyze_decision_patterns",
    "analyze_reward_analysis",
    "plot_action_frequencies",
    "plot_sequence_patterns",
    "plot_decision_patterns",
    "plot_reward_distributions",
]
