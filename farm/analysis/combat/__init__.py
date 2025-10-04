"""
Combat analysis module.

Provides comprehensive analysis of combat behavior including:
- Combat encounter patterns and success rates
- Agent combat performance and rankings
- Damage distribution and efficiency metrics
- Temporal patterns in combat behavior
"""

from farm.analysis.combat.module import combat_module, CombatModule
from farm.analysis.combat.compute import (
    compute_combat_statistics,
    compute_agent_combat_performance,
    compute_combat_efficiency_metrics,
    compute_combat_temporal_patterns,
)
from farm.analysis.combat.analyze import (
    analyze_combat_overview,
    analyze_agent_combat_performance,
    analyze_combat_efficiency,
    analyze_combat_temporal_patterns,
)
from farm.analysis.combat.plot import (
    plot_combat_overview,
    plot_combat_success_rate,
    plot_agent_combat_performance,
    plot_combat_efficiency,
    plot_damage_distribution,
    plot_combat_temporal_patterns,
)
from farm.analysis.combat.data import (
    process_combat_data,
    process_combat_metrics_data,
    process_agent_combat_stats,
)

__all__ = [
    "combat_module",
    "CombatModule",
    "compute_combat_statistics",
    "compute_agent_combat_performance",
    "compute_combat_efficiency_metrics",
    "compute_combat_temporal_patterns",
    "analyze_combat_overview",
    "analyze_agent_combat_performance",
    "analyze_combat_efficiency",
    "analyze_combat_temporal_patterns",
    "plot_combat_overview",
    "plot_combat_success_rate",
    "plot_agent_combat_performance",
    "plot_combat_efficiency",
    "plot_damage_distribution",
    "plot_combat_temporal_patterns",
    "process_combat_data",
    "process_combat_metrics_data",
    "process_agent_combat_stats",
]
