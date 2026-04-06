"""
System dynamics analysis: cross-domain coupling and unified reporting.
"""

from farm.analysis.system_dynamics.module import SystemDynamicsModule, system_dynamics_module
from farm.analysis.system_dynamics.data import process_system_dynamics_data
from farm.analysis.system_dynamics.compute import (
    resource_population_coupling,
    synthesize_system_dynamics,
    feedback_loop_candidates,
    action_reward_lag_coupling,
)

__all__ = [
    "SystemDynamicsModule",
    "system_dynamics_module",
    "process_system_dynamics_data",
    "resource_population_coupling",
    "synthesize_system_dynamics",
    "feedback_loop_candidates",
    "action_reward_lag_coupling",
]
