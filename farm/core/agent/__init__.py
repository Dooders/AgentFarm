"""
Agent module with component-based architecture.

This module provides a modular, composable agent system following SOLID principles.
Agents are built from pluggable components and configurable behaviors.
"""

from farm.core.agent.core import AgentCore
from farm.core.agent.factory import AgentFactory

# Components
from farm.core.agent.components import (
    IAgentComponent,
    MovementComponent,
    ResourceComponent,
    CombatComponent,
    PerceptionComponent,
    ReproductionComponent,
)

# Behaviors
from farm.core.agent.behaviors import (
    IAgentBehavior,
    DefaultAgentBehavior,
    LearningAgentBehavior,
)

# Configuration
from farm.core.agent.config import (
    AgentConfig,
    MovementConfig,
    ResourceConfig,
    CombatConfig,
    PerceptionConfig,
    ReproductionConfig,
)

# State
from farm.core.agent.state import StateManager

__all__ = [
    # Core
    "AgentCore",
    "AgentFactory",
    # Components
    "IAgentComponent",
    "MovementComponent",
    "ResourceComponent",
    "CombatComponent",
    "PerceptionComponent",
    "ReproductionComponent",
    # Behaviors
    "IAgentBehavior",
    "DefaultAgentBehavior",
    "LearningAgentBehavior",
    # Configuration
    "AgentConfig",
    "MovementConfig",
    "ResourceConfig",
    "CombatConfig",
    "PerceptionConfig",
    "ReproductionConfig",
    # State
    "StateManager",
]