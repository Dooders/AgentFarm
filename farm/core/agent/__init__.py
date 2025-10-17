"""Agent system module with component-based architecture."""

import sys

from .behaviors import DefaultAgentBehavior, IAgentBehavior, LearningAgentBehavior
from .components import (
    AgentComponent,
    CombatComponent,
    MovementComponent,
    PerceptionComponent,
    ReproductionComponent,
    ResourceComponent,
)
from .config import AgentComponentConfig
from .core import AgentCore
from .factory import AgentFactory
from .interfaces import IAgentComponent
from .services import AgentServices
from .state import StateManager, StateSnapshot


__all__ = [
    # Core
    "AgentCore",
    "AgentFactory",
    # Interfaces
    "IAgentComponent",
    "IAgentBehavior",
    # Components
    "AgentComponent",
    "MovementComponent",
    "ResourceComponent",
    "CombatComponent",
    "PerceptionComponent",
    "ReproductionComponent",
    # Behaviors
    "DefaultAgentBehavior",
    "LearningAgentBehavior",
    # Services
    "AgentServices",
    # Config
    "AgentComponentConfig",
    # State
    "StateManager",
    "StateSnapshot",
]
