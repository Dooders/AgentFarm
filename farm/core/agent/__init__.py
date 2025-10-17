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

# For backward compatibility: re-export BaseAgent from the legacy module file
# farm/core/agent.py contains the original BaseAgent implementation
# We need to import it explicitly to avoid shadowing by this package
BaseAgent = None
try:
    # Get the agent module file (farm.core.agent as a module, not this package)
    # We import it with a different name to avoid conflicts
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "farm_core_agent_module",
        __file__.replace("__init__.py", "../agent.py"),
    )
    if spec and spec.loader:
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        BaseAgent = getattr(agent_module, "BaseAgent", None)
except Exception:
    # If this fails, BaseAgent won't be available - that's okay for the new architecture
    pass

__all__ = [
    # Core
    "AgentCore",
    "AgentFactory",
    "BaseAgent",  # Legacy, for backward compatibility
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
