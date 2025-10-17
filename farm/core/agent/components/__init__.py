"""
Agent components providing specific capabilities.

Components follow the Single Responsibility Principle, each handling
one specific aspect of agent behavior (movement, resources, combat, etc.).
"""

from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.components.movement import MovementComponent
from farm.core.agent.components.resource import ResourceComponent
from farm.core.agent.components.combat import CombatComponent
from farm.core.agent.components.perception import PerceptionComponent
from farm.core.agent.components.reproduction import ReproductionComponent

__all__ = [
    "IAgentComponent",
    "MovementComponent",
    "ResourceComponent",
    "CombatComponent",
    "PerceptionComponent",
    "ReproductionComponent",
]