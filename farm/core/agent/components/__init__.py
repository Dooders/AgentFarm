"""Agent component system."""

from .base import AgentComponent
from .combat import CombatComponent
from .movement import MovementComponent
from .perception import PerceptionComponent
from .reproduction import ReproductionComponent
from .resource import ResourceComponent

__all__ = [
    "AgentComponent",
    "MovementComponent",
    "ResourceComponent",
    "CombatComponent",
    "PerceptionComponent",
    "ReproductionComponent",
]
