"""Agent component system."""

from .base import AgentComponent
from .combat import CombatComponent
from .communication import CommunicationComponent, Message, MessageType
from .movement import MovementComponent
from .perception import PerceptionComponent
from .reproduction import ReproductionComponent
from .resource import ResourceComponent

__all__ = [
    "AgentComponent",
    "CommunicationComponent",
    "Message",
    "MessageType",
    "MovementComponent",
    "ResourceComponent",
    "CombatComponent",
    "PerceptionComponent",
    "ReproductionComponent",
]
