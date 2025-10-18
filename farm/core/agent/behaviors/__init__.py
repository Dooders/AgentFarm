"""Agent behavior system."""

from .base import IAgentBehavior
from .default import DefaultAgentBehavior
from .learning import LearningAgentBehavior

__all__ = [
    "IAgentBehavior",
    "DefaultAgentBehavior",
    "LearningAgentBehavior",
]
