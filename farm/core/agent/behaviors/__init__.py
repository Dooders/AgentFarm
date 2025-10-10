"""
Agent behavior strategies.

Behaviors define how agents make decisions and act each turn.
Different behaviors can be swapped to create different agent types.
"""

from farm.core.agent.behaviors.base_behavior import IAgentBehavior
from farm.core.agent.behaviors.default_behavior import DefaultAgentBehavior
from farm.core.agent.behaviors.learning_behavior import LearningAgentBehavior

__all__ = [
    "IAgentBehavior",
    "DefaultAgentBehavior",
    "LearningAgentBehavior",
]