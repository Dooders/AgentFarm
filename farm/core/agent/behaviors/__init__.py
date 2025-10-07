"""
Agent behavior strategies.

Behaviors define how agents make decisions and act each turn.
Different behaviors can be swapped to create different agent types.
"""

from farm.core.agent.behaviors.base_behavior import IAgentBehavior

__all__ = [
    "IAgentBehavior",
]