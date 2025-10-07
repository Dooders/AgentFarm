"""
Agent components providing specific capabilities.

Components follow the Single Responsibility Principle, each handling
one specific aspect of agent behavior (movement, resources, combat, etc.).
"""

from farm.core.agent.components.base import IAgentComponent

__all__ = [
    "IAgentComponent",
]