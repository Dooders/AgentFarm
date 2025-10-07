"""
Agent actions with proper object-oriented design.

Actions are objects that encapsulate behavior, validation, and execution logic.
This replaces the function-based action system with a proper class hierarchy.
"""

from farm.core.agent.actions.base import IAction

__all__ = [
    "IAction",
]