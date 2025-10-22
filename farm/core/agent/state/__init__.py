"""
Agent state management module.

This module is deprecated. Use farm.core.state.AgentStateManager instead.
"""

import warnings

# Deprecated - use farm.core.state.AgentStateManager instead
from farm.core.state import AgentStateManager

warnings.warn(
    "The 'farm.core.agent.state' module is deprecated and will be removed in a future release. "
    "Use 'farm.core.state.AgentStateManager' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "AgentStateManager",
]
