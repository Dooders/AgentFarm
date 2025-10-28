"""
Default agent behavior - random action selection.

Simple behavior for testing and baseline comparison.
"""

import random
from typing import Optional

import torch

from farm.core.action import Action

from .base import IAgentBehavior


class DefaultAgentBehavior(IAgentBehavior):
    """
    Default behavior that randomly selects actions.
    
    Used for:
    - Testing and debugging
    - Baseline comparison
    - Agents without learned policies
    """
    
    def __init__(self):
        """Initialize default behavior."""
        self.action_history = []
    
    def decide_action(
        self,
        core,
        state: torch.Tensor,
        enabled_actions: Optional[list[Action]] = None,
    ) -> Action:
        """
        Select a random action.
        
        Args:
            core: Agent core
            state: Current state (unused)
            enabled_actions: Optional list of allowed actions
        
        Returns:
            Randomly selected Action
        """
        actions = enabled_actions if enabled_actions else core.actions
        
        if not actions:
            raise ValueError("No actions available to choose from")
        
        # Use per-agent RNG if available, fallback to global random
        if hasattr(core, '_py_rng'):
            action = core._py_rng.choice(actions)
        else:
            action = random.choice(actions)
        
        self.action_history.append(action.name)
        return action
    
    def update(
        self,
        state: torch.Tensor,
        action: Action,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """
        No learning in default behavior.
        
        Args:
            state: State before action
            action: Action executed
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
        """
        pass
    
    def reset(self) -> None:
        """Reset behavior state."""
        self.action_history = []
