"""
Default agent behavior - random action selection.

Simple behavior for testing and baseline comparison.
"""

from typing import Optional

import torch

from farm.core.action import Action, weighted_random_choice

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
        Select an action using weighted random selection based on action weights.
        
        Args:
            core: Agent core (can be None if enabled_actions is provided)
            state: Current state (unused)
            enabled_actions: Optional list of allowed actions
        
        Returns:
            Weighted randomly selected Action
        """
        # Use enabled_actions if provided, otherwise use core.actions
        if enabled_actions is not None:
            actions_list = enabled_actions
        elif core is not None:
            actions_list = core.actions
        else:
            raise ValueError("Either core or enabled_actions must be provided")
        
        # Pass actions_list as the actions parameter
        # weighted_random_choice uses the first parameter when the second is None
        action = weighted_random_choice(actions_list)
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
