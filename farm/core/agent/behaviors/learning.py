"""
Learning agent behavior - reinforcement learning based.

Wraps the DecisionModule for RL-based action selection and learning.
"""

from typing import Optional

import torch

from farm.core.action import Action, action_registry, action_name_to_index
from farm.core.decision.decision import DecisionModule
from farm.utils.logging import get_logger

from .base import IAgentBehavior

logger = get_logger(__name__)


class LearningAgentBehavior(IAgentBehavior):
    """
    RL-based behavior using DecisionModule for intelligent action selection.
    
    Integrates with existing DecisionModule to provide:
    - DDQN, PPO, and other RL algorithms
    - Curriculum learning support
    - Experience replay and learning
    """
    
    def __init__(self, decision_module: DecisionModule):
        """
        Initialize learning behavior.
        
        Args:
            decision_module: DecisionModule for RL-based decision making
        """
        self.decision_module = decision_module
        self.last_action_index = None
        self.action_registry = action_registry
    
    def decide_action(
        self,
        core,
        state: torch.Tensor,
        enabled_actions: Optional[list[Action]] = None,
    ) -> Action:
        """
        Use DecisionModule to select action with RL.
        
        Args:
            core: Agent core
            state: Current observation tensor
            enabled_actions: Optional list of allowed actions (for curriculum)
        
        Returns:
            Action selected by RL algorithm
        """
        actions = enabled_actions if enabled_actions else core.actions
        
        if not actions:
            raise ValueError("No actions available to choose from")
        
        # Get enabled action indices if curriculum is used
        if enabled_actions:
            enabled_action_indices = [core.actions.index(action) for action in enabled_actions]
        else:
            enabled_action_indices = None
        
        # Use DecisionModule to select action
        action_index = self.decision_module.decide_action(state, enabled_action_indices)
        
        # Handle curriculum case - action_index is relative to enabled_actions
        if enabled_actions:
            action = enabled_actions[action_index]
        else:
            action = core.actions[action_index]
        
        self.last_action_index = action_index
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
        Update DecisionModule with experience.
        
        Args:
            state: State before action
            action: Action executed
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
        """
        print(f"DEBUG: Learning behavior update called: action={action.name}, reward={reward}")
        try:
            # Convert action to index
            action_index = action_name_to_index(action.name)
            print(f"DEBUG: Converted action {action.name} to index {action_index}")
            
            # Update decision module
            self.decision_module.update(
                state=state,
                action=action_index,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        except Exception as e:
            # Log but don't crash on update failure
            logger.warning(f"Failed to update decision module: {e}", exc_info=True)
    
    def reset(self) -> None:
        """Reset behavior state."""
        self.last_action_index = None
