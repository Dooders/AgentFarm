"""
Learning agent behavior - reinforcement learning based.

Wraps the DecisionModule for RL-based action selection and learning.
"""

from typing import Optional

import numpy as np
import torch

from farm.core.action import Action, action_registry, action_name_to_index
from farm.core.decision.action_weight_policy import (
    compute_action_weights,
    extract_signals,
)
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

        # Build the per-action weight vector, then scale by state-aware
        # multiplier / threshold genes (Chromosome B).  ``core.actions[i].weight``
        # already reflects evolved base-weight genes via
        # ``AgentCore.refresh_action_weights_from_decision_config``; this layer
        # adds the situation-dependent biasing on top.
        num_actions = self.decision_module.num_actions
        base_weights = np.zeros(num_actions, dtype=np.float64)
        action_names = ["" for _ in range(num_actions)]
        for i, action in enumerate(core.actions):
            if i < num_actions:
                base_weights[i] = action.weight
                action_names[i] = action.name

        decision_config = getattr(getattr(core, "config", None), "decision", None)
        if decision_config is not None:
            try:
                signals = extract_signals(core)
                action_weights = compute_action_weights(
                    base_weights=base_weights,
                    action_names=action_names,
                    decision_config=decision_config,
                    signals=signals,
                    enabled_actions=enabled_action_indices,
                )
            except Exception as exc:
                logger.debug(
                    "action_weight_policy_fallback",
                    agent_id=getattr(core, "agent_id", "unknown"),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                action_weights = base_weights
        else:
            action_weights = base_weights

        total_weight = np.sum(action_weights)
        if total_weight > 0:
            action_weights = action_weights / total_weight
        else:
            action_weights = np.ones(num_actions) / num_actions

        # Use DecisionModule to select action with weights
        action_index = self.decision_module.decide_action(
            state, enabled_action_indices, action_weights=action_weights
        )
        
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
        try:
            # Convert action to index
            action_index = action_name_to_index(action.name)
            
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
