"""
Agent behavior interfaces.

Defines the behavior strategy interface for agent decision-making.
Allows pluggable decision-making algorithms (RL, random, heuristic, etc.).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from farm.core.action import Action

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class IAgentBehavior(ABC):
    """
    Interface for agent behavior/decision-making strategies.
    
    Implementations are strategies for selecting actions based on agent state.
    This follows the Strategy pattern and allows easy swapping of behavior algorithms.
    
    Different implementations could include:
    - Random action selection
    - Reinforcement learning (DDQN, PPO, etc.)
    - Heuristic-based rules
    - Scripted behavior
    """
    
    @abstractmethod
    def decide_action(
        self,
        core: "AgentCore",
        state: torch.Tensor,
        enabled_actions: Optional[list[Action]] = None,
    ) -> Action:
        """
        Decide which action to take based on current state.
        
        Args:
            core: The agent core making the decision
            state: Current state tensor (typically observation from environment)
            enabled_actions: Optional list of allowed actions (for curriculum learning).
                           If None, all actions are allowed.
        
        Returns:
            Action: The action to execute
        """
        pass
    
    @abstractmethod
    def update(
        self,
        state: torch.Tensor,
        action: Action,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Update behavior with new experience (for learning behaviors).
        
        Args:
            state: State before action
            action: Action executed
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset behavior state (e.g., for episode boundaries)."""
        pass
