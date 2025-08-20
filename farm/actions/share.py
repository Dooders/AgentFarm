"""
Share action module for AgentFarm.

This module handles resource sharing between agents using Deep Q-Learning to learn
optimal sharing strategies. Updated to use the new profile-based configuration system.
"""

from typing import TYPE_CHECKING, Optional, Tuple, List
import logging

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder, DEVICE
from farm.core.profiles import DQNProfile

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ShareActionSpace:
    """Defines the available sharing actions and their corresponding amounts."""
    
    NO_SHARE: int = 0
    SHARE_LOW: int = 1    # Share minimum amount (1 resource)
    SHARE_MEDIUM: int = 2  # Share moderate amount (2 resources)
    SHARE_HIGH: int = 3   # Share larger amount (3 resources)


class ShareQNetwork(BaseQNetwork):
    """Neural network architecture for sharing Q-value approximation."""
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_size: int = 64,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        super().__init__(input_dim, 4, hidden_size, shared_encoder)  # 4 sharing actions


class ShareModule(BaseDQNModule):
    """
    Deep Q-Learning module for resource sharing.
    
    This module learns optimal sharing strategies, deciding how much to share
    with nearby agents based on cooperation history and current needs.
    """
    
    def __init__(
        self,
        dqn_profile: DQNProfile,
        rewards: dict = None,
        costs: dict = None,
        thresholds: dict = None,
        device: torch.device = DEVICE,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        """
        Initialize the sharing module.
        
        Args:
            dqn_profile: DQN learning configuration profile
            rewards: Sharing-specific reward values
            costs: Sharing-specific costs
            thresholds: Sharing-specific thresholds
            device: Computation device  
            shared_encoder: Optional shared encoder for efficiency
        """
        # Set default reward/cost/threshold values
        self.rewards = {
            "success": 0.3,
            "altruism_bonus": 0.2,
            **(rewards or {})
        }
        
        self.costs = {
            "failure_penalty": -0.1,
            **(costs or {})
        }
        
        self.thresholds = {
            "share_range": 30.0,
            "min_share_amount": 1,
            "max_resources": 30,
            "cooperation_memory": 100,
            **(thresholds or {})
        }
        
        super().__init__(
            input_dim=8,
            output_dim=4,
            dqn_profile=dqn_profile,
            device=device,
            shared_encoder=shared_encoder
        )
        
        self._setup_action_space()
        
        # Cooperation tracking
        self.cooperation_history = {}
    
    def _setup_action_space(self) -> None:
        """Initialize action space mapping."""
        self.action_space = ShareActionSpace()
        self.action_names = {
            0: "NO_SHARE",
            1: "SHARE_LOW",
            2: "SHARE_MEDIUM", 
            3: "SHARE_HIGH"
        }
    
    def get_share_decision(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[int, Optional["BaseAgent"], int]:
        """
        Make sharing decision for the agent.
        
        Args:
            agent: The agent making the decision
            state: Current environment state tensor
            
        Returns:
            Tuple of (share_amount, target_agent, action_taken)
        """
        action = self.select_action(state)
        
        if action == ShareActionSpace.NO_SHARE:
            return 0, None, action
        
        # Find nearby agents to share with
        nearby_agents = self._get_nearby_agents(agent)
        if not nearby_agents:
            return 0, None, action
        
        # Select target agent (prefer those with low resources)
        target_agent = self._select_target(agent, nearby_agents)
        
        # Calculate share amount based on action
        share_amount = self._calculate_share_amount(agent, action)
        
        return share_amount, target_agent, action
    
    def _get_nearby_agents(self, agent: "BaseAgent") -> List["BaseAgent"]:
        """Get agents within sharing range."""
        share_range = self.thresholds["share_range"]
        nearby_agents = agent.environment.get_nearby_agents(agent.position, share_range)
        
        # Filter out self and dead agents
        return [a for a in nearby_agents if a.agent_id != agent.agent_id and a.alive]
    
    def _select_target(
        self, agent: "BaseAgent", nearby_agents: List["BaseAgent"]
    ) -> "BaseAgent":
        """Select the best target for sharing (prioritize those with low resources)."""
        if not nearby_agents:
            return None
        
        # Sort by resource level (ascending) and cooperation score
        def target_score(target):
            resource_need = 1.0 - (target.resource_level / self.thresholds["max_resources"])
            cooperation_score = self._get_cooperation_score(target.agent_id)
            return resource_need * 0.7 + cooperation_score * 0.3
        
        return max(nearby_agents, key=target_score)
    
    def _calculate_share_amount(self, agent: "BaseAgent", action: int) -> int:
        """Calculate how much to share based on action."""
        share_amounts = {
            ShareActionSpace.SHARE_LOW: 1,
            ShareActionSpace.SHARE_MEDIUM: 2,
            ShareActionSpace.SHARE_HIGH: 3
        }
        
        max_share = share_amounts.get(action, 0)
        
        # Can't share more than agent has
        return min(max_share, agent.resource_level)
    
    def _get_cooperation_score(self, agent_id: str) -> float:
        """Get cooperation score for an agent based on history."""
        if agent_id not in self.cooperation_history:
            return 0.5  # Neutral score for unknown agents
        
        history = self.cooperation_history[agent_id]
        if len(history) == 0:
            return 0.5
        
        return sum(history) / len(history)
    
    def update_cooperation(self, agent_id: str, cooperative: bool) -> None:
        """Update cooperation history for an agent."""
        if agent_id not in self.cooperation_history:
            self.cooperation_history[agent_id] = []
        
        history = self.cooperation_history[agent_id]
        history.append(1.0 if cooperative else 0.0)
        
        # Keep only recent history
        max_memory = self.thresholds["cooperation_memory"]
        if len(history) > max_memory:
            history.pop(0)


def share_action(agent: "BaseAgent") -> None:
    """
    Execute sharing action for an agent.
    
    This function handles the complete sharing sequence including target selection,
    resource transfer, and cooperation tracking.
    
    Args:
        agent: The agent performing the sharing action
    """
    # Get sharing configuration from agent's config
    action_config = agent.environment.config.get_action_config("share")
    share_module = agent.share_module
    
    step_number = agent.environment.time
    resources_before = agent.resource_level
    
    # Get current state and make sharing decision
    state = agent.get_state().to_tensor(agent.device)
    share_amount, target_agent, action_taken = share_module.get_share_decision(agent, state)
    
    reward = 0.0
    
    if share_amount > 0 and target_agent and agent.resource_level >= share_amount:
        # Execute the share
        agent.resource_level -= share_amount
        target_agent.resource_level += share_amount
        
        # Calculate reward
        base_reward = action_config.rewards.get("success", 0.3)
        
        # Altruism bonus if target was in need
        if target_agent.resource_level < action_config.thresholds.get("max_resources", 30) * 0.3:
            base_reward += action_config.rewards.get("altruism_bonus", 0.2)
        
        # Amount multiplier
        reward = base_reward * (share_amount / action_config.thresholds.get("min_share_amount", 1))
        
        # Update cooperation tracking
        share_module.update_cooperation(target_agent.agent_id, True)
        
    elif action_taken != ShareActionSpace.NO_SHARE:
        # Intended to share but couldn't (no target or insufficient resources)
        reward = action_config.costs.get("failure_penalty", -0.1)
    
    # Store experience for learning
    next_state = agent.get_state().to_tensor(agent.device)
    
    share_module.store_experience(
        state=state,
        action=action_taken,
        reward=reward,
        next_state=next_state,
        done=False,
        step_number=step_number,
        agent_id=agent.agent_id,
        module_type="share",
        action_taken_mapped=share_module.action_names.get(action_taken, str(action_taken))
    )
    
    # Train the module periodically
    if len(share_module.memory) > share_module.profile.batch_size:
        if step_number % 4 == 0:  # Train every 4 steps
            share_module.train(
                step_number=step_number,
                agent_id=agent.agent_id,
                module_type="share"
            )