"""
Gather action module for AgentFarm.

This module handles resource gathering using Deep Q-Learning to learn optimal
gathering strategies. Updated to use the new profile-based configuration system.
"""

from typing import TYPE_CHECKING, Optional, Tuple
import logging

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder, DEVICE
from farm.core.profiles import DQNProfile

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent
    from farm.core.resources import Resource

logger = logging.getLogger(__name__)


class GatherActionSpace:
    """Defines the possible actions for the gathering decision process."""
    
    GATHER: int = 0  # Attempt gathering
    WAIT: int = 1    # Wait for better opportunity  
    SKIP: int = 2    # Skip gathering this step


class GatherQNetwork(BaseQNetwork):
    """Neural network architecture for gathering Q-value approximation."""
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_size: int = 64,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        super().__init__(input_dim, 3, hidden_size, shared_encoder)  # 3 gather actions


class GatherModule(BaseDQNModule):
    """
    Deep Q-Learning module for resource gathering.
    
    This module learns optimal gathering strategies, deciding when to gather,
    wait for better opportunities, or skip gathering entirely.
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
        Initialize the gathering module.
        
        Args:
            dqn_profile: DQN learning configuration profile
            rewards: Gathering-specific reward values
            costs: Gathering-specific costs
            thresholds: Gathering-specific thresholds
            device: Computation device
            shared_encoder: Optional shared encoder for efficiency
        """
        # Set default reward/cost/threshold values
        self.rewards = {
            "success": 1.0,
            "failure_penalty": -0.1,
            "efficiency_bonus": 0.5,
            **(rewards or {})
        }
        
        self.costs = {
            "base": -0.05,
            "movement_penalty": 0.3,
            **(costs or {})
        }
        
        self.thresholds = {
            "min_resource": 0.1,
            "max_wait_steps": 5,
            "gathering_range": 30,
            **(thresholds or {})
        }
        
        super().__init__(
            input_dim=8,
            output_dim=3,
            dqn_profile=dqn_profile,
            device=device,
            shared_encoder=shared_encoder
        )
        
        self._setup_action_space()
        
        # Gathering state tracking
        self.wait_steps = 0
        self.last_target = None
    
    def _setup_action_space(self) -> None:
        """Initialize action space mapping."""
        self.action_space = GatherActionSpace()
        self.action_names = {
            0: "GATHER",
            1: "WAIT",
            2: "SKIP"
        }
    
    def get_gather_decision(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[bool, Optional["Resource"]]:
        """
        Make gathering decision for the agent.
        
        Args:
            agent: The agent making the decision
            state: Current environment state tensor
            
        Returns:
            Tuple of (should_gather, target_resource)
        """
        action = self.select_action(state)
        
        if action == GatherActionSpace.GATHER:
            target_resource = self._find_best_resource(agent)
            if target_resource and target_resource.amount >= self.thresholds["min_resource"]:
                self.wait_steps = 0
                self.last_target = target_resource
                return True, target_resource
            else:
                # No valid target, treat as SKIP
                return False, None
                
        elif action == GatherActionSpace.WAIT:
            self.wait_steps += 1
            # Force gather if waited too long
            if self.wait_steps >= self.thresholds["max_wait_steps"]:
                target_resource = self._find_best_resource(agent)
                self.wait_steps = 0
                return True, target_resource
            return False, None
            
        else:  # SKIP
            self.wait_steps = 0
            return False, None
    
    def _find_best_resource(self, agent: "BaseAgent") -> Optional["Resource"]:
        """Find the best resource for gathering."""
        gathering_range = self.thresholds.get("gathering_range", 30)
        nearby_resources = agent.environment.get_nearby_resources(
            agent.position, gathering_range
        )
        
        if not nearby_resources:
            return None
        
        # Score resources by amount and distance
        def score_resource(resource):
            if resource.amount <= 0:
                return -1
                
            distance = np.sqrt(
                (agent.position[0] - resource.position[0])**2 +
                (agent.position[1] - resource.position[1])**2
            )
            
            # Higher score for more resources and closer distance
            return resource.amount / (1 + distance * 0.1)
        
        return max(nearby_resources, key=score_resource)
    
    def calculate_gather_reward(
        self,
        agent: "BaseAgent", 
        initial_resources: float,
        target_resource: Optional["Resource"],
    ) -> float:
        """Calculate reward for gathering attempt."""
        # Base reward/penalty
        resource_change = agent.resource_level - initial_resources
        
        if resource_change > 0:
            # Successful gathering
            base_reward = self.rewards["success"]
            
            # Efficiency bonus for gathering larger amounts
            efficiency_bonus = (resource_change - 1) * self.rewards["efficiency_bonus"]
            
            return base_reward + efficiency_bonus
        else:
            # Failed gathering
            return self.rewards["failure_penalty"]


def gather_action(agent: "BaseAgent") -> None:
    """
    Execute gathering action for an agent.
    
    This function handles the complete gathering sequence including resource
    selection, gathering execution, and reward calculation.
    
    Args:
        agent: The agent performing the gathering action
    """
    # Get gathering configuration from agent's config
    action_config = agent.environment.config.get_action_config("gather")
    gather_module = agent.gather_module
    
    step_number = agent.environment.time
    resources_before = agent.resource_level
    
    # Get current state and make gathering decision
    state = agent.get_state().to_tensor(agent.device)
    should_gather, target_resource = gather_module.get_gather_decision(agent, state)
    
    reward = 0.0
    
    if should_gather and target_resource:
        # Attempt to gather from target resource
        max_gather = min(
            agent.environment.config.max_gather_amount,
            target_resource.amount,
            agent.environment.config.max_resource_amount - agent.resource_level
        )
        
        if max_gather > 0:
            # Successful gathering
            agent.resource_level += max_gather
            target_resource.consume(max_gather)
            
            # Calculate reward
            reward = gather_module.calculate_gather_reward(
                agent, resources_before, target_resource
            )
        else:
            # No resources could be gathered
            reward = action_config.rewards.get("failure_penalty", -0.1)
    else:
        # Decided not to gather (WAIT or SKIP)
        reward = 0.0  # Neutral reward for waiting/skipping
    
    # Store experience for learning
    next_state = agent.get_state().to_tensor(agent.device)
    
    # Map decision to action index
    if should_gather:
        action_taken = GatherActionSpace.GATHER
    elif gather_module.wait_steps > 0:
        action_taken = GatherActionSpace.WAIT
    else:
        action_taken = GatherActionSpace.SKIP
    
    gather_module.store_experience(
        state=state,
        action=action_taken,
        reward=reward,
        next_state=next_state,
        done=False,
        step_number=step_number,
        agent_id=agent.agent_id,
        module_type="gather",
        action_taken_mapped=gather_module.action_names.get(action_taken, str(action_taken))
    )
    
    # Train the module periodically
    if len(gather_module.memory) > gather_module.profile.batch_size:
        if step_number % 4 == 0:  # Train every 4 steps
            gather_module.train(
                step_number=step_number,
                agent_id=agent.agent_id,
                module_type="gather"
            )