"""
Select action module for AgentFarm.

This module handles the meta-decision of which action type to execute using
Deep Q-Learning. Updated to use the new profile-based configuration system.
"""

from typing import TYPE_CHECKING, List
import logging

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder, DEVICE
from farm.core.profiles import DQNProfile, AgentBehaviorProfile
from farm.core.action import Action

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SelectQNetwork(BaseQNetwork):
    """Neural network architecture for action selection Q-value approximation."""
    
    def __init__(self, input_dim: int, num_actions: int, hidden_size: int = 64, 
                 shared_encoder: Optional[SharedEncoder] = None) -> None:
        super().__init__(input_dim, num_actions, hidden_size, shared_encoder)


class SelectModule(BaseDQNModule):
    """
    Deep Q-Learning module for action selection.
    
    This module learns which action type to choose in different situations,
    combining behavior profile weights with learned Q-values.
    """
    
    def __init__(
        self,
        num_actions: int,
        dqn_profile: DQNProfile,
        behavior_profile: AgentBehaviorProfile,
        device: torch.device = DEVICE,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        """
        Initialize the selection module.
        
        Args:
            num_actions: Number of possible actions
            dqn_profile: DQN learning configuration profile
            behavior_profile: Agent behavior profile for action weights
            device: Computation device
            shared_encoder: Optional shared encoder for efficiency
        """
        self.behavior_profile = behavior_profile
        
        super().__init__(
            input_dim=8,
            output_dim=num_actions,
            dqn_profile=dqn_profile,
            device=device,
            shared_encoder=shared_encoder
        )
    
    def select_action(
        self, agent: "BaseAgent", actions: List[Action], state: torch.Tensor
    ) -> Action:
        """
        Select which action to execute based on current state.
        
        Args:
            agent: The agent making the decision
            actions: Available actions to choose from
            state: Current environment state tensor
            
        Returns:
            Selected action to execute
        """
        # Get base probabilities from behavior profile
        base_probs = self._get_base_probabilities(actions)
        
        # Adjust probabilities based on agent state
        adjusted_probs = self._adjust_probabilities(agent, base_probs, actions)
        
        # Get Q-values from network
        with torch.no_grad():
            state_tensor = state.to(self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()
        
        # Combine probabilities and Q-values
        final_probs = self._combine_probs_and_qvalues(adjusted_probs, q_values)
        
        # Select action based on combined probabilities
        if np.random.random() < self.epsilon:
            # Exploration: use adjusted probabilities
            action_index = np.random.choice(len(actions), p=adjusted_probs)
        else:
            # Exploitation: use combined probabilities
            action_index = np.random.choice(len(actions), p=final_probs)
        
        return actions[action_index]
    
    def _get_base_probabilities(self, actions: List[Action]) -> List[float]:
        """Get base action probabilities from behavior profile."""
        action_weights = {
            "move": self.behavior_profile.move_weight,
            "gather": self.behavior_profile.gather_weight,
            "share": self.behavior_profile.share_weight,
            "attack": self.behavior_profile.attack_weight,
            "reproduce": self.behavior_profile.reproduce_weight
        }
        
        probs = []
        for action in actions:
            weight = action_weights.get(action.name, 0.1)  # Default weight
            probs.append(weight)
        
        # Normalize probabilities
        total = sum(probs)
        return [p / total for p in probs] if total > 0 else [1.0 / len(probs)] * len(probs)
    
    def _adjust_probabilities(
        self, agent: "BaseAgent", base_probs: List[float], actions: List[Action]
    ) -> List[float]:
        """Adjust probabilities based on agent state and environment."""
        adjusted_probs = base_probs.copy()
        
        # Get agent state info
        resource_ratio = agent.resource_level / 30.0  # Assume max 30 resources
        health_ratio = agent.current_health / agent.starting_health
        
        # Find nearby resources and agents
        nearby_resources = agent.environment.get_nearby_resources(agent.position, 30.0)
        nearby_agents = agent.environment.get_nearby_agents(agent.position, 30.0)
        
        has_nearby_resources = len(nearby_resources) > 0
        has_nearby_agents = len(nearby_agents) > 1  # Exclude self
        
        for i, action in enumerate(actions):
            action_name = action.name
            multiplier = 1.0
            
            if action_name == "move":
                # Increase move probability if no resources nearby
                if not has_nearby_resources:
                    multiplier *= 1.5
                    
            elif action_name == "gather":
                # Increase gather probability if low resources and resources nearby
                if resource_ratio < 0.3 and has_nearby_resources:
                    multiplier *= 1.5
                elif not has_nearby_resources:
                    multiplier *= 0.1  # Very low if no resources
                    
            elif action_name == "share":
                # Adjust sharing based on wealth and cooperation tendency
                if resource_ratio > 0.7:  # Wealthy
                    multiplier *= 1.0 + self.behavior_profile.cooperation_tendency
                elif resource_ratio < 0.3:  # Poor
                    multiplier *= 0.5
                if not has_nearby_agents:
                    multiplier *= 0.1  # Can't share if alone
                    
            elif action_name == "attack":
                # Adjust attack based on aggression and desperation
                aggression_mult = 0.5 + self.behavior_profile.aggression_level
                if resource_ratio < 0.3:  # Desperate
                    multiplier *= 1.4 * aggression_mult
                else:
                    multiplier *= 0.6 * aggression_mult
                if not has_nearby_agents:
                    multiplier *= 0.1  # Can't attack if alone
                    
            elif action_name == "reproduce":
                # Adjust reproduction based on resources and health
                if resource_ratio > 0.7 and health_ratio > 0.5:
                    multiplier *= 1.4
                elif resource_ratio < 0.5:
                    multiplier *= 0.3
            
            adjusted_probs[i] *= multiplier
        
        # Renormalize
        total = sum(adjusted_probs)
        return [p / total for p in adjusted_probs] if total > 0 else base_probs
    
    def _combine_probs_and_qvalues(
        self, probs: List[float], q_values: np.ndarray
    ) -> List[float]:
        """Combine behavior probabilities with learned Q-values."""
        # Normalize Q-values to probabilities using softmax
        exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        q_probs = exp_q / np.sum(exp_q)
        
        # Combine with behavior probabilities (weighted average)
        alpha = 0.7  # Weight for Q-values vs behavior probs
        combined = alpha * q_probs + (1 - alpha) * np.array(probs)
        
        # Renormalize
        return combined / np.sum(combined)


def create_selection_state(agent: "BaseAgent") -> torch.Tensor:
    """Create state representation for action selection."""
    # Create state vector with relevant factors for action selection
    nearby_resources = agent.environment.get_nearby_resources(agent.position, 30.0)
    nearby_agents = agent.environment.get_nearby_agents(agent.position, 30.0)
    
    state_vector = [
        # Agent state
        agent.resource_level / 30.0,  # Normalized resources
        agent.current_health / agent.starting_health,  # Health ratio
        agent.age / 1000.0,  # Normalized age
        
        # Environment state
        len(nearby_resources) / 10.0,  # Nearby resource count
        len(nearby_agents) / 10.0,  # Nearby agent count
        
        # Resource availability
        total_nearby_resources = sum(r.amount for r in nearby_resources) / 100.0,
        
        # Population pressure
        len(agent.environment.agents) / agent.environment.config.max_population,
        
        # Time factor
        agent.environment.time / 1000.0  # Normalized time
    ]
    
    return torch.tensor(state_vector, dtype=torch.float32, device=agent.device)