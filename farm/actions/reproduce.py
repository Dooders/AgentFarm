"""
Reproduce action module for AgentFarm.

This module handles agent reproduction using Deep Q-Learning to learn optimal
reproduction timing. Updated to use the new profile-based configuration system.
"""

from typing import TYPE_CHECKING, Optional, Tuple
import logging

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder, DEVICE
from farm.core.profiles import DQNProfile

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReproduceActionSpace:
    """Defines the action space for reproduction decisions."""
    
    WAIT: int = 0      # Wait for better conditions
    REPRODUCE: int = 1  # Attempt reproduction


class ReproduceQNetwork(BaseQNetwork):
    """Neural network architecture for reproduction Q-value approximation."""
    
    def __init__(self, input_dim: int = 8, hidden_size: int = 64, 
                 shared_encoder: Optional[SharedEncoder] = None) -> None:
        super().__init__(input_dim, 2, hidden_size, shared_encoder)  # 2 reproduction actions


class ReproduceModule(BaseDQNModule):
    """
    Deep Q-Learning module for reproduction decisions.
    
    This module learns optimal reproduction timing, considering agent health,
    resources, population density, and environmental conditions.
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
        Initialize the reproduction module.
        
        Args:
            dqn_profile: DQN learning configuration profile
            rewards: Reproduction-specific reward values
            costs: Reproduction-specific costs
            thresholds: Reproduction-specific thresholds
            device: Computation device
            shared_encoder: Optional shared encoder for efficiency
        """
        # Set default reward/cost/threshold values
        self.rewards = {
            "success": 1.0,
            "failure_penalty": -0.2,
            "offspring_survival": 0.5,
            "population_balance": 0.3,
            **(rewards or {})
        }
        
        self.costs = {
            "offspring_cost": 3,
            **(costs or {})
        }
        
        self.thresholds = {
            "min_health": 0.5,
            "min_resources": 8,
            "ideal_density_radius": 50.0,
            "max_local_density": 0.7,
            "min_space_required": 20.0,
            **(thresholds or {})
        }
        
        super().__init__(
            input_dim=8,
            output_dim=2,
            dqn_profile=dqn_profile,
            device=device,
            shared_encoder=shared_encoder
        )
        
        self._setup_action_space()
    
    def _setup_action_space(self) -> None:
        """Initialize action space mapping."""
        self.action_space = ReproduceActionSpace()
        self.action_names = {
            0: "WAIT",
            1: "REPRODUCE"
        }
    
    def get_reproduction_decision(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Make reproduction decision for the agent.
        
        Args:
            agent: The agent making the decision
            state: Current environment state tensor
            
        Returns:
            Tuple of (should_reproduce, confidence_score)
        """
        action = self.select_action(state)
        
        if action == ReproduceActionSpace.REPRODUCE:
            # Check if reproduction conditions are met
            if _check_reproduction_conditions(agent, self.thresholds):
                return True, 0.8
            else:
                # Wanted to reproduce but conditions not met
                return False, 0.2
        else:
            # Decided to wait
            return False, 0.6


def reproduce_action(agent: "BaseAgent") -> None:
    """
    Execute reproduction action for an agent.
    
    This function handles the complete reproduction sequence including condition
    checking, offspring creation, and reward calculation.
    
    Args:
        agent: The agent attempting reproduction
    """
    # Get reproduction configuration from agent's config
    action_config = agent.environment.config.get_action_config("reproduce")
    reproduce_module = agent.reproduce_module
    
    step_number = agent.environment.time
    resources_before = agent.resource_level
    
    # Get current state and make reproduction decision
    state = _get_reproduce_state(agent)
    should_reproduce, confidence = reproduce_module.get_reproduction_decision(agent, state)
    
    reward = 0.0
    offspring = None
    
    if should_reproduce:
        # Check reproduction conditions
        if _check_reproduction_conditions(agent, action_config.thresholds):
            # Attempt reproduction
            offspring_cost = action_config.costs.get("offspring_cost", 3)
            
            if agent.resource_level >= offspring_cost:
                # Create offspring
                offspring = agent.create_offspring()
                
                if offspring:
                    # Successful reproduction
                    agent.resource_level -= offspring_cost
                    reward = _calculate_reproduction_reward(agent, offspring, action_config)
                    
                    # Log reproduction event
                    agent.environment.db.log_reproduction_event(
                        step_number=step_number,
                        parent_id=agent.agent_id,
                        success=True,
                        parent_resources_before=resources_before,
                        parent_resources_after=agent.resource_level,
                        offspring_id=offspring.agent_id,
                        offspring_initial_resources=offspring.resource_level,
                        failure_reason=None,
                        parent_position=agent.position,
                        parent_generation=agent.generation,
                        offspring_generation=offspring.generation
                    )
                else:
                    # Failed to create offspring
                    reward = action_config.rewards.get("failure_penalty", -0.2)
            else:
                # Insufficient resources
                reward = action_config.rewards.get("failure_penalty", -0.2)
        else:
            # Conditions not met
            reward = action_config.rewards.get("failure_penalty", -0.2)
    else:
        # Decided to wait - neutral reward
        reward = 0.0
    
    # Store experience for learning
    next_state = _get_reproduce_state(agent)
    action_taken = ReproduceActionSpace.REPRODUCE if should_reproduce else ReproduceActionSpace.WAIT
    
    reproduce_module.store_experience(
        state=state,
        action=action_taken,
        reward=reward,
        next_state=next_state,
        done=False,
        step_number=step_number,
        agent_id=agent.agent_id,
        module_type="reproduce",
        action_taken_mapped=reproduce_module.action_names.get(action_taken, str(action_taken))
    )
    
    # Train the module periodically
    if len(reproduce_module.memory) > reproduce_module.profile.batch_size:
        if step_number % 4 == 0:  # Train every 4 steps
            reproduce_module.train(
                step_number=step_number,
                agent_id=agent.agent_id,
                module_type="reproduce"
            )


def _get_reproduce_state(agent: "BaseAgent") -> torch.Tensor:
    """Get state representation for reproduction decisions."""
    # Create state vector with relevant reproduction factors
    state_vector = [
        # Agent state
        agent.resource_level / 30.0,  # Normalized resources
        agent.current_health / agent.starting_health,  # Health ratio
        agent.age / 1000.0,  # Normalized age
        
        # Population state
        len(agent.environment.agents) / agent.environment.config.max_population,
        
        # Local density
        nearby_agents = agent.environment.get_nearby_agents(agent.position, 50.0)
        len(nearby_agents) / 10.0,  # Normalized local density
        
        # Resource availability
        nearby_resources = agent.environment.get_nearby_resources(agent.position, 50.0)
        total_nearby_resources = sum(r.amount for r in nearby_resources) / 100.0,
        
        # Environmental factors
        agent.environment.time / 1000.0,  # Normalized time
        
        # Generation info
        agent.generation / 10.0  # Normalized generation
    ]
    
    return torch.tensor(state_vector, dtype=torch.float32, device=agent.device)


def _check_reproduction_conditions(agent: "BaseAgent", thresholds: dict) -> bool:
    """Check if agent meets reproduction conditions."""
    # Health check
    health_ratio = agent.current_health / agent.starting_health
    if health_ratio < thresholds.get("min_health", 0.5):
        return False
    
    # Resource check
    min_resources = thresholds.get("min_resources", 8)
    if agent.resource_level < min_resources:
        return False
    
    # Population density check
    density_radius = thresholds.get("ideal_density_radius", 50.0)
    max_density = thresholds.get("max_local_density", 0.7)
    
    nearby_agents = agent.environment.get_nearby_agents(agent.position, density_radius)
    local_density = len(nearby_agents) / (density_radius * density_radius * 3.14159)
    
    if local_density > max_density:
        return False
    
    # Space requirement check
    min_space = thresholds.get("min_space_required", 20.0)
    if nearby_agents:
        closest_distance = min(
            np.sqrt((agent.position[0] - other.position[0])**2 + 
                   (agent.position[1] - other.position[1])**2)
            for other in nearby_agents if other.agent_id != agent.agent_id
        )
        if closest_distance < min_space:
            return False
    
    return True


def _calculate_reproduction_reward(agent: "BaseAgent", offspring: "BaseAgent", action_config) -> float:
    """Calculate reward for successful reproduction."""
    base_reward = action_config.rewards.get("success", 1.0)
    
    # Offspring survival bonus (based on initial resources given)
    survival_bonus = (offspring.resource_level / 10.0) * action_config.rewards.get("offspring_survival", 0.5)
    
    # Population balance bonus
    total_population = len(agent.environment.agents)
    optimal_population = agent.environment.config.max_population * 0.6
    balance_factor = 1.0 - abs(total_population - optimal_population) / optimal_population
    balance_bonus = balance_factor * action_config.rewards.get("population_balance", 0.3)
    
    return base_reward + survival_bonus + balance_bonus