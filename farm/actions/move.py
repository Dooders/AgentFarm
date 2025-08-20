"""
Movement action module for AgentFarm.

This module handles agent movement using Deep Q-Learning to learn optimal
movement policies. Updated to use the new profile-based configuration system.
"""

from typing import TYPE_CHECKING, Optional, Tuple
import logging
import random

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder, DEVICE
from farm.core.profiles import DQNProfile

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent
    from farm.core.environment import Environment
    from farm.core.resources import Resource
    from farm.core.state import ModelState

logger = logging.getLogger(__name__)


class MoveActionSpace:
    """Constants defining the discrete action space for movement."""
    
    RIGHT: int = 0
    LEFT: int = 1  
    UP: int = 2
    DOWN: int = 3


class MoveQNetwork(BaseQNetwork):
    """Neural network architecture for movement Q-value approximation."""
    
    def __init__(self, input_dim: int, hidden_size: int = 64, 
                 shared_encoder: Optional[SharedEncoder] = None) -> None:
        super().__init__(input_dim, 4, hidden_size, shared_encoder)  # 4 movement actions


class MoveModule(BaseDQNModule):
    """
    Deep Q-Learning module for movement actions.
    
    This module learns optimal movement strategies through experience,
    with rewards/costs configured through the profile system.
    """
    
    def __init__(
        self,
        dqn_profile: DQNProfile,
        rewards: dict = None,
        costs: dict = None,
        device: torch.device = DEVICE,
        db: Optional["SimulationDatabase"] = None,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        """
        Initialize the movement module.
        
        Args:
            dqn_profile: DQN learning configuration profile
            rewards: Movement-specific reward values
            costs: Movement-specific costs  
            device: Computation device
            db: Database for logging
            shared_encoder: Optional shared encoder for efficiency
        """
        # Set default reward/cost values
        self.rewards = {
            "approach_resource": 0.3,
            "retreat_penalty": -0.2,
            **(rewards or {})
        }
        
        self.costs = {
            "base": -0.1,
            **(costs or {})
        }
        
        super().__init__(
            input_dim=8,
            output_dim=4,
            dqn_profile=dqn_profile,
            device=device,
            db=db,
            shared_encoder=shared_encoder
        )
        
        self._setup_action_space()
    
    def _setup_action_space(self) -> None:
        """Initialize action space mapping."""
        self.action_space = MoveActionSpace()
        self.action_names = {
            0: "RIGHT",
            1: "LEFT",
            2: "UP", 
            3: "DOWN"
        }
    
    def get_movement(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Get movement direction for the agent.
        
        Args:
            agent: The agent requesting movement
            state: Current environment state tensor
            
        Returns:
            Tuple of (dx, dy) movement values
        """
        action = self.select_action(state)
        
        # Map action to movement
        movement_map = {
            MoveActionSpace.RIGHT: (1, 0),
            MoveActionSpace.LEFT: (-1, 0),
            MoveActionSpace.UP: (0, 1),
            MoveActionSpace.DOWN: (0, -1)
        }
        
        return movement_map.get(action, (0, 0))
    
    def get_state(self) -> "ModelState":
        """Get current model state for monitoring."""
        from farm.core.state import ModelState
        
        return ModelState(
            learning_rate=self.profile.learning_rate,
            epsilon=self.epsilon,
            latest_loss=self.losses[-1] if self.losses else None,
            latest_reward=self.episode_rewards[-1] if self.episode_rewards else None,
            memory_size=len(self.memory),
            memory_capacity=self.profile.memory_size,
            steps=self.training_step,
            architecture={"hidden_size": self.profile.hidden_size, "output_dim": self.output_dim},
            training_metrics={
                "avg_loss": float(np.mean(self.losses[-100:])) if len(self.losses) >= 10 else 0.0,
                "avg_reward": float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 10 else 0.0
            }
        )


def move_action(agent: "BaseAgent") -> None:
    """
    Execute movement action for an agent.
    
    This function handles the complete movement sequence including direction selection,
    boundary checking, and reward calculation.
    
    Args:
        agent: The agent performing the movement action
    """
    # Get movement configuration from agent's config
    action_config = agent.environment.config.get_action_config("move")
    move_module = agent.move_module
    
    step_number = agent.environment.time
    initial_position = agent.position
    resources_before = agent.resource_level
    
    # Get current state
    state = _ensure_tensor(agent.get_state(), agent.device)
    
    # Get movement from module
    dx, dy = move_module.get_movement(agent, state)
    
    # Calculate new position with boundary checking
    new_x = max(0, min(agent.environment.width - 1, agent.position[0] + dx))
    new_y = max(0, min(agent.environment.height - 1, agent.position[1] + dy))
    new_position = (new_x, new_y)
    
    # Update agent position
    agent.position = new_position
    
    # Apply movement cost
    base_cost = action_config.costs.get("base", -0.1)
    agent.resource_level = max(0, agent.resource_level + base_cost)
    
    # Calculate movement reward
    reward = _calculate_movement_reward(agent, initial_position, new_position)
    
    # Store experience for learning
    next_state = _ensure_tensor(agent.get_state(), agent.device)
    
    # Determine action index from movement
    action_map = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}
    action = action_map.get((dx, dy), 0)
    
    move_module.store_experience(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=False,
        step_number=step_number,
        agent_id=agent.agent_id,
        module_type="move",
        action_taken_mapped=move_module.action_names.get(action, str(action))
    )
    
    # Train the module periodically  
    if len(move_module.memory) > move_module.profile.batch_size:
        if step_number % 4 == 0:  # Train every 4 steps
            move_module.train(
                step_number=step_number,
                agent_id=agent.agent_id,
                module_type="move"
            )


def _calculate_movement_reward(
    agent: "BaseAgent",
    initial_position: Tuple[float, float],
    new_position: Tuple[float, float],
) -> float:
    """
    Calculate reward for movement based on resource proximity.
    
    Args:
        agent: The agent that moved
        initial_position: Position before movement
        new_position: Position after movement
        
    Returns:
        Calculated movement reward
    """
    action_config = agent.environment.config.get_action_config("move")
    
    # Base cost is already applied, calculate proximity bonus/penalty
    reward = 0.0
    
    # Find closest resource for reward calculation
    closest_resource = _find_closest_resource(agent.environment, new_position)
    
    if closest_resource:
        # Calculate distance change
        old_distance = _calculate_distance(initial_position, closest_resource.position)
        new_distance = _calculate_distance(new_position, closest_resource.position)
        
        distance_change = old_distance - new_distance
        
        if distance_change > 0:
            # Moved closer to resource
            reward += action_config.rewards.get("approach_resource", 0.3)
        elif distance_change < 0:
            # Moved away from resource
            reward += action_config.rewards.get("retreat_penalty", -0.2)
    
    return reward


def _find_closest_resource(
    environment: "Environment", position: Tuple[float, float]
) -> Optional["Resource"]:
    """Find the closest resource to a given position."""
    if not environment.resources:
        return None
        
    closest_resource = None
    min_distance = float('inf')
    
    for resource in environment.resources:
        if resource.amount > 0:  # Only consider non-depleted resources
            distance = _calculate_distance(position, resource.position)
            if distance < min_distance:
                min_distance = distance
                closest_resource = resource
                
    return closest_resource


def _calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def _ensure_tensor(state: Any, device: torch.device) -> torch.Tensor:
    """Ensure state is a tensor on the correct device."""
    if isinstance(state, torch.Tensor):
        return state.to(device)
    elif hasattr(state, 'to_tensor'):
        return state.to_tensor(device)
    else:
        # Convert to tensor if it's a list/array
        return torch.tensor(state, dtype=torch.float32, device=device)


def _store_and_train(agent: "BaseAgent", state: Any, reward: float) -> None:
    """Store experience and train if enough samples available."""
    # This is a legacy function - functionality moved to move_action
    pass