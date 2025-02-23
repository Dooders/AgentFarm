"""Movement learning and execution module using Deep Q-Learning (DQN).

This module implements a Deep Q-Learning approach for agents to learn optimal movement
policies in a 2D environment. It provides both the neural network architecture and
the training/execution logic.

Key Components:
    - MoveQNetwork: Neural network architecture for Q-value approximation
    - MoveModule: Main class handling training, action selection, and movement execution
    - Experience Replay: Stores and samples past experiences for stable learning
    - Target Network: Separate network for computing stable Q-value targets

Technical Details:
    - State Space: N-dimensional vector representing agent's current state
    - Action Space: 4 discrete actions (right, left, up, down)
    - Learning Algorithm: Deep Q-Learning with experience replay
    - Exploration: Epsilon-greedy strategy with decay
    - Hardware Acceleration: Automatic GPU usage when available

Example Usage:
    ```python
    # Initialize configuration
    config = MovementConfig(
        learning_rate=0.001,
        memory_size=10000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        target_update_freq=100
    )
    
    # Create and use module
    move_module = MoveModule(config)
    next_position = move_module.get_movement(agent, current_state)
    move_module.store_experience(state, action, reward, next_state)
    loss = move_module.train(batch)
    ```

Dependencies:
    - torch: Deep learning framework for neural network implementation
    - random: For exploration strategy implementation
    - collections.deque: For experience replay buffer management
"""

import logging
import random
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork
from farm.database.database import SimulationDatabase

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent
    from farm.core.environment import Environment
    from farm.core.resources import Resource
    from farm.core.state import ModelState

logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoveConfig(BaseDQNConfig):
    """Configuration specific to movement."""

    move_base_cost: float = -0.1
    move_resource_approach_reward: float = 0.3
    move_resource_retreat_penalty: float = -0.2


DEFAULT_MOVE_CONFIG = MoveConfig()


class MoveActionSpace:
    RIGHT: int = 0
    LEFT: int = 1
    UP: int = 2
    DOWN: int = 3


class MoveQNetwork(BaseQNetwork):
    """Movement-specific Q-network."""

    def __init__(self, input_dim: int, hidden_size: int = 64) -> None:
        super().__init__(
            input_dim, output_dim=4, hidden_size=hidden_size
        )  # 4 movement actions


class MoveModule(BaseDQNModule):
    """Movement-specific DQN module."""

    def __init__(
        self,
        config: MoveConfig = DEFAULT_MOVE_CONFIG,
        device: torch.device = DEVICE,
        db: Optional["SimulationDatabase"] = None,
    ) -> None:
        #! need to make input_dim dynamic to input size
        super().__init__(input_dim=8, output_dim=4, config=config, device=device, db=db)
        self._setup_action_space()

    def _setup_action_space(self) -> None:
        """Initialize movement-specific action space."""
        self.action_space = {
            MoveActionSpace.RIGHT: (1, 0),
            MoveActionSpace.LEFT: (-1, 0),
            MoveActionSpace.UP: (0, 1),
            MoveActionSpace.DOWN: (0, -1),
        }

    def get_movement(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[float, float]:
        """Determine next movement position using learned policy."""
        # Convert state to tensor if needed and store for later use
        if not isinstance(state, torch.Tensor):
            if hasattr(state, "to_tensor"):
                state = state.to_tensor(self.device)
            else:
                state = torch.FloatTensor(state).to(self.device)

        action = self.select_action(state)
        dx, dy = self.action_space[action]

        # Scale movement by agent's max_movement
        dx *= agent.max_movement
        dy *= agent.max_movement

        # Calculate new position within bounds
        new_x = max(0, min(agent.environment.width, agent.position[0] + dx))
        new_y = max(0, min(agent.environment.height, agent.position[1] + dy))

        # Store state for learning (already a tensor)
        self.previous_state = state
        self.previous_action = action

        return (new_x, new_y)

    def get_state(self) -> "ModelState":
        """Get current state of the move module.

        Returns:
            ModelState: Current state including learning parameters and metrics

        Example:
            >>> state = move_module.get_state()
            >>> print(f"Current epsilon: {state.epsilon}")
        """
        from core.state import ModelState  # Import locally to avoid circle

        return ModelState.from_move_module(self)

    def select_action(
        self, state_tensor: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select action using epsilon-greedy with adaptive temperature.

        Uses a temperature-based exploration strategy where the temperature is derived
        from the current epsilon value. This creates smoother exploration behavior
        compared to uniform random selection.

        Args:
            state_tensor (torch.Tensor): Current state observation
            epsilon (Optional[float]): Override default epsilon value

        Returns:
            int: Selected action index from MoveActionSpace

        Notes:
            - During exploration, uses softmax with temperature for weighted random selection
            - Temperature scales with epsilon for adaptive exploration behavior
            - During exploitation, selects action with highest Q-value
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            # Temperature-based random action selection
            q_values = self.q_network(state_tensor)
            temperature = max(0.1, epsilon)  # Use epsilon as temperature

            # Apply softmax with temperature
            probabilities = torch.softmax(q_values / temperature, dim=0)

            # Convert to numpy for random choice
            action_probs = probabilities.detach().cpu().numpy()
            return np.random.choice(len(action_probs), p=action_probs)

        with torch.no_grad():
            return self.q_network(state_tensor).argmax().item()


def move_action(agent: "BaseAgent") -> None:
    """Execute movement using optimized Deep Q-Learning based policy."""
    # Get state and ensure it's a tensor
    state = _ensure_tensor(agent.get_state(), agent.move_module.device)

    # Get movement and update position
    initial_position = agent.position
    new_position = agent.move_module.get_movement(agent, state)

    # Collect action for database
    if agent.environment.db is not None:
        agent.environment.db.logger.log_agent_action(
            step_number=agent.environment.time,
            agent_id=agent.agent_id,
            action_type="move",
            position_before=initial_position,
            position_after=new_position,
            resources_before=agent.resource_level,
            resources_after=agent.resource_level - DEFAULT_MOVE_CONFIG.move_base_cost,
            reward=DEFAULT_MOVE_CONFIG.move_base_cost,
            details={
                "distance_moved": _calculate_distance(initial_position, new_position)
            },
        )

    # Update position
    agent.position = new_position

    # Calculate reward and store experience
    reward = _calculate_movement_reward(agent, initial_position, new_position)
    agent.total_reward += reward
    _store_and_train(agent, state, reward)


def _calculate_movement_reward(
    agent: "BaseAgent",
    initial_position: Tuple[float, float],
    new_position: Tuple[float, float],
) -> float:
    """Calculate reward for movement based on resource proximity."""
    # Base cost for moving
    reward = DEFAULT_MOVE_CONFIG.move_base_cost

    # Calculate movement distance
    distance_moved = np.sqrt(
        (new_position[0] - initial_position[0]) ** 2
        + (new_position[1] - initial_position[1]) ** 2
    )

    if distance_moved > 0:
        closest_resource = _find_closest_resource(agent.environment, new_position)
        if closest_resource:
            old_distance = _calculate_distance(
                closest_resource.position, initial_position
            )
            new_distance = _calculate_distance(closest_resource.position, new_position)
            reward += (
                DEFAULT_MOVE_CONFIG.move_resource_approach_reward
                if new_distance < old_distance
                else DEFAULT_MOVE_CONFIG.move_resource_retreat_penalty
            )

    return reward


def _find_closest_resource(
    environment: "Environment", position: Tuple[float, float]
) -> Optional["Resource"]:
    """Find the closest non-depleted resource."""
    active_resources = [r for r in environment.resources if not r.is_depleted()]
    if not active_resources:
        return None

    return min(
        active_resources, key=lambda r: _calculate_distance(r.position, position)
    )


def _calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def _ensure_tensor(state: Any, device: torch.device) -> torch.Tensor:
    """Ensure state is a tensor on the correct device."""
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if hasattr(state, "to_tensor"):
        return state.to_tensor(device)
    return torch.FloatTensor(state).to(device)


def _store_and_train(agent: "BaseAgent", state: Any, reward: float) -> None:
    """Store experience and perform training if possible."""
    if agent.move_module.previous_state is not None:
        next_state = _ensure_tensor(agent.get_state(), agent.move_module.device)

        # Map action number to direction string
        direction_map = {
            MoveActionSpace.RIGHT: "right",
            MoveActionSpace.LEFT: "left",
            MoveActionSpace.UP: "up",
            MoveActionSpace.DOWN: "down",
        }
        direction = direction_map[agent.move_module.previous_action]

        agent.move_module.store_experience(
            step_number=agent.environment.time,
            agent_id=agent.agent_id,
            module_type="move",
            module_id=agent.move_module.module_id,
            state=agent.move_module.previous_state,
            action=agent.move_module.previous_action,
            action_taken_mapped=direction,
            reward=reward,
            next_state=next_state,
            done=False,
        )

        if len(agent.move_module.memory) >= 2:
            batch_size = min(32, len(agent.move_module.memory))
            agent.move_module.train(random.sample(agent.move_module.memory, batch_size))
