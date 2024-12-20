"""Attack learning and execution module using Deep Q-Learning (DQN).

This module implements a Deep Q-Learning approach for agents to learn optimal attack
policies in a multi-agent environment. It provides both the neural network architecture
and the training/execution logic.

Key Components:
    - AttackQNetwork: Neural network architecture for Q-value approximation
    - AttackModule: Main class handling training, action selection, and attack execution
    - Experience Replay: Stores and samples past experiences for stable learning
    - Target Network: Separate network for computing stable Q-value targets

Technical Details:
    - State Space: N-dimensional vector representing agent's current state
    - Action Space: 5 discrete actions (attack up/down/left/right, defend)
    - Learning Algorithm: Deep Q-Learning with experience replay
    - Exploration: Epsilon-greedy strategy with decay
    - Hardware Acceleration: Automatic GPU usage when available
"""

import logging
from typing import TYPE_CHECKING

from actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttackConfig(BaseDQNConfig):
    """Configuration specific to attacks."""

    attack_base_cost: float = -0.2
    attack_success_reward: float = 1.0
    attack_failure_penalty: float = -0.3
    attack_defense_threshold: float = 0.3
    attack_defense_boost: float = 2.0


DEFAULT_ATTACK_CONFIG = AttackConfig()


class AttackActionSpace:
    """Define available attack actions."""

    ATTACK_RIGHT: int = 0
    ATTACK_LEFT: int = 1
    ATTACK_UP: int = 2
    ATTACK_DOWN: int = 3
    DEFEND: int = 4


class AttackQNetwork(BaseQNetwork):
    """Attack-specific Q-network."""

    def __init__(self, input_dim: int, hidden_size: int = 64) -> None:
        super().__init__(
            input_dim, output_dim=5, hidden_size=hidden_size
        )  # 5 attack actions


class AttackModule(BaseDQNModule):
    """Attack-specific DQN module."""

    def __init__(
        self,
        config: AttackConfig = DEFAULT_ATTACK_CONFIG,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(input_dim=6, output_dim=5, config=config, device=device)
        self._setup_action_space()

    def _setup_action_space(self) -> None:
        """Initialize attack-specific action space."""
        self.action_space = {
            AttackActionSpace.ATTACK_RIGHT: (1, 0),
            AttackActionSpace.ATTACK_LEFT: (-1, 0),
            AttackActionSpace.ATTACK_UP: (0, 1),
            AttackActionSpace.ATTACK_DOWN: (0, -1),
            AttackActionSpace.DEFEND: (0, 0),
        }

    def select_action(self, state: torch.Tensor, health_ratio: float) -> int:
        """Override select_action to include health-based defense boost."""
        if torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)
                if health_ratio < self.config.attack_defense_threshold:
                    q_values[
                        AttackActionSpace.DEFEND
                    ] *= self.config.attack_defense_boost
                return q_values.argmax().item()
        return torch.randint(0, len(self.action_space), (1,)).item()


def attack_action(agent: "BaseAgent") -> None:
    """Execute attack action using the AttackModule."""
    # Get current state and health ratio
    state = agent.get_state()
    health_ratio = agent.current_health / agent.starting_health
    initial_resources = agent.resource_level

    # Select attack action
    action = agent.attack_module.select_action(
        state.to_tensor(agent.attack_module.device), health_ratio
    )

    # Handle defense action
    if action == AttackActionSpace.DEFEND:
        agent.is_defending = True

        # Log defense action
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                step_number=agent.environment.time,
                agent_id=agent.agent_id,
                action_type="defend",
                position_before=agent.position,
                position_after=agent.position,
                resources_before=initial_resources,
                resources_after=initial_resources,
                reward=0,
                details={"is_defending": True}
            )

        logger.debug(f"Agent {id(agent)} took defensive stance")
        return

    # Calculate attack target position
    target_pos = agent.calculate_attack_position(action)

    # Find potential targets using KD-tree
    targets = agent.environment.get_nearby_agents(target_pos, agent.config.attack_range)

    if not targets:
        # Log failed attack
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                step_number=agent.environment.time,
                agent_id=agent.agent_id,
                action_type="attack",
                position_before=agent.position,
                position_after=agent.position,
                resources_before=initial_resources,
                resources_after=initial_resources,
                reward=0,
                details={"success": False, "reason": "no_targets"}
            )
        return

    # Execute attack logic...
