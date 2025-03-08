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

from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork
from farm.loggers.attack_logger import AttackLogger

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

import numpy as np
import torch
import random

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

    # Initialize attack logger
    attack_logger = AttackLogger(agent.environment.db)

    # Select attack action
    action = agent.attack_module.select_action(
        state.to_tensor(agent.attack_module.device), health_ratio
    )

    # Handle defense action
    if action == AttackActionSpace.DEFEND:
        agent.is_defending = True
        attack_logger.log_defense(
            step_number=agent.environment.time,
            agent=agent,
            resources_before=initial_resources,
            resources_after=initial_resources,
        )
        return

    # Calculate attack target position
    target_pos = agent.calculate_attack_position(action)

    # Find potential targets using KD-tree
    targets = agent.environment.get_nearby_agents(target_pos, agent.config.attack_range)

    if not targets:
        attack_logger.log_attack_attempt(
            step_number=agent.environment.time,
            agent=agent,
            action_target_id=None,
            target_position=target_pos,
            resources_before=initial_resources,
            resources_after=initial_resources,
            success=False,
            targets_found=0,
            reason="no_targets",
        )
        return

    # Calculate attack cost and apply it
    attack_cost = agent.config.attack_base_cost * agent.resource_level
    agent.resource_level += attack_cost

    # Initialize attack statistics
    total_damage_dealt = 0.0
    successful_hits = 0
    
    # Increment combat encounters counter
    agent.environment.combat_encounters += 1
    agent.environment.combat_encounters_this_step += 1

    # Filter out self from targets
    valid_targets = [target for target in targets if target.agent_id != agent.agent_id]
    
    # If no valid targets remain after filtering
    if not valid_targets:
        # Log attack outcome with no valid targets
        attack_logger.log_attack_attempt(
            step_number=agent.environment.time,
            agent=agent,
            action_target_id=None,
            target_position=target_pos,
            resources_before=initial_resources,
            resources_after=agent.resource_level,
            success=False,
            targets_found=0,
            reason="no_valid_targets",
        )
        return
        
    # Select a random target
    target = random.choice(valid_targets)
    
    # Calculate base damage
    base_damage = agent.attack_strength * (
        agent.resource_level / agent.starting_health
    )

    # Apply defensive reduction if target is defending
    if target.is_defending:
        base_damage *= 1 - target.defense_strength

    # Apply damage and track statistics
    if target.take_damage(base_damage):
        total_damage_dealt += base_damage
        successful_hits += 1

    # Update successful attacks counter if any hits were successful
    if successful_hits > 0:
        agent.environment.successful_attacks += 1
        agent.environment.successful_attacks_this_step += 1

    # Log attack outcome
    attack_logger.log_attack_attempt(
        step_number=agent.environment.time,
        agent=agent,
        action_target_id=target.agent_id,
        target_position=target_pos,
        resources_before=initial_resources,
        resources_after=agent.resource_level,
        success=successful_hits > 0,
        targets_found=len(valid_targets),
        damage_dealt=total_damage_dealt,
        reason="hit" if successful_hits > 0 else "missed",
    )
