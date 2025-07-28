"""Attack learning and execution module using Deep Q-Learning (DQN).

This module implements a Deep Q-Learning approach for agents to learn optimal attack
policies in a multi-agent environment. It provides both the neural network architecture
and the training/execution logic for combat interactions between agents.

Key Components:
    - AttackQNetwork: Neural network architecture for Q-value approximation
    - AttackModule: Main class handling training, action selection, and attack execution
    - Experience Replay: Stores and samples past experiences for stable learning
    - Target Network: Separate network for computing stable Q-value targets
    - AttackLogger: Comprehensive logging of attack attempts and outcomes

Technical Details:
    - State Space: 6-dimensional vector representing agent's current state (position, health, resources)
    - Action Space: 5 discrete actions (attack up/down/left/right, defend)
    - Learning Algorithm: Deep Q-Learning with experience replay and soft target updates
    - Exploration: Epsilon-greedy strategy with decay for balanced exploration/exploitation
    - Hardware Acceleration: Automatic GPU usage when available for neural network operations
    - Combat Mechanics: Damage calculation, defensive stance, health tracking, and death detection

The module integrates with the broader simulation through:
    - Environment spatial queries for target detection
    - Agent health and resource management
    - Combat statistics tracking
    - Comprehensive logging for analysis and debugging
"""

import logging
from typing import TYPE_CHECKING

from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork
from farm.loggers.attack_logger import AttackLogger

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttackConfig(BaseDQNConfig):
    """Configuration class for attack-specific parameters and learning settings.

    Extends BaseDQNConfig with attack-specific hyperparameters that control
    combat mechanics, reward structures, and learning behavior.

    Attributes:
        attack_base_cost: Base resource cost for attempting an attack
        attack_success_reward: Reward multiplier for successful attacks
        attack_failure_penalty: Penalty for failed attack attempts
        attack_defense_threshold: Health ratio threshold for defensive behavior
        attack_defense_boost: Multiplier for defense action when health is low
    """

    attack_base_cost: float = -0.2
    attack_success_reward: float = 1.0
    attack_failure_penalty: float = -0.3
    attack_defense_threshold: float = 0.3
    attack_defense_boost: float = 2.0


DEFAULT_ATTACK_CONFIG = AttackConfig()


class AttackActionSpace:
    """Defines the available attack actions and their corresponding indices.

    Provides a mapping between action indices and their semantic meaning
    for the attack module. Actions include directional attacks and defensive stance.

    Attributes:
        ATTACK_RIGHT: Attack to the right (positive x-direction)
        ATTACK_LEFT: Attack to the left (negative x-direction)
        ATTACK_UP: Attack upward (positive y-direction)
        ATTACK_DOWN: Attack downward (negative y-direction)
        DEFEND: Take defensive stance (no movement, damage reduction)
    """

    ATTACK_RIGHT: int = 0
    ATTACK_LEFT: int = 1
    ATTACK_UP: int = 2
    ATTACK_DOWN: int = 3
    DEFEND: int = 4


class AttackQNetwork(BaseQNetwork):
    """Neural network architecture for attack Q-value approximation.

    Extends BaseQNetwork with attack-specific configuration, providing
    Q-value estimates for the 5 possible attack actions (4 directions + defend).

    The network takes a 6-dimensional state vector as input and outputs
    Q-values for each possible attack action, enabling the agent to learn
    optimal attack strategies through reinforcement learning.
    """

    def __init__(self, input_dim: int, hidden_size: int = 64) -> None:
        super().__init__(
            input_dim, output_dim=5, hidden_size=hidden_size
        )  # 5 attack actions


class AttackModule(BaseDQNModule):
    """Deep Q-Network module specialized for attack action learning and execution.

    Extends BaseDQNModule with attack-specific functionality including:
    - Health-based defensive behavior enhancement
    - Combat action space management
    - Attack-specific configuration handling

    The module learns optimal attack strategies through experience replay
    and epsilon-greedy exploration, adapting its policy based on combat outcomes
    and agent health status.
    """

    def __init__(
        self,
        config: AttackConfig = DEFAULT_ATTACK_CONFIG,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(input_dim=6, output_dim=5, config=config, device=device)
        self._setup_action_space()
        # Store the attack-specific config for access to attack attributes
        self.attack_config = config

    def _setup_action_space(self) -> None:
        """Initialize the attack action space with directional vectors.

        Creates a mapping from action indices to directional vectors used
        for calculating attack target positions. Each vector represents
        the direction and magnitude of the attack.
        """
        self.action_space = {
            AttackActionSpace.ATTACK_RIGHT: (1, 0),
            AttackActionSpace.ATTACK_LEFT: (-1, 0),
            AttackActionSpace.ATTACK_UP: (0, 1),
            AttackActionSpace.ATTACK_DOWN: (0, -1),
            AttackActionSpace.DEFEND: (0, 0),
        }

    def select_action(self, state: torch.Tensor, health_ratio: float) -> int:
        """Select an attack action using epsilon-greedy strategy with health-based defense boost.

        Overrides the base select_action method to incorporate health-based defensive
        behavior. When the agent's health ratio falls below the defense threshold,
        the Q-value for the defend action is boosted to encourage defensive behavior.

        Args:
            state: Current agent state tensor
            health_ratio: Current health as a ratio of starting health (0.0 to 1.0)

        Returns:
            int: Selected action index from AttackActionSpace
        """
        if torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)
                if health_ratio < self.attack_config.attack_defense_threshold:
                    q_values[
                        AttackActionSpace.DEFEND
                    ] *= self.attack_config.attack_defense_boost
                return q_values.argmax().item()
        return int(torch.randint(0, len(self.action_space), (1,)).item())


def attack_action(agent: "BaseAgent") -> None:
    """Execute an attack action using the agent's AttackModule.

    This function orchestrates the complete attack process including:
    1. State evaluation and action selection
    2. Target detection and validation
    3. Damage calculation and application
    4. Combat statistics tracking
    5. Comprehensive logging of outcomes

    The attack process involves:
    - Selecting an action (attack direction or defend) using the DQN
    - Finding potential targets within attack range
    - Calculating and applying damage to selected targets
    - Updating agent resources and health
    - Logging all combat interactions for analysis

    Args:
        agent: The agent performing the attack action

    Note:
        This function handles both offensive attacks and defensive actions.
        Defensive actions set the agent's defending flag but don't consume resources.
    """
    # Get current state and health ratio for decision making
    state = agent.get_state()
    health_ratio = agent.current_health / agent.starting_health
    initial_resources = agent.resource_level

    # Initialize attack logger for tracking combat outcomes
    attack_logger = AttackLogger(agent.environment.db)

    # Select attack action using DQN with health-based defense boost
    action = agent.attack_module.select_action(
        state.to_tensor(agent.attack_module.device), health_ratio
    )

    # Handle defensive action (no resource cost, sets defending flag)
    if action == AttackActionSpace.DEFEND:
        agent.is_defending = True
        attack_logger.log_defense(
            step_number=agent.environment.time,
            agent=agent,
            resources_before=initial_resources,
            resources_after=initial_resources,
        )
        return

    # Calculate attack target position based on selected action direction
    target_pos = agent.calculate_attack_position(action)

    # Safety check for config to prevent attribute access errors
    if agent.config is None:
        logger.error(f"Agent {agent.agent_id} has no config, skipping attack action")
        return

    # Find potential targets within attack range using spatial indexing
    targets = agent.environment.get_nearby_agents(target_pos, agent.config.attack_range)

    if not targets:
        attack_logger.log_attack_attempt(
            step_number=agent.environment.time,
            agent=agent,
            action_target_id="",  # Empty string instead of None
            target_position=target_pos,
            resources_before=initial_resources,
            resources_after=initial_resources,
            success=False,
            targets_found=0,
            reason="no_targets",
        )
        return

        # Calculate and apply attack resource cost
    attack_cost = agent.config.attack_base_cost * agent.resource_level
    agent.resource_level += attack_cost

    # Initialize combat statistics tracking
    total_damage_dealt = 0.0
    successful_hits = 0

    # Update global combat encounter counters
    agent.environment.combat_encounters += 1
    agent.environment.combat_encounters_this_step += 1

    # Filter out self-targeting to prevent agents from attacking themselves
    valid_targets = [target for target in targets if target.agent_id != agent.agent_id]

    # If no valid targets remain after filtering
    if not valid_targets:
        # Log attack outcome with no valid targets
        attack_logger.log_attack_attempt(
            step_number=agent.environment.time,
            agent=agent,
            action_target_id="",  # Empty string instead of None
            target_position=target_pos,
            resources_before=initial_resources,
            resources_after=agent.resource_level,
            success=False,
            targets_found=0,
            reason="no_valid_targets",
        )
        return

        # Select a random target from valid candidates for attack
    target = random.choice(valid_targets)

    # Calculate base damage based on attack strength and resource ratio
    base_damage = agent.attack_strength * (agent.resource_level / agent.starting_health)

    # Apply defensive damage reduction if target is in defensive stance
    if target.is_defending:
        base_damage *= 1 - target.defense_strength

    # Apply damage to target and track combat outcomes
    if target.take_damage(base_damage):
        total_damage_dealt += base_damage
        successful_hits += 1

    # Update global successful attack counters if damage was dealt
    if successful_hits > 0:
        agent.environment.successful_attacks += 1
        agent.environment.successful_attacks_this_step += 1

    # Log comprehensive attack outcome for analysis and debugging
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
