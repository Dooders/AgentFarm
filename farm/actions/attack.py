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
    - State Space: 8-dimensional vector representing agent's current state (position, health, resources, etc.)
    - Action Space: 5 discrete actions (attack up/down/left/right, defend)
    - Learning Algorithm: Deep Q-Learning with experience replay and soft target updates
    - Exploration: Epsilon-greedy strategy with multiplicative decay (epsilon *= decay_rate)
    - Network Architecture: 2-layer feedforward network with ReLU activations and configurable hidden size
    - Hardware Acceleration: Automatic GPU usage when available for neural network operations
    - Combat Mechanics: Damage calculation, defensive stance, health tracking, and death detection
    - Reward Structure: Success rewards (+1.0), failure penalties (-0.3), and health-based defense boosts

The module integrates with the broader simulation through:
    - Environment spatial queries for target detection
    - Agent health and resource management
    - Combat statistics tracking
    - Comprehensive logging for analysis and debugging
"""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder
from farm.actions.config import DEFAULT_ATTACK_CONFIG, AttackConfig
from farm.core.action import action_registry
from farm.loggers.attack_logger import AttackLogger

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

import random

import torch

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _initialize_attack_action(agent: "BaseAgent") -> Tuple[float, float, AttackLogger]:
    """Initialize common attack action variables and logger.

    Args:
        agent: The agent performing the attack action

    Returns:
        Tuple containing (initial_resources, health_ratio, attack_logger)
    """
    initial_resources = agent.resource_level
    health_ratio = agent.current_health / agent.starting_health
    attack_logger = AttackLogger(agent.environment.db)
    return initial_resources, health_ratio, attack_logger


def _validate_agent_config(agent: "BaseAgent") -> bool:
    """Validate that the agent has a valid configuration.

    Args:
        agent: The agent to validate

    Returns:
        True if config is valid, False otherwise
    """
    if agent.config is None:
        logger.error(f"Agent {agent.agent_id} has no config, skipping attack action")
        return False
    return True


def _find_valid_targets(
    agent: "BaseAgent", target_position: Tuple[float, float]
) -> List["BaseAgent"]:
    """Find valid attack targets within range.

    Args:
        agent: The agent looking for targets
        target_position: Position to search around

    Returns:
        List of valid targets (excluding self)
    """
    if agent.config is None:
        return []
    targets = (
        agent.spatial_service.get_nearby_agents(target_position, agent.config.range)
        if getattr(agent, "spatial_service", None) is not None
        else []
    )
    return [target for target in targets if target.agent_id != agent.agent_id]


def _log_no_targets_attempt(
    attack_logger: AttackLogger,
    agent: "BaseAgent",
    target_position: Tuple[float, float],
    initial_resources: float,
    reason: str,
) -> None:
    """Log an attack attempt with no targets.

    Args:
        attack_logger: Logger instance
        agent: The attacking agent
        target_position: Position where attack was attempted
        initial_resources: Resources before attack
        reason: Reason for failure
    """
    attack_logger.log_attack_attempt(
        step_number=agent.environment.time,
        agent=agent,
        action_target_id=None,
        target_position=target_position,
        resources_before=initial_resources,
        resources_after=initial_resources,
        success=False,
        targets_found=0,
        reason=reason,
    )
    # Log interaction edge (attempt with no target)
    if getattr(agent, "logging_service", None) is not None:
        agent.logging_service.log_interaction_edge(
            source_type="agent",
            source_id=agent.agent_id,
            target_type="agent",
            target_id="none",
            interaction_type="attack_attempt",
            action_type="attack",
            details={
                "success": False,
                "reason": reason,
                "targets_found": 0,
                "target_position": target_position,
            },
        )


def _apply_attack_cost(agent: "BaseAgent") -> None:
    """Apply the resource cost for attempting an attack.

    Args:
        agent: The agent performing the attack
    """
    if agent.config is None:
        return
    attack_cost = agent.config.base_cost * agent.resource_level
    agent.resource_level += attack_cost


def _update_combat_counters(agent: "BaseAgent") -> None:
    """Update global combat encounter counters.

    Args:
        agent: The agent involved in combat
    """
    if getattr(agent, "metrics_service", None) is not None:
        agent.metrics_service.record_combat_encounter()


def _calculate_and_apply_damage(
    agent: "BaseAgent", target: "BaseAgent"
) -> Tuple[float, int]:
    """Calculate and apply damage to a target.

    Args:
        agent: The attacking agent
        target: The target agent

    Returns:
        Tuple of (total_damage_dealt, successful_hits)
    """
    # Calculate base damage based on attack strength and resource ratio
    base_damage = agent.attack_strength * (agent.resource_level / agent.starting_health)

    # Apply defensive damage reduction if target is in defensive stance
    if target.is_defending:
        base_damage *= 1 - target.defense_strength

    # Apply damage to target and track combat outcomes
    total_damage_dealt = 0.0
    successful_hits = 0

    if target.take_damage(base_damage):
        total_damage_dealt += base_damage
        successful_hits += 1

    # Update global successful attack counters if damage was dealt
    if successful_hits > 0 and getattr(agent, "metrics_service", None) is not None:
        agent.metrics_service.record_successful_attack()

    return total_damage_dealt, successful_hits


def _log_attack_outcome(
    attack_logger: AttackLogger,
    agent: "BaseAgent",
    target: "BaseAgent",
    target_position: Tuple[float, float],
    initial_resources: float,
    final_resources: float,
    total_damage_dealt: float,
    successful_hits: int,
    valid_targets_count: int,
) -> None:
    """Log the outcome of an attack attempt.

    Args:
        attack_logger: Logger instance
        agent: The attacking agent
        target: The target agent
        target_position: Position where attack was attempted
        initial_resources: Resources before attack
        final_resources: Resources after attack
        total_damage_dealt: Total damage dealt
        successful_hits: Number of successful hits
        valid_targets_count: Number of valid targets found
    """
    attack_logger.log_attack_attempt(
        step_number=agent.environment.time,
        agent=agent,
        action_target_id=target.agent_id,
        target_position=target_position,
        resources_before=initial_resources,
        resources_after=final_resources,
        success=successful_hits > 0,
        targets_found=valid_targets_count,
        damage_dealt=total_damage_dealt,
        reason="hit" if successful_hits > 0 else "missed",
    )
    # Log interaction edge for attack outcome
    if getattr(agent, "logging_service", None) is not None:
        agent.logging_service.log_interaction_edge(
            source_type="agent",
            source_id=agent.agent_id,
            target_type="agent",
            target_id=target.agent_id,
            interaction_type="attack" if successful_hits > 0 else "attack_failed",
            action_type="attack",
            details={
                "success": successful_hits > 0,
                "damage_dealt": total_damage_dealt,
                "targets_found": valid_targets_count,
                "target_position": target_position,
            },
        )


def _find_closest_target(
    agent: "BaseAgent", valid_targets: List["BaseAgent"]
) -> Optional["BaseAgent"]:
    """Find the closest target using greedy search.

    Args:
        agent: The attacking agent
        valid_targets: List of valid targets

    Returns:
        The closest target, or None if no targets found
    """
    closest_target = None
    min_distance = float("inf")

    for target in valid_targets:
        # Calculate distance to target
        dx = target.position[0] - agent.position[0]
        dy = target.position[1] - agent.position[1]
        distance = (dx * dx + dy * dy) ** 0.5  # Euclidean distance

        if distance < min_distance:
            min_distance = distance
            closest_target = target

    return closest_target


def _calculate_attack_direction(agent: "BaseAgent", target: "BaseAgent") -> int:
    """Calculate the attack direction towards a target.

    Args:
        agent: The attacking agent
        target: The target agent

    Returns:
        Attack action index (ATTACK_RIGHT, ATTACK_LEFT, ATTACK_UP, ATTACK_DOWN)
    """
    dx = target.position[0] - agent.position[0]
    dy = target.position[1] - agent.position[1]

    # Determine attack direction based on which axis has the larger difference
    if abs(dx) > abs(dy):
        # Attack horizontally
        return (
            AttackActionSpace.ATTACK_RIGHT if dx > 0 else AttackActionSpace.ATTACK_LEFT
        )
    else:
        # Attack vertically
        return AttackActionSpace.ATTACK_UP if dy > 0 else AttackActionSpace.ATTACK_DOWN


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

    The network takes an 8-dimensional state vector as input and outputs
    Q-values for each possible attack action, enabling the agent to learn
    optimal attack strategies through reinforcement learning.

    Architecture:
        - Input layer: 8 dimensions (agent state)
        - Hidden layer: Configurable size (default: 64) with ReLU activation
        - Output layer: 5 dimensions (attack actions) with linear activation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        super().__init__(
            input_dim,
            output_dim=5,
            hidden_size=hidden_size,
            shared_encoder=shared_encoder,
        )  # 5 attack actions


class AttackModule(BaseDQNModule):
    """Deep Q-Network module specialized for attack action learning and execution.

    Extends BaseDQNModule with attack-specific functionality including:
    - Health-based defensive behavior enhancement (boosts defense Q-value when health < threshold)
    - Combat action space management with directional attack vectors
    - Attack-specific configuration handling and reward structures

    The module learns optimal attack strategies through experience replay
    and epsilon-greedy exploration, adapting its policy based on combat outcomes
    and agent health status. When an agent's health ratio falls below the defense
    threshold, the Q-value for the defend action is multiplied by the defense boost
    to encourage defensive behavior.
    """

    def __init__(
        self,
        config: AttackConfig = DEFAULT_ATTACK_CONFIG,
        device: torch.device = DEVICE,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        super().__init__(input_dim=8, output_dim=5, config=config, device=device)
        self._setup_action_space()
        # Store the attack-specific config for access to attack attributes
        self.attack_config = config

        # Initialize Q-networks with shared encoder if provided
        self.q_network = AttackQNetwork(
            input_dim=8,
            hidden_size=config.dqn_hidden_size,
            shared_encoder=shared_encoder,
        ).to(device)
        self.target_network = AttackQNetwork(
            input_dim=8,
            hidden_size=config.dqn_hidden_size,
            shared_encoder=shared_encoder,
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

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
        the Q-value for the defend action is multiplied by the defense boost to
        encourage defensive behavior.

        Args:
            state: Current agent state tensor (8-dimensional)
            health_ratio: Current health as a ratio of starting health (0.0 to 1.0)

        Returns:
            int: Selected action index from AttackActionSpace (0-4)
        """
        # Use random.random() instead of torch.rand() for deterministic testing
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)
                if health_ratio < self.attack_config.defense_threshold:
                    q_values[
                        AttackActionSpace.DEFEND
                    ] *= self.attack_config.defense_boost
                return q_values.argmax().item()
        return random.randint(0, len(self.action_space) - 1)


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
    # Initialize common variables
    initial_resources, health_ratio, attack_logger = _initialize_attack_action(agent)

    # Select attack action using DQN with health-based defense boost
    action = agent.attack_module.select_action(
        agent.get_state().to_tensor(agent.attack_module.device), health_ratio
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

    # Validate agent configuration
    if not _validate_agent_config(agent):
        return

    # Find valid targets
    valid_targets = _find_valid_targets(agent, target_pos)

    if not valid_targets:
        _log_no_targets_attempt(
            attack_logger, agent, target_pos, initial_resources, "no_targets"
        )
        return

    # Apply attack cost and update combat counters
    _apply_attack_cost(agent)
    _update_combat_counters(agent)

    # Select a random target from valid candidates for attack
    target = random.choice(valid_targets)

    # Calculate and apply damage
    total_damage_dealt, successful_hits = _calculate_and_apply_damage(agent, target)

    # Log comprehensive attack outcome for analysis and debugging
    _log_attack_outcome(
        attack_logger,
        agent,
        target,
        target_pos,
        initial_resources,
        agent.resource_level,
        total_damage_dealt,
        successful_hits,
        len(valid_targets),
    )


def simple_attack_action(agent: "BaseAgent") -> None:
    """
    This function is a simple attack action that does not use the DQN.
    It performs a greedy search to find and attack the closest agent.

    Args:
        agent: The agent performing the attack action

    Returns:
        None
    """
    # Initialize common variables
    initial_resources, health_ratio, attack_logger = _initialize_attack_action(agent)

    # Validate agent configuration
    if not _validate_agent_config(agent):
        return

    # Find valid targets around the agent's position
    valid_targets = _find_valid_targets(agent, agent.position)

    if not valid_targets:
        _log_no_targets_attempt(
            attack_logger, agent, agent.position, initial_resources, "no_valid_targets"
        )
        return

    # Greedy search: find the closest agent
    closest_target = _find_closest_target(agent, valid_targets)

    if closest_target is None:
        _log_no_targets_attempt(
            attack_logger, agent, agent.position, initial_resources, "no_closest_target"
        )
        return

    # Calculate attack direction towards the closest target
    action = _calculate_attack_direction(agent, closest_target)

    # Calculate attack target position based on selected action direction
    target_pos = agent.calculate_attack_position(action)

    # Apply attack cost and update combat counters
    _apply_attack_cost(agent)
    _update_combat_counters(agent)

    # Calculate and apply damage
    total_damage_dealt, successful_hits = _calculate_and_apply_damage(
        agent, closest_target
    )

    # Log comprehensive attack outcome for analysis and debugging
    _log_attack_outcome(
        attack_logger,
        agent,
        closest_target,
        target_pos,
        initial_resources,
        agent.resource_level,
        total_damage_dealt,
        successful_hits,
        len(valid_targets),
    )


action_registry.register("attack", 0.1, attack_action)
action_registry.register("simple_attack", 0.1, simple_attack_action)
