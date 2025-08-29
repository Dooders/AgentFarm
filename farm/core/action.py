import logging
import math
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from farm.core.agent import BaseAgent

logger = logging.getLogger(__name__)


# Helper Functions for Common Action Patterns


def validate_agent_config(agent: "BaseAgent", action_name: str) -> bool:
    """Validate that agent has a configuration. Logs error and returns False if missing.

    Args:
        agent: The agent to validate
        action_name: Name of the action for logging purposes

    Returns:
        bool: True if config is valid, False otherwise
    """
    if agent.config is None:
        logger.error(
            f"Agent {agent.agent_id} has no config, skipping {action_name} action"
        )
        return False
    return True


def calculate_euclidean_distance(pos1: tuple, pos2: tuple) -> float:
    """Calculate Euclidean distance between two positions.

    Args:
        pos1: First position as (x, y) tuple
        pos2: Second position as (x, y) tuple

    Returns:
        float: Euclidean distance between positions
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.sqrt(dx * dx + dy * dy)


def find_closest_entity(
    agent: "BaseAgent", entities: list, entity_type: str = "target"
) -> tuple:
    """Find the closest entity to the agent from a list of entities.

    Args:
        agent: The agent looking for closest entity
        entities: List of entities with position attribute
        entity_type: Type name for logging purposes

    Returns:
        tuple: (closest_entity, distance) or (None, inf) if no valid entities
    """
    if not entities:
        return None, float("inf")

    closest_entity = None
    min_distance = float("inf")

    for entity in entities:
        distance = calculate_euclidean_distance(agent.position, entity.position)
        if distance < min_distance:
            min_distance = distance
            closest_entity = entity

    if closest_entity is None:
        logger.debug(f"Agent {agent.agent_id} could not find a closest {entity_type}")
        return None, float("inf")

    return closest_entity, min_distance


def log_interaction_safely(agent: "BaseAgent", **kwargs) -> None:
    """Safely log an interaction edge, handling environments without logging infrastructure.

    Args:
        agent: The agent performing the action
        **kwargs: Arguments to pass to log_interaction_edge
    """
    try:
        if hasattr(agent, "logging_service") and agent.logging_service:
            agent.logging_service.log_interaction_edge(**kwargs)
    except Exception:
        pass


def check_resource_requirement(
    agent: "BaseAgent",
    required_amount: float,
    action_name: str,
    requirement_type: str = "resources",
) -> bool:
    """Check if agent has sufficient resources for an action.

    Args:
        agent: The agent to check
        required_amount: Minimum required resource level
        action_name: Name of the action for logging
        requirement_type: Type of requirement for logging

    Returns:
        bool: True if agent has sufficient resources, False otherwise
    """
    if agent.resource_level < required_amount:
        logger.debug(
            f"Agent {agent.agent_id} has insufficient {requirement_type} "
            f"({agent.resource_level} < {required_amount}) for {action_name}"
        )
        return False
    return True


"""Action management module for agent behaviors in a multi-agent environment.

This module defines the core Action class and implements various agent behaviors
including movement, resource gathering, sharing, and combat. Each action has
associated costs, rewards, and conditions for execution.

Key Components:
    - ActionType: Enumeration of available agent actions
    - Action: Base class for defining executable agent behaviors
    - Movement: Deep Q-Learning based movement with rewards
    - Gathering: Resource collection from environment nodes
    - Sharing: Resource distribution between nearby agents
    - Combat: Competitive resource acquisition through attacks

Technical Details:
    - Range-based interactions (gathering: config-based, sharing: 30 units, attack: 20 units)
    - Resource-based action costs and rewards
    - Numpy-based distance calculations
    - Automatic tensor-numpy conversion for state handling
"""


class ActionType(IntEnum):
    """Enumeration of available agent actions.

    Actions define the possible behaviors agents can take in the environment.
    Each action has specific effects on the agent and its surroundings.

    Attributes:
        DEFEND (0): Agent enters defensive stance, reducing damage from attacks
        ATTACK (1): Agent attempts to attack nearby enemies
        GATHER (2): Agent collects resources from nearby resource nodes
        SHARE (3): Agent shares resources with nearby allies
        MOVE (4): Agent moves to a new position within the environment
        REPRODUCE (5): Agent attempts to create offspring if conditions are met
        PASS (6): Agent takes no action this turn
    """

    DEFEND = 0
    ATTACK = 1
    GATHER = 2
    SHARE = 3
    MOVE = 4
    REPRODUCE = 5
    PASS = 6


class Action:
    """Base class for defining executable agent behaviors.

    Encapsulates a named action with an associated weight for action selection
    and an execution function that defines the behavior.

    Args:
        name (str): Identifier for the action (e.g., "move", "gather")
        weight (float): Selection probability weight for action choice
        function (callable): Function implementing the action behavior
            Must accept agent as first parameter and support *args, **kwargs

    Example:
        ```python
        move = Action("move", 0.4, move_action)
        move.execute(agent, additional_arg=value)
        ```
    """

    def __init__(self, name, weight, function):
        """
        Initialize an action with a name, weight, and associated function.

        Parameters:
        - name (str): The name of the action (e.g., "move", "gather").
        - weight (float): The weight or likelihood of selecting this action.
        - function (callable): The function to execute when this action is chosen.
        """
        self.name = name
        self.weight = weight
        self.function = function

    def execute(self, agent, *args, **kwargs):
        """Execute the action's behavior function.

        Calls the associated function with the agent and any additional parameters.

        Args:
            agent: Agent instance performing the action
            *args: Variable positional arguments for the action function
            **kwargs: Variable keyword arguments for the action function
        """
        self.function(agent, *args, **kwargs)


class action_registry:
    _registry = {}

    @classmethod
    def register(cls, name: str, weight: float, function: Callable) -> None:
        """Register a new action in the global registry.

        Args:
            name: Unique name for the action
            weight: Selection weight
            function: The function to execute
        """
        cls._registry[name] = Action(name, weight, function)

    @classmethod
    def get(cls, name: str) -> Action | None:
        """Get a registered action by name."""
        return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> List[Action]:
        """Get all registered actions."""
        return list(cls._registry.values())


def attack_action(agent: "BaseAgent") -> None:
    """Execute the attack action for the given agent.

    Finds the closest agent within attack range and attacks it using the spatial index
    for efficient proximity queries.
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "attack"):
        return

    # Get attack range from config
    attack_range = getattr(agent.config, "attack_range", 20.0)

    # Find nearby agents using spatial index
    nearby_agents = agent.spatial_service.get_nearby_agents(agent.position, attack_range)

    # Filter out self and find valid targets
    valid_targets = [
        target
        for target in nearby_agents
        if target.agent_id != agent.agent_id and target.alive
    ]

    if not valid_targets:
        logger.debug(
            f"Agent {agent.agent_id} found no valid targets within range {attack_range}"
        )
        return

    # Find the closest target using helper function
    closest_target, min_distance = find_closest_entity(agent, valid_targets, "target")

    if closest_target is None:
        return

    # Calculate attack direction towards the closest target
    dx = closest_target.position[0] - agent.position[0]
    dy = closest_target.position[1] - agent.position[1]

    # Determine attack direction (this could be used for more complex attack mechanics)
    if abs(dx) > abs(dy):
        attack_direction = "horizontal"
    else:
        attack_direction = "vertical"

    # Calculate damage based on agent's attack strength and health ratio
    health_ratio = agent.current_health / agent.starting_health
    base_damage = agent.attack_strength * health_ratio

    # Apply defensive damage reduction if target is defending
    if closest_target.is_defending:
        defense_reduction = getattr(closest_target, "defense_strength", 0.5)
        base_damage *= 1.0 - defense_reduction

    # Apply damage to target
    actual_damage = closest_target.take_damage(base_damage)

    # Update combat statistics
    if actual_damage > 0 and hasattr(agent, "metrics_service") and agent.metrics_service:
        try:
            agent.metrics_service.record_combat_encounter()
            agent.metrics_service.record_successful_attack()
        except Exception:
            pass

    # Log the attack interaction using helper function
    log_interaction_safely(
        agent,
        source_type="agent",
        source_id=agent.agent_id,
        target_type="agent",
        target_id=closest_target.agent_id,
        interaction_type="attack" if actual_damage > 0 else "attack_failed",
        action_type="attack",
        details={
            "success": actual_damage > 0,
            "damage_dealt": actual_damage,
            "distance": min_distance,
            "attack_range": attack_range,
            "target_position": closest_target.position,
            "attacker_position": agent.position,
            "attack_direction": attack_direction,
        },
    )

    logger.debug(
        f"Agent {agent.agent_id} attacked {closest_target.agent_id} "
        f"at distance {min_distance:.2f}, dealt {actual_damage:.2f} damage"
    )


def gather_action(agent: "BaseAgent") -> None:
    """Execute the gather action for the given agent.

    Simple rule-based gathering that finds the nearest resource and gathers from it.
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "gather"):
        return

    # Get gathering range from config
    gathering_range = getattr(agent.config, "gathering_range", 30)

    # Find nearby resources using spatial index
    nearby_resources = agent.spatial_service.get_nearby_resources(agent.position, gathering_range)

    # Filter out depleted resources
    available_resources = [
        r for r in nearby_resources if not r.is_depleted() and r.amount > 0
    ]

    if not available_resources:
        logger.debug(
            f"Agent {agent.agent_id} found no available resources within range {gathering_range}"
        )
        return

    # Find the closest resource using helper function
    closest_resource, min_distance = find_closest_entity(
        agent, available_resources, "resource"
    )

    if closest_resource is None:
        return

    # Record initial resource levels
    initial_resources = agent.resource_level
    resource_amount_before = closest_resource.amount

    # Determine how much to gather
    max_gather = getattr(agent.config, "max_amount", 10)
    gather_amount = min(max_gather, closest_resource.amount)

    if gather_amount <= 0:
        logger.debug(f"Agent {agent.agent_id} cannot gather from depleted resource")
        return

    # Perform the gathering
    try:
        # Consume from resource
        actual_gathered = closest_resource.consume(gather_amount)

        # Add to agent's resources
        agent.resource_level += actual_gathered

        # Calculate simple reward based on amount gathered
        reward = actual_gathered * 0.1  # Simple reward per unit gathered
        agent.total_reward += reward

        # Log successful gather action using helper function
        log_interaction_safely(
            agent,
            source_type="agent",
            source_id=agent.agent_id,
            target_type="resource",
            target_id=str(getattr(closest_resource, "resource_id", "unknown")),
            interaction_type="gather",
            action_type="gather",
            details={
                "amount_gathered": actual_gathered,
                "resource_before": resource_amount_before,
                "resource_after": closest_resource.amount,
                "distance": min_distance,
                "success": True,
            },
        )

        logger.debug(
            f"Agent {agent.agent_id} gathered {actual_gathered} resources from {min_distance:.2f} units away"
        )

    except Exception as e:
        logger.error(f"Gathering failed for agent {agent.agent_id}: {str(e)}")

        # Log failed gather action using helper function
        log_interaction_safely(
            agent,
            source_type="agent",
            source_id=agent.agent_id,
            target_type="resource",
            target_id=str(getattr(closest_resource, "resource_id", "unknown")),
            interaction_type="gather_failed",
            action_type="gather",
            details={
                "reason": "gathering_error",
                "error": str(e),
                "success": False,
            },
        )


def share_action(agent: "BaseAgent") -> None:
    """Execute the share action for the given agent.

    Simple rule-based sharing that finds agents in need and shares resources with them.
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "share"):
        return

    # Get sharing range from config
    share_range = getattr(agent.config, "share_range", 30)

    # Find nearby agents using spatial index
    nearby_agents = agent.spatial_service.get_nearby_agents(agent.position, share_range)

    # Filter out self and find valid targets
    valid_targets = [
        target
        for target in nearby_agents
        if target.agent_id != agent.agent_id and target.alive
    ]

    if not valid_targets:
        logger.debug(
            f"Agent {agent.agent_id} found no valid targets within range {share_range}"
        )
        return

    # Find the agent with the lowest resource level (simple need-based selection)
    target = min(valid_targets, key=lambda a: a.resource_level)

    # Determine share amount (simple fixed amount if agent has enough resources)
    share_amount = getattr(agent.config, "share_amount", 2)
    min_keep = getattr(agent.config, "min_keep_resources", 5)

    # Only share if agent has enough resources to keep minimum and share
    if not check_resource_requirement(
        agent, min_keep + share_amount, "share", "resources to share"
    ):
        return

    # Execute sharing
    agent.resource_level -= share_amount
    target.resource_level += share_amount

    # Calculate simple reward
    reward = share_amount * 0.05  # Small reward per resource shared
    agent.total_reward += reward

    # Update environment's resources_shared counter
    if hasattr(agent, "metrics_service") and agent.metrics_service:
        try:
            agent.metrics_service.record_resources_shared(share_amount)
        except Exception:
            pass

    # Log the share interaction using helper function
    log_interaction_safely(
        agent,
        source_type="agent",
        source_id=agent.agent_id,
        target_type="agent",
        target_id=target.agent_id,
        interaction_type="share",
        action_type="share",
        details={
            "amount_shared": share_amount,
            "target_resources_before": target.resource_level - share_amount,
            "target_resources_after": target.resource_level,
            "success": True,
        },
    )

    logger.debug(
        f"Agent {agent.agent_id} shared {share_amount} resources with {target.agent_id} "
        f"(target now has {target.resource_level} resources)"
    )


def move_action(agent: "BaseAgent") -> None:
    """Execute the move action for the given agent.

    Simple rule-based movement that randomly selects a direction
    and moves the agent within environment bounds.
    """
    import random

    # Validate agent configuration
    if not validate_agent_config(agent, "move"):
        return

    # Define movement directions (right, left, up, down)
    directions = [
        (1, 0),  # Right
        (-1, 0),  # Left
        (0, 1),  # Up
        (0, -1),  # Down
    ]

    # Randomly select a direction
    dx, dy = random.choice(directions)

    # Get movement distance from config
    move_distance = getattr(agent.config, "max_movement", 1)

    # Calculate new position
    new_x = agent.position[0] + dx * move_distance
    new_y = agent.position[1] + dy * move_distance

    # Ensure position stays within environment bounds
    # Bound by config width/height if provided
    if hasattr(agent, "config") and agent.config:
        env_width = getattr(agent.config, "width", None)
        env_height = getattr(agent.config, "height", None)
        if env_width is not None and env_height is not None:
            new_x = max(0, min(env_width - 1, new_x))
            new_y = max(0, min(env_height - 1, new_y))

    new_position = (new_x, new_y)

    # Check if the new position is valid
    if hasattr(agent, "validation_service") and agent.validation_service and agent.validation_service.is_valid_position(new_position):
        # Move the agent
        agent.update_position(new_position)
        logger.debug(
            f"Agent {agent.agent_id} moved from {agent.position} to {new_position}"
        )
    else:
        logger.debug(
            f"Agent {agent.agent_id} could not move to invalid position {new_position}"
        )


def reproduce_action(agent: "BaseAgent") -> None:
    """Execute the reproduce action for the given agent.

    Simple rule-based reproduction that checks resource requirements
    and uses a random chance for reproduction attempts.
    """
    import random

    # Validate agent configuration
    if not validate_agent_config(agent, "reproduce"):
        return

    # Check if agent has enough resources for reproduction
    min_resources = getattr(agent.config, "min_reproduction_resources", 8)
    if not check_resource_requirement(agent, min_resources, "reproduce"):
        return

    # Simple rule: 50% chance to attempt reproduction if conditions are met
    if random.random() < 0.5:
        success = agent.reproduce()
        if success:
            logger.debug(f"Agent {agent.agent_id} successfully reproduced")
        else:
            logger.debug(f"Agent {agent.agent_id} reproduction attempt failed")
    else:
        logger.debug(f"Agent {agent.agent_id} chose not to reproduce this turn")


def defend_action(agent: "BaseAgent") -> None:
    """Execute the defend action for the given agent.

    Simple rule-based defense that allows agents to enter a defensive stance,
    reducing damage taken and providing recovery benefits.
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "defend"):
        return

    # Get defense parameters from config
    defense_duration = getattr(agent.config, "defense_duration", 3)
    defense_healing = getattr(agent.config, "defense_healing", 1)
    defense_cost = getattr(agent.config, "defense_cost", 0)

    # Check if agent has enough resources for defense cost
    if not check_resource_requirement(agent, defense_cost, "defend"):
        return

    # Pay the defense cost
    if defense_cost > 0:
        agent.resource_level -= defense_cost

    # Enter defensive state
    agent.is_defending = True
    agent.defense_timer = defense_duration

    # Apply healing/recovery effect
    if defense_healing > 0 and hasattr(agent, "current_health"):
        max_health = getattr(agent, "starting_health", agent.current_health)
        healing_amount = min(defense_healing, max_health - agent.current_health)
        if healing_amount > 0:
            agent.current_health += healing_amount
            logger.debug(
                f"Agent {agent.agent_id} healed {healing_amount} health while defending"
            )

    # Calculate simple reward for defensive action
    reward = 0.02  # Small reward for successful defense
    agent.total_reward += reward

    # Log the defend action using helper function
    log_interaction_safely(
        agent,
        source_type="agent",
        source_id=agent.agent_id,
        target_type="agent",
        target_id=agent.agent_id,  # Self-targeted for defense
        interaction_type="defend",
        action_type="defend",
        details={
            "duration": defense_duration,
            "healing": defense_healing,
            "cost": defense_cost,
            "success": True,
        },
    )

    logger.debug(
        f"Agent {agent.agent_id} entered defensive stance for {defense_duration} turns "
        f"(cost: {defense_cost}, healing: {defense_healing})"
    )


def pass_action(agent: "BaseAgent") -> None:
    """Execute the pass action for the given agent.

    Simple action where the agent does nothing this turn.
    Useful for waiting, observing, or strategic inaction.
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "pass"):
        return

    # Calculate small reward for strategic inaction
    reward = 0.01  # Minimal reward for passing
    agent.total_reward += reward

    # Log the pass action using helper function
    log_interaction_safely(
        agent,
        source_type="agent",
        source_id=agent.agent_id,
        target_type="agent",
        target_id=agent.agent_id,  # Self-targeted for pass
        interaction_type="pass",
        action_type="pass",
        details={
            "reason": "strategic_inaction",
            "success": True,
        },
    )

    logger.debug(f"Agent {agent.agent_id} chose to pass this turn")


action_registry.register("attack", 0.1, attack_action)
action_registry.register("move", 0.4, move_action)
action_registry.register("gather", 0.3, gather_action)
action_registry.register("reproduce", 0.15, reproduce_action)
action_registry.register("share", 0.2, share_action)
action_registry.register("defend", 0.25, defend_action)
action_registry.register("pass", 0.05, pass_action)
