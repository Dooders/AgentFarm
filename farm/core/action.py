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

from farm.utils.logging_config import get_logger

logger = get_logger(__name__)
import math
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from farm.core.agent import BaseAgent

from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


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
    except Exception as e:
        logger.debug(
            "interaction_logging_failed",
            agent_id=getattr(agent, "agent_id", "unknown"),
            error_type=type(e).__name__,
            error_message=str(e),
        )


def validate_action_result(agent: "BaseAgent", action_name: str, result: dict) -> dict:
    """Validate the results of an action execution to ensure intended effects occurred.

    This function performs post-action validation to verify that the action had the
    expected effects on the agent and environment. It checks for inconsistencies
    between the reported action result and the actual agent state.

    Args:
        agent: The agent that performed the action
        action_name: Name of the action that was executed
        result: The result dictionary returned by the action

    Returns:
        dict: Validation result containing validation status and any issues found
    """
    validation_result = {"valid": True, "issues": [], "warnings": []}

    try:
        # Basic validation - check that result has required fields
        if not isinstance(result, dict):
            validation_result["valid"] = False
            validation_result["issues"].append("Action result is not a dictionary")
            return validation_result

        if "success" not in result:
            validation_result["valid"] = False
            validation_result["issues"].append("Action result missing 'success' field")
            return validation_result

        # If action reported failure, no further validation needed
        if not result["success"]:
            return validation_result

        # Action-specific validations
        if action_name == "move":
            validation_result.update(_validate_move_action(agent, result))
        elif action_name == "gather":
            validation_result.update(_validate_gather_action(agent, result))
        elif action_name == "attack":
            validation_result.update(_validate_attack_action(agent, result))
        elif action_name == "share":
            validation_result.update(_validate_share_action(agent, result))
        elif action_name == "defend":
            validation_result.update(_validate_defend_action(agent, result))
        elif action_name == "reproduce":
            validation_result.update(_validate_reproduce_action(agent, result))
        elif action_name == "pass":
            validation_result.update(_validate_pass_action(agent, result))

    except Exception as e:
        validation_result["valid"] = False
        validation_result["issues"].append(f"Validation exception: {str(e)}")
        logger.warning(f"Post-action validation failed for {action_name}: {str(e)}")

    return validation_result


def _validate_move_action(agent: "BaseAgent", result: dict) -> dict:
    """Validate move action results."""
    validation = {"valid": True, "issues": [], "warnings": []}

    if "details" not in result:
        validation["valid"] = False
        validation["issues"].append("Move action result missing details")
        return validation

    details = result["details"]

    # Check position change
    if "old_position" in details and "new_position" in details:
        old_pos = details["old_position"]
        new_pos = details["new_position"]
        actual_pos = agent.position

        # Position should have changed if move was successful
        if old_pos == new_pos and old_pos == actual_pos:
            validation["warnings"].append(
                "Agent position did not change after successful move"
            )

        # Current position should match reported new position
        if actual_pos != new_pos:
            validation["issues"].append(
                f"Agent position mismatch: expected {new_pos}, got {actual_pos}"
            )
            validation["valid"] = False

    return validation


def _validate_gather_action(agent: "BaseAgent", result: dict) -> dict:
    """Validate gather action results."""
    validation = {"valid": True, "issues": [], "warnings": []}

    if "details" not in result:
        validation["valid"] = False
        validation["issues"].append("Gather action result missing details")
        return validation

    details = result["details"]

    # Check resource increase
    if "agent_resources_before" in details and "amount_gathered" in details:
        expected_resources = (
            details["agent_resources_before"] + details["amount_gathered"]
        )
        actual_resources = agent.resource_level

        if (
            abs(actual_resources - expected_resources) > 0.01
        ):  # Allow small floating point differences
            validation["issues"].append(
                f"Resource mismatch: expected {expected_resources}, got {actual_resources}"
            )
            validation["valid"] = False

    return validation


def _validate_attack_action(agent: "BaseAgent", result: dict) -> dict:
    """Validate attack action results."""
    validation = {"valid": True, "issues": [], "warnings": []}

    if "details" not in result:
        validation["valid"] = False
        validation["issues"].append("Attack action result missing details")
        return validation

    details = result["details"]

    # Check if target was reported as damaged
    if "target_id" in details and "damage_dealt" in details:
        damage = details["damage_dealt"]
        if damage <= 0:
            validation["warnings"].append("Attack dealt no damage")

    return validation


def _validate_share_action(agent: "BaseAgent", result: dict) -> dict:
    """Validate share action results."""
    validation = {"valid": True, "issues": [], "warnings": []}

    if "details" not in result:
        validation["valid"] = False
        validation["issues"].append("Share action result missing details")
        return validation

    details = result["details"]

    # Check resource decrease
    if "agent_resources_before" in details and "amount_shared" in details:
        expected_resources = (
            details["agent_resources_before"] - details["amount_shared"]
        )
        actual_resources = agent.resource_level

        if abs(actual_resources - expected_resources) > 0.01:
            validation["issues"].append(
                f"Resource mismatch: expected {expected_resources}, got {actual_resources}"
            )
            validation["valid"] = False

    return validation


def _validate_defend_action(agent: "BaseAgent", result: dict) -> dict:
    """Validate defend action results."""
    validation = {"valid": True, "issues": [], "warnings": []}

    if "details" not in result:
        validation["valid"] = False
        validation["issues"].append("Defend action result missing details")
        return validation

    details = result["details"]

    # Check defensive state
    if not getattr(agent, "is_defending", False):
        validation["issues"].append(
            "Agent is not in defensive state after defend action"
        )
        validation["valid"] = False

    # Check resource cost
    if "resources_before" in details and "cost" in details:
        expected_resources = details["resources_before"] - details["cost"]
        actual_resources = agent.resource_level

        if abs(actual_resources - expected_resources) > 0.01:
            validation["issues"].append(
                f"Resource mismatch: expected {expected_resources}, got {actual_resources}"
            )
            validation["valid"] = False

    return validation


def _validate_reproduce_action(agent: "BaseAgent", result: dict) -> dict:
    """Validate reproduce action results."""
    validation = {"valid": True, "issues": [], "warnings": []}

    if "details" not in result:
        validation["valid"] = False
        validation["issues"].append("Reproduce action result missing details")
        return validation

    details = result["details"]

    # Check resource cost
    if "resources_before" in details and "offspring_cost" in details:
        expected_resources = details["resources_before"] - details["offspring_cost"]
        actual_resources = agent.resource_level

        if abs(actual_resources - expected_resources) > 0.01:
            validation["issues"].append(
                f"Resource mismatch: expected {expected_resources}, got {actual_resources}"
            )
            validation["valid"] = False

    return validation


def _validate_pass_action(agent: "BaseAgent", result: dict) -> dict:
    """Validate pass action results."""
    validation = {"valid": True, "issues": [], "warnings": []}

    # Pass action should not change agent state significantly
    # This is more of a sanity check than strict validation

    if "details" not in result:
        validation["valid"] = False
        validation["issues"].append("Pass action result missing details")
        return validation

    return validation


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

    def execute(self, agent, *args, **kwargs) -> dict:
        """Execute the action's behavior function with success/failure tracking.

        Calls the associated function with the agent and any additional parameters,
        capturing success status and any error information.

        Args:
            agent: Agent instance performing the action
            *args: Variable positional arguments for the action function
            **kwargs: Variable keyword arguments for the action function

        Returns:
            dict: Execution result containing:
                - success: bool indicating if action succeeded
                - error: str error message if failed (None if success)
                - details: dict with action-specific result information
        """
        try:
            # Execute the action function
            result = self.function(agent, *args, **kwargs)

            # If function returns a dict, use it; otherwise assume success
            if isinstance(result, dict):
                return result
            else:
                logger.warning(
                    f"Action {self.name} for agent {agent.agent_id} returned {type(result).__name__} instead of dict. "
                    f"Assuming success. Please update action function to return proper result dictionary."
                )
                return {"success": True, "error": None, "details": {}}

        except Exception as e:
            logger.error(
                f"Action {self.name} failed for agent {agent.agent_id}: {str(e)}"
            )
            return {
                "success": False,
                "error": str(e),
                "details": {"exception_type": type(e).__name__},
            }


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
        """Get a registered action by name.

        Args:
            name: The name of the action to retrieve

        Returns:
            The Action object if found, None if no action with that name exists
        """
        return cls._registry.get(name)

    @classmethod
    def get_all(cls, normalized: bool = True) -> List[Action]:
        """Get all registered actions.

        Args:
            normalized: If True, normalize action weights to sum to 1. Default is True.
        """
        actions = list(cls._registry.values())

        if normalized and actions:
            # Normalize weights
            total_weight = sum(action.weight for action in actions)
            for action in actions:
                action.weight /= total_weight

        return actions


def attack_action(agent: "BaseAgent") -> dict:
    """Execute the attack action for the given agent.

    This action implements aggressive behavior where the agent seeks out and attacks
    the nearest enemy within its attack range. The action uses spatial indexing for
    efficient proximity queries to find valid targets.

    Behavior details:
    - Uses configurable attack_range (default: 20.0 units) to find nearby agents
    - Filters targets to exclude self and non-living agents
    - Selects the closest valid target using Euclidean distance
    - Calculates damage based on agent's attack strength and current health ratio
    - Applies defensive damage reduction (50% reduction when target is defending)
    - Records combat metrics and logs the interaction for analysis
    - Provides small reward bonus for successful attacks

    Returns:
        dict: Action result containing success status and details
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "attack"):
        return {
            "success": False,
            "error": "Invalid agent configuration for attack action",
            "details": {},
        }

    # Get attack range from config
    attack_range = getattr(agent.config, "attack_range", 20.0)

    try:
        # Find nearby agents using spatial index
        nearby = agent.spatial_service.get_nearby(
            agent.position, attack_range, ["agents"]
        )
        nearby_agents = nearby.get("agents", [])

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
            return {
                "success": False,
                "error": "No valid targets found within attack range",
                "details": {"attack_range": attack_range},
            }

        # Find the closest target using helper function
        closest_target, min_distance = find_closest_entity(
            agent, valid_targets, "target"
        )

        if closest_target is None:
            return {
                "success": False,
                "error": "Could not find closest target",
                "details": {"attack_range": attack_range},
            }

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
        if (
            actual_damage > 0
            and hasattr(agent, "metrics_service")
            and agent.metrics_service
        ):
            try:
                agent.metrics_service.record_combat_encounter()
                agent.metrics_service.record_successful_attack()
            except Exception as e:
                logger.warning(f"Failed to record combat metrics: {e}")

        # Log the attack interaction using helper function
        log_interaction_safely(
            agent,
            source_type="agent",
            source_id=agent.agent_id,
            target_type="agent",
            target_id=closest_target.agent_id,
            interaction_type="attack" if actual_damage > 0 else "attack_failed",
            action_type="attack",
            details={},  # Minimal details to avoid duplication with action-specific logging
        )

        logger.debug(
            f"Agent {agent.agent_id} attacked {closest_target.agent_id} "
            f"at distance {min_distance:.2f}, dealt {actual_damage:.2f} damage"
        )

        return {
            "success": actual_damage > 0,
            "error": None if actual_damage > 0 else "Attack dealt no damage",
            "details": {
                "damage_dealt": actual_damage,
                "target_id": closest_target.agent_id,
                "distance": min_distance,
                "attack_range": attack_range,
                "target_position": closest_target.position,
                "attacker_position": agent.position,
                "attack_direction": attack_direction,
                "target_defending": closest_target.is_defending,
                "target_alive": closest_target.alive,
            },
        }

    except Exception as e:
        logger.error(f"Attack action failed for agent {agent.agent_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Attack action exception: {str(e)}",
            "details": {"exception_type": type(e).__name__},
        }


def gather_action(agent: "BaseAgent") -> dict:
    """Execute the gather action for the given agent.

    This action implements resource collection behavior where the agent seeks out
    and gathers resources from the nearest available resource node. The action uses
    spatial indexing for efficient proximity queries to find resource locations.

    Behavior details:
    - Uses configurable gathering_range (default: 30 units) to find nearby resources
    - Filters resources to exclude depleted or empty resource nodes
    - Selects the closest available resource using Euclidean distance
    - Gathers up to max_gather amount (configurable, default: 10) or available amount
    - Transfers gathered resources directly to agent's resource pool
    - Calculates reward based on amount gathered (0.1 per unit)
    - Records resource gathering metrics and logs the interaction for analysis

    Returns:
        dict: Action result containing success status and details
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "gather"):
        return {
            "success": False,
            "error": "Invalid agent configuration for gather action",
            "details": {},
        }

    # Get gathering range from config
    gathering_range = getattr(agent.config, "gathering_range", 30)

    try:
        # Find nearby resources using spatial index
        nearby = agent.spatial_service.get_nearby(
            agent.position, gathering_range, ["resources"]
        )
        nearby_resources = nearby.get("resources", [])

        # Filter out depleted resources
        available_resources = [
            r for r in nearby_resources if not r.is_depleted() and r.amount > 0
        ]

        if not available_resources:
            logger.debug(
                f"Agent {agent.agent_id} found no available resources within range {gathering_range}"
            )
            return {
                "success": False,
                "error": "No available resources found within gathering range",
                "details": {"gathering_range": gathering_range},
            }

        # Find the closest resource using helper function
        closest_resource, min_distance = find_closest_entity(
            agent, available_resources, "resource"
        )

        if closest_resource is None:
            return {
                "success": False,
                "error": "Could not find closest resource",
                "details": {"gathering_range": gathering_range},
            }

        # Record initial resource levels
        initial_agent_resources = agent.resource_level
        resource_amount_before = closest_resource.amount

        # Determine how much to gather
        # Prefer config.max_gather_amount if present; fall back to 10
        max_gather = getattr(agent.config, "max_gather_amount", 10)
        gather_amount = min(max_gather, closest_resource.amount)

        if gather_amount <= 0:
            logger.debug(f"Agent {agent.agent_id} cannot gather from depleted resource")
            return {
                "success": False,
                "error": "Resource is depleted or has no amount available",
                "details": {
                    "resource_amount": closest_resource.amount,
                    "max_gather": max_gather,
                },
            }

        # Perform the gathering
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
            details={},  # Minimal details to avoid duplication with action-specific logging
        )

        logger.debug(
            f"Agent {agent.agent_id} gathered {actual_gathered} resources from {min_distance:.2f} units away"
        )

        return {
            "success": actual_gathered > 0,
            "error": (
                None if actual_gathered > 0 else "No resources were actually gathered"
            ),
            "details": {
                "amount_gathered": actual_gathered,
                "resource_before": resource_amount_before,
                "resource_after": closest_resource.amount,
                "agent_resources_before": initial_agent_resources,
                "agent_resources_after": agent.resource_level,
                "distance": min_distance,
                "gathering_range": gathering_range,
                "resource_id": getattr(closest_resource, "resource_id", "unknown"),
                "resource_depleted": closest_resource.is_depleted(),
            },
        }

    except Exception as e:
        logger.error(f"Gathering failed for agent {agent.agent_id}: {str(e)}")

        # Log failed gather action using helper function
        log_interaction_safely(
            agent,
            source_type="agent",
            source_id=agent.agent_id,
            target_type="resource",
            target_id="unknown",
            interaction_type="gather_failed",
            action_type="gather",
            details={},  # Minimal details to avoid duplication
        )

        return {
            "success": False,
            "error": f"Gathering action exception: {str(e)}",
            "details": {"exception_type": type(e).__name__},
        }


def share_action(agent: "BaseAgent") -> dict:
    """Execute the share action for the given agent.

    This action implements cooperative behavior where the agent shares resources
    with nearby allies. The agent finds the most resource-depleted nearby agent
    and transfers a portion of its resources to help that agent survive.

    Behavior details:
    - Uses configurable share_range (default: 30 units) to find nearby agents
    - Filters targets to exclude self and non-living agents
    - Selects the agent with the lowest resource level (greatest need)
    - Shares a fixed share_amount (configurable, default: 2) or available amount
    - Requires agent to keep minimum resources (min_keep_resources, default: 5)
    - Transfers resources directly from agent's pool to target's pool
    - Calculates small reward for cooperative behavior (0.05 per resource shared)
    - Records resource sharing metrics and logs the interaction for analysis

    Returns:
        dict: Action result containing success status and details
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "share"):
        return {
            "success": False,
            "error": "Invalid agent configuration for share action",
            "details": {},
        }

    # Get sharing range from config
    share_range = getattr(agent.config, "share_range", 30)

    try:
        # Find nearby agents using spatial index
        nearby = agent.spatial_service.get_nearby(
            agent.position, share_range, ["agents"]
        )
        nearby_agents = nearby.get("agents", [])

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
            return {
                "success": False,
                "error": "No valid targets found within sharing range",
                "details": {"share_range": share_range},
            }

        # Find the agent with the lowest resource level (simple need-based selection)
        target = min(valid_targets, key=lambda a: a.resource_level)

        # Determine share amount (simple fixed amount if agent has enough resources)
        share_amount = getattr(agent.config, "share_amount", 2)
        min_keep = getattr(agent.config, "min_keep_resources", 5)

        # Only share if agent has enough resources to keep minimum and share
        if not check_resource_requirement(
            agent, min_keep + share_amount, "share", "resources to share"
        ):
            return {
                "success": False,
                "error": f"Insufficient resources to share while maintaining minimum of {min_keep}",
                "details": {
                    "agent_resources": agent.resource_level,
                    "required_total": min_keep + share_amount,
                    "share_amount": share_amount,
                    "min_keep": min_keep,
                },
            }

        # Record initial resource levels
        agent_resources_before = agent.resource_level
        target_resources_before = target.resource_level

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
            except Exception as e:
                logger.warning(f"Failed to record sharing metrics: {e}")

        # Log the share interaction using helper function
        log_interaction_safely(
            agent,
            source_type="agent",
            source_id=agent.agent_id,
            target_type="agent",
            target_id=target.agent_id,
            interaction_type="share",
            action_type="share",
            details={},  # Minimal details to avoid duplication with action-specific logging
        )

        logger.debug(
            f"Agent {agent.agent_id} shared {share_amount} resources with {target.agent_id} "
            f"(target now has {target.resource_level} resources)"
        )

        return {
            "success": True,
            "error": None,
            "details": {
                "amount_shared": share_amount,
                "agent_resources_before": agent_resources_before,
                "agent_resources_after": agent.resource_level,
                "target_resources_before": target_resources_before,
                "target_resources_after": target.resource_level,
                "target_id": target.agent_id,
                "share_range": share_range,
                "reward_earned": reward,
            },
        }

    except Exception as e:
        logger.error(f"Share action failed for agent {agent.agent_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Share action exception: {str(e)}",
            "details": {"exception_type": type(e).__name__},
        }


def move_action(agent: "BaseAgent") -> dict:
    """Execute the move action for the given agent.

    This action implements basic movement behavior where the agent randomly selects
    a direction and attempts to move to a new position within the environment.
    The movement is constrained by environment boundaries and validation rules.

    Behavior details:
    - Randomly selects from four cardinal directions (up, down, left, right)
    - Uses configurable max_movement distance (default: 1 unit) for movement magnitude
    - Calculates new position by applying movement vector to current position
    - Validates new position against environment boundaries and obstacles
    - Updates agent's position if the move is valid, otherwise logs failure
    - Marks spatial index as dirty to trigger updates for nearby agent queries

    Returns:
        dict: Action result containing success status and details
    """
    import random

    # Validate agent configuration
    if not validate_agent_config(agent, "move"):
        return {
            "success": False,
            "error": "Invalid agent configuration for move action",
            "details": {},
        }

    # Define movement directions (right, left, up, down)
    directions = [
        (1, 0),  # Right
        (-1, 0),  # Left
        (0, 1),  # Up
        (0, -1),  # Down
    ]

    try:
        # Randomly select a direction
        dx, dy = random.choice(directions)

        # Get movement distance from config
        move_distance = getattr(agent.config, "max_movement", 1)

        # Calculate new position
        new_x = agent.position[0] + dx * move_distance
        new_y = agent.position[1] + dy * move_distance

        # Record original position
        original_position = agent.position

        # Ensure position stays within environment bounds
        # Bound by config width/height if provided
        if hasattr(agent, "config") and agent.config:
            env_width = getattr(agent.config, "width", None)
            env_height = getattr(agent.config, "height", None)
            if env_width is not None and env_height is not None:
                new_x = max(0, min(int(env_width) - 1, new_x))
                new_y = max(0, min(int(env_height) - 1, new_y))

        new_position = (new_x, new_y)

        # Check if the new position is valid
        if (
            hasattr(agent, "validation_service")
            and agent.validation_service
            and agent.validation_service.is_valid_position(new_position)
        ):
            # Move the agent
            agent.update_position(new_position)

            # Log successful move
            log_interaction_safely(
                agent,
                source_type="agent",
                source_id=agent.agent_id,
                target_type="position",
                target_id=f"{new_position[0]},{new_position[1]}",
                interaction_type="move",
                action_type="move",
                details={},  # Minimal details to avoid duplication with action-specific logging
            )

            logger.debug(
                f"Agent {agent.agent_id} moved from {original_position} to {new_position}"
            )

            return {
                "success": True,
                "error": None,
                "details": {
                    "old_position": original_position,
                    "new_position": new_position,
                    "distance": move_distance,
                    "direction": (dx, dy),
                    "movement_vector": (dx * move_distance, dy * move_distance),
                    "position_changed": original_position != new_position,
                },
            }
        else:
            # Log failed move
            log_interaction_safely(
                agent,
                source_type="agent",
                source_id=agent.agent_id,
                target_type="position",
                target_id=f"{new_position[0]},{new_position[1]}",
                interaction_type="move_failed",
                action_type="move",
                details={},  # Minimal details to avoid duplication
            )

            logger.debug(
                f"Agent {agent.agent_id} could not move to invalid position {new_position}"
            )

            return {
                "success": False,
                "error": "Invalid position - move blocked by boundaries or obstacles",
                "details": {
                    "attempted_position": new_position,
                    "current_position": original_position,
                    "distance": move_distance,
                    "direction": (dx, dy),
                    "movement_vector": (dx * move_distance, dy * move_distance),
                    "validation_service_available": hasattr(agent, "validation_service")
                    and agent.validation_service is not None,
                },
            }

    except Exception as e:
        logger.error(f"Move action failed for agent {agent.agent_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Move action exception: {str(e)}",
            "details": {"exception_type": type(e).__name__},
        }


def reproduce_action(agent: "BaseAgent") -> dict:
    """Execute the reproduce action for the given agent.

    This action implements reproductive behavior where the agent attempts to create
    offspring if it has sufficient resources. The action consolidates resource checking,
    probability-based decision making, and offspring creation into a complete lifecycle.

    Behavior details:
    - Checks minimum resource requirements (min_reproduction_resources, default: 8)
    - Verifies total cost including offspring_cost (default: 5) can be met
    - Applies reproduction chance probability (reproduction_chance, default: 0.5)
    - Creates offspring using agent's _create_offspring method if conditions met
    - Transfers offspring_cost from parent to pay for reproduction
    - Records reproduction event with detailed metrics and logging
    - Offspring inherits parent's position and receives initial resources

    Returns:
        dict: Action result containing success status and details
    """
    import random

    # Validate agent configuration
    if not validate_agent_config(agent, "reproduce"):
        return {
            "success": False,
            "error": "Invalid agent configuration for reproduce action",
            "details": {},
        }

    # Get reproduction parameters from config
    min_resources = getattr(agent.config, "min_reproduction_resources", 8)
    offspring_cost = getattr(agent.config, "offspring_cost", 5)
    reproduction_chance = getattr(agent.config, "reproduction_chance", 0.5)

    # Check total resource requirements (minimum + offspring cost)
    total_required = min_resources + offspring_cost

    try:
        if not check_resource_requirement(agent, total_required, "reproduce"):
            return {
                "success": False,
                "error": f"Insufficient resources for reproduction (need {total_required}, have {agent.resource_level})",
                "details": {
                    "agent_resources": agent.resource_level,
                    "min_resources": min_resources,
                    "offspring_cost": offspring_cost,
                    "total_required": total_required,
                },
            }

        # Random chance to attempt reproduction
        reproduction_roll = random.random()
        if reproduction_roll >= reproduction_chance:
            logger.debug(
                f"Agent {agent.agent_id} chose not to reproduce this turn (roll: {reproduction_roll:.3f} >= chance: {reproduction_chance})"
            )
            return {
                "success": False,
                "error": f"Reproduction chance not met (roll: {reproduction_roll:.3f} >= chance: {reproduction_chance})",
                "details": {
                    "reproduction_roll": reproduction_roll,
                    "reproduction_chance": reproduction_chance,
                    "agent_resources": agent.resource_level,
                    "total_required": total_required,
                },
            }

        # Record state before reproduction
        resources_before = agent.resource_level
        generation_before = getattr(agent, "generation", 0)

        # Attempt reproduction using the agent's method
        success = agent.reproduce()

        if success:
            # Log successful reproduction
            log_interaction_safely(
                agent,
                source_type="agent",
                source_id=agent.agent_id,
                target_type="agent",
                target_id="offspring",  # Will be replaced with actual ID in reproduce method
                interaction_type="reproduce",
                action_type="reproduce",
                details={},  # Minimal details to avoid duplication with reproduction event logging
            )

            logger.debug(f"Agent {agent.agent_id} successfully reproduced")
            return {
                "success": True,
                "error": None,
                "details": {
                    "resources_before": resources_before,
                    "resources_after": agent.resource_level,
                    "offspring_cost": offspring_cost,
                    "min_resources": min_resources,
                    "parent_generation": generation_before,
                    "reproduction_roll": reproduction_roll,
                    "reproduction_chance": reproduction_chance,
                },
            }
        else:
            logger.debug(f"Agent {agent.agent_id} reproduction attempt failed")
            return {
                "success": False,
                "error": "Reproduction failed - offspring creation unsuccessful",
                "details": {
                    "resources_before": resources_before,
                    "resources_after": agent.resource_level,  # May have changed if partial creation occurred
                    "offspring_cost": offspring_cost,
                    "min_resources": min_resources,
                    "parent_generation": generation_before,
                    "reproduction_roll": reproduction_roll,
                    "reproduction_chance": reproduction_chance,
                },
            }

    except Exception as e:
        logger.error(f"Reproduce action failed for agent {agent.agent_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Reproduce action exception: {str(e)}",
            "details": {"exception_type": type(e).__name__},
        }


def defend_action(agent: "BaseAgent") -> dict:
    """Execute the defend action for the given agent.

    This action implements defensive behavior where the agent enters a protective
    stance that reduces incoming damage and provides healing benefits. The defense
    lasts for a configurable duration and consumes resources to activate.

    Behavior details:
    - Checks and pays defense cost (defense_cost, default: 0) if configured
    - Activates defensive stance by setting is_defending flag to True
    - Sets defense timer to configured duration (defense_duration, default: 3 turns)
    - Applies healing effect (defense_healing, default: 1 health) if configured
    - Healing is capped by maximum health to prevent overhealing
    - Provides small reward (0.02) for successful defensive action
    - Records defense activation with detailed logging and metrics

    Returns:
        dict: Action result containing success status and details
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "defend"):
        return {
            "success": False,
            "error": "Invalid agent configuration for defend action",
            "details": {},
        }

    # Get defense parameters from config
    defense_duration = getattr(agent.config, "defense_duration", 3)
    defense_healing = getattr(agent.config, "defense_healing", 1)
    defense_cost = getattr(agent.config, "defense_cost", 0)

    try:
        # Check if agent has enough resources for defense cost
        if not check_resource_requirement(agent, defense_cost, "defend"):
            return {
                "success": False,
                "error": f"Insufficient resources for defense cost (need {defense_cost}, have {agent.resource_level})",
                "details": {
                    "agent_resources": agent.resource_level,
                    "defense_cost": defense_cost,
                    "defense_duration": defense_duration,
                    "defense_healing": defense_healing,
                },
            }

        # Record state before defense
        resources_before = agent.resource_level
        health_before = getattr(agent, "current_health", 0)
        was_defending_before = getattr(agent, "is_defending", False)
        defense_timer_before = getattr(agent, "defense_timer", 0)

        # Pay the defense cost
        if defense_cost > 0:
            agent.resource_level -= defense_cost

        # Enter defensive state
        agent.is_defending = True
        agent.defense_timer = defense_duration

        # Apply healing/recovery effect
        healing_applied = 0
        if defense_healing > 0 and hasattr(agent, "current_health"):
            max_health = getattr(agent, "starting_health", agent.current_health)
            healing_amount = min(defense_healing, max_health - agent.current_health)
            if healing_amount > 0:
                agent.current_health += healing_amount
                healing_applied = healing_amount
                logger.debug(
                    f"Agent {agent.agent_id} healed {healing_amount} health while defending"
                )

        # Calculate simple reward for defensive action
        # Use config value if available, otherwise fall back to default
        reward = getattr(
            getattr(agent.config, "action_rewards", None), "defend_success_reward", 0.02
        )
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
            details={},  # Minimal details to avoid duplication with action-specific logging
        )

        logger.debug(
            f"Agent {agent.agent_id} entered defensive stance for {defense_duration} turns "
            f"(cost: {defense_cost}, healing: {healing_applied})"
        )

        return {
            "success": True,
            "error": None,
            "details": {
                "duration": defense_duration,
                "healing": defense_healing,
                "healing_applied": healing_applied,
                "cost": defense_cost,
                "resources_before": resources_before,
                "resources_after": agent.resource_level,
                "health_before": health_before,
                "health_after": getattr(agent, "current_health", 0),
                "was_defending_before": was_defending_before,
                "defense_timer_before": defense_timer_before,
                "reward_earned": reward,
            },
        }

    except Exception as e:
        logger.error(f"Defend action failed for agent {agent.agent_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Defend action exception: {str(e)}",
            "details": {"exception_type": type(e).__name__},
        }


def pass_action(agent: "BaseAgent") -> dict:
    """Execute the pass action for the given agent.

    This action implements strategic inaction where the agent chooses to do nothing
    during its turn. While seemingly passive, passing can be a deliberate strategy
    for observation, resource conservation, or waiting for better opportunities.

    Behavior details:
    - Agent takes no physical action (no movement, combat, or resource interaction)
    - Conserves energy by avoiding action costs and resource consumption
    - Provides minimal reward (0.01) to encourage occasional strategic inaction
    - Records pass action in interaction logs for behavioral analysis
    - Useful for agents that are waiting, observing environment changes,
      conserving resources, or implementing more sophisticated strategies

    Returns:
        dict: Action result containing success status and details
    """
    # Validate agent configuration
    if not validate_agent_config(agent, "pass"):
        return {
            "success": False,
            "error": "Invalid agent configuration for pass action",
            "details": {},
        }

    try:
        # Calculate small reward for strategic inaction
        # Use config value if available, otherwise fall back to default
        reward = getattr(
            getattr(agent.config, "action_rewards", None), "pass_action_reward", 0.01
        )
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
            details={},  # Minimal details to avoid duplication with action-specific logging
        )

        logger.debug(f"Agent {agent.agent_id} chose to pass this turn")

        return {
            "success": True,
            "error": None,
            "details": {
                "reason": "strategic_inaction",
                "reward_earned": reward,
                "agent_resources": agent.resource_level,
                "agent_health": getattr(agent, "current_health", 0),
                "agent_position": agent.position,
            },
        }

    except Exception as e:
        logger.error(f"Pass action failed for agent {agent.agent_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Pass action exception: {str(e)}",
            "details": {"exception_type": type(e).__name__},
        }


# Centralized action space utilities
def get_action_space() -> dict[str, int]:
    """Get the centralized mapping of action names to indices.

    Returns:
        dict[str, int]: Mapping from action name strings to ActionType enum values
    """
    return {
        "defend": ActionType.DEFEND.value,
        "attack": ActionType.ATTACK.value,
        "gather": ActionType.GATHER.value,
        "share": ActionType.SHARE.value,
        "move": ActionType.MOVE.value,
        "reproduce": ActionType.REPRODUCE.value,
        "pass": ActionType.PASS.value,
    }


def action_name_to_index(action_name: str) -> int:
    """Convert action name to index using the centralized action space.

    Args:
        action_name: Name of the action (case-insensitive)

    Returns:
        int: Action index from ActionType enum, defaults to DEFEND (0) if unknown
    """
    action_space = get_action_space()
    return action_space.get(action_name.lower(), ActionType.DEFEND.value)


def get_action_names() -> list[str]:
    """Get list of all valid action names in the action space.

    Returns:
        list[str]: List of action names in the order defined by ActionType enum
    """
    return list(get_action_space().keys())


def get_action_count() -> int:
    """Get the total number of actions in the action space.

    Returns:
        int: Number of actions defined in the action space
    """
    return len(ActionType)


action_registry.register("attack", 0.1, attack_action)
action_registry.register("move", 0.4, move_action)
action_registry.register("gather", 0.3, gather_action)
action_registry.register("reproduce", 0.15, reproduce_action)
action_registry.register("share", 0.2, share_action)
action_registry.register("defend", 0.25, defend_action)
action_registry.register("pass", 0.05, pass_action)
