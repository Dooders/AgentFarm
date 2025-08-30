from typing import Any, Dict, List, Optional, Tuple

from farm.core.services.interfaces import (
    IAgentLifecycleService,
    ILoggingService,
    IMetricsService,
    ISpatialQueryService,
    ITimeService,
    IValidationService,
)


class EnvironmentValidationService(IValidationService):
    """Service for validating positions and other environmental constraints.

    This service provides validation functionality by delegating to the
    environment's validation methods.
    """

    def __init__(self, environment: Any) -> None:
        """Initialize the validation service with an environment instance.

        Args:
            environment: The environment object that contains validation methods.
        """
        self._env = environment

    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Check if a given position is valid within the environment.

        Args:
            position: A tuple of (x, y) coordinates to validate.

        Returns:
            True if the position is valid, False otherwise.
        """
        return self._env.is_valid_position(position)


class EnvironmentMetricsService(IMetricsService):
    """Service for recording and tracking simulation metrics.

    This service provides methods to record various events and metrics
    that occur during simulation execution.
    """

    def __init__(self, environment: Any) -> None:
        """Initialize the metrics service with an environment instance.

        Args:
            environment: The environment object that contains metrics recording methods.
        """
        self._env = environment

    def record_birth(self) -> None:
        """Record that an agent birth event has occurred."""
        self._env.record_birth()

    def record_death(self) -> None:
        """Record that an agent death event has occurred."""
        self._env.record_death()

    def record_combat_encounter(self) -> None:
        """Record that a combat encounter between agents has occurred."""
        self._env.record_combat_encounter()

    def record_successful_attack(self) -> None:
        """Record that a successful attack has occurred in combat."""
        self._env.record_successful_attack()

    def record_resources_shared(self, amount: float) -> None:
        """Record that resources have been shared between agents.

        Args:
            amount: The amount of resources that were shared.
        """
        self._env.record_resources_shared(amount)


class EnvironmentAgentLifecycleService(IAgentLifecycleService):
    """Service for managing the lifecycle of agents in the environment.

    This service handles adding and removing agents from the simulation,
    as well as generating unique agent identifiers.
    """

    def __init__(self, environment: Any) -> None:
        """Initialize the agent lifecycle service with an environment instance.

        Args:
            environment: The environment object that manages agent lifecycle operations.
        """
        self._env = environment

    def add_agent(self, agent: Any) -> None:
        """Add an agent to the environment.

        Args:
            agent: The agent object to add to the simulation.
        """
        self._env.add_agent(agent)

    def remove_agent(self, agent: Any) -> None:
        """Remove an agent from the environment.

        Args:
            agent: The agent object to remove from the simulation.
        """
        self._env.remove_agent(agent)

    def get_next_agent_id(self) -> str:
        """Generate and return the next unique agent identifier.

        Returns:
            A unique string identifier for a new agent.
        """
        return self._env.get_next_agent_id()


class EnvironmentTimeService(ITimeService):
    """Service for accessing the current simulation time.

    This service provides a clean interface to query the current
    time step of the simulation.
    """

    def __init__(self, environment: Any) -> None:
        """Initialize the time service with an environment instance.

        Args:
            environment: The environment object that tracks simulation time.
        """
        self._env = environment

    def current_time(self) -> int:
        """Get the current simulation time step.

        Returns:
            The current time step as an integer.
        """
        return self._env.time


class EnvironmentLoggingService(ILoggingService):
    """Service for logging various simulation events and interactions.

    This service handles logging of agent interactions, reproduction events,
    and agent lifecycle changes to the simulation database.
    """

    def __init__(self, environment: Any) -> None:
        """Initialize the logging service with an environment instance.

        Args:
            environment: The environment object that contains logging methods.
        """
        self._env = environment

    def log_interaction_edge(
        self,
        source_type: str,
        source_id: str | int,
        target_type: str,
        target_id: str | int,
        interaction_type: str,
        action_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an interaction between two entities in the simulation.

        Args:
            source_type: The type of the source entity (e.g., "agent", "resource").
            source_id: The identifier of the source entity.
            target_type: The type of the target entity (e.g., "agent", "resource").
            target_id: The identifier of the target entity.
            interaction_type: The type of interaction (e.g., "combat", "sharing").
            action_type: Optional specific action type within the interaction.
            details: Optional dictionary of additional details about the interaction.
        """
        # delegate to environment's method which already checks DB presence
        self._env.log_interaction_edge(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            interaction_type=interaction_type,
            action_type=action_type,
            details=details,
        )

    def log_reproduction_event(
        self,
        step_number: int,
        parent_id: str,
        success: bool,
        parent_resources_before: float,
        parent_resources_after: float,
        offspring_id: Optional[str] = None,
        offspring_initial_resources: Optional[float] = None,
        failure_reason: Optional[str] = None,
        parent_position: Optional[Tuple[float, float]] = None,
        parent_generation: Optional[int] = None,
        offspring_generation: Optional[int] = None,
    ) -> None:
        """Log a reproduction attempt by an agent.

        Args:
            step_number: The current simulation step when reproduction occurred.
            parent_id: The identifier of the parent agent attempting reproduction.
            success: Whether the reproduction attempt was successful.
            parent_resources_before: Parent's resource level before reproduction.
            parent_resources_after: Parent's resource level after reproduction.
            offspring_id: The identifier of the offspring if reproduction succeeded.
            offspring_initial_resources: Initial resources allocated to offspring.
            failure_reason: Reason for reproduction failure if unsuccessful.
            parent_position: Position of the parent agent during reproduction.
            parent_generation: Generation number of the parent agent.
            offspring_generation: Generation number of the offspring.
        """
        # delegate to environment's method which already checks DB presence
        self._env.log_reproduction_event(
            step_number=step_number,
            parent_id=parent_id,
            success=success,
            parent_resources_before=parent_resources_before,
            parent_resources_after=parent_resources_after,
            offspring_id=offspring_id,
            offspring_initial_resources=offspring_initial_resources,
            failure_reason=failure_reason,
            parent_position=parent_position,
            parent_generation=parent_generation,
            offspring_generation=offspring_generation,
        )

    def update_agent_death(
        self, agent_id: str, death_time: int, cause: str = "starvation"
    ) -> None:
        """Update the death information for an agent in the database.

        Args:
            agent_id: The identifier of the agent that died.
            death_time: The simulation step when the agent died.
            cause: The cause of death (default: "starvation").
        """
        # delegate to environment's method which already checks DB presence
        self._env.update_agent_death(agent_id, death_time, cause)


class SpatialIndexAdapter(ISpatialQueryService):
    """Adapter that exposes `SpatialIndex` as an `ISpatialQueryService`.

    This adapter provides a clean interface to spatial indexing functionality,
    allowing spatial queries to be performed on agents and resources.
    """

    def __init__(self, spatial_index: Any) -> None:
        """Initialize the spatial index adapter.

        Args:
            spatial_index: The spatial index object that implements the core spatial operations.
        """
        self._index = spatial_index

    def get_nearby_agents(
        self, position: Tuple[float, float], radius: float
    ) -> List[Any]:
        """Get all agents within a specified radius using the spatial index.

        Args:
            position: A tuple of (x, y) coordinates representing the center point.
            radius: The search radius around the position.

        Returns:
            A list of agents within the specified radius.
        """
        return self._index.get_nearby_agents(position, radius)

    def get_nearby_resources(
        self, position: Tuple[float, float], radius: float
    ) -> List[Any]:
        """Get all resources within a specified radius using the spatial index.

        Args:
            position: A tuple of (x, y) coordinates representing the center point.
            radius: The search radius around the position.

        Returns:
            A list of resources within the specified radius.
        """
        return self._index.get_nearby_resources(position, radius)

    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional[Any]:
        """Find the nearest resource to a given position using the spatial index.

        Args:
            position: A tuple of (x, y) coordinates to search from.

        Returns:
            The nearest resource object, or None if no resources are found.
        """
        return self._index.get_nearest_resource(position)

    def mark_positions_dirty(self) -> None:
        """Mark that agent/resource positions have changed and spatial index needs updating.

        This method should be called when positions are modified to ensure
        that the spatial index remains accurate for subsequent queries.
        """
        self._index.mark_positions_dirty()


__all__ = [
    "EnvironmentValidationService",
    "EnvironmentMetricsService",
    "EnvironmentAgentLifecycleService",
    "EnvironmentTimeService",
    "EnvironmentLoggingService",
    "SpatialIndexAdapter",
]
