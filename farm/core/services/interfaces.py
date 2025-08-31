from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ISpatialQueryService(ABC):
    """Interface for spatial queries in the environment.

    Provides methods to query nearby entities and spatial validity without
    coupling callers to a concrete `Environment` implementation. This abstraction
    allows for different spatial indexing strategies (e.g., quad trees, spatial
    hashing, brute force) to be swapped without affecting dependent code.
    """

    @abstractmethod
    def get_nearby_agents(
        self, position: Tuple[float, float], radius: float
    ) -> List[Any]:
        """Find all agents within a specified radius of a given position.

        Args:
            position: A tuple of (x, y) coordinates representing the center point.
            radius: The search radius around the position. Must be non-negative.

        Returns:
            A list of agents found within the specified radius. The list may be
            empty if no agents are found. Agents are typically returned in no
            particular order.

        Raises:
            ValueError: If radius is negative.
        """
        pass

    @abstractmethod
    def get_nearby_resources(
        self, position: Tuple[float, float], radius: float
    ) -> List[Any]:
        """Find all resources within a specified radius of a given position.

        Args:
            position: A tuple of (x, y) coordinates representing the center point.
            radius: The search radius around the position. Must be non-negative.

        Returns:
            A list of resources found within the specified radius. The list may be
            empty if no resources are found. Resources are typically returned in no
            particular order.

        Raises:
            ValueError: If radius is negative.
        """
        pass

    @abstractmethod
    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional[Any]:
        """Find the nearest resource to a given position.

        Args:
            position: A tuple of (x, y) coordinates representing the search point.

        Returns:
            The nearest resource object if any exists, None otherwise. Distance
            is typically calculated using Euclidean distance, but implementation
            may vary based on the spatial indexing strategy used.
        """
        pass

    @abstractmethod
    def mark_positions_dirty(self) -> None:
        """Mark spatial structures as needing an update after position changes.

        This method should be called whenever agent or resource positions have
        been modified externally. It signals to the spatial indexing system that
        cached spatial data may be stale and needs recalculation on the next query.

        Note:
            Calling this method does not immediately update spatial structures.
            Updates are typically deferred until the next spatial query to avoid
            unnecessary computation when multiple position changes occur in sequence.
        """
        pass


class IConfigService(ABC):
    """Interface for accessing configuration values and environment settings.

    Centralizes configuration lookup to enable dependency injection and
    improve testability by avoiding direct environment access in modules.
    """

    @abstractmethod
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a configuration value by key.

        Args:
            key: Configuration or environment variable name
            default: Value to return if the key is not found

        Returns:
            The configuration value as a string if found, otherwise default.
        """
        pass

    @abstractmethod
    def get_analysis_module_paths(self, env_var: str = "FARM_ANALYSIS_MODULES") -> List[str]:
        """Get analysis module import paths from configuration.

        Args:
            env_var: Name of the configuration/environment variable containing
                     a comma-separated list of import paths

        Returns:
            List of import path strings. Empty if none configured.
        """
        pass

    @abstractmethod
    def get_openai_api_key(self) -> Optional[str]:
        """Return the OpenAI API key from configuration if available."""
        pass


class IValidationService(ABC):
    """Interface for common validation checks related to the environment.

    Provides methods to validate positions, actions, and other environmental
    constraints without coupling to specific environment implementations.
    This allows validation logic to be reused across different environment types.
    """

    @abstractmethod
    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Check if a position is valid within the environment boundaries.

        Args:
            position: A tuple of (x, y) coordinates to validate.

        Returns:
            True if the position is within valid environment bounds and accessible,
            False otherwise. A position might be invalid if it's outside the
            environment boundaries or blocked by obstacles (depending on implementation).

        Note:
            The exact definition of "valid" may vary by environment implementation.
            For example, some environments may allow positions outside nominal bounds
            while others may have complex boundary conditions.
        """
        pass


class IMetricsService(ABC):
    """Interface for recording simulation metrics and counters.

    Provides a standardized way to track key simulation events and statistics.
    Implementations may store metrics in memory, write to databases, or integrate
    with external monitoring systems. This abstraction allows metrics collection
    to be configured independently of the core simulation logic.
    """

    @abstractmethod
    def record_birth(self) -> None:
        """Record that a new agent has been born in the simulation.

        This should be called whenever an agent is created through reproduction
        or initial population seeding. Used to track population growth and
        reproduction rates over time.
        """
        pass

    @abstractmethod
    def record_death(self) -> None:
        """Record that an agent has died in the simulation.

        This should be called whenever an agent is removed from the simulation
        due to starvation, combat, or other causes. Used to track population
        decline and mortality rates.
        """
        pass

    @abstractmethod
    def record_combat_encounter(self) -> None:
        """Record that a combat encounter has occurred between agents.

        This should be called whenever two or more agents engage in combat,
        regardless of the outcome. Used to track conflict frequency and
        interaction patterns in the simulation.
        """
        pass

    @abstractmethod
    def record_successful_attack(self) -> None:
        """Record that an attack in combat was successful.

        This should be called when one agent successfully damages another
        in combat. Used to track combat effectiveness and survival strategies.
        Note: This is separate from combat_encounter to allow analysis of
        attack success rates.
        """
        pass

    @abstractmethod
    def record_resources_shared(self, amount: float) -> None:
        """Record that resources have been shared between agents.

        Args:
            amount: The amount of resources that were shared. Should be a
                positive value representing the quantity transferred.

        This should be called when agents cooperate by sharing resources.
        Used to track cooperative behavior and resource distribution patterns.
        """
        pass


class IAgentLifecycleService(ABC):
    """Interface for agent lifecycle operations in the environment.

    Manages the addition, removal, and identification of agents within the
    simulation. This abstraction allows different agent management strategies
    (e.g., ID generation schemes, agent storage mechanisms) to be implemented
    without affecting dependent code.
    """

    @abstractmethod
    def add_agent(self, agent: Any) -> None:
        """Add a new agent to the simulation environment.

        Args:
            agent: The agent object to add to the environment. The agent should
                be properly initialized with valid state before being added.

        Note:
            The agent should have a valid ID assigned before or during this call.
            Implementations may perform validation to ensure the agent is in
            a suitable state for addition to the simulation.
        """
        pass

    @abstractmethod
    def remove_agent(self, agent: Any) -> None:
        """Remove an agent from the simulation environment.

        Args:
            agent: The agent object to remove from the environment. The agent
                should currently exist in the simulation.

        Note:
            After removal, the agent should no longer participate in simulation
            activities. Implementations may perform cleanup operations or
            trigger related events (e.g., metrics recording).
        """
        pass

    @abstractmethod
    def get_next_agent_id(self) -> str:
        """Generate and return the next unique agent identifier.

        Returns:
            A unique string identifier for a new agent. The ID should be unique
            within the current simulation run and follow a consistent format
            to allow for easy identification and debugging.

        Note:
            This method should be thread-safe if the simulation supports
            concurrent agent creation. The ID generation strategy may vary
            by implementation (e.g., sequential numbering, UUIDs, timestamps).
        """
        pass


class ITimeService(ABC):
    """Interface for accessing simulation time without direct environment coupling.

    Provides a standardized way to access the current simulation time step.
    This abstraction allows different time management strategies (e.g., real-time
    simulation, turn-based, accelerated time) to be implemented without affecting
    dependent code.
    """

    @abstractmethod
    def current_time(self) -> int:
        """Get the current simulation time step.

        Returns:
            The current time step as an integer. Time steps typically start from 0
            and increment monotonically as the simulation progresses. The exact
            interpretation of time steps may vary by simulation implementation.

        Note:
            This method should return a consistent value throughout a single
            simulation step. Time should only advance when explicitly progressed
            by the simulation engine.
        """
        pass


class ILoggingService(ABC):
    """Interface for logging key simulation events to external sinks (e.g., database).

    Provides methods to record significant simulation events for analysis and
    debugging. Implementations may write to databases, files, or external
    monitoring systems. This abstraction allows logging configuration to be
    changed without affecting core simulation logic.
    """

    @abstractmethod
    def log_interaction_edge(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        interaction_type: str,
        action_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an interaction between two simulation entities.

        Args:
            source_type: The type of the initiating entity (e.g., "agent", "resource").
            source_id: Unique identifier of the initiating entity.
            target_type: The type of the target entity (e.g., "agent", "resource").
            target_id: Unique identifier of the target entity.
            interaction_type: The category of interaction (e.g., "combat", "sharing",
                "communication").
            action_type: Optional specific action within the interaction type
                (e.g., "attack", "defend", "share_resources").
            details: Optional dictionary containing additional context-specific
                information about the interaction (e.g., damage dealt, resources shared).

        Note:
            This method is useful for building interaction networks and analyzing
            behavioral patterns in the simulation. The details parameter allows
            for flexible extension without changing the interface.
        """
        pass

    @abstractmethod
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
        """Log a reproduction attempt and its outcome.

        Args:
            step_number: The simulation time step when the reproduction occurred.
            parent_id: Unique identifier of the parent agent attempting reproduction.
            success: True if reproduction was successful, False otherwise.
            parent_resources_before: Parent's resource level before reproduction attempt.
            parent_resources_after: Parent's resource level after reproduction attempt.
            offspring_id: Unique identifier of the offspring if reproduction succeeded.
            offspring_initial_resources: Initial resource level assigned to offspring.
            failure_reason: Description of why reproduction failed (if applicable).
            parent_position: Position of the parent agent at time of reproduction.
            parent_generation: Generation number of the parent agent.
            offspring_generation: Generation number assigned to the offspring.

        Note:
            This comprehensive logging enables detailed analysis of reproduction
            patterns, resource costs, and evolutionary dynamics. Failed reproduction
            attempts are logged with failure_reason to distinguish different causes.
        """
        pass

    @abstractmethod
    def update_agent_death(
        self, agent_id: str, death_time: int, cause: str = "starvation"
    ) -> None:
        """Log or update the death of an agent.

        Args:
            agent_id: Unique identifier of the deceased agent.
            death_time: Simulation time step when the agent died.
            cause: The reason for the agent's death. Common causes include
                "starvation", "combat", "old_age", or custom causes defined
                by specific simulation rules.

        Note:
            This method may be called retrospectively to update death information
            that wasn't available at the time of agent removal, or it may be
            called immediately upon agent death. The cause parameter helps
            categorize mortality patterns in the simulation.
        """
        pass


__all__ = [
    "ISpatialQueryService",
    "IValidationService",
    "IMetricsService",
    "IAgentLifecycleService",
    "ITimeService",
    "ILoggingService",
]
