from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ISpatialQueryService(ABC):
    """Interface for spatial queries in the environment.

    Provides methods to query nearby entities and spatial validity without
    coupling callers to a concrete `Environment` implementation.
    """

    @abstractmethod
    def get_nearby_agents(self, position: Tuple[float, float], radius: float) -> List[Any]:
        """Find agents within the given radius of a position."""
        pass

    @abstractmethod
    def get_nearby_resources(self, position: Tuple[float, float], radius: float) -> List[Any]:
        """Find resources within the given radius of a position."""
        pass

    @abstractmethod
    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional[Any]:
        """Find the nearest resource to a position if any exist."""
        pass

    @abstractmethod
    def mark_positions_dirty(self) -> None:
        """Mark spatial structures as needing an update after position changes."""
        pass


class IValidationService(ABC):
    """Interface for common validation checks related to the environment."""

    @abstractmethod
    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Return True if position is inside the environment bounds."""
        pass


class IMetricsService(ABC):
    """Interface for recording simulation metrics and counters."""

    @abstractmethod
    def record_birth(self) -> None:
        pass

    @abstractmethod
    def record_death(self) -> None:
        pass

    @abstractmethod
    def record_combat_encounter(self) -> None:
        pass

    @abstractmethod
    def record_successful_attack(self) -> None:
        pass

    @abstractmethod
    def record_resources_shared(self, amount: float) -> None:
        pass


class IAgentLifecycleService(ABC):
    """Interface for agent lifecycle operations in the environment."""

    @abstractmethod
    def add_agent(self, agent: Any) -> None:
        pass

    @abstractmethod
    def remove_agent(self, agent: Any) -> None:
        pass

    @abstractmethod
    def get_next_agent_id(self) -> str:
        pass


class ITimeService(ABC):
    """Interface for accessing simulation time without direct environment coupling."""

    @abstractmethod
    def current_time(self) -> int:
        pass


class ILoggingService(ABC):
    """Interface for logging key simulation events to external sinks (e.g., database)."""

    @abstractmethod
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
        pass

    @abstractmethod
    def update_agent_death(self, agent_id: str, death_time: int, cause: str = "starvation") -> None:
        pass


__all__ = [
    "ISpatialQueryService",
    "IValidationService",
    "IMetricsService",
    "IAgentLifecycleService",
    "ITimeService",
    "ILoggingService",
]

