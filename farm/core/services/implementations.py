from typing import Any, Dict, List, Optional, Tuple

from farm.core.services.interfaces import (
    IAgentLifecycleService,
    ILoggingService,
    IMetricsService,
    ISpatialQueryService,
    ITimeService,
    IValidationService,
)


class EnvironmentSpatialQueryService(ISpatialQueryService):
    def __init__(self, environment: Any) -> None:
        self._env = environment

    def get_nearby_agents(self, position: Tuple[float, float], radius: float) -> List[Any]:
        return self._env.get_nearby_agents(position, radius)

    def get_nearby_resources(self, position: Tuple[float, float], radius: float) -> List[Any]:
        return self._env.get_nearby_resources(position, radius)

    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional[Any]:
        return self._env.get_nearest_resource(position)


class EnvironmentValidationService(IValidationService):
    def __init__(self, environment: Any) -> None:
        self._env = environment

    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        return self._env.is_valid_position(position)


class EnvironmentMetricsService(IMetricsService):
    def __init__(self, environment: Any) -> None:
        self._env = environment

    def record_birth(self) -> None:
        self._env.record_birth()

    def record_death(self) -> None:
        self._env.record_death()

    def record_combat_encounter(self) -> None:
        self._env.record_combat_encounter()

    def record_successful_attack(self) -> None:
        self._env.record_successful_attack()

    def record_resources_shared(self, amount: float) -> None:
        self._env.record_resources_shared(amount)


class EnvironmentAgentLifecycleService(IAgentLifecycleService):
    def __init__(self, environment: Any) -> None:
        self._env = environment

    def add_agent(self, agent: Any) -> None:
        self._env.add_agent(agent)

    def remove_agent(self, agent: Any) -> None:
        self._env.remove_agent(agent)

    def get_next_agent_id(self) -> str:
        return self._env.get_next_agent_id()


class EnvironmentTimeService(ITimeService):
    def __init__(self, environment: Any) -> None:
        self._env = environment

    def current_time(self) -> int:
        return self._env.time


class EnvironmentLoggingService(ILoggingService):
    def __init__(self, environment: Any) -> None:
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
        if self._env.db is None:
            return
        self._env.db.log_reproduction_event(
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

    def update_agent_death(self, agent_id: str, death_time: int, cause: str = "starvation") -> None:
        if self._env.db is None:
            return
        self._env.db.update_agent_death(agent_id, death_time, cause)


__all__ = [
    "EnvironmentSpatialQueryService",
    "EnvironmentValidationService",
    "EnvironmentMetricsService",
    "EnvironmentAgentLifecycleService",
    "EnvironmentTimeService",
    "EnvironmentLoggingService",
]

