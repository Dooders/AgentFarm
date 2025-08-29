from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from farm.core.environment import Environment


@runtime_checkable
class MetricsService(Protocol):
    def record_birth(self) -> None: ...
    def record_death(self) -> None: ...
    def record_combat_encounter(self) -> None: ...
    def record_successful_attack(self) -> None: ...
    def record_resources_shared(self, amount: float) -> None: ...


@runtime_checkable
class LoggingService(Protocol):
    def log_interaction_edge(self, **kwargs) -> None: ...


class EnvironmentMetricsAdapter(MetricsService):
    """Adapter that delegates metrics recording to the Environment instance.

    Uses environment's own methods to preserve side-effects (e.g., counters).
    """

    def __init__(self, environment: "Environment") -> None:
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


class EnvironmentLoggingAdapter(LoggingService):
    """Adapter that delegates interaction-edge logging to the Environment.

    Environment encapsulates DB availability and step/time handling.
    """

    def __init__(self, environment: "Environment") -> None:
        self._env = environment

    def log_interaction_edge(self, **kwargs) -> None:
        self._env.log_interaction_edge(**kwargs)