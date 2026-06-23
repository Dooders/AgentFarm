"""Population cap helpers shared by reproduction gates."""

from __future__ import annotations

from typing import Any, Optional, Tuple


def get_current_population(environment: Any) -> int:
    """Return the current alive-agent count for cap checks."""
    # Prefer the PettingZoo-style agent-id list when available (O(1) len);
    # fall back to alive agent objects for environments without `agents`.
    if hasattr(environment, "agents"):
        return len(getattr(environment, "agents", []))
    return len(getattr(environment, "alive_agent_objects", []))


def get_population_cap_status(environment: Any) -> Optional[Tuple[bool, int, float]]:
    """Return ``(at_cap, current, max)`` when a positive cap is configured."""
    env_config = getattr(environment, "config", None)
    population_config = getattr(env_config, "population", None)
    max_population = getattr(population_config, "max_population", None)
    if not (
        isinstance(max_population, (int, float))
        and not isinstance(max_population, bool)
        and max_population > 0
    ):
        return None

    current_population = get_current_population(environment)
    return current_population >= max_population, current_population, float(max_population)


def is_population_at_cap(environment: Any) -> bool:
    """True when the environment has a positive cap and alive count is at/above it."""
    status = get_population_cap_status(environment)
    return status is not None and status[0]
