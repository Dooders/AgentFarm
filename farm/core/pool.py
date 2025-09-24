"""Object pooling utilities for agents and actions.

Provides reusable pools to mitigate heavy object creation/destruction costs.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Type

from farm.core.action import Action


class AgentPool:
    """Pool for reusing agent instances.

    The pool stores previously used agent objects and reinitializes them on acquire,
    reducing memory churn and GC pressure from frequent allocations.
    """

    def __init__(
        self,
        agent_cls: Type[Any],
        *,
        max_size: Optional[int] = None,
        preload: int = 0,
    ) -> None:
        self._agent_cls = agent_cls
        self._pool: List[Any] = []
        self._max_size = max_size
        self.total_created = 0
        self.total_reused = 0

        # Optionally preload empty instances; they will be reset on acquire
        for _ in range(max(0, preload)):
            self._pool.append(agent_cls.__new__(agent_cls))

    def acquire(self, **kwargs) -> Any:
        """Get an agent from the pool or create a new one if empty.

        Keyword args are passed to the agent's reset (preferred) or __init__.
        """
        if self._pool:
            agent = self._pool.pop()
            self.total_reused += 1
            # Use reset if available, otherwise call __init__ to reinitialize
            if hasattr(agent, "reset") and callable(getattr(agent, "reset")):
                agent.reset(**kwargs)
                return agent
            # Fallback: construct a fresh instance if reset is not available
            agent = self._agent_cls(**kwargs)
            self.total_created += 1
            return agent

        # Pool empty: create new instance
        agent = self._agent_cls(**kwargs)
        self.total_created += 1
        return agent

    def release(self, agent: Any) -> None:
        """Return an agent to the pool after preparing it for reuse."""
        try:
            if hasattr(agent, "prepare_for_release") and callable(
                getattr(agent, "prepare_for_release")
            ):
                agent.prepare_for_release()
        except Exception:
            # Never let pooling interfere with simulation lifecycle
            pass

        if self._max_size is not None and len(self._pool) >= self._max_size:
            # Drop if pool is full
            return
        self._pool.append(agent)

    def size(self) -> int:
        return len(self._pool)

    def capacity(self) -> Optional[int]:
        return self._max_size

    def clear(self) -> None:
        self._pool.clear()


class ActionPool:
    """Lightweight pool for Action objects.

    Actions are small, but when thousands of agents are reconstructed from genomes,
    pooling avoids repeated allocations.
    """

    def __init__(self) -> None:
        self._pools: Dict[str, List[Action]] = {}

    def acquire(self, name: str, weight: float, function: Callable) -> Action:
        bucket = self._pools.get(name)
        if bucket:
            action = bucket.pop()
            action.name = name
            action.weight = weight
            action.function = function
            return action
        return Action(name, weight, function)

    def release(self, action: Action) -> None:
        bucket = self._pools.setdefault(action.name, [])
        bucket.append(action)


# Global singleton for convenience in genome reconstruction
global_action_pool = ActionPool()


def pooling_enabled() -> bool:
    """Global on/off switch controlled via environment variable.

    Set FARM_DISABLE_POOLING=1 to disable pooling paths (for benchmarks/tests).
    """
    return os.getenv("FARM_DISABLE_POOLING", "0") != "1"

