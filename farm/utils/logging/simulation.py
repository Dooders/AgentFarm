"""Type-safe simulation logger with Protocol for better IDE support.

This module provides typed logging interfaces for simulation events,
enabling better IDE support, type checking, and documentation.

Usage:
    from farm.utils.simulation_logger import TypedSimulationLogger, SimulationLogger

    def run_simulation(logger: SimulationLogger):
        logger.log_agent_action(
            agent_id="agent_001",
            action="move",
            success=True,
            duration_ms=50.0
        )
"""

from typing import Any, Optional, Protocol, runtime_checkable

import structlog


@runtime_checkable
class SimulationLogger(Protocol):
    """Type-safe protocol for simulation logging.

    This protocol defines the interface for simulation-specific logging
    methods, enabling type checking and better IDE support.
    """

    def log_agent_action(
        self,
        agent_id: str,
        action: str,
        success: bool,
        duration_ms: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """Log agent action with type safety."""

    def log_population_change(
        self, population: int, change: int, step: int, **kwargs: Any
    ) -> None:
        """Log population change."""

    def log_resource_update(
        self, total_resources: float, active_nodes: int, step: int, **kwargs: Any
    ) -> None:
        """Log resource update."""

    def log_simulation_event(self, event: str, step: int, **kwargs: Any) -> None:
        """Log general simulation event."""

    def log_performance_metric(
        self, metric_name: str, value: float, unit: str = "ms", **kwargs: Any
    ) -> None:
        """Log performance metric."""

    def log_experiment_event(
        self, event: str, experiment_id: str, **kwargs: Any
    ) -> None:
        """Log experiment-related event."""


class TypedSimulationLogger:
    """Typed logger wrapper for simulation events.

    Provides type-safe logging methods for common simulation events
    with proper parameter validation and IDE support.
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger):
        """Initialize typed simulation logger.

        Args:
            logger: The underlying structlog logger
        """
        self.logger = logger

    def log_agent_action(
        self,
        agent_id: str,
        action: str,
        success: bool,
        duration_ms: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """Log agent action with type safety.

        Args:
            agent_id: Unique identifier for the agent
            action: Action performed by the agent
            success: Whether the action was successful
            duration_ms: Optional duration of the action in milliseconds
            **kwargs: Additional context data
        """
        self.logger.info(
            "agent_action",
            agent_id=agent_id,
            action=action,
            success=success,
            duration_ms=duration_ms,
            **kwargs
        )

    def log_population_change(
        self, population: int, change: int, step: int, **kwargs: Any
    ) -> None:
        """Log population change.

        Args:
            population: Current population count
            change: Change in population (positive for births, negative for deaths)
            step: Simulation step number
            **kwargs: Additional context data
        """
        self.logger.info(
            "population_changed",
            population=population,
            change=change,
            step=step,
            **kwargs
        )

    def log_resource_update(
        self, total_resources: float, active_nodes: int, step: int, **kwargs: Any
    ) -> None:
        """Log resource update.

        Args:
            total_resources: Total available resources
            active_nodes: Number of active resource nodes
            step: Simulation step number
            **kwargs: Additional context data
        """
        self.logger.debug(
            "resources_updated",
            total=total_resources,
            active=active_nodes,
            step=step,
            **kwargs
        )

    def log_simulation_event(self, event: str, step: int, **kwargs: Any) -> None:
        """Log general simulation event.

        Args:
            event: Event name
            step: Simulation step number
            **kwargs: Additional context data
        """
        self.logger.info("simulation_event", event_name=event, step=step, **kwargs)

    def log_performance_metric(
        self, metric_name: str, value: float, unit: str = "ms", **kwargs: Any
    ) -> None:
        """Log performance metric.

        Args:
            metric_name: Name of the performance metric
            value: Metric value
            unit: Unit of measurement (default: "ms")
            **kwargs: Additional context data
        """
        self.logger.info(
            "performance_metric", metric=metric_name, value=value, unit=unit, **kwargs
        )

    def log_experiment_event(
        self, event: str, experiment_id: str, **kwargs: Any
    ) -> None:
        """Log experiment-related event.

        Args:
            event: Event name
            experiment_id: Unique identifier for the experiment
            **kwargs: Additional context data
        """
        self.logger.info(
            "experiment_event", event_name=event, experiment_id=experiment_id, **kwargs
        )

    def bind(self, **kwargs: Any) -> "TypedSimulationLogger":
        """Bind context variables and return new TypedSimulationLogger.

        Args:
            **kwargs: Context variables to bind

        Returns:
            New TypedSimulationLogger with bound context
        """
        bound_logger = self.logger.bind(**kwargs)
        return TypedSimulationLogger(bound_logger)


def get_simulation_logger(name: str = "") -> TypedSimulationLogger:
    """Get a typed simulation logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        TypedSimulationLogger instance
    """
    logger = structlog.get_logger(name)
    return TypedSimulationLogger(logger)


# Re-export for convenience
__all__ = [
    "SimulationLogger",
    "TypedSimulationLogger",
    "get_simulation_logger",
]
