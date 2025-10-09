"""
Configuration monitoring and observability.

This module provides comprehensive logging, metrics, and monitoring
capabilities for the configuration system.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import SimulationConfig
from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigMetrics:
    """Configuration operation metrics."""

    operation: str
    duration: float
    success: bool
    cache_hit: bool = False
    config_size: int = 0
    error_type: Optional[str] = None
    environment: Optional[str] = None
    profile: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ConfigMonitor:
    """
    Configuration system monitor with metrics and logging.

    Provides comprehensive observability into configuration operations,
    performance metrics, and error tracking.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration monitor.

        Args:
            logger: Logger instance to use (creates default if None)
        """
        from farm.utils.logging import get_logger
        self.logger = logger or get_logger("config_monitor")
        self.metrics: List[ConfigMetrics] = []
        self.max_metrics_history = 1000

    def log_config_operation(
        self,
        operation: str,
        config: Optional[SimulationConfig] = None,
        environment: Optional[str] = None,
        profile: Optional[str] = None,
        duration: Optional[float] = None,
        success: bool = True,
        cache_hit: bool = False,
        error: Optional[Exception] = None,
        **kwargs,
    ) -> None:
        """
        Log a configuration operation.

        Args:
            operation: Operation name (load, save, version, etc.)
            config: Configuration involved in operation
            environment: Environment name
            profile: Profile name
            duration: Operation duration in seconds
            success: Whether operation succeeded
            cache_hit: Whether this was a cache hit
            error: Exception if operation failed
            **kwargs: Additional metadata
        """
        # Create metrics record
        metrics = ConfigMetrics(
            operation=operation,
            duration=duration or 0.0,
            success=success,
            cache_hit=cache_hit,
            config_size=len(config.to_dict()) if config else 0,
            error_type=type(error).__name__ if error else None,
            environment=environment,
            profile=profile,
        )

        # Store metrics
        self.metrics.append(metrics)
        if len(self.metrics) > self.max_metrics_history:
            self.metrics.pop(0)

        # Log based on operation type and success
        if success:
            if cache_hit:
                self.logger.debug(
                    f"Config {operation} cache hit: env={environment}, profile={profile}, "
                    f"duration={duration:.3f}s"
                )
            else:
                self.logger.info(
                    "config_operation_success",
                    operation=operation,
                    environment=environment,
                    profile=profile,
                    duration_seconds=round(duration, 3),
                )
        else:
            self.logger.error(
                "config_operation_failed",
                operation=operation,
                environment=environment,
                profile=profile,
                error_type=type(error).__name__ if error else "Unknown",
                duration_seconds=round(duration, 3),
            )
            if error:
                self.logger.debug(
                    f"Config {operation} error details: {error}", exc_info=True
                )

    @contextmanager
    def measure_operation(
        self,
        operation: str,
        config: Optional[SimulationConfig] = None,
        environment: Optional[str] = None,
        profile: Optional[str] = None,
        **kwargs,
    ):
        """
        Context manager to measure and log configuration operations.

        Args:
            operation: Operation name
            config: Configuration involved
            environment: Environment name
            profile: Profile name
            **kwargs: Additional metadata

        Yields:
            Dict to store operation results
        """
        start_time = time.time()
        result = {"success": True, "cache_hit": False, "error": None}

        try:
            yield result
        except Exception as e:
            result["success"] = False
            result["error"] = e
            raise
        finally:
            duration = time.time() - start_time
            self.log_config_operation(
                operation=operation,
                config=config,
                environment=environment,
                profile=profile,
                duration=duration,
                **result,
                **kwargs,
            )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of configuration operations.

        Returns:
            Dictionary with metrics summary
        """
        if not self.metrics:
            return {"total_operations": 0}

        total_ops = len(self.metrics)
        successful_ops = sum(1 for m in self.metrics if m.success)
        cache_hits = sum(1 for m in self.metrics if m.cache_hit)
        avg_duration = sum(m.duration for m in self.metrics) / total_ops

        # Group by operation type
        operations = {}
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)

        operation_stats = {}
        for op_name, op_metrics in operations.items():
            op_success = sum(1 for m in op_metrics if m.success)
            op_avg_duration = sum(m.duration for m in op_metrics) / len(op_metrics)
            operation_stats[op_name] = {
                "count": len(op_metrics),
                "success_rate": op_success / len(op_metrics),
                "avg_duration": op_avg_duration,
            }

        return {
            "total_operations": total_ops,
            "success_rate": successful_ops / total_ops,
            "cache_hit_rate": cache_hits / total_ops if total_ops > 0 else 0,
            "avg_duration": avg_duration,
            "operation_stats": operation_stats,
        }

    def get_recent_errors(self, limit: int = 10) -> List[ConfigMetrics]:
        """
        Get recent configuration errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error metrics
        """
        errors = [m for m in self.metrics if not m.success]
        return sorted(errors[-limit:], key=lambda x: x.timestamp)

    def get_performance_trends(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance trends for operations.

        Args:
            operation: Specific operation to analyze (None for all)

        Returns:
            Dictionary with performance trend data
        """
        relevant_metrics = [
            m
            for m in self.metrics
            if (operation is None or m.operation == operation) and m.success
        ]

        if len(relevant_metrics) < 2:
            return {"insufficient_data": True}

        # Sort by timestamp
        relevant_metrics.sort(key=lambda x: x.timestamp)

        # Calculate moving averages
        window_size = min(10, len(relevant_metrics) // 2)
        durations = [m.duration for m in relevant_metrics]

        moving_avg = []
        for i in range(window_size, len(durations) + 1):
            window = durations[i - window_size : i]
            moving_avg.append(sum(window) / len(window))

        return {
            "operation": operation or "all",
            "total_samples": len(relevant_metrics),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "moving_avg_window": window_size,
            "moving_avg_trend": (
                moving_avg[-10:] if len(moving_avg) > 10 else moving_avg
            ),
        }

    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file.

        Args:
            filepath: Path to export metrics
        """
        import json

        metrics_data = [vars(m) for m in self.metrics]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"exported_at": time.time(), "metrics": metrics_data},
                f,
                indent=2,
                default=str,
            )


# Global monitor instance
_global_monitor = ConfigMonitor()


def get_global_monitor() -> ConfigMonitor:
    """Get the global configuration monitor instance."""
    return _global_monitor


def monitor_config_operation(operation: str, **kwargs):
    """
    Decorator to monitor configuration operations.

    Args:
        operation: Operation name
        **kwargs: Additional metadata to log
    """

    def decorator(func):
        def wrapper(*args, **func_kwargs):
            monitor = get_global_monitor()

            # Extract relevant info from arguments
            config = None
            environment = None
            profile = None

            # Try to extract config-related arguments
            if "config" in func_kwargs:
                config = func_kwargs["config"]
            if "environment" in func_kwargs:
                environment = func_kwargs["environment"]
            if "profile" in func_kwargs:
                profile = func_kwargs["profile"]

            with monitor.measure_operation(
                operation=operation,
                config=config,
                environment=environment,
                profile=profile,
                **kwargs,
            ) as result:
                try:
                    return func(*args, **func_kwargs)
                except Exception as e:
                    result["error"] = e
                    raise

        return wrapper

    return decorator


# Apply monitoring to key configuration operations
@monitor_config_operation("load_centralized")
def monitored_load_centralized_config(*args, **kwargs):
    """Monitored version of centralized config loading."""
    return SimulationConfig.from_centralized_config(*args, **kwargs)


@monitor_config_operation("version_config")
def monitored_version_config(*args, **kwargs):
    """Monitored version of config versioning."""
    return SimulationConfig.version_config(*args, **kwargs)


@monitor_config_operation("instantiate_template")
def monitored_template_instantiate(*args, **kwargs):
    """Monitored version of template instantiation."""
    from .template import ConfigTemplate

    return ConfigTemplate.instantiate(*args, **kwargs)


# Health check functions
def get_config_system_health() -> Dict[str, Any]:
    """
    Get overall health status of the configuration system.

    Returns:
        Dictionary with health metrics
    """
    monitor = get_global_monitor()
    metrics = monitor.get_metrics_summary()

    # Determine health status
    success_rate = metrics.get("success_rate", 0)
    avg_duration = metrics.get("avg_duration", float("inf"))

    if success_rate >= 0.99 and avg_duration < 1.0:
        status = "healthy"
    elif success_rate >= 0.95:
        status = "warning"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "success_rate": success_rate,
        "avg_operation_time": avg_duration,
        "total_operations": metrics.get("total_operations", 0),
        "cache_hit_rate": metrics.get("cache_hit_rate", 0),
        "recent_errors": len(monitor.get_recent_errors(5)),
    }


def log_config_system_status() -> None:
    """Log the current status of the configuration system."""
    monitor = get_global_monitor()
    health = get_config_system_health()

    monitor.logger.info(
        f"Config system health: {health['status']} | "
        f"Success rate: {health['success_rate']:.1%} | "
        f"Avg time: {health['avg_operation_time']:.3f}s | "
        f"Total ops: {health['total_operations']} | "
        f"Cache hits: {health['cache_hit_rate']:.1%} | "
        f"Recent errors: {health['recent_errors']}"
    )
