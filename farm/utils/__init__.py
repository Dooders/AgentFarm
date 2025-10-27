"""Utilities package for AgentFarm.

This package provides common utilities including:
- Structured logging configuration and helpers
- Configuration utilities
- Identity management
- Spatial and mathematical utilities
- And more
"""

from farm.utils.logging import (
    # Configuration
    configure_logging,
    get_logger,
    bind_context,
    unbind_context,
    clear_context,
    # Decorators
    log_performance,
    log_errors,
    # Context managers
    log_context,
    log_step,
    log_simulation,
    log_experiment,
    # Specialized loggers
    AgentLogger,
    DatabaseLogger,
    LogSampler,
    PerformanceMonitor,
)

from farm.utils.spatial import bilinear_distribute_value

__all__ = [
    # Logging configuration
    "configure_logging",
    "get_logger",
    "bind_context",
    "unbind_context",
    "clear_context",
    # Logging utilities
    "log_performance",
    "log_errors",
    "log_context",
    "log_step",
    "log_simulation",
    "log_experiment",
    "LogSampler",
    "AgentLogger",
    "DatabaseLogger",
    "PerformanceMonitor",
    # Spatial utilities
    "bilinear_distribute_value",
]
