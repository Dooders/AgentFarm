"""Unified logging package for AgentFarm.

This package provides structured logging capabilities using structlog with:
- Core logging configuration and utilities
- Optional advanced features (async, correlation IDs, testing)
- Specialized loggers for different components

Basic Usage:
    from farm.utils.logging import configure_logging, get_logger
    
    configure_logging(environment="development", log_dir="logs")
    logger = get_logger(__name__)
    logger.info("event_occurred", detail="some detail")

Advanced Usage:
    # With metrics tracking
    from farm.utils.logging import configure_logging
    configure_logging(
        environment="production",
        enable_metrics=True,
        enable_sampling=True
    )
    
    # Correlation IDs
    from farm.utils.logging.correlation import add_correlation_id
    corr_id = add_correlation_id()
    
    # Async logging
    from farm.utils.logging.async_logger import get_async_logger
    async_logger = get_async_logger(__name__)
    
    # Testing
    from farm.utils.logging.test_helpers import capture_logs
    with capture_logs() as logs:
        logger.info("test_event")
"""

# Core configuration and logger functions
from farm.utils.logging.config import (
    configure_logging,
    get_logger,
    bind_context,
    unbind_context,
    clear_context,
    # Custom context classes (optional)
    FastContext,
    ThreadSafeContext,
    MemoryEfficientContext,
    # Metrics (optional)
    get_metrics_summary,
    reset_metrics,
)

# Core utilities
from farm.utils.logging.utils import (
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

# Export main interfaces
__all__ = [
    # Configuration
    "configure_logging",
    "get_logger",
    "bind_context",
    "unbind_context",
    "clear_context",
    # Context classes
    "FastContext",
    "ThreadSafeContext",
    "MemoryEfficientContext",
    # Metrics
    "get_metrics_summary",
    "reset_metrics",
    # Decorators
    "log_performance",
    "log_errors",
    # Context managers
    "log_context",
    "log_step",
    "log_simulation",
    "log_experiment",
    # Specialized loggers
    "AgentLogger",
    "DatabaseLogger",
    "LogSampler",
    "PerformanceMonitor",
]

# Optional features are available via explicit imports:
# - farm.utils.logging.async_logger - Async logging wrappers
# - farm.utils.logging.correlation - Correlation ID tracking
# - farm.utils.logging.test_helpers - Testing utilities
# - farm.utils.logging.simulation - Typed simulation logger

