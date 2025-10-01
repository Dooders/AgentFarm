"""Centralized structured logging configuration for AgentFarm.

This module provides a structured logging setup using structlog that replaces
the standard logging module throughout the codebase. It offers:

- Structured, context-rich log entries
- Multiple output formats (console, JSON file, plain text)
- Automatic context binding (simulation_id, step_number, agent_id, etc.)
- Performance-optimized logging with sampling capabilities
- Environment-specific configurations (dev, production, testing)
- Integration with existing Python logging infrastructure

Usage:
    # At application startup
    from farm.utils.logging_config import configure_logging
    configure_logging(environment="development", log_dir="logs")
    
    # In your modules
    from farm.utils.logging_config import get_logger
    logger = get_logger(__name__)
    
    # Basic logging
    logger.info("simulation_started", simulation_id="sim_001", num_agents=100)
    
    # With bound context
    logger = logger.bind(simulation_id="sim_001", step=42)
    logger.info("agent_action", agent_id="agent_123", action="move")
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.types import EventDict, Processor


def add_timestamp(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to log entries."""
    from datetime import datetime
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_log_level(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict if not present."""
    if method_name == "warn":
        # Normalize warn to warning
        event_dict["level"] = "warning"
    else:
        event_dict["level"] = method_name
    return event_dict


def add_logger_name(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add logger name to event dict."""
    record = event_dict.get("_record")
    if record:
        event_dict["logger"] = record.name
    return event_dict


def censor_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Censor sensitive information from logs."""
    sensitive_keys = {"password", "token", "secret", "api_key", "auth"}
    
    for key in list(event_dict.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            event_dict[key] = "***REDACTED***"
    
    return event_dict


def extract_exception_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Extract exception information if present."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        event_dict["exception"] = structlog.processors.format_exc_info(logger, method_name, {"exc_info": exc_info})
    return event_dict


class PerformanceLogger:
    """Processor to track and log performance metrics."""
    
    def __init__(self, slow_threshold_ms: float = 100.0):
        self.slow_threshold_ms = slow_threshold_ms
    
    def __call__(self, logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        """Add performance warning for slow operations."""
        duration_ms = event_dict.get("duration_ms")
        if duration_ms and duration_ms > self.slow_threshold_ms:
            event_dict["performance_warning"] = "slow_operation"
        return event_dict


def configure_logging(
    environment: str = "development",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    json_logs: bool = False,
    enable_colors: bool = True,
    include_caller_info: bool = True,
) -> None:
    """Configure structured logging for the application.
    
    Args:
        environment: Environment name (development, production, testing)
        log_dir: Directory for log files. If None, only console output
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output JSON formatted logs to file
        enable_colors: Enable colored console output (development only)
        include_caller_info: Include file/line/function info in logs
    """
    # Determine log level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    numeric_level = level_map.get(log_level.upper(), logging.INFO)
    
    # Configure standard library logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
        force=True,
    )
    
    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,  # Merge context variables
        add_log_level,  # Add log level
        add_logger_name,  # Add logger name
        add_timestamp,  # Add timestamp
        structlog.processors.StackInfoRenderer(),  # Render stack info
        extract_exception_info,  # Extract exception info
        censor_sensitive_data,  # Censor sensitive data
        PerformanceLogger(slow_threshold_ms=100.0),  # Performance warnings
    ]
    
    # Add caller info if requested
    if include_caller_info:
        processors.append(structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ))
    
    # Configure output format based on environment
    if environment == "production" or json_logs:
        # JSON format for production/analysis
        processors.append(structlog.processors.JSONRenderer())
        console_renderer = structlog.processors.JSONRenderer()
    elif environment == "testing":
        # Simpler format for testing
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
        console_renderer = structlog.dev.ConsoleRenderer(colors=False)
    else:
        # Development: pretty console output with colors
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=enable_colors,
                exception_formatter=structlog.dev.rich_traceback if enable_colors else structlog.dev.plain_traceback,
            )
        )
        console_renderer = structlog.dev.ConsoleRenderer(colors=enable_colors)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup file logging if log_dir is specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Setup file handler for JSON logs
        if json_logs or environment == "production":
            json_handler = logging.FileHandler(log_path / "application.json.log")
            json_handler.setLevel(numeric_level)
            json_formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=[
                    add_timestamp,
                    add_log_level,
                    structlog.processors.StackInfoRenderer(),
                ],
            )
            json_handler.setFormatter(json_formatter)
            logging.root.addHandler(json_handler)
        
        # Always create a plain text log file
        text_handler = logging.FileHandler(log_path / "application.log")
        text_handler.setLevel(numeric_level)
        text_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False),
            foreign_pre_chain=[
                add_timestamp,
                add_log_level,
                structlog.processors.StackInfoRenderer(),
            ],
        )
        text_handler.setFormatter(text_formatter)
        logging.root.addHandler(text_handler)
    
    # Configure specific loggers
    _configure_third_party_loggers(numeric_level)
    
    # Log configuration complete
    logger = get_logger(__name__)
    logger.info(
        "logging_configured",
        environment=environment,
        log_level=log_level,
        log_dir=str(log_dir) if log_dir else None,
        json_logs=json_logs,
    )


def _configure_third_party_loggers(level: int) -> None:
    """Configure third-party library loggers to reduce noise."""
    # Reduce noise from common libraries
    noisy_loggers = [
        "urllib3",
        "werkzeug",
        "socketio",
        "engineio",
        "matplotlib",
        "PIL",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Set SQLAlchemy to INFO (shows queries in debug mode)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        Structured logger instance with context binding capabilities
        
    Example:
        logger = get_logger(__name__)
        logger.info("event_occurred", detail="some detail")
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables that will be included in all subsequent logs.
    
    This uses structlog's contextvars to bind context that persists across
    the entire execution context (including async operations).
    
    Args:
        **kwargs: Key-value pairs to bind to logging context
        
    Example:
        bind_context(simulation_id="sim_001", experiment_id="exp_42")
        # All subsequent logs will include these fields
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Unbind previously bound context variables.
    
    Args:
        *keys: Keys to unbind from logging context
        
    Example:
        unbind_context("simulation_id", "step_number")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


# Re-export structlog for convenience
__all__ = [
    "configure_logging",
    "get_logger",
    "bind_context",
    "unbind_context",
    "clear_context",
]
