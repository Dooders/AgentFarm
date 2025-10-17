"""Centralized structured logging configuration for AgentFarm.

This module provides a unified structured logging setup using structlog that replaces
the standard logging module throughout the codebase. It offers:

- Structured, context-rich log entries
- Multiple output formats (console, JSON file, plain text)
- Automatic context binding (simulation_id, step_number, agent_id, etc.)
- Performance-optimized logging with optional sampling capabilities
- Environment-specific configurations (dev, production, testing)
- Integration with existing Python logging infrastructure
- Optional advanced features: metrics tracking, log rotation, custom context classes

Usage:
    # Basic setup
    from farm.utils.logging_config import configure_logging, get_logger
    configure_logging(environment="development", log_dir="logs")

    # With advanced features
    configure_logging(
        environment="production",
        enable_metrics=True,
        enable_sampling=True,
        sample_rate=0.1
    )

    # In your modules
    logger = get_logger(__name__)
    logger.info("simulation_started", simulation_id="sim_001", num_agents=100)

    # With bound context
    logger = logger.bind(simulation_id="sim_001", step=42)
    logger.info("agent_action", agent_id="agent_123", action="move")
"""

import logging
import os
import statistics
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

# =============================================================================
# CUSTOM CONTEXT CLASSES (OPTIONAL)
# =============================================================================


class FastContext(dict):
    """Optimized context dict for high-performance logging.

    This context class is optimized for frequent context operations
    in high-throughput logging scenarios. It uses __slots__ to reduce
    memory overhead and provides efficient child context creation.

    Features:
    - Memory efficient with __slots__
    - Fast child context creation
    - Optimized for frequent updates
    - Compatible with structlog context requirements
    """

    __slots__ = ()  # No __dict__, saves memory

    def __init__(self, *args, **kwargs):
        """Initialize fast context.

        Args:
            *args: Positional arguments for dict initialization
            **kwargs: Keyword arguments for dict initialization
        """
        super().__init__(*args, **kwargs)

    def new_child(self, child: Optional[Dict[str, Any]] = None) -> "FastContext":
        """Create child context efficiently.

        Args:
            child: Optional dict to merge into new context

        Returns:
            New FastContext with merged data
        """
        new = self.copy()
        if child:
            new.update(child)
        return new

    def copy(self) -> "FastContext":
        """Create a copy of this context.

        Returns:
            New FastContext with same data
        """
        return FastContext(self)


class ThreadSafeContext(dict):
    """Thread-safe context class for parallel simulations.

    This context class provides thread-safe operations for use
    in multi-threaded simulation scenarios.
    """

    __slots__ = ("_lock",)

    def __init__(self, *args, **kwargs):
        """Initialize thread-safe context."""
        super().__init__(*args, **kwargs)
        import threading

        self._lock = threading.RLock()

    def new_child(self, child: Optional[Dict[str, Any]] = None) -> "ThreadSafeContext":
        """Create child context with thread safety.

        Args:
            child: Optional dict to merge into new context

        Returns:
            New ThreadSafeContext with merged data
        """
        with self._lock:
            new = self.copy()
            if child:
                new.update(child)
            return new

    def copy(self) -> "ThreadSafeContext":
        """Create a thread-safe copy of this context.

        Returns:
            New ThreadSafeContext with same data
        """
        with self._lock:
            return ThreadSafeContext(self)

    def update(self, other: Dict[str, Any]) -> None:
        """Update context with thread safety.

        Args:
            other: Dict to merge into this context
        """
        with self._lock:
            super().update(other)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item with thread safety."""
        with self._lock:
            super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        """Get item with thread safety."""
        with self._lock:
            return super().__getitem__(key)

    def __delitem__(self, key: str) -> None:
        """Delete item with thread safety."""
        with self._lock:
            super().__delitem__(key)


class MemoryEfficientContext(dict):
    """Memory-efficient context class for large-scale simulations.

    This context class is optimized for memory usage in scenarios
    with many concurrent contexts or large context data.
    """

    __slots__ = ("_frozen",)

    def __init__(self, *args, **kwargs):
        """Initialize memory-efficient context."""
        super().__init__(*args, **kwargs)
        self._frozen = False

    def freeze(self) -> "MemoryEfficientContext":
        """Freeze the context to prevent further modifications.

        Returns:
            Self for method chaining
        """
        self._frozen = True
        return self

    def new_child(self, child: Optional[Dict[str, Any]] = None) -> "MemoryEfficientContext":
        """Create child context efficiently.

        Args:
            child: Optional dict to merge into new context

        Returns:
            New MemoryEfficientContext with merged data
        """
        new = self.copy()
        if child:
            new.update(child)
        return new

    def copy(self) -> "MemoryEfficientContext":
        """Create a copy of this context.

        Returns:
            New MemoryEfficientContext with same data
        """
        return MemoryEfficientContext(self)

    def update(self, other: Dict[str, Any]) -> None:
        """Update context if not frozen.

        Args:
            other: Dict to merge into this context

        Raises:
            RuntimeError: If context is frozen
        """
        if self._frozen:
            raise RuntimeError("Cannot update frozen context")
        super().update(other)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item if not frozen.

        Args:
            key: Key to set
            value: Value to set

        Raises:
            RuntimeError: If context is frozen
        """
        if self._frozen:
            raise RuntimeError("Cannot modify frozen context")
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete item if not frozen.

        Args:
            key: Key to delete

        Raises:
            RuntimeError: If context is frozen
        """
        if self._frozen:
            raise RuntimeError("Cannot modify frozen context")
        super().__delitem__(key)


# =============================================================================
# CORE PROCESSORS
# =============================================================================


def add_log_level(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict."""
    if method_name == "warn":
        event_dict["level"] = "warning"
    else:
        event_dict["level"] = method_name
    return event_dict


def add_log_level_number(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add numeric log level for filtering in log aggregation tools."""
    level_map = {
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40,
        "critical": 50,
    }
    level_name = event_dict.get("level", "info")
    event_dict["level_num"] = level_map.get(level_name, 20)
    return event_dict


def add_logger_name(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add logger name to event dict."""
    record = event_dict.get("_record")
    if record:
        event_dict["logger"] = record.name
    elif hasattr(logger, "name"):
        event_dict["logger"] = logger.name
    return event_dict


def add_process_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add process ID and thread ID to log entries."""
    import os
    import threading

    event_dict["process_id"] = os.getpid()
    event_dict["thread_id"] = threading.get_ident()
    event_dict["thread_name"] = threading.current_thread().name
    return event_dict


def censor_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Censor sensitive information from logs."""
    sensitive_keys = {"password", "token", "secret", "api_key", "auth", "key"}

    for key in list(event_dict.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            event_dict[key] = "***REDACTED***"

    return event_dict


# =============================================================================
# OPTIONAL PROCESSORS
# =============================================================================


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


class MetricsProcessor:
    """Processor to track metrics from logs for analysis."""

    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.start_time = time.time()
        self.event_counts: Dict[str, int] = {}

    def __call__(self, logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        """Track metrics and add runtime."""
        event_name = event_dict.get("event", event_dict.get("message", "unknown"))

        # Count events
        self.event_counts[event_name] = self.event_counts.get(event_name, 0) + 1

        # Track duration metrics
        if "duration_ms" in event_dict:
            if event_name not in self.metrics:
                self.metrics[event_name] = []
            self.metrics[event_name].append(event_dict["duration_ms"])

        # Add runtime to all logs
        event_dict["runtime_seconds"] = round(time.time() - self.start_time, 2)

        return event_dict

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            "event_counts": self.event_counts,
            "duration_metrics": {},
            "total_runtime": time.time() - self.start_time,
        }

        for event, durations in self.metrics.items():
            if durations:
                summary["duration_metrics"][event] = {
                    "count": len(durations),
                    "mean": round(statistics.mean(durations), 2),
                    "median": round(statistics.median(durations), 2),
                    "max": round(max(durations), 2),
                    "min": round(min(durations), 2),
                }

        return summary

    def reset(self):
        """Reset metrics."""
        self.metrics.clear()
        self.event_counts.clear()
        self.start_time = time.time()


class SamplingProcessor:
    """Processor to sample high-frequency logs."""

    # Configurable minimum sample rate to avoid enormous sampling intervals
    # Increased from 0.001 to 0.01 to reduce sampling interval from 1000 to 100
    MIN_SAMPLE_RATE = 0.01

    def __init__(
        self,
        sample_rate: float = 1.0,
        events_to_sample: Optional[set] = None,
        min_sample_rate: Optional[float] = None,
    ):
        """Initialize sampling processor.

        Args:
            sample_rate: Fraction of events to log (minimum 0.01, maximum 1.0)
            events_to_sample: Set of event names to sample (None or empty = sample all)
            min_sample_rate: Override minimum sample rate (defaults to 0.01)

        Raises:
            ValueError: If sample_rate is not within valid range or is 0.0
        """
        # Use provided minimum or class default
        min_rate = min_sample_rate if min_sample_rate is not None else self.MIN_SAMPLE_RATE

        # Explicit check for zero to provide clear error message
        if sample_rate == 0.0:
            raise ValueError(
                f"sample_rate cannot be 0.0 as it would cause division by zero. Use a value between {min_rate} and 1.0"
            )

        # Enforce practical minimum to avoid enormous sampling intervals
        if not min_rate <= sample_rate <= 1.0:
            raise ValueError(
                f"sample_rate must be between {min_rate} and 1.0, got {sample_rate}. "
                "Note: sample_rate cannot be 0.0 as it would cause division by zero"
            )

        self.sample_rate = float(sample_rate)
        # None or empty set means "sample all events"
        self.events_to_sample = set(events_to_sample) if events_to_sample else set()
        self.counter: Dict[str, int] = {}
        # Pre-compute interval once for performance; 1 means log all
        # Safe division since we've validated sample_rate > 0
        self._sample_interval: int = 1 if self.sample_rate >= 1.0 else max(1, int(round(1.0 / self.sample_rate)))

    def _should_sample(self, event: str) -> bool:
        # If events_to_sample is empty, sample all events; otherwise only those included
        return not self.events_to_sample or event in self.events_to_sample

    def __call__(self, logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        event = event_dict.get("event", event_dict.get("message", ""))

        if not event:
            return event_dict

        if self._should_sample(event):
            count = self.counter.get(event, 0) + 1
            self.counter[event] = count

            # Keep first event; then log every Nth event based on interval
            if self._sample_interval > 1 and (count - 1) % self._sample_interval != 0:
                raise structlog.DropEvent

            event_dict["sampled"] = self._sample_interval > 1
            event_dict["sample_rate"] = self.sample_rate

        return event_dict


# =============================================================================
# METRICS MANAGER
# =============================================================================


class MetricsManager:
    """Manages global metrics processor instance."""

    def __init__(self):
        self._processor: Optional[MetricsProcessor] = None

    def set_processor(self, processor: MetricsProcessor) -> None:
        """Set the metrics processor."""
        self._processor = processor

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        if self._processor is None:
            return {"error": "Metrics not enabled"}
        return self._processor.get_summary()

    def reset(self) -> None:
        """Reset metrics counters."""
        if self._processor is not None:
            self._processor.reset()


_metrics_manager = MetricsManager()


def get_metrics_summary() -> Dict[str, Any]:
    """Get summary of logged metrics.

    Returns:
        Dictionary with event counts and duration metrics
    """
    return _metrics_manager.get_summary()


def reset_metrics():
    """Reset metrics counters."""
    _metrics_manager.reset()


# =============================================================================
# UNIFIED CONFIGURATION
# =============================================================================


def configure_logging(
    environment: str = "development",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    json_logs: bool = False,
    enable_colors: bool = True,
    # Core options
    include_caller_info: bool = False,
    include_process_info: bool = False,
    # Optional advanced features
    enable_metrics: bool = False,
    enable_sampling: bool = False,
    sample_rate: float = 0.1,
    events_to_sample: Optional[set] = None,
    # Context class selection
    use_threadlocal: bool = False,
    context_class: Optional[type] = None,
    # File rotation
    enable_log_rotation: bool = True,
    max_log_size_mb: int = 100,
    backup_count: int = 5,
    # Performance
    slow_threshold_ms: float = 100.0,
    # Console output control
    disable_console: bool = False,
) -> None:
    """Configure unified structured logging with core and optional features.

    Args:
        environment: Environment name (development, production, testing)
        log_dir: Directory for log files (None = console only)
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output JSON formatted logs to file
        enable_colors: Enable colored console output (development only)
        include_caller_info: Include file/line/function info (expensive)
        include_process_info: Include process/thread IDs
        enable_metrics: Track metrics from logs
        enable_sampling: Enable log sampling for high-frequency events
        sample_rate: Sampling rate (0.001 to 1.0)
        events_to_sample: Set of event names to sample (None = all)
        use_threadlocal: Use thread-local context (for parallel experiments)
        context_class: Custom context class (overrides use_threadlocal)
        enable_log_rotation: Enable log rotation
        max_log_size_mb: Max log file size before rotation
        backup_count: Number of backup log files to keep
        slow_threshold_ms: Threshold for slow operation warnings
        disable_console: Disable console output (useful when using tqdm progress bars)
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

    # Configure standard library logging
    if disable_console:
        # When console is disabled, use a null stream to suppress output
        logging.basicConfig(
            format="%(message)s",
            level=numeric_level,
            stream=open(os.devnull, "w"),
            force=True,
        )
    else:
        logging.basicConfig(
            format="%(message)s",
            level=numeric_level,
            stream=sys.stdout,
            force=True,
        )

    # Create metrics processor if enabled
    if enable_metrics:
        _metrics_manager.set_processor(MetricsProcessor())

    # Build optimized processor chain
    processors: list[Processor] = [
        # PHASE 1: Context merging (cheap)
        (structlog.threadlocal.merge_threadlocal if use_threadlocal else structlog.contextvars.merge_contextvars),
        # PHASE 2: Early metadata (cheap)
        add_log_level,
        add_log_level_number,
        # PHASE 3: Filter disabled levels ASAP (performance optimization)
        structlog.stdlib.filter_by_level,
        # PHASE 4: Sampling (if enabled)
        *([SamplingProcessor(sample_rate, events_to_sample)] if enable_sampling else []),
        # PHASE 5: Unicode safety
        structlog.processors.UnicodeDecoder(),
        # PHASE 6: Add metadata
        add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        # PHASE 7: Stack and exception info
        structlog.processors.StackInfoRenderer(),
        structlog.processors.dict_tracebacks,
        # PHASE 8: Security
        censor_sensitive_data,
        # PHASE 9: Custom processors
        *([_metrics_manager._processor] if _metrics_manager._processor else []),
        PerformanceLogger(slow_threshold_ms=slow_threshold_ms),
        # PHASE 10: Process info (if enabled)
        *([add_process_info] if include_process_info else []),
        # PHASE 11: Callsite info (expensive - only if requested)
        *(
            [
                structlog.processors.CallsiteParameterAdder(
                    {
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    }
                )
            ]
            if include_caller_info
            else []
        ),
    ]

    # Add final renderer based on environment
    if disable_console:
        # When console is disabled, use a null renderer that doesn't output anything
        def null_renderer(logger, method_name, event_dict):
            return ""  # Return empty string to suppress output

        processors.append(null_renderer)
    elif environment == "production" or json_logs:
        processors.append(structlog.processors.JSONRenderer())
    elif environment == "testing":
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=enable_colors,
                exception_formatter=(structlog.dev.rich_traceback if enable_colors else structlog.dev.plain_traceback),
            )
        )

    # Determine context class
    if context_class is not None:
        final_context_class = context_class
    elif use_threadlocal:
        final_context_class = structlog.threadlocal.wrap_dict(dict)
    else:
        final_context_class = dict

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=final_context_class,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Setup file logging with rotation
    if log_dir:
        _setup_rotating_file_handlers(
            log_dir=log_dir,
            log_level=numeric_level,
            json_logs=json_logs or environment == "production",
            enable_rotation=enable_log_rotation,
            max_bytes=max_log_size_mb * 1024 * 1024,
            backup_count=backup_count,
        )

    # Configure third-party loggers
    _configure_third_party_loggers(numeric_level)

    # Log configuration complete
    logger = structlog.get_logger(__name__)
    logger.info(
        "logging_configured",
        environment=environment,
        log_level=log_level,
        log_dir=str(log_dir) if log_dir else None,
        features={
            "json_logs": json_logs,
            "metrics": enable_metrics,
            "sampling": enable_sampling,
            "rotation": enable_log_rotation,
            "disable_console": disable_console,
            "threadlocal": use_threadlocal,
            "custom_context": context_class is not None,
        },
    )


def _setup_rotating_file_handlers(
    log_dir: str,
    log_level: int,
    json_logs: bool = False,
    enable_rotation: bool = True,
    max_bytes: int = 100 * 1024 * 1024,  # 100 MB
    backup_count: int = 5,
) -> None:
    """Setup rotating file handlers for logs.

    Args:
        log_dir: Directory for log files
        log_level: Numeric log level
        json_logs: Whether to create JSON log files
        enable_rotation: Enable log rotation
        max_bytes: Max file size before rotation
        backup_count: Number of backup files to keep
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # JSON log handler
    if json_logs:
        if enable_rotation:
            json_handler = RotatingFileHandler(
                log_path / "application.json.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
        else:
            json_handler = logging.FileHandler(log_path / "application.json.log")

        json_handler.setLevel(log_level)
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                add_log_level,
                structlog.processors.StackInfoRenderer(),
            ],
        )
        json_handler.setFormatter(json_formatter)
        logging.root.addHandler(json_handler)

    # Plain text log handler
    if enable_rotation:
        text_handler = RotatingFileHandler(
            log_path / "application.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
    else:
        text_handler = logging.FileHandler(log_path / "application.log")

    text_handler.setLevel(log_level)
    text_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
        foreign_pre_chain=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            add_log_level,
            structlog.processors.StackInfoRenderer(),
        ],
    )
    text_handler.setFormatter(text_formatter)
    logging.root.addHandler(text_handler)


def _configure_third_party_loggers(level: int) -> None:
    """Configure third-party library loggers to reduce noise."""
    noisy_loggers = [
        "urllib3",
        "werkzeug",
        "socketio",
        "engineio",
        "matplotlib",
        "PIL",
        "asyncio",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set SQLAlchemy to WARNING
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Structured logger instance with context binding capabilities
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables that will be included in all subsequent logs.

    This uses structlog's contextvars to bind context that persists across
    the entire execution context (including async operations).

    Args:
        **kwargs: Key-value pairs to bind to logging context
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Unbind previously bound context variables.

    Args:
        *keys: Keys to unbind from logging context
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "configure_logging",
    "get_logger",
    "bind_context",
    "unbind_context",
    "clear_context",
    # Custom context classes (optional)
    "FastContext",
    "ThreadSafeContext",
    "MemoryEfficientContext",
    # Metrics (if enabled)
    "get_metrics_summary",
    "reset_metrics",
]
