"""Enhanced logging configuration with advanced structlog features.

This module provides an upgraded logging configuration that uses advanced
structlog features for better performance, testing, and observability.

Usage:
    # Replace this:
    from farm.utils.logging_config import configure_logging
    
    # With this:
    from farm.utils.logging_config_enhanced import configure_logging_enhanced as configure_logging
    
    # Or use directly:
    from farm.utils.logging_config_enhanced import configure_logging_enhanced
    configure_logging_enhanced(environment="production", enable_metrics=True)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional
import time
import statistics

import structlog
from structlog.types import EventDict, Processor


# =============================================================================
# ENHANCED PROCESSORS
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
    elif hasattr(logger, 'name'):
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
    
    def __init__(self, sample_rate: float = 1.0, events_to_sample: Optional[set] = None):
        """Initialize sampling processor.
        
        Args:
            sample_rate: Fraction of events to log (0.0 to 1.0)
            events_to_sample: Set of event names to sample (None = sample all)
        
        Raises:
            ValueError: If sample_rate is not between 0 and 1
        """
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be between 0.0 (exclusive) and 1.0, got {sample_rate}")
        self.sample_rate = sample_rate
        self.events_to_sample = events_to_sample or set()
        self.counter = {}
    
    def __call__(self, logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        event = event_dict.get("event", event_dict.get("message", ""))
        
        # Sample specific events
        if self.events_to_sample and event in self.events_to_sample:
            if event not in self.counter:
                self.counter[event] = 0
            
            self.counter[event] += 1
            
            # Drop events based on sample rate
            if self.sample_rate < 1.0:
                # Safe division - sample_rate is validated to be > 0 in __init__
                sample_interval = int(1.0 / self.sample_rate)
                if self.counter[event] % sample_interval != 0:
                    raise structlog.DropEvent
            
            # Add sampling info
            event_dict["sampled"] = True
            event_dict["sample_rate"] = self.sample_rate
        
        return event_dict


# =============================================================================
# GLOBAL METRICS INSTANCE (for retrieval)
# =============================================================================

_global_metrics_processor: Optional[MetricsProcessor] = None


def get_metrics_summary() -> Dict[str, Any]:
    """Get summary of logged metrics.
    
    Returns:
        Dictionary with event counts and duration metrics
    """
    if _global_metrics_processor is None:
        return {"error": "Metrics not enabled"}
    return _global_metrics_processor.get_summary()


def reset_metrics():
    """Reset metrics counters."""
    if _global_metrics_processor is not None:
        _global_metrics_processor.reset()


# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================

def configure_logging_enhanced(
    environment: str = "development",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    json_logs: bool = False,
    enable_colors: bool = True,
    include_caller_info: bool = False,  # Expensive, default False
    include_process_info: bool = False,
    enable_metrics: bool = False,
    use_threadlocal: bool = False,  # For parallel simulations
    enable_sampling: bool = False,
    sample_rate: float = 0.1,
    events_to_sample: Optional[set] = None,
    enable_log_rotation: bool = True,
    max_log_size_mb: int = 100,
    backup_count: int = 5,
) -> None:
    """Configure enhanced structured logging with advanced features.
    
    Args:
        environment: Environment name (development, production, testing)
        log_dir: Directory for log files (None = console only)
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output JSON formatted logs to file
        enable_colors: Enable colored console output (development only)
        include_caller_info: Include file/line/function info (expensive)
        include_process_info: Include process/thread IDs
        enable_metrics: Track metrics from logs
        use_threadlocal: Use thread-local context (for parallel experiments)
        enable_sampling: Enable log sampling for high-frequency events
        sample_rate: Sampling rate (0.0 to 1.0)
        events_to_sample: Set of event names to sample
        enable_log_rotation: Enable log rotation
        max_log_size_mb: Max log file size before rotation
        backup_count: Number of backup log files to keep
    """
    global _global_metrics_processor
    
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
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
        force=True,
    )
    
    # Create metrics processor if enabled
    if enable_metrics:
        _global_metrics_processor = MetricsProcessor()
    
    # Build optimized processor chain
    processors: list[Processor] = [
        # PHASE 1: Context merging (cheap)
        structlog.threadlocal.merge_threadlocal if use_threadlocal
        else structlog.contextvars.merge_contextvars,
        
        # PHASE 2: Early metadata (cheap)
        add_log_level,
        add_log_level_number,
        
        # PHASE 3: Filter disabled levels ASAP (performance optimization)
        structlog.stdlib.filter_by_level,
        
        # PHASE 4: Sampling (if enabled)
        *(
            [SamplingProcessor(sample_rate, events_to_sample)]
            if enable_sampling
            else []
        ),
        
        # PHASE 5: Unicode safety
        structlog.processors.UnicodeDecoder(),
        
        # PHASE 6: Add metadata
        add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        
        # PHASE 7: Stack and exception info
        structlog.processors.StackInfoRenderer(),
        structlog.processors.dict_tracebacks,  # Better than format_exc_info
        
        # PHASE 8: Security
        censor_sensitive_data,
        
        # PHASE 9: Custom processors
        *([_global_metrics_processor] if _global_metrics_processor else []),
        PerformanceLogger(slow_threshold_ms=100.0),
        
        # PHASE 10: Process info (if enabled)
        *(
            [add_process_info]  # Add process ID, thread ID, thread name
            if include_process_info
            else []
        ),
        
        # PHASE 11: Callsite info (expensive - only in dev)
        *(
            [structlog.processors.CallsiteParameterAdder({
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            })]
            if include_caller_info
            else []
        ),
        
        # PHASE 12: Event renaming (optional)
        # structlog.processors.EventRenamer(to="message"),  # Uncomment if desired
        
        # PHASE 13: Unicode output
        structlog.processors.UnicodeEncoder(encoding="utf-8", errors="replace"),
    ]
    
    # Add final renderer based on environment
    if environment == "production" or json_logs:
        processors.append(structlog.processors.JSONRenderer())
    elif environment == "testing":
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=enable_colors,
                exception_formatter=(
                    structlog.dev.rich_traceback if enable_colors
                    else structlog.dev.plain_traceback
                ),
            )
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=(
            structlog.threadlocal.wrap_dict(dict) if use_threadlocal
            else dict
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup file logging with rotation
    if log_dir:
        setup_rotating_file_handlers(
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
            "threadlocal": use_threadlocal,
            "sampling": enable_sampling,
            "rotation": enable_log_rotation,
        },
    )


def setup_rotating_file_handlers(
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
    """Bind context variables that will be included in all subsequent logs."""
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Unbind previously bound context variables."""
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


# Re-export for convenience
__all__ = [
    "configure_logging_enhanced",
    "get_logger",
    "bind_context",
    "unbind_context",
    "clear_context",
    "get_metrics_summary",
    "reset_metrics",
]
