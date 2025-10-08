"""Testing utilities for structlog in AgentFarm.

This module provides testing helpers for verifying logging behavior
in unit tests and integration tests.

Usage:
    from farm.utils.logging_test_helpers import capture_logs, assert_log_contains

    def test_simulation_logging():
        with capture_logs() as logs:
            run_simulation()

            # Assert expected logs
            assert_log_contains(logs, "simulation_started")
            assert_log_contains(logs, "simulation_completed")
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import structlog
from structlog.testing import LogCapture

# Import custom processors from config
from farm.utils.logging.config import (
    add_log_level_number,
    add_logger_name,
    censor_sensitive_data,
)


@contextmanager
def capture_logs():
    """Context manager to capture logs during tests.

    Yields:
        LogCapture: Object containing captured log entries

    Example:
        with capture_logs() as logs:
            logger.info("test_event", value=42)
            assert len(logs.entries) == 1
            assert logs.entries[0]["event"] == "test_event"
    """
    cap = LogCapture()

    # Store original configuration
    old_config = structlog.get_config()

    # Store original logging configuration
    old_handlers = logging.root.handlers[:]
    old_level = logging.root.level

    try:
        # Clear existing handlers to avoid double logging
        logging.root.handlers.clear()
        logging.root.setLevel(logging.DEBUG)

        # Configure structlog with LogCapture as the final processor
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                add_log_level_number,
                structlog.stdlib.filter_by_level,
                structlog.processors.UnicodeDecoder(),
                add_logger_name,
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.dict_tracebacks,
                censor_sensitive_data,
                cap,  # This captures the logs
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,  # Don't cache to ensure fresh loggers
        )

        yield cap
    finally:
        # Restore original structlog configuration
        structlog.configure(**old_config)

        # Restore original logging configuration
        logging.root.handlers.clear()
        logging.root.handlers.extend(old_handlers)
        logging.root.setLevel(old_level)


def assert_log_contains(
    logs: LogCapture, event: str, level: Optional[str] = None, **kwargs
) -> None:
    """Assert that logs contain a specific event with optional fields.

    Args:
        logs: LogCapture instance from capture_logs()
        event: Event name to search for
        level: Optional log level to match
        **kwargs: Additional fields that must be present in the log entry

    Raises:
        AssertionError: If the expected log entry is not found

    Example:
        assert_log_contains(logs, "agent_action", level="info", agent_id="agent_001")
    """
    found = False
    for entry in logs.entries:
        if entry.get("event") == event:
            # Check level if specified
            if level is not None and entry.get("level") != level:
                continue

            # Check additional fields
            if all(entry.get(key) == value for key, value in kwargs.items()):
                found = True
                break

    if not found:
        # Create helpful error message
        expected = {"event": event}
        if level:
            expected["level"] = level
        expected.update(kwargs)

        available_events = [entry.get("event") for entry in logs.entries]
        raise AssertionError(
            f"Expected log entry not found: {expected}\n"
            f"Available events: {available_events}\n"
            f"All entries: {logs.entries}"
        )


def assert_log_count(logs: LogCapture, event: str, expected_count: int) -> None:
    """Assert that a specific event appears exactly N times.

    Args:
        logs: LogCapture instance from capture_logs()
        event: Event name to count
        expected_count: Expected number of occurrences

    Raises:
        AssertionError: If the count doesn't match
    """
    actual_count = sum(1 for entry in logs.entries if entry.get("event") == event)
    if actual_count != expected_count:
        raise AssertionError(
            f"Expected {expected_count} occurrences of '{event}', "
            f"but found {actual_count}"
        )


def get_log_entries_by_event(logs: LogCapture, event: str) -> List[Dict[str, Any]]:
    """Get all log entries for a specific event.

    Args:
        logs: LogCapture instance from capture_logs()
        event: Event name to filter by

    Returns:
        List of log entries matching the event
    """
    return [entry for entry in logs.entries if entry.get("event") == event]


def get_log_entries_by_level(logs: LogCapture, level: str) -> List[Dict[str, Any]]:
    """Get all log entries for a specific level.

    Args:
        logs: LogCapture instance from capture_logs()
        level: Log level to filter by

    Returns:
        List of log entries matching the level
    """
    return [entry for entry in logs.entries if entry.get("level") == level]


def assert_no_errors(logs: LogCapture) -> None:
    """Assert that no error or critical level logs were captured.

    Args:
        logs: LogCapture instance from capture_logs()

    Raises:
        AssertionError: If any error or critical logs are found
    """
    error_entries = get_log_entries_by_level(logs, "error") + get_log_entries_by_level(
        logs, "critical"
    )
    if error_entries:
        raise AssertionError(f"Found error/critical logs: {error_entries}")


def assert_no_warnings(logs: LogCapture) -> None:
    """Assert that no warning level logs were captured.

    Args:
        logs: LogCapture instance from capture_logs()

    Raises:
        AssertionError: If any warning logs are found
    """
    warning_entries = get_log_entries_by_level(logs, "warning")
    if warning_entries:
        raise AssertionError(f"Found warning logs: {warning_entries}")


class LoggingTestMixin:
    """Mixin class for test cases that need logging verification.

    Provides convenient methods for testing logging behavior.
    """

    def capture_logs(self):
        """Get a log capture context manager."""
        return capture_logs()

    def assert_log_contains(self, logs: LogCapture, event: str, **kwargs):
        """Assert that logs contain a specific event."""
        assert_log_contains(logs, event, **kwargs)

    def assert_log_count(self, logs: LogCapture, event: str, expected_count: int):
        """Assert that a specific event appears exactly N times."""
        assert_log_count(logs, event, expected_count)

    def assert_no_errors(self, logs: LogCapture):
        """Assert that no error logs were captured."""
        assert_no_errors(logs)

    def assert_no_warnings(self, logs: LogCapture):
        """Assert that no warning logs were captured."""
        assert_no_warnings(logs)


# Re-export for convenience
__all__ = [
    "capture_logs",
    "assert_log_contains",
    "assert_log_count",
    "get_log_entries_by_event",
    "get_log_entries_by_level",
    "assert_no_errors",
    "assert_no_warnings",
    "LoggingTestMixin",
]
