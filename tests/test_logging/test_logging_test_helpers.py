"""Comprehensive tests for farm.utils.logging.test_helpers module.

Tests for the testing utilities themselves to ensure they work correctly.
"""

import pytest

from farm.utils.logging import configure_logging, get_logger
from farm.utils.logging.test_helpers import (
    LoggingTestMixin,
    assert_log_contains,
    assert_log_count,
    assert_no_errors,
    assert_no_warnings,
    capture_logs,
    get_log_entries_by_event,
    get_log_entries_by_level,
)


class TestCaptureLogsContextManager:
    """Test capture_logs context manager."""

    def test_capture_logs_basic(self):
        """Test basic log capture functionality."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event", value=42)

            assert len(logs.entries) > 0
            assert logs.entries[0]["event"] == "test_event"
            assert logs.entries[0]["value"] == 42

    def test_capture_logs_empty(self):
        """Test capturing when no logs are generated."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            # No logs generated
            pass

        assert len(logs.entries) == 0

    def test_capture_logs_multiple_events(self):
        """Test capturing multiple log events."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1", value=1)
            logger.warning("event2", value=2)
            logger.error("event3", value=3)

            assert len(logs.entries) == 3

    def test_capture_logs_restores_config(self):
        """Test that original logging config is restored."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        # Capture config before
        import structlog

        original_config = structlog.get_config()

        with capture_logs() as logs:
            logger.info("test_event")

        # Config should be restored
        restored_config = structlog.get_config()
        assert original_config == restored_config

    def test_capture_logs_with_exception(self):
        """Test that config is restored even on exception."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        import structlog

        original_config = structlog.get_config()

        try:
            with capture_logs() as logs:
                logger.info("before_error")
                raise ValueError("Test error")
        except ValueError:
            pass

        # Config should still be restored
        restored_config = structlog.get_config()
        assert original_config == restored_config


class TestAssertLogContains:
    """Test assert_log_contains function."""

    def test_assert_log_contains_basic(self):
        """Test basic log assertion."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event", value=42)

            # Should not raise
            assert_log_contains(logs, "test_event")

    def test_assert_log_contains_with_level(self):
        """Test log assertion with level check."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event", value=42)

            # Should not raise
            assert_log_contains(logs, "test_event", level="info")

    def test_assert_log_contains_with_fields(self):
        """Test log assertion with field checks."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event", value=42, name="test")

            # Should not raise
            assert_log_contains(logs, "test_event", value=42, name="test")

    def test_assert_log_contains_fails_missing_event(self):
        """Test that assertion fails when event is missing."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("other_event")

            with pytest.raises(AssertionError, match="Expected log entry not found"):
                assert_log_contains(logs, "missing_event")

    def test_assert_log_contains_fails_wrong_level(self):
        """Test that assertion fails with wrong level."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event")

            with pytest.raises(AssertionError, match="Expected log entry not found"):
                assert_log_contains(logs, "test_event", level="error")

    def test_assert_log_contains_fails_wrong_field(self):
        """Test that assertion fails with wrong field value."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event", value=42)

            with pytest.raises(AssertionError, match="Expected log entry not found"):
                assert_log_contains(logs, "test_event", value=99)

    def test_assert_log_contains_multiple_matches(self):
        """Test that assertion succeeds when multiple events match."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event", value=1)
            logger.info("test_event", value=2)
            logger.info("test_event", value=3)

            # Should match the first one
            assert_log_contains(logs, "test_event", value=1)


class TestAssertLogCount:
    """Test assert_log_count function."""

    def test_assert_log_count_exact(self):
        """Test exact count assertion."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event")
            logger.info("test_event")
            logger.info("test_event")

            # Should not raise
            assert_log_count(logs, "test_event", 3)

    def test_assert_log_count_zero(self):
        """Test zero count assertion."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("other_event")

            # Should not raise
            assert_log_count(logs, "missing_event", 0)

    def test_assert_log_count_fails_wrong_count(self):
        """Test that assertion fails with wrong count."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event")
            logger.info("test_event")

            with pytest.raises(
                AssertionError, match="Expected 3 occurrences.*but found 2"
            ):
                assert_log_count(logs, "test_event", 3)


class TestGetLogEntriesByEvent:
    """Test get_log_entries_by_event function."""

    def test_get_entries_basic(self):
        """Test getting entries by event name."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1", value=1)
            logger.info("event2", value=2)
            logger.info("event1", value=3)

            event1_logs = get_log_entries_by_event(logs, "event1")

            assert len(event1_logs) == 2
            assert event1_logs[0]["value"] == 1
            assert event1_logs[1]["value"] == 3

    def test_get_entries_empty(self):
        """Test getting entries when none match."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("other_event")

            missing_logs = get_log_entries_by_event(logs, "missing_event")

            assert len(missing_logs) == 0


class TestGetLogEntriesByLevel:
    """Test get_log_entries_by_level function."""

    def test_get_entries_by_level(self):
        """Test getting entries by log level."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1")
            logger.warning("event2")
            logger.info("event3")
            logger.error("event4")

            info_logs = get_log_entries_by_level(logs, "info")
            warning_logs = get_log_entries_by_level(logs, "warning")
            error_logs = get_log_entries_by_level(logs, "error")

            assert len(info_logs) == 2
            assert len(warning_logs) == 1
            assert len(error_logs) == 1

    def test_get_entries_by_level_empty(self):
        """Test getting entries when no logs at that level."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1")

            critical_logs = get_log_entries_by_level(logs, "critical")

            assert len(critical_logs) == 0


class TestAssertNoErrors:
    """Test assert_no_errors function."""

    def test_assert_no_errors_passes(self):
        """Test that assertion passes when no errors."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1")
            logger.warning("event2")

            # Should not raise
            assert_no_errors(logs)

    def test_assert_no_errors_fails_on_error(self):
        """Test that assertion fails when errors present."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1")
            logger.error("error_event")

            with pytest.raises(AssertionError, match="Found error/critical logs"):
                assert_no_errors(logs)

    def test_assert_no_errors_fails_on_critical(self):
        """Test that assertion fails when critical logs present."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1")
            logger.critical("critical_event")

            with pytest.raises(AssertionError, match="Found error/critical logs"):
                assert_no_errors(logs)


class TestAssertNoWarnings:
    """Test assert_no_warnings function."""

    def test_assert_no_warnings_passes(self):
        """Test that assertion passes when no warnings."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1")
            logger.debug("event2")

            # Should not raise
            assert_no_warnings(logs)

    def test_assert_no_warnings_fails_on_warning(self):
        """Test that assertion fails when warnings present."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("event1")
            logger.warning("warning_event")

            with pytest.raises(AssertionError, match="Found warning logs"):
                assert_no_warnings(logs)


class TestLoggingTestMixin:
    """Test LoggingTestMixin class."""

    def test_mixin_capture_logs(self):
        """Test mixin capture_logs method."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        mixin = LoggingTestMixin()

        with mixin.capture_logs() as logs:
            logger.info("test_event")

            assert len(logs.entries) > 0

    def test_mixin_assert_log_contains(self):
        """Test mixin assert_log_contains method."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        mixin = LoggingTestMixin()

        with mixin.capture_logs() as logs:
            logger.info("test_event", value=42)

            # Should not raise
            mixin.assert_log_contains(logs, "test_event", value=42)

    def test_mixin_assert_log_count(self):
        """Test mixin assert_log_count method."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        mixin = LoggingTestMixin()

        with mixin.capture_logs() as logs:
            logger.info("test_event")
            logger.info("test_event")

            # Should not raise
            mixin.assert_log_count(logs, "test_event", 2)

    def test_mixin_assert_no_errors(self):
        """Test mixin assert_no_errors method."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        mixin = LoggingTestMixin()

        with mixin.capture_logs() as logs:
            logger.info("test_event")

            # Should not raise
            mixin.assert_no_errors(logs)

    def test_mixin_assert_no_warnings(self):
        """Test mixin assert_no_warnings method."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        mixin = LoggingTestMixin()

        with mixin.capture_logs() as logs:
            logger.info("test_event")

            # Should not raise
            mixin.assert_no_warnings(logs)


class TestHelperEdgeCases:
    """Test edge cases for test helpers."""

    def test_empty_logs_assertions(self):
        """Test assertions with empty logs."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            # No logs generated
            pass

        # These should all pass
        assert_no_errors(logs)
        assert_no_warnings(logs)
        assert_log_count(logs, "any_event", 0)

        # This should fail
        with pytest.raises(AssertionError):
            assert_log_contains(logs, "any_event")

    def test_logs_with_missing_fields(self):
        """Test handling logs with missing fields."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event")  # No extra fields

            # Should fail when looking for missing field
            with pytest.raises(AssertionError):
                assert_log_contains(logs, "test_event", missing_field="value")

    def test_logs_with_none_values(self):
        """Test handling logs with None values."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            logger.info("test_event", value=None)

            # Should match None values
            assert_log_contains(logs, "test_event", value=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
