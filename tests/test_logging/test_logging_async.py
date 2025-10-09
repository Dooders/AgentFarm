"""Comprehensive tests for farm.utils.logging.async_logger module.

Tests for:
- AsyncLogger class
- AsyncLoggingContext context manager
- get_async_logger function
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
import structlog

from farm.utils.logging import configure_logging
from farm.utils.logging.async_logger import (
    AsyncLogger,
    AsyncLoggingContext,
    get_async_logger,
)
from farm.utils.logging.test_helpers import assert_log_contains, capture_logs


class TestAsyncLogger:
    """Test AsyncLogger class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test AsyncLogger initialization."""
        configure_logging(environment="testing")
        logger = structlog.get_logger(__name__)
        async_logger = AsyncLogger(logger)

        assert async_logger.logger is logger
        assert async_logger.executor is not None

    @pytest.mark.asyncio
    async def test_initialization_with_executor(self):
        """Test AsyncLogger initialization with custom executor."""
        configure_logging(environment="testing")
        logger = structlog.get_logger(__name__)
        executor = ThreadPoolExecutor(max_workers=2)

        async_logger = AsyncLogger(logger, executor=executor)

        assert async_logger.executor is executor

        # Cleanup
        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_debug_logging(self):
        """Test async debug logging."""
        configure_logging(environment="testing", log_level="DEBUG")
        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            await async_logger.debug("debug_event", value=42)

            assert_log_contains(logs, "debug_event", level="debug", value=42)

        async_logger.close()

    @pytest.mark.asyncio
    async def test_info_logging(self):
        """Test async info logging."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            await async_logger.info("info_event", value=42)

            assert_log_contains(logs, "info_event", level="info", value=42)

        async_logger.close()

    @pytest.mark.asyncio
    async def test_warning_logging(self):
        """Test async warning logging."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            await async_logger.warning("warning_event", message="test warning")

            assert_log_contains(
                logs, "warning_event", level="warning", message="test warning"
            )

        async_logger.close()

    @pytest.mark.asyncio
    async def test_error_logging(self):
        """Test async error logging."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            await async_logger.error("error_event", error="test error")

            assert_log_contains(logs, "error_event", level="error", error="test error")

        async_logger.close()

    @pytest.mark.asyncio
    async def test_critical_logging(self):
        """Test async critical logging."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            await async_logger.critical("critical_event", message="critical issue")

            assert_log_contains(
                logs, "critical_event", level="critical", message="critical issue"
            )

        async_logger.close()

    @pytest.mark.asyncio
    async def test_bind_context(self):
        """Test binding context variables."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            bound_logger = async_logger.bind(simulation_id="sim_001", step=42)
            await bound_logger.info("bound_event")

            assert_log_contains(logs, "bound_event", simulation_id="sim_001", step=42)

        async_logger.close()
        bound_logger.close()

    @pytest.mark.asyncio
    async def test_multiple_async_logs(self):
        """Test multiple async log calls."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            await asyncio.gather(
                async_logger.info("event1", value=1),
                async_logger.info("event2", value=2),
                async_logger.info("event3", value=3),
            )

            assert_log_contains(logs, "event1", value=1)
            assert_log_contains(logs, "event2", value=2)
            assert_log_contains(logs, "event3", value=3)

        async_logger.close()

    @pytest.mark.asyncio
    async def test_concurrent_logging(self):
        """Test concurrent logging from multiple tasks."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        async def log_task(task_id):
            for i in range(5):
                await async_logger.info("task_event", task_id=task_id, iteration=i)

        with capture_logs() as logs:
            await asyncio.gather(
                log_task("task1"),
                log_task("task2"),
                log_task("task3"),
            )

            # Should have 15 log entries total (3 tasks * 5 iterations)
            task_events = [e for e in logs.entries if e.get("event") == "task_event"]
            assert len(task_events) == 15

        async_logger.close()

    @pytest.mark.asyncio
    async def test_close_executor(self):
        """Test closing executor."""
        configure_logging(environment="testing")
        logger = structlog.get_logger(__name__)
        executor = ThreadPoolExecutor(max_workers=1)
        async_logger = AsyncLogger(logger, executor=executor)

        await async_logger.info("test_event")

        async_logger.close()

        # Executor should be shutdown
        assert executor._shutdown


class TestAsyncLoggingContext:
    """Test AsyncLoggingContext context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test AsyncLoggingContext as context manager."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            async with AsyncLoggingContext() as logger:
                await logger.info("context_event", value=42)

            assert_log_contains(logs, "context_event", value=42)

    @pytest.mark.asyncio
    async def test_context_cleanup(self):
        """Test that executor is cleaned up after context."""
        configure_logging(environment="testing")
        context = AsyncLoggingContext(max_workers=2)

        async with context as logger:
            await logger.info("test_event")
            executor = context.executor

        # Executor should be shutdown and cleared
        assert context.executor is None
        assert context.logger is None
        assert executor._shutdown

    @pytest.mark.asyncio
    async def test_context_with_error(self):
        """Test that context cleanup happens even on error."""
        configure_logging(environment="testing")
        context = AsyncLoggingContext()

        with pytest.raises(ValueError):
            async with context as logger:
                await logger.info("before_error")
                raise ValueError("Test error")

        # Executor should still be cleaned up
        assert context.executor is None
        assert context.logger is None

    @pytest.mark.asyncio
    async def test_custom_max_workers(self):
        """Test AsyncLoggingContext with custom max_workers."""
        configure_logging(environment="testing")

        async with AsyncLoggingContext(max_workers=3) as logger:
            assert isinstance(logger, AsyncLogger)
            assert logger.executor is not None

    @pytest.mark.asyncio
    async def test_multiple_logs_in_context(self):
        """Test multiple log calls within context."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            async with AsyncLoggingContext() as logger:
                await logger.info("event1")
                await logger.warning("event2")
                await logger.error("event3")

            assert len(logs.entries) >= 3
            assert_log_contains(logs, "event1", level="info")
            assert_log_contains(logs, "event2", level="warning")
            assert_log_contains(logs, "event3", level="error")


class TestGetAsyncLogger:
    """Test get_async_logger function."""

    @pytest.mark.asyncio
    async def test_get_async_logger_default(self):
        """Test get_async_logger with default parameters."""
        configure_logging(environment="testing")
        logger = get_async_logger()

        assert isinstance(logger, AsyncLogger)
        assert logger.executor is not None

        logger.close()

    @pytest.mark.asyncio
    async def test_get_async_logger_with_name(self):
        """Test get_async_logger with logger name."""
        configure_logging(environment="testing")
        logger = get_async_logger(__name__)

        assert isinstance(logger, AsyncLogger)

        with capture_logs() as logs:
            await logger.info("test_event")
            assert len(logs.entries) > 0

        logger.close()

    @pytest.mark.asyncio
    async def test_get_async_logger_with_max_workers(self):
        """Test get_async_logger with custom max_workers."""
        configure_logging(environment="testing")
        logger = get_async_logger(__name__, max_workers=3)

        assert isinstance(logger, AsyncLogger)

        logger.close()


class TestAsyncLoggingPerformance:
    """Test async logging performance characteristics."""

    @pytest.mark.asyncio
    async def test_non_blocking_behavior(self):
        """Test that async logging doesn't block main execution."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        start_time = asyncio.get_event_loop().time()

        # Log many events asynchronously
        tasks = [async_logger.info(f"event_{i}", iteration=i) for i in range(100)]
        await asyncio.gather(*tasks)

        elapsed = asyncio.get_event_loop().time() - start_time

        # Should complete relatively quickly due to async execution
        assert elapsed < 5.0  # Generous timeout

        async_logger.close()

    @pytest.mark.asyncio
    async def test_async_logging_in_simulation_loop(self):
        """Test async logging in a simulated async loop."""
        configure_logging(environment="testing")
        async_logger = get_async_logger(__name__)

        async def simulation_step(step_num):
            await async_logger.info("step_started", step=step_num)
            # Simulate some async work
            await asyncio.sleep(0.001)
            await async_logger.info("step_completed", step=step_num)

        with capture_logs() as logs:
            # Run multiple simulation steps concurrently
            await asyncio.gather(*[simulation_step(i) for i in range(10)])

            # Should have start and complete logs for each step
            started_logs = [e for e in logs.entries if e.get("event") == "step_started"]
            completed_logs = [
                e for e in logs.entries if e.get("event") == "step_completed"
            ]

            assert len(started_logs) == 10
            assert len(completed_logs) == 10

        async_logger.close()


class TestAsyncLoggerIntegration:
    """Test AsyncLogger integration with other logging features."""

    @pytest.mark.asyncio
    async def test_with_context_variables(self):
        """Test AsyncLogger with context variables."""
        configure_logging(environment="testing")
        from farm.utils.logging import bind_context, clear_context

        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            bind_context(simulation_id="sim_001")
            await async_logger.info("event_with_context")
            clear_context()

            assert_log_contains(logs, "event_with_context", simulation_id="sim_001")

        async_logger.close()

    @pytest.mark.asyncio
    async def test_with_metrics_enabled(self):
        """Test AsyncLogger with metrics enabled."""
        configure_logging(environment="testing", enable_metrics=True)
        from farm.utils.logging import get_metrics_summary

        async_logger = get_async_logger(__name__)

        await async_logger.info("metric_event1", duration_ms=100.0)
        await async_logger.info("metric_event2", duration_ms=200.0)

        # Give async logs time to process
        await asyncio.sleep(0.1)

        metrics = get_metrics_summary()
        assert "event_counts" in metrics

        async_logger.close()

    @pytest.mark.asyncio
    async def test_with_correlation_id(self):
        """Test AsyncLogger with correlation IDs."""
        configure_logging(environment="testing")
        from farm.utils.logging.correlation import (
            add_correlation_id,
            clear_correlation_id,
        )

        async_logger = get_async_logger(__name__)

        with capture_logs() as logs:
            corr_id = add_correlation_id("test_correlation")
            await async_logger.info("correlated_event")
            clear_correlation_id()

            assert_log_contains(logs, "correlated_event", correlation_id=corr_id)

        async_logger.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
