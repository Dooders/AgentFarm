"""Async logging support for non-blocking performance-critical simulations.

This module provides async wrappers for structlog to enable non-blocking
logging in performance-critical simulation scenarios.

Usage:
    from farm.utils.logging_async import AsyncLogger

    async def run_simulation_async():
        logger = AsyncLogger(structlog.get_logger())

        await logger.info("simulation_started", simulation_id="sim_001")
        # ... simulation code ...
        await logger.info("simulation_completed", simulation_id="sim_001")
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

import structlog


class AsyncLogger:
    """Async wrapper for structlog logger.

    Provides non-blocking logging by executing log calls in a thread pool.
    This is useful for performance-critical simulations where logging
    should not block the main execution thread.

    Example:
        logger = AsyncLogger(structlog.get_logger())
        await logger.info("event", data="value")
    """

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        executor: Optional[ThreadPoolExecutor] = None,
        max_workers: int = 1,
    ):
        """Initialize async logger.

        Args:
            logger: The structlog logger to wrap
            executor: Optional thread pool executor. If None, creates one.
            max_workers: Number of worker threads (only used if executor is None)
        """
        self.logger_name = getattr(logger, "_context", {}).get("logger_name", "")
        self.logger = logger
        self.executor = executor or ThreadPoolExecutor(max_workers=max_workers)
        self.loop = asyncio.get_event_loop()

    async def debug(self, event: str, **kwargs: Any) -> None:
        """Async debug log."""
        await self._log_async("debug", event, **kwargs)

    async def info(self, event: str, **kwargs: Any) -> None:
        """Async info log."""
        await self._log_async("info", event, **kwargs)

    async def warning(self, event: str, **kwargs: Any) -> None:
        """Async warning log."""
        await self._log_async("warning", event, **kwargs)

    async def error(self, event: str, **kwargs: Any) -> None:
        """Async error log."""
        await self._log_async("error", event, **kwargs)

    async def critical(self, event: str, **kwargs: Any) -> None:
        """Async critical log."""
        await self._log_async("critical", event, **kwargs)

    async def _log_async(self, level: str, event: str, **kwargs: Any) -> None:
        """Execute log call in thread pool with context variables."""
        # Capture context variables from the current thread
        context_vars = structlog.contextvars.get_contextvars()

        def _log_with_context():
            # Set context variables in the worker thread
            if context_vars:
                structlog.contextvars.bind_contextvars(**context_vars)
            # Use the bound logger if it has bound context, otherwise get a fresh logger
            if hasattr(self.logger, "_context") and self.logger._context:
                # Use the bound logger that has context
                logger_to_use = self.logger
            else:
                # Get a fresh logger instance that uses the current structlog configuration
                logger_to_use = structlog.get_logger(self.logger_name)
            # Execute the log call
            getattr(logger_to_use, level)(event, **kwargs)

        await self.loop.run_in_executor(self.executor, _log_with_context)

    def bind(self, **kwargs: Any) -> "AsyncLogger":
        """Bind context variables and return new AsyncLogger.

        Args:
            **kwargs: Context variables to bind

        Returns:
            New AsyncLogger with bound context
        """
        bound_logger = self.logger.bind(**kwargs)
        return AsyncLogger(bound_logger, self.executor)

    def close(self) -> None:
        """Close the thread pool executor."""
        if self.executor:
            self.executor.shutdown(wait=True)


class AsyncLoggingContext:
    """Context manager for async logging with automatic cleanup.

    Example:
        async with AsyncLoggingContext() as logger:
            await logger.info("operation_started")
            # ... async operations ...
            await logger.info("operation_completed")
        # Executor is automatically cleaned up
    """

    def __init__(self, max_workers: int = 1):
        """Initialize async logging context.

        Args:
            max_workers: Number of worker threads for logging
        """
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.logger: Optional[AsyncLogger] = None

    async def __aenter__(self) -> AsyncLogger:
        """Enter async context and return AsyncLogger."""
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.logger = AsyncLogger(structlog.get_logger(), self.executor)
        return self.logger

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and cleanup executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        self.logger = None


def get_async_logger(name: str = "", max_workers: int = 1) -> AsyncLogger:
    """Get an async logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module
        max_workers: Number of worker threads for async logging

    Returns:
        AsyncLogger instance
    """
    logger = structlog.get_logger(name)
    return AsyncLogger(logger, max_workers=max_workers)


# Re-export for convenience
__all__ = [
    "AsyncLogger",
    "AsyncLoggingContext",
    "get_async_logger",
]
