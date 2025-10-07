"""
Lightweight logging facade to decouple modules from specific logging configs.

This module exposes a stable interface that proxies to structlog so callers
can import `get_logger` without caring whether the base or enhanced logging
configuration is active.
"""
from typing import Any

import structlog


def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:  # pragma: no cover
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:  # pragma: no cover
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:  # pragma: no cover
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:  # pragma: no cover
    structlog.contextvars.clear_contextvars()


__all__ = [
    "get_logger",
    "bind_context",
    "unbind_context",
    "clear_context",
] 
