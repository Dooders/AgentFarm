"""Correlation ID support for distributed tracing in AgentFarm.

This module provides correlation ID functionality to track related events
across simulation runs, API calls, and distributed operations.

Usage:
    from farm.utils.logging_correlation import add_correlation_id, get_correlation_id
    
    # Start a new operation with correlation ID
    correlation_id = add_correlation_id()
    
    # All subsequent logs will include the correlation_id
    logger.info("simulation_started", simulation_id="sim_001")
    
    # Pass correlation ID to sub-operations
    analyze_results(correlation_id=correlation_id)
"""

import uuid
from typing import Optional
import structlog


def add_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Add correlation ID to track related operations.
    
    Args:
        correlation_id: Optional custom correlation ID. If None, generates a new one.
        
    Returns:
        The correlation ID that was set
        
    Example:
        # Generate new correlation ID
        corr_id = add_correlation_id()
        
        # Use custom correlation ID
        corr_id = add_correlation_id("custom_operation_123")
    """
    if correlation_id is None:
        # Generate short, readable correlation ID
        correlation_id = str(uuid.uuid4())[:8]
    
    # Bind to context variables so it appears in all subsequent logs
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context.
    
    Returns:
        Current correlation ID if set, None otherwise
    """
    # Get from context variables
    context = structlog.contextvars.get_contextvars()
    return context.get("correlation_id")


def clear_correlation_id() -> None:
    """Clear the current correlation ID from context."""
    structlog.contextvars.unbind_contextvars("correlation_id")


def with_correlation_id(correlation_id: Optional[str] = None):
    """Context manager to temporarily set a correlation ID.
    
    Args:
        correlation_id: Optional correlation ID. If None, generates a new one.
        
    Example:
        with with_correlation_id("batch_001"):
            logger.info("processing_batch")  # Will include correlation_id
            process_items()
        # correlation_id is automatically cleared
    """
    class CorrelationContext:
        def __init__(self, corr_id: Optional[str]):
            self.corr_id = corr_id
            self.original_corr_id = get_correlation_id()
        
        def __enter__(self):
            if self.corr_id is None:
                self.corr_id = str(uuid.uuid4())[:8]
            add_correlation_id(self.corr_id)
            return self.corr_id
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.original_corr_id is None:
                clear_correlation_id()
            else:
                add_correlation_id(self.original_corr_id)
    
    return CorrelationContext(correlation_id)


def bind_correlation_context(**kwargs):
    """Bind correlation context along with other context variables.
    
    Args:
        **kwargs: Context variables to bind, including optional correlation_id
        
    Example:
        bind_correlation_context(
            correlation_id="exp_001",
            experiment_id="exp_001",
            batch_id="batch_001"
        )
    """
    structlog.contextvars.bind_contextvars(**kwargs)


# Re-export for convenience
__all__ = [
    "add_correlation_id",
    "get_correlation_id", 
    "clear_correlation_id",
    "with_correlation_id",
    "bind_correlation_context",
]
