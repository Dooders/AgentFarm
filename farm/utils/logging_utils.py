"""Logging utilities and helpers for AgentFarm.

This module provides utility functions, decorators, and context managers
for enhanced structured logging capabilities throughout the application.

Features:
- Performance logging decorators
- Context managers for automatic context binding
- Log sampling for high-frequency operations
- Specialized loggers for different subsystems
"""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar, cast

import structlog

from farm.utils.logging_config import bind_context, get_logger, unbind_context

# Type variable for generic function decoration
F = TypeVar('F', bound=Callable[..., Any])


def log_performance(
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    slow_threshold_ms: float = 100.0,
) -> Callable[[F], F]:
    """Decorator to log function performance metrics.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        slow_threshold_ms: Threshold in ms to log slow operations
        
    Example:
        @log_performance(operation_name="agent_step", slow_threshold_ms=50.0)
        def step(self, action):
            # ... implementation
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            op_name = operation_name or func.__name__
            
            # Prepare log context
            log_context = {"operation": op_name}
            if log_args:
                # Be careful with large objects
                log_context["args"] = str(args)[:200]  # Truncate
                log_context["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                log_context["duration_ms"] = round(duration_ms, 2)
                log_context["status"] = "success"
                
                if log_result and result is not None:
                    log_context["result"] = str(result)[:200]  # Truncate
                
                # Log at different levels based on duration
                if duration_ms > slow_threshold_ms:
                    logger.warning("operation_slow", **log_context)
                else:
                    logger.debug("operation_complete", **log_context)
                
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                log_context["duration_ms"] = round(duration_ms, 2)
                log_context["status"] = "error"
                log_context["error"] = str(e)
                log_context["error_type"] = type(e).__name__
                
                logger.error("operation_failed", **log_context, exc_info=True)
                raise
        
        return cast(F, wrapper)
    return decorator


def log_errors(logger_name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to automatically log errors with context.
    
    Args:
        logger_name: Name for logger (defaults to function module)
        
    Example:
        @log_errors()
        def risky_operation(self, data):
            # ... implementation
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(logger_name or func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "unhandled_exception",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                raise
        return cast(F, wrapper)
    return decorator


@contextmanager
def log_context(**kwargs: Any):
    """Context manager to temporarily bind logging context.
    
    Args:
        **kwargs: Context key-value pairs to bind
        
    Example:
        with log_context(simulation_id="sim_001", step=42):
            # All logs here will include simulation_id and step
            logger.info("agent_moved", agent_id="agent_123")
    """
    try:
        bind_context(**kwargs)
        yield
    finally:
        unbind_context(*kwargs.keys())


@contextmanager
def log_step(step_number: int, **extra_context: Any):
    """Context manager for logging a simulation step.
    
    Args:
        step_number: Current step number
        **extra_context: Additional context to bind
        
    Example:
        with log_step(step_number=42, simulation_id="sim_001"):
            # All logs will include step_number and simulation_id
            process_agents()
    """
    logger = get_logger("farm.simulation")
    context = {"step": step_number, **extra_context}
    
    try:
        bind_context(**context)
        logger.debug("step_started", step=step_number)
        start_time = time.perf_counter()
        yield
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.debug("step_completed", step=step_number, duration_ms=round(duration_ms, 2))
    except Exception as e:
        logger.error(
            "step_failed",
            step=step_number,
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        raise
    finally:
        unbind_context(*context.keys())


@contextmanager
def log_simulation(simulation_id: str, **config: Any):
    """Context manager for logging an entire simulation run.
    
    Args:
        simulation_id: Unique simulation identifier
        **config: Simulation configuration to log
        
    Example:
        with log_simulation(simulation_id="sim_001", num_agents=100, num_steps=1000):
            run_simulation()
    """
    logger = get_logger("farm.simulation")
    
    try:
        bind_context(simulation_id=simulation_id)
        logger.info("simulation_started", simulation_id=simulation_id, **config)
        start_time = time.perf_counter()
        yield
        duration_s = time.perf_counter() - start_time
        logger.info(
            "simulation_completed",
            simulation_id=simulation_id,
            duration_seconds=round(duration_s, 2),
        )
    except Exception as e:
        logger.error(
            "simulation_failed",
            simulation_id=simulation_id,
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        raise
    finally:
        unbind_context("simulation_id")


@contextmanager
def log_experiment(experiment_id: str, experiment_name: str, **config: Any):
    """Context manager for logging an experiment (multiple simulations).
    
    Args:
        experiment_id: Unique experiment identifier
        experiment_name: Human-readable experiment name
        **config: Experiment configuration to log
        
    Example:
        with log_experiment(
            experiment_id="exp_001",
            experiment_name="parameter_sweep",
            num_iterations=10
        ):
            run_experiment()
    """
    logger = get_logger("farm.experiment")
    
    try:
        bind_context(experiment_id=experiment_id, experiment_name=experiment_name)
        logger.info(
            "experiment_started",
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            **config,
        )
        start_time = time.perf_counter()
        yield
        duration_s = time.perf_counter() - start_time
        logger.info(
            "experiment_completed",
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            duration_seconds=round(duration_s, 2),
        )
    except Exception as e:
        logger.error(
            "experiment_failed",
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        raise
    finally:
        unbind_context("experiment_id", "experiment_name")


class LogSampler:
    """Sample high-frequency logs to reduce noise.
    
    Example:
        sampler = LogSampler(sample_rate=0.1)  # Log 10% of events
        
        for i in range(1000):
            if sampler.should_log():
                logger.debug("high_frequency_event", iteration=i)
    """
    
    def __init__(self, sample_rate: float = 1.0, min_interval_ms: float = 0):
        """Initialize log sampler.
        
        Args:
            sample_rate: Fraction of events to log (0.0 to 1.0)
            min_interval_ms: Minimum time between logs in milliseconds
        """
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.min_interval_ms = min_interval_ms
        self.last_log_time = 0.0
        self.counter = 0
    
    def should_log(self) -> bool:
        """Determine if this event should be logged."""
        self.counter += 1
        
        # Check time-based sampling
        if self.min_interval_ms > 0:
            current_time = time.perf_counter() * 1000
            if current_time - self.last_log_time < self.min_interval_ms:
                return False
            self.last_log_time = current_time
        
        # Check rate-based sampling
        if self.sample_rate < 1.0:
            import random
            return random.random() < self.sample_rate
        
        return True
    
    def reset(self) -> None:
        """Reset sampler state."""
        self.counter = 0
        self.last_log_time = 0.0


class AgentLogger:
    """Specialized logger for agent-related events with automatic context binding."""
    
    def __init__(self, agent_id: str, agent_type: str = "unknown"):
        """Initialize agent logger.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (system, independent, control, etc.)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.logger = get_logger("farm.agent").bind(
            agent_id=agent_id,
            agent_type=agent_type,
        )
    
    def log_action(
        self,
        action_type: str,
        success: bool = True,
        reward: Optional[float] = None,
        **details: Any,
    ) -> None:
        """Log an agent action.
        
        Args:
            action_type: Type of action taken
            success: Whether action was successful
            reward: Reward received (if any)
            **details: Additional action details
        """
        self.logger.info(
            "agent_action",
            action_type=action_type,
            success=success,
            reward=reward,
            **details,
        )
    
    def log_state_change(
        self,
        state_type: str,
        old_value: Any,
        new_value: Any,
        **context: Any,
    ) -> None:
        """Log a state change.
        
        Args:
            state_type: Type of state that changed
            old_value: Previous value
            new_value: New value
            **context: Additional context
        """
        self.logger.info(
            "agent_state_change",
            state_type=state_type,
            old_value=old_value,
            new_value=new_value,
            **context,
        )
    
    def log_interaction(
        self,
        interaction_type: str,
        target_id: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Log an interaction with another agent or resource.
        
        Args:
            interaction_type: Type of interaction
            target_id: ID of interaction target
            **details: Additional interaction details
        """
        self.logger.info(
            "agent_interaction",
            interaction_type=interaction_type,
            target_id=target_id,
            **details,
        )
    
    def log_death(self, cause: str, **context: Any) -> None:
        """Log agent death.
        
        Args:
            cause: Cause of death
            **context: Additional context
        """
        self.logger.info("agent_died", cause=cause, **context)
    
    def log_birth(self, parent_ids: Optional[list[str]] = None, **context: Any) -> None:
        """Log agent birth/creation.
        
        Args:
            parent_ids: IDs of parent agents (if any)
            **context: Additional context
        """
        self.logger.info("agent_born", parent_ids=parent_ids, **context)


__all__ = [
    "log_performance",
    "log_errors",
    "log_context",
    "log_step",
    "log_simulation",
    "log_experiment",
    "LogSampler",
    "AgentLogger",
]
