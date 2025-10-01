"""
Examples demonstrating structured logging with structlog in AgentFarm.

Run this file to see various logging patterns in action.
"""

import time
from farm.utils import (
    configure_logging,
    get_logger,
    bind_context,
    unbind_context,
    log_context,
    log_simulation,
    log_step,
    log_performance,
    log_errors,
    AgentLogger,
    LogSampler,
)


def example_basic_logging():
    """Example 1: Basic structured logging."""
    print("\n=== Example 1: Basic Logging ===\n")
    
    logger = get_logger(__name__)
    
    # Simple event logging with context
    logger.info("simulation_started", num_agents=100, num_steps=1000)
    logger.debug("spatial_index_built", num_items=250, build_time_ms=45.2)
    logger.warning("resource_depleted", resource_id="res_001", remaining=0)
    logger.error("agent_collision", agent_a="agent_123", agent_b="agent_456")


def example_context_binding():
    """Example 2: Context binding."""
    print("\n=== Example 2: Context Binding ===\n")
    
    logger = get_logger(__name__)
    
    # Bind global context
    bind_context(simulation_id="sim_001", experiment_id="exp_42")
    
    logger.info("step_started", step=1)
    logger.info("agent_spawned", agent_id="agent_001")
    
    # Unbind specific context
    unbind_context("experiment_id")
    
    logger.info("step_completed", step=1)
    
    # Clean up
    unbind_context("simulation_id")


def example_context_managers():
    """Example 3: Using context managers."""
    print("\n=== Example 3: Context Managers ===\n")
    
    logger = get_logger(__name__)
    
    # Scoped context
    with log_context(simulation_id="sim_002", environment="test"):
        logger.info("entering_context")
        
        with log_step(step_number=1):
            logger.info("processing_agents", num_agents=50)
        
        logger.info("exiting_context")
    
    # Context is automatically cleaned up
    logger.info("outside_context")


def example_simulation_context():
    """Example 4: Simulation-level context."""
    print("\n=== Example 4: Simulation Context ===\n")
    
    logger = get_logger(__name__)
    
    with log_simulation(
        simulation_id="sim_003",
        num_agents=100,
        num_steps=1000,
        seed=42,
    ):
        logger.info("initializing_environment")
        
        for step in range(3):  # Just 3 steps for demo
            with log_step(step_number=step):
                logger.info("processing_step")
                time.sleep(0.01)  # Simulate work
        
        logger.info("finalizing_results")


@log_performance(operation_name="slow_computation", slow_threshold_ms=10.0)
def slow_function():
    """Example function with performance logging."""
    time.sleep(0.05)  # Simulate slow operation
    return "result"


@log_errors()
def risky_function(value):
    """Example function with error logging."""
    if value < 0:
        raise ValueError("Value must be non-negative")
    return value * 2


def example_decorators():
    """Example 5: Using decorators."""
    print("\n=== Example 5: Decorators ===\n")
    
    logger = get_logger(__name__)
    
    # Performance decorator
    logger.info("calling_slow_function")
    result = slow_function()
    logger.info("slow_function_returned", result=result)
    
    # Error decorator
    try:
        risky_function(-1)
    except ValueError:
        logger.info("caught_expected_error")
    
    result = risky_function(5)
    logger.info("risky_function_succeeded", result=result)


def example_agent_logger():
    """Example 6: Specialized agent logger."""
    print("\n=== Example 6: Agent Logger ===\n")
    
    # Create agent logger with bound context
    agent_logger = AgentLogger(agent_id="agent_007", agent_type="system")
    
    # Log agent events
    agent_logger.log_birth(parent_ids=["agent_001", "agent_002"], generation=2)
    
    agent_logger.log_action(
        action_type="move",
        success=True,
        reward=0.5,
        position=(10.5, 20.3),
    )
    
    agent_logger.log_interaction(
        interaction_type="resource_share",
        target_id="agent_008",
        amount=15,
    )
    
    agent_logger.log_state_change(
        state_type="health",
        old_value=100,
        new_value=85,
        reason="combat",
    )
    
    agent_logger.log_death(cause="starvation", lifetime_steps=342)


def example_log_sampling():
    """Example 7: Log sampling for high-frequency events."""
    print("\n=== Example 7: Log Sampling ===\n")
    
    logger = get_logger(__name__)
    
    # Create sampler that logs 10% of events
    sampler = LogSampler(sample_rate=0.1)
    
    logger.info("starting_high_frequency_loop")
    
    logged_count = 0
    for i in range(100):
        if sampler.should_log():
            logger.debug("high_frequency_event", iteration=i)
            logged_count += 1
    
    logger.info("completed_loop", total_iterations=100, logged_events=logged_count)
    
    # Time-based sampling
    time_sampler = LogSampler(min_interval_ms=50)  # Max once per 50ms
    
    logger.info("starting_time_based_sampling")
    logged_count = 0
    
    for i in range(50):
        if time_sampler.should_log():
            logger.debug("time_sampled_event", iteration=i)
            logged_count += 1
        time.sleep(0.01)  # 10ms between iterations
    
    logger.info("completed_time_sampling", logged_events=logged_count)


def example_bound_logger():
    """Example 8: Logger with permanent bindings."""
    print("\n=== Example 8: Bound Logger ===\n")
    
    # Create logger with permanent context
    base_logger = get_logger(__name__)
    component_logger = base_logger.bind(
        component="spatial_index",
        version="2.0",
    )
    
    # All logs from this logger include component and version
    component_logger.info("index_initialized", capacity=1000)
    component_logger.info("item_added", item_id="item_001")
    component_logger.warning("index_rebuild_needed", reason="capacity_exceeded")
    component_logger.info("index_rebuilt", new_capacity=2000, duration_ms=123.4)


def example_error_logging():
    """Example 9: Comprehensive error logging."""
    print("\n=== Example 9: Error Logging ===\n")
    
    logger = get_logger(__name__)
    
    try:
        # Simulate an error
        data = {"key": "value"}
        result = data["missing_key"]
    except KeyError as e:
        logger.error(
            "data_access_error",
            error_type=type(e).__name__,
            error_message=str(e),
            data_keys=list(data.keys()),
            attempted_key="missing_key",
            exc_info=True,
        )
    
    try:
        # Simulate another error
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.error(
            "calculation_error",
            operation="division",
            numerator=10,
            denominator=0,
            error_type=type(e).__name__,
            exc_info=True,
        )


def example_nested_contexts():
    """Example 10: Nested context for hierarchical logging."""
    print("\n=== Example 10: Nested Contexts ===\n")
    
    logger = get_logger(__name__)
    
    # Experiment > Simulation > Step > Agent hierarchy
    with log_context(experiment_id="exp_001", experiment_name="parameter_sweep"):
        logger.info("experiment_started", num_simulations=3)
        
        for sim_idx in range(2):  # Just 2 sims for demo
            with log_context(simulation_id=f"sim_{sim_idx}", iteration=sim_idx):
                logger.info("simulation_started")
                
                for step in range(2):  # Just 2 steps for demo
                    with log_context(step=step):
                        logger.info("step_processing", num_agents=50)
                        
                        # Agent-level logging
                        agent_logger = logger.bind(agent_id="agent_001")
                        agent_logger.debug("agent_action", action="move")
                
                logger.info("simulation_completed")
        
        logger.info("experiment_completed")


def main():
    """Run all examples."""
    # Configure logging for examples
    configure_logging(
        environment="development",
        log_level="DEBUG",
        enable_colors=True,
    )
    
    logger = get_logger(__name__)
    logger.info("starting_logging_examples")
    
    # Run all examples
    example_basic_logging()
    example_context_binding()
    example_context_managers()
    example_simulation_context()
    example_decorators()
    example_agent_logger()
    example_log_sampling()
    example_bound_logger()
    example_error_logging()
    example_nested_contexts()
    
    logger.info("completed_all_examples")
    
    print("\n=== Examples Complete ===")
    print("Check the logs directory for JSON output (if configured)")


if __name__ == "__main__":
    main()
