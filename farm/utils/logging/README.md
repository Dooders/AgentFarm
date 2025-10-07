# AgentFarm Logging Module

A unified, structured logging system for AgentFarm built on [structlog](https://www.structlog.org/), providing high-performance, context-rich logging for complex simulations and experiments.

## Features

- **Structured Logging**: JSON-serializable log entries with rich context
- **Performance Optimized**: Minimal overhead with optional sampling and async logging
- **Context Management**: Automatic context binding for simulations, experiments, and agents
- **Multiple Output Formats**: Console (colored/plain), JSON files, and plain text
- **Advanced Capabilities**: Metrics tracking, correlation IDs, log sampling, and rotation
- **Type-Safe Interfaces**: Typed loggers with IDE support and type checking
- **Testing Utilities**: Comprehensive test helpers for verifying logging behavior

## Quick Start

### Basic Setup

```python
from farm.utils.logging import configure_logging, get_logger

# Configure logging for development
configure_logging(
    environment="development",
    log_dir="logs",
    log_level="INFO"
)

# Get a logger
logger = get_logger(__name__)

# Log structured events
logger.info("simulation_started", simulation_id="sim_001", num_agents=100)
logger.debug("agent_action", agent_id="agent_001", action="move", success=True)
```

### Production Setup

```python
configure_logging(
    environment="production",
    log_dir="logs",
    log_level="INFO",
    json_logs=True,              # JSON output for log aggregation
    enable_metrics=True,          # Track performance metrics
    enable_sampling=True,         # Sample high-frequency logs
    sample_rate=0.1,              # Log 10% of sampled events
    enable_log_rotation=True,     # Rotate logs
    max_log_size_mb=100,          # 100MB per log file
    backup_count=5                # Keep 5 backup files
)
```

## Module Structure

```
farm/utils/logging/
├── __init__.py           # Main exports and package interface
├── config.py             # Core configuration and processors
├── utils.py              # Decorators, context managers, specialized loggers
├── simulation.py         # Type-safe simulation logging
├── async_logger.py       # Async logging for non-blocking operations
├── correlation.py        # Correlation ID tracking for distributed tracing
└── test_helpers.py       # Testing utilities
```

## Core Usage

### Getting Loggers

```python
from farm.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("event_occurred", key="value", count=42)
```

### Context Binding

Bind context that persists across all subsequent log calls:

```python
from farm.utils.logging import bind_context, unbind_context, clear_context

# Bind context
bind_context(simulation_id="sim_001", step=42)
logger.info("agent_moved")  # Automatically includes simulation_id and step

# Unbind specific keys
unbind_context("step")

# Clear all context
clear_context()
```

### Context Managers

#### Log Simulation

```python
from farm.utils.logging import log_simulation

with log_simulation(simulation_id="sim_001", num_agents=100, num_steps=1000):
    # All logs include simulation_id
    run_simulation()
    # Automatically logs start, end, duration, and any errors
```

#### Log Experiment

```python
from farm.utils.logging import log_experiment

with log_experiment(
    experiment_id="exp_001",
    experiment_name="parameter_sweep",
    num_iterations=10
):
    run_experiment()
```

#### Log Step

```python
from farm.utils.logging import log_step

for step in range(1000):
    with log_step(step_number=step, simulation_id="sim_001"):
        # All logs include step number
        process_agents()
```

#### Generic Context

```python
from farm.utils.logging import log_context

with log_context(operation="data_processing", batch_id="batch_001"):
    process_data()
```

## Decorators

### Performance Logging

```python
from farm.utils.logging import log_performance

@log_performance(operation_name="agent_step", slow_threshold_ms=50.0)
def step(self, action):
    # Function implementation
    pass
    # Automatically logs duration, warns if > 50ms
```

### Error Logging

```python
from farm.utils.logging import log_errors

@log_errors()
def risky_operation(data):
    # Function implementation
    pass
    # Automatically logs any exceptions with full context
```

## Specialized Loggers

### Agent Logger

```python
from farm.utils.logging import AgentLogger

agent_logger = AgentLogger(agent_id="agent_001", agent_type="system")

agent_logger.log_action(
    action_type="move",
    success=True,
    reward=10.5,
    target_position=(100, 200)
)

agent_logger.log_state_change(
    state_type="energy",
    old_value=80.0,
    new_value=70.0
)

agent_logger.log_interaction(
    interaction_type="combat",
    target_id="agent_002",
    outcome="victory"
)

agent_logger.log_death(cause="starvation", age=150)
agent_logger.log_birth(parent_ids=["agent_001", "agent_002"])
```

### Database Logger

```python
from farm.utils.logging import DatabaseLogger

db_logger = DatabaseLogger(db_path="simulation.db", simulation_id="sim_001")

db_logger.log_query(
    query_type="insert",
    table="agents",
    duration_ms=5.2,
    rows=100
)

db_logger.log_transaction(status="commit", duration_ms=15.3)
```

### Type-Safe Simulation Logger

```python
from farm.utils.logging.simulation import get_simulation_logger

sim_logger = get_simulation_logger(__name__)

sim_logger.log_agent_action(
    agent_id="agent_001",
    action="move",
    success=True,
    duration_ms=2.5
)

sim_logger.log_population_change(
    population=250,
    change=-5,
    step=100
)

sim_logger.log_performance_metric(
    metric_name="step_time",
    value=125.5,
    unit="ms"
)
```

## Advanced Features

### Metrics Tracking

Enable metrics to track event counts and performance statistics:

```python
from farm.utils.logging import configure_logging, get_metrics_summary, reset_metrics

configure_logging(enable_metrics=True)

# Run simulation...

# Get metrics summary
summary = get_metrics_summary()
print(summary)
# {
#     "event_counts": {"agent_action": 10000, "step_completed": 100},
#     "duration_metrics": {
#         "step_completed": {
#             "count": 100,
#             "mean": 125.5,
#             "median": 120.0,
#             "max": 250.0,
#             "min": 80.0
#         }
#     },
#     "total_runtime": 15.3
# }

# Reset metrics
reset_metrics()
```

### Log Sampling

Reduce noise from high-frequency events:

```python
from farm.utils.logging import LogSampler

# Sample 10% of events
sampler = LogSampler(sample_rate=0.1)

for i in range(10000):
    if sampler.should_log():
        logger.debug("high_frequency_event", iteration=i)

# Or use time-based sampling
time_sampler = LogSampler(min_interval_ms=100)  # Max one log per 100ms

for i in range(10000):
    if time_sampler.should_log():
        logger.debug("frequent_event", iteration=i)
```

Alternatively, use built-in sampling:

```python
configure_logging(
    enable_sampling=True,
    sample_rate=0.1,
    events_to_sample={"agent_action", "resource_update"}  # Only sample these events
)
```

### Correlation IDs

Track related operations across distributed systems:

```python
from farm.utils.logging.correlation import (
    add_correlation_id,
    get_correlation_id,
    with_correlation_id
)

# Add correlation ID
corr_id = add_correlation_id()  # Auto-generated
# or
corr_id = add_correlation_id("custom_id_123")  # Custom

# All subsequent logs include correlation_id
logger.info("operation_started")

# Use in context manager
with with_correlation_id("batch_001"):
    logger.info("processing_batch")
    process_items()
# correlation_id automatically cleared
```

### Async Logging

Non-blocking logging for performance-critical code:

```python
from farm.utils.logging.async_logger import get_async_logger, AsyncLoggingContext

# Method 1: Direct async logger
async_logger = get_async_logger(__name__)

async def run_simulation():
    await async_logger.info("simulation_started", simulation_id="sim_001")
    # ... async simulation code ...
    await async_logger.info("simulation_completed")

# Method 2: Context manager with auto-cleanup
async def run_with_context():
    async with AsyncLoggingContext() as logger:
        await logger.info("operation_started")
        # ... async operations ...
        await logger.info("operation_completed")
    # Executor automatically cleaned up
```

### Performance Monitoring

Detailed performance tracking with checkpoints:

```python
from farm.utils.logging import PerformanceMonitor

with PerformanceMonitor(operation="complex_computation") as monitor:
    # Step 1
    load_data()
    monitor.checkpoint("data_loaded")
    
    # Step 2
    process_data()
    monitor.checkpoint("data_processed")
    
    # Step 3
    save_results()
    monitor.checkpoint("results_saved")
    
# Automatically logs total duration and all checkpoints
```

### Custom Context Classes

For specialized performance scenarios:

```python
from farm.utils.logging import configure_logging, FastContext, ThreadSafeContext

# Fast context (low memory overhead)
configure_logging(context_class=FastContext)

# Thread-safe context (for parallel simulations)
configure_logging(use_threadlocal=True)
# or
configure_logging(context_class=ThreadSafeContext)
```

## Configuration Options

### `configure_logging()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `environment` | str | "development" | Environment: development, production, testing |
| `log_dir` | str | None | Directory for log files (None = console only) |
| `log_level` | str | "INFO" | Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `json_logs` | bool | False | Output JSON formatted logs to file |
| `enable_colors` | bool | True | Enable colored console output (dev only) |
| `include_caller_info` | bool | False | Include file/line/function info (expensive) |
| `include_process_info` | bool | False | Include process/thread IDs |
| `enable_metrics` | bool | False | Track metrics from logs |
| `enable_sampling` | bool | False | Enable log sampling for high-frequency events |
| `sample_rate` | float | 0.1 | Sampling rate (0.001 to 1.0) |
| `events_to_sample` | set | None | Set of event names to sample (None = all) |
| `use_threadlocal` | bool | False | Use thread-local context (for parallel experiments) |
| `context_class` | type | None | Custom context class (overrides use_threadlocal) |
| `enable_log_rotation` | bool | True | Enable log rotation |
| `max_log_size_mb` | int | 100 | Max log file size before rotation |
| `backup_count` | int | 5 | Number of backup log files to keep |
| `slow_threshold_ms` | float | 100.0 | Threshold for slow operation warnings |

## Testing

### Capture and Verify Logs

```python
from farm.utils.logging.test_helpers import (
    capture_logs,
    assert_log_contains,
    assert_log_count,
    assert_no_errors,
    assert_no_warnings
)

def test_simulation_logging():
    with capture_logs() as logs:
        run_simulation()
        
        # Assert specific log exists
        assert_log_contains(logs, "simulation_started")
        assert_log_contains(logs, "agent_action", level="info", agent_id="agent_001")
        
        # Assert log count
        assert_log_count(logs, "step_completed", 100)
        
        # Assert no errors
        assert_no_errors(logs)
        assert_no_warnings(logs)
        
        # Manual inspection
        assert len(logs.entries) > 0
        assert logs.entries[0]["event"] == "simulation_started"
```

### Get Specific Log Entries

```python
from farm.utils.logging.test_helpers import (
    get_log_entries_by_event,
    get_log_entries_by_level
)

with capture_logs() as logs:
    run_simulation()
    
    # Get all agent actions
    agent_actions = get_log_entries_by_event(logs, "agent_action")
    assert len(agent_actions) == 1000
    
    # Get all errors
    errors = get_log_entries_by_level(logs, "error")
    assert len(errors) == 0
```

### Test Mixin

```python
from farm.utils.logging.test_helpers import LoggingTestMixin
import unittest

class SimulationTestCase(unittest.TestCase, LoggingTestMixin):
    def test_simulation_logs_correctly(self):
        with self.capture_logs() as logs:
            run_simulation()
            
            self.assert_log_contains(logs, "simulation_started")
            self.assert_log_count(logs, "step_completed", 100)
            self.assert_no_errors(logs)
```

## Best Practices

### 1. Use Structured Events

**Good:**
```python
logger.info("agent_action", agent_id="agent_001", action="move", success=True)
```

**Bad:**
```python
logger.info(f"Agent {agent_id} performed {action} successfully")
```

### 2. Bind Context Early

```python
# At simulation start
bind_context(simulation_id="sim_001")

# In agent code - context is automatically included
logger.info("agent_moved", agent_id=self.id)
```

### 3. Use Appropriate Log Levels

- **DEBUG**: Detailed diagnostic information (sampling recommended)
- **INFO**: General informational messages about progress
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error events that might still allow operation to continue
- **CRITICAL**: Serious errors that may prevent operation

### 4. Sample High-Frequency Logs

```python
# Enable sampling for debug logs in tight loops
configure_logging(
    enable_sampling=True,
    sample_rate=0.01,  # 1% sampling
    events_to_sample={"agent_action", "resource_update"}
)
```

### 5. Use Specialized Loggers

Use `AgentLogger`, `DatabaseLogger`, etc. for domain-specific logging with automatic context binding.

### 6. Don't Log Sensitive Data

The logging system automatically redacts fields containing: `password`, `token`, `secret`, `api_key`, `auth`, `key`.

### 7. Measure Performance

```python
@log_performance(slow_threshold_ms=100.0)
def expensive_operation():
    # Automatically warns if operation takes > 100ms
    pass
```

### 8. Clean Up Context

```python
# Bad - context leaks
bind_context(temp_value=123)
run_operation()
# temp_value still in context

# Good - context cleaned up
with log_context(temp_value=123):
    run_operation()
# temp_value automatically removed
```

## Migration from Standard Logging

If you have existing code using Python's standard `logging` module:

```python
# Old code
import logging
logger = logging.getLogger(__name__)
logger.info(f"Processing agent {agent_id}")

# New code
from farm.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("processing_agent", agent_id=agent_id)
```

The logging system is compatible with existing standard library logging calls, but structured logging provides better queryability and analysis.

## Performance Considerations

1. **Context Binding**: Fast operation, use liberally
2. **Log Sampling**: Essential for high-frequency events (>1000 events/second)
3. **Async Logging**: Use for I/O-intensive logging in performance-critical paths
4. **Caller Info**: Expensive, only enable in development when needed
5. **Metrics Tracking**: Small overhead, acceptable for most use cases

## Environment-Specific Configurations

### Development

```python
configure_logging(
    environment="development",
    log_level="DEBUG",
    enable_colors=True,
    include_caller_info=True  # Helpful for debugging
)
```

### Production

```python
configure_logging(
    environment="production",
    log_level="INFO",
    json_logs=True,
    enable_metrics=True,
    enable_log_rotation=True,
    enable_sampling=True,
    sample_rate=0.1
)
```

### Testing

```python
configure_logging(
    environment="testing",
    log_level="WARNING",  # Reduce noise in tests
    enable_colors=False
)
```

## Troubleshooting

### Logs Not Appearing

1. Check log level: `configure_logging(log_level="DEBUG")`
2. Ensure logging is configured: `configure_logging()` must be called before logging
3. Check if event is being sampled: Disable sampling or increase sample_rate

### Performance Issues

1. Enable sampling for high-frequency logs
2. Use async logging for I/O-intensive operations
3. Disable caller info in production
4. Use appropriate log levels (DEBUG logs are filtered efficiently)

### Context Not Appearing

1. Ensure `bind_context()` is called before logging
2. Check if using thread-local context in multi-threaded code
3. Verify context isn't cleared prematurely

## API Reference

See individual module docstrings for detailed API documentation:

- `farm.utils.logging.config` - Core configuration and processors
- `farm.utils.logging.utils` - Utilities, decorators, and specialized loggers
- `farm.utils.logging.simulation` - Type-safe simulation logging
- `farm.utils.logging.async_logger` - Async logging support
- `farm.utils.logging.correlation` - Correlation ID tracking
- `farm.utils.logging.test_helpers` - Testing utilities

## Contributing

When adding new logging features:

1. Follow structured logging patterns (event names + context)
2. Add tests in `tests/test_logging/`
3. Update this README with examples
4. Consider performance implications
5. Add type hints and docstrings

## License

Part of the AgentFarm project. See main project LICENSE for details.

