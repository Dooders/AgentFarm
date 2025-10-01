# Structured Logging Quick Reference

Quick reference for using structured logging in AgentFarm.

## Setup

```python
from farm.utils import configure_logging, get_logger

# At application startup
configure_logging(environment="development", log_level="INFO")

# In your module
logger = get_logger(__name__)
```

## Basic Logging

```python
# Event-style logging (preferred)
logger.info("event_name", key1=value1, key2=value2)

# Common levels
logger.debug("detailed_info", ...)      # Diagnostic information
logger.info("important_event", ...)     # General events
logger.warning("warning_event", ...)    # Warnings
logger.error("error_event", ...)        # Errors
logger.critical("critical_event", ...)  # Critical failures
```

## Context Binding

```python
from farm.utils import bind_context, unbind_context, log_context

# Global context (persists across all logs)
bind_context(simulation_id="sim_001")
logger.info("event")  # Includes simulation_id
unbind_context("simulation_id")

# Scoped context (automatic cleanup)
with log_context(simulation_id="sim_001"):
    logger.info("event")  # Includes simulation_id
# Auto-cleaned up here

# Logger-level binding (permanent for this logger)
my_logger = logger.bind(component="spatial_index")
my_logger.info("event")  # Always includes component
```

## Context Managers

```python
from farm.utils import log_simulation, log_step, log_experiment

# Simulation
with log_simulation(simulation_id="sim_001", num_agents=100):
    run_simulation()

# Step
with log_step(step_number=42, simulation_id="sim_001"):
    process_step()

# Experiment
with log_experiment(experiment_id="exp_001", experiment_name="test"):
    run_experiment()
```

## Decorators

```python
from farm.utils import log_performance, log_errors

# Performance logging
@log_performance(operation_name="compute", slow_threshold_ms=100.0)
def my_function():
    pass

# Error logging
@log_errors()
def risky_function():
    pass
```

## Specialized Loggers

```python
from farm.utils import AgentLogger, LogSampler

# Agent logger
agent_logger = AgentLogger(agent_id="agent_001", agent_type="system")
agent_logger.log_action("move", success=True, reward=0.5)
agent_logger.log_interaction("share", target_id="agent_002")
agent_logger.log_death(cause="starvation")

# Log sampling (10% of events)
sampler = LogSampler(sample_rate=0.1)
if sampler.should_log():
    logger.debug("high_frequency_event")
```

## Error Logging

```python
try:
    risky_operation()
except Exception as e:
    logger.error(
        "operation_failed",
        error_type=type(e).__name__,
        error_message=str(e),
        exc_info=True,  # Include traceback
    )
```

## Common Patterns

### Simulation Start
```python
with log_simulation(simulation_id="sim_001", num_agents=100, num_steps=1000):
    logger.info("environment_initialized", grid_size=(100, 100))
    for step in range(num_steps):
        with log_step(step_number=step):
            process_step()
```

### Agent Actions
```python
agent_logger = AgentLogger(agent_id, agent_type)
agent_logger.log_action(
    action_type="move",
    success=True,
    reward=0.5,
    position=(10, 20),
)
```

### Database Operations
```python
logger.info("database_query_started", query_type="select", table="agents")
# ... execute query
logger.info("database_query_completed", duration_ms=45.2, rows_returned=100)
```

### Performance Tracking
```python
@log_performance(operation_name="spatial_index_rebuild", slow_threshold_ms=100.0)
def rebuild_index():
    # Will automatically log duration and warn if > 100ms
    pass
```

## Migration Quick Guide

```python
# OLD
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="...")
logger.info(f"Agent {agent_id} moved to {pos}")
logger.error(f"Error: {e}", exc_info=True)

# NEW
from farm.utils import get_logger, configure_logging
configure_logging(environment="development", log_level="INFO")
logger = get_logger(__name__)
logger.info("agent_moved", agent_id=agent_id, position=pos)
logger.error("error_occurred", error_type=type(e).__name__, 
            error_message=str(e), exc_info=True)
```

## Best Practices

✅ **DO:**
- Use event names: `logger.info("simulation_started", ...)`
- Include rich context: `logger.info("event", agent_id=id, position=pos, ...)`
- Bind context early: `bind_context(simulation_id=sim_id)`
- Use appropriate log levels
- Sample high-frequency logs: `LogSampler(sample_rate=0.1)`

❌ **DON'T:**
- Use f-strings: `logger.info(f"Event: {value}")`  
- Log messages without context: `logger.info("Something happened")`
- Log sensitive data: `logger.info("login", password=pwd)`
- Log large objects without truncation
- Skip error context: `logger.error("failed")`

## Output Formats

### Development Console
```
2025-10-01T12:34:56Z [info] simulation_started simulation_id=sim_001 num_agents=100
```

### Production JSON
```json
{"timestamp": "2025-10-01T12:34:56Z", "level": "info", "event": "simulation_started", "simulation_id": "sim_001", "num_agents": 100}
```

## Common Fields

Include these fields for consistency:

| Field | When | Example |
|-------|------|---------|
| `simulation_id` | All simulation logs | `"sim_001"` |
| `step` or `step_number` | Per-step logs | `42` |
| `agent_id` | Agent-related logs | `"agent_123"` |
| `agent_type` | Agent logs | `"system"` |
| `error_type` | Error logs | `"ValueError"` |
| `error_message` | Error logs | `"Invalid input"` |
| `duration_ms` | Performance logs | `123.45` |
| `operation` | Performance logs | `"rebuild_index"` |

## Configuration Options

```python
configure_logging(
    environment="development",     # "development", "production", "testing"
    log_dir="logs",               # Directory for log files (None = console only)
    log_level="INFO",             # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    json_logs=False,              # True for JSON output
    enable_colors=True,           # Colored console output
    include_caller_info=True,     # Include file/line/function info
)
```

## CLI Arguments (run_simulation.py)

```bash
python run_simulation.py \
  --log-level DEBUG \
  --json-logs \
  --environment production \
  --steps 1000
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Logs not appearing | Call `configure_logging()` first |
| Context not persisting | Use `bind_context()` or context managers |
| Too many logs | Use `LogSampler` or increase log level |
| Performance impact | Sample high-frequency logs, use DEBUG level |

## Links

- Full Guide: [docs/logging_guide.md](logging_guide.md)
- Examples: [examples/logging_examples.py](../examples/logging_examples.py)
- Migration: [docs/LOGGING_MIGRATION.md](LOGGING_MIGRATION.md)
