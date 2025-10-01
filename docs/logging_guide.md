# Structured Logging Guide for AgentFarm

This guide explains how to use the new structured logging system powered by `structlog` in AgentFarm.

## Table of Contents

- [Why Structured Logging?](#why-structured-logging)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Context Binding](#context-binding)
- [Performance Logging](#performance-logging)
- [Specialized Loggers](#specialized-loggers)
- [Best Practices](#best-practices)
- [Migration from Standard Logging](#migration-from-standard-logging)

## Why Structured Logging?

Structured logging provides several advantages over traditional text-based logging:

- **Searchable & Filterable**: Every log entry is a structured dict with searchable fields
- **Rich Context**: Automatically include simulation_id, step_number, agent_id, etc.
- **Machine Readable**: JSON output for analysis tools and log aggregators
- **Better Debugging**: Easily trace events across distributed systems
- **Performance Metrics**: Built-in performance tracking and slow operation detection
- **Type Safety**: Better IDE support and fewer string formatting bugs

## Quick Start

### 1. Basic Logging

```python
from farm.utils import get_logger

logger = get_logger(__name__)

# Simple logging
logger.info("simulation_started", num_agents=100, num_steps=1000)
logger.warning("resource_low", resource_id="res_001", level=10)
logger.error("agent_died", agent_id="agent_123", cause="starvation")
```

Output (development mode):
```
2025-10-01T12:34:56.789Z [info     ] simulation_started    num_agents=100 num_steps=1000
2025-10-01T12:34:57.123Z [warning  ] resource_low          resource_id=res_001 level=10
2025-10-01T12:34:58.456Z [error    ] agent_died            agent_id=agent_123 cause=starvation
```

### 2. Context Binding

```python
from farm.utils import get_logger, bind_context, unbind_context

logger = get_logger(__name__)

# Bind context that persists across logs
bind_context(simulation_id="sim_001", experiment_id="exp_42")

logger.info("step_started", step=1)  # Includes simulation_id and experiment_id
logger.info("step_started", step=2)  # Still includes the context

# Unbind when done
unbind_context("simulation_id", "experiment_id")
```

### 3. Context Managers

```python
from farm.utils import log_simulation, log_step

# Simulation context
with log_simulation(simulation_id="sim_001", num_agents=100, num_steps=1000):
    # All logs here include simulation_id
    
    for step in range(1000):
        with log_step(step_number=step):
            # All logs here include both simulation_id and step
            process_agents()
```

## Configuration

Configure logging at application startup:

```python
from farm.utils import configure_logging

# Development: Pretty console output with colors
configure_logging(
    environment="development",
    log_dir="logs",
    log_level="DEBUG",
    enable_colors=True,
)

# Production: JSON logs for analysis
configure_logging(
    environment="production",
    log_dir="/var/log/agentfarm",
    log_level="INFO",
    json_logs=True,
)

# Testing: Simple format, no colors
configure_logging(
    environment="testing",
    log_level="WARNING",
)
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `environment` | Environment name (development/production/testing) | "development" |
| `log_dir` | Directory for log files (None = console only) | None |
| `log_level` | Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL) | "INFO" |
| `json_logs` | Output JSON formatted logs to file | False |
| `enable_colors` | Enable colored console output | True |
| `include_caller_info` | Include file/line/function info | True |

## Basic Usage

### Event-Style Logging

Instead of traditional messages, use event names with context:

```python
# ❌ Old way (traditional logging)
logger.info(f"Agent {agent_id} moved from {old_pos} to {new_pos}")

# ✅ New way (structured logging)
logger.info(
    "agent_moved",
    agent_id=agent_id,
    old_position=old_pos,
    new_position=new_pos,
)
```

### Log Levels

```python
logger.debug("detailed_operation", operation="rebuild_spatial_index", items=1000)
logger.info("simulation_started", config=config_dict)
logger.warning("performance_degraded", fps=15, expected_fps=30)
logger.error("database_error", error=str(e), query=sql)
logger.critical("system_failure", reason="out_of_memory")
```

### Exception Logging

```python
try:
    risky_operation()
except Exception as e:
    logger.error(
        "operation_failed",
        operation="risky_operation",
        error_type=type(e).__name__,
        error_message=str(e),
        exc_info=True,  # Include full traceback
    )
```

## Context Binding

### Global Context

Use `bind_context()` to add fields to all subsequent logs:

```python
from farm.utils import bind_context, unbind_context, clear_context

# Bind context
bind_context(simulation_id="sim_001", user_id="user_123")

logger.info("event_occurred")  # Includes simulation_id and user_id

# Unbind specific keys
unbind_context("user_id")

# Clear all context
clear_context()
```

### Scoped Context

Use context managers for automatic cleanup:

```python
from farm.utils import log_context

with log_context(simulation_id="sim_001", step=42):
    logger.info("agent_action", action="move")
    # Includes simulation_id and step
    
# Context automatically cleaned up after exiting block
logger.info("other_event")  # No simulation_id or step
```

### Logger-Level Binding

```python
# Create a logger with permanent bindings
logger = get_logger(__name__).bind(
    component="spatial_index",
    version="2.0",
)

# All logs from this logger include component and version
logger.info("index_rebuilt", num_items=5000)
```

## Performance Logging

### Performance Decorator

Automatically log function execution time:

```python
from farm.utils import log_performance

@log_performance(operation_name="agent_step", slow_threshold_ms=50.0)
def step(self, action):
    # Implementation
    pass
```

Output:
```
[debug] operation_complete    operation=agent_step duration_ms=23.45 status=success
```

If operation is slow:
```
[warning] operation_slow      operation=agent_step duration_ms=156.78 status=success performance_warning=slow_operation
```

### Error Logging Decorator

Automatically log unhandled exceptions:

```python
from farm.utils import log_errors

@log_errors()
def risky_function(data):
    # If this raises an exception, it's automatically logged
    process(data)
```

## Specialized Loggers

### Simulation Logging

```python
from farm.utils import log_simulation

with log_simulation(
    simulation_id="sim_001",
    num_agents=100,
    num_steps=1000,
):
    run_simulation()
```

Logs:
```
[info] simulation_started    simulation_id=sim_001 num_agents=100 num_steps=1000
... simulation logs (all include simulation_id) ...
[info] simulation_completed  simulation_id=sim_001 duration_seconds=45.67
```

### Step Logging

```python
from farm.utils import log_step

for step_num in range(1000):
    with log_step(step_number=step_num, simulation_id="sim_001"):
        # All logs include step and simulation_id
        process_step()
```

### Experiment Logging

```python
from farm.utils import log_experiment

with log_experiment(
    experiment_id="exp_001",
    experiment_name="parameter_sweep",
    num_iterations=10,
):
    run_experiment()
```

### Agent Logger

Specialized logger for agent events:

```python
from farm.utils import AgentLogger

class MyAgent:
    def __init__(self, agent_id, agent_type):
        self.logger = AgentLogger(agent_id, agent_type)
    
    def move(self, new_position):
        self.logger.log_action(
            action_type="move",
            success=True,
            reward=0.5,
            position=new_position,
        )
    
    def interact(self, other_agent):
        self.logger.log_interaction(
            interaction_type="share_resource",
            target_id=other_agent.id,
            amount=10,
        )
    
    def die(self, cause):
        self.logger.log_death(cause=cause, final_resources=self.resources)
```

### Log Sampling

Reduce log volume for high-frequency events:

```python
from farm.utils import LogSampler

# Log only 10% of events
sampler = LogSampler(sample_rate=0.1)

for i in range(10000):
    if sampler.should_log():
        logger.debug("high_frequency_event", iteration=i)

# Or use minimum interval (log at most once per 100ms)
sampler = LogSampler(min_interval_ms=100)
```

## Best Practices

### 1. Use Event Names, Not Messages

```python
# ❌ Don't
logger.info("The simulation has started")

# ✅ Do
logger.info("simulation_started")
```

### 2. Include Rich Context

```python
# ❌ Don't
logger.error("Agent failed")

# ✅ Do
logger.error(
    "agent_action_failed",
    agent_id=agent.id,
    agent_type=agent.type,
    action=action,
    reason="insufficient_resources",
    current_resources=agent.resources,
)
```

### 3. Use Structured Data

```python
# ❌ Don't
logger.info(f"Position: ({x}, {y})")

# ✅ Do
logger.info("agent_position", x=x, y=y)
# Or
logger.info("agent_position", position=(x, y))
```

### 4. Bind Context Early

```python
# At simulation start
bind_context(simulation_id=sim_id)

# At agent creation
agent_logger = logger.bind(agent_id=agent.id, agent_type=agent.type)

# In loops
with log_step(step_number=step):
    # Process step
    pass
```

### 5. Use Appropriate Log Levels

- **DEBUG**: Detailed diagnostic information (spatial index rebuilds, cache hits)
- **INFO**: General informational events (simulation started, agent born)
- **WARNING**: Warning events that might need attention (low resources, slow operations)
- **ERROR**: Error events that need investigation (failed actions, database errors)
- **CRITICAL**: Critical failures requiring immediate action (system crash)

### 6. Don't Log Sensitive Data

The logging system automatically censors common sensitive fields (password, token, secret, api_key), but be careful:

```python
# ❌ Don't
logger.info("user_login", password=password)

# ✅ Do
logger.info("user_login", username=username)  # password auto-censored
```

### 7. Sample High-Frequency Logs

```python
# For per-agent, per-step operations
sampler = LogSampler(sample_rate=0.01)  # 1% of events

for agent in agents:
    if sampler.should_log():
        logger.debug("agent_state", agent_id=agent.id, state=agent.state)
```

## Migration from Standard Logging

### Replace Imports

```python
# ❌ Old
import logging
logger = logging.getLogger(__name__)

# ✅ New
from farm.utils import get_logger
logger = get_logger(__name__)
```

### Replace basicConfig

```python
# ❌ Old
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ✅ New
from farm.utils import configure_logging
configure_logging(
    environment="development",
    log_level="INFO",
)
```

### Replace String Formatting

```python
# ❌ Old
logger.info(f"Agent {agent_id} moved to {position}")
logger.error("Error: %s", str(e))

# ✅ New
logger.info("agent_moved", agent_id=agent_id, position=position)
logger.error("error_occurred", error=str(e), error_type=type(e).__name__)
```

### Replace exc_info

```python
# ❌ Old
logger.error(f"Operation failed: {e}", exc_info=True)

# ✅ New
logger.error(
    "operation_failed",
    error_type=type(e).__name__,
    error_message=str(e),
    exc_info=True,
)
```

## Output Formats

### Development (Console)
```
2025-10-01T12:34:56.789Z [info     ] simulation_started    simulation_id=sim_001 num_agents=100
```

### Production (JSON)
```json
{
  "timestamp": "2025-10-01T12:34:56.789Z",
  "level": "info",
  "event": "simulation_started",
  "simulation_id": "sim_001",
  "num_agents": 100,
  "logger": "farm.simulation",
  "filename": "simulation.py",
  "func_name": "run_simulation",
  "lineno": 123
}
```

### Testing (Simple)
```
[info] simulation_started simulation_id=sim_001 num_agents=100
```

## Integration with Analysis Tools

The JSON log format can be easily imported into analysis tools:

```python
import json
import pandas as pd

# Load logs as DataFrame
with open("logs/application.json.log") as f:
    logs = [json.loads(line) for line in f]
    df = pd.DataFrame(logs)

# Analyze
slow_ops = df[df['duration_ms'] > 100]
errors_by_type = df[df['level'] == 'error'].groupby('error_type').size()
```

## Examples

See the [examples directory](../examples/logging_examples.py) for complete working examples.

## Troubleshooting

### Logs not appearing

Ensure logging is configured before any other imports:
```python
from farm.utils import configure_logging
configure_logging()  # Must be first!
```

### Context not persisting

Use `bind_context()` for global context or context managers for scoped context:
```python
from farm.utils import bind_context
bind_context(simulation_id="sim_001")  # Persists globally
```

### Performance impact

Use sampling for high-frequency logs:
```python
from farm.utils import LogSampler
sampler = LogSampler(sample_rate=0.1)
```

### Third-party library noise

The logging config automatically reduces noise from common libraries, but you can adjust:
```python
import logging
logging.getLogger("noisy_library").setLevel(logging.ERROR)
```
