# Structured Logging with Structlog

AgentFarm now uses **structured logging** powered by [structlog](https://www.structlog.org/) to provide rich, context-aware, machine-readable logs.

## 🎯 Why Structured Logging?

- **Searchable & Filterable**: Every log is a structured dict with searchable fields
- **Rich Context**: Automatically include simulation_id, step_number, agent_id, etc.
- **Machine Readable**: JSON output for analysis tools and log aggregators  
- **Better Debugging**: Easily trace events across distributed systems
- **Performance Metrics**: Built-in performance tracking and slow operation detection
- **Type Safety**: Better IDE support and fewer string formatting bugs

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

This installs `structlog>=24.1.0` and `python-json-logger>=2.0.0`.

### Basic Usage

```python
from farm.utils import configure_logging, get_logger

# Configure at startup
configure_logging(environment="development", log_level="INFO")

# Get logger
logger = get_logger(__name__)

# Log events with context
logger.info("simulation_started", num_agents=100, num_steps=1000)
logger.warning("resource_low", resource_id="res_001", level=10)
logger.error("agent_died", agent_id="agent_123", cause="starvation")
```

### Output

**Development (colored console):**
```
2025-10-01T12:34:56.789Z [info     ] simulation_started    num_agents=100 num_steps=1000
2025-10-01T12:34:57.123Z [warning  ] resource_low          resource_id=res_001 level=10
2025-10-01T12:34:58.456Z [error    ] agent_died            agent_id=agent_123 cause=starvation
```

**Production (JSON):**
```json
{"timestamp": "2025-10-01T12:34:56.789Z", "level": "info", "event": "simulation_started", "num_agents": 100, "num_steps": 1000, "logger": "farm.simulation"}
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Quick Reference](docs/LOGGING_QUICK_REFERENCE.md) | Cheat sheet for common logging patterns |
| [Full Guide](docs/logging_guide.md) | Comprehensive guide with examples |
| [Migration Checklist](docs/LOGGING_MIGRATION.md) | Step-by-step migration from standard logging |
| [Examples](examples/logging_examples.py) | Runnable examples demonstrating all features |

## 🔧 Configuration

Configure logging at application startup:

```python
from farm.utils import configure_logging

configure_logging(
    environment="development",  # "development", "production", "testing"
    log_dir="logs",            # Directory for log files
    log_level="INFO",          # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    json_logs=False,           # True for JSON output
    enable_colors=True,        # Colored console output
)
```

### CLI Arguments

```bash
# Development with debug logs
python run_simulation.py --log-level DEBUG --steps 1000

# Production with JSON logs
python run_simulation.py --environment production --json-logs --steps 1000
```

## 🎨 Key Features

### 1. Context Binding

Automatically include context in all logs:

```python
from farm.utils import bind_context, log_context

# Global context
bind_context(simulation_id="sim_001")
logger.info("event")  # Includes simulation_id

# Scoped context
with log_context(simulation_id="sim_001"):
    logger.info("event")  # Includes simulation_id
# Auto-cleaned up
```

### 2. Context Managers

```python
from farm.utils import log_simulation, log_step

with log_simulation(simulation_id="sim_001", num_agents=100):
    for step in range(1000):
        with log_step(step_number=step):
            process_step()  # All logs include simulation_id and step
```

### 3. Performance Logging

```python
from farm.utils import log_performance

@log_performance(operation_name="rebuild_index", slow_threshold_ms=100.0)
def rebuild_spatial_index():
    # Automatically logs duration and warns if slow
    pass
```

### 4. Specialized Loggers

```python
from farm.utils import AgentLogger

agent_logger = AgentLogger(agent_id="agent_001", agent_type="system")
agent_logger.log_action("move", success=True, reward=0.5, position=(10, 20))
agent_logger.log_interaction("share", target_id="agent_002", amount=10)
agent_logger.log_death(cause="starvation", lifetime_steps=342)
```

### 5. Log Sampling

Reduce noise from high-frequency events:

```python
from farm.utils import LogSampler

sampler = LogSampler(sample_rate=0.1)  # Log 10% of events

for i in range(10000):
    if sampler.should_log():
        logger.debug("high_frequency_event", iteration=i)
```

## 📊 Log Analysis

JSON logs can be easily analyzed:

```python
import json
import pandas as pd

# Load logs as DataFrame
with open("logs/application.json.log") as f:
    logs = [json.loads(line) for line in f]
    df = pd.DataFrame(logs)

# Analyze
slow_operations = df[df['duration_ms'] > 100]
errors_by_type = df[df['level'] == 'error'].groupby('error_type').size()
agent_actions = df[df['event'] == 'agent_action']['action_type'].value_counts()
```

## 🔄 Migration from Standard Logging

### Before (Standard Logging)

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"Simulation started with {num_agents} agents")
logger.error(f"Error: {e}", exc_info=True)
```

### After (Structured Logging)

```python
from farm.utils import configure_logging, get_logger

configure_logging(environment="development", log_level="INFO")
logger = get_logger(__name__)

logger.info("simulation_started", num_agents=num_agents)
logger.error(
    "error_occurred",
    error_type=type(e).__name__,
    error_message=str(e),
    exc_info=True,
)
```

## ✅ Best Practices

**DO:**
- ✅ Use event names: `logger.info("simulation_started", ...)`
- ✅ Include rich context: `logger.info("event", agent_id=id, position=pos)`
- ✅ Bind context early: `bind_context(simulation_id=sim_id)`
- ✅ Use appropriate log levels
- ✅ Sample high-frequency logs: `LogSampler(sample_rate=0.1)`

**DON'T:**
- ❌ Use f-strings: `logger.info(f"Event: {value}")`
- ❌ Log messages without context: `logger.info("Something happened")`
- ❌ Log sensitive data without censoring
- ❌ Log large objects without truncation

## 🧪 Examples

Run the comprehensive examples:

```bash
python examples/logging_examples.py
```

This demonstrates:
- Basic logging
- Context binding
- Context managers
- Performance decorators
- Agent logger
- Log sampling
- Error logging
- Nested contexts

## 📈 Current Status

**Phase 1: Foundation** ✅ COMPLETED

- ✅ Structlog integration
- ✅ Centralized configuration
- ✅ Logging utilities and helpers
- ✅ Updated entry points (main.py, run_simulation.py)
- ✅ Documentation and examples

**Phase 2: Core Modules** 🔄 IN PROGRESS

See [LOGGING_MIGRATION.md](docs/LOGGING_MIGRATION.md) for the complete migration checklist.

**Files with logging:** 91  
**Files migrated:** 2 (main.py, run_simulation.py)  
**Remaining:** 89

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Logs not appearing | Call `configure_logging()` at startup |
| Context not persisting | Use `bind_context()` or context managers |
| Too many logs | Use `LogSampler` or increase log level |
| Performance impact | Sample high-frequency logs, use DEBUG level |
| Third-party noise | Adjust log levels in `configure_logging()` |

## 🔗 Resources

- [Structlog Documentation](https://www.structlog.org/)
- [Logging Guide](docs/logging_guide.md)
- [Quick Reference](docs/LOGGING_QUICK_REFERENCE.md)
- [Migration Guide](docs/LOGGING_MIGRATION.md)
- [Examples](examples/logging_examples.py)

## 📝 License

Same as AgentFarm project license.
