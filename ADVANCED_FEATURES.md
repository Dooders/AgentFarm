# Advanced Structlog Features

This document describes advanced features available in the AgentFarm structured logging system.

## 🚀 Advanced Loggers

### DatabaseLogger

Specialized logger for database operations with automatic context binding.

```python
from farm.utils import DatabaseLogger

# Create database logger
db_logger = DatabaseLogger(
    db_path="/path/to/simulation.db",
    simulation_id="sim_001"
)

# Log queries
db_logger.log_query(
    query_type="select",
    table="agents",
    duration_ms=12.3,
    rows=100,
)

# Log transactions
db_logger.log_transaction(
    status="commit",
    duration_ms=5.2,
    rows_affected=50,
)
```

**Output:**
```json
{
  "event": "database_query",
  "query_type": "select",
  "table": "agents",
  "duration_ms": 12.3,
  "rows": 100,
  "db_path": "/path/to/simulation.db",
  "simulation_id": "sim_001"
}
```

### PerformanceMonitor

Context manager for detailed performance monitoring with checkpoints.

```python
from farm.utils import PerformanceMonitor

with PerformanceMonitor("data_processing") as monitor:
    # Load data
    data = load_data()
    monitor.checkpoint("load_data")
    
    # Transform
    transformed = transform(data)
    monitor.checkpoint("transform_data")
    
    # Save
    save(transformed)
    monitor.checkpoint("save_results")

# Automatically logs:
# - operation_completed with total duration
# - All checkpoints with elapsed times
```

**Output:**
```json
{
  "event": "checkpoint",
  "operation": "data_processing",
  "checkpoint": "load_data",
  "elapsed_ms": 15.2
}
{
  "event": "checkpoint",
  "operation": "data_processing",
  "checkpoint": "transform_data",
  "elapsed_ms": 28.7
}
{
  "event": "operation_completed",
  "operation": "data_processing",
  "duration_ms": 45.3,
  "checkpoints": 3
}
```

## 🔧 Enhanced Configuration

### Process/Thread Information

Enable process and thread IDs for parallel execution debugging:

```python
from farm.utils import configure_logging

configure_logging(
    environment="production",
    log_level="INFO",
    include_process_info=True,  # Add PID and thread info
)
```

**Output:**
```json
{
  "event": "simulation_started",
  "process_id": 12345,
  "thread_id": 139876543210,
  "thread_name": "MainThread",
  "simulation_id": "sim_001"
}
```

This is especially useful for:
- Parallel experiment runners
- Multi-threaded simulations
- Debugging race conditions
- Process isolation issues

## 🎯 Advanced Context Patterns

### Hierarchical Context

Build complex context hierarchies:

```python
from farm.utils import bind_context, log_context, get_logger

logger = get_logger(__name__)

# Level 1: Experiment
bind_context(experiment_id="exp_001", experiment_name="learning_rates")

with log_context(simulation_id="sim_001", iteration=1):
    # Level 2: Simulation
    
    with log_context(step=42):
        # Level 3: Step
        
        with log_context(agent_id="agent_123", agent_type="system"):
            # Level 4: Agent
            logger.info("decision_made", action="move", reward=0.5)
            # Includes: experiment_id, experiment_name, simulation_id, 
            #           iteration, step, agent_id, agent_type
```

### Temporary Context Override

Temporarily override context:

```python
from farm.utils import log_context, bind_context, get_logger

logger = get_logger(__name__)

bind_context(environment="production")

# Temporarily override for specific section
with log_context(environment="testing"):
    logger.info("test_event")  # Has environment="testing"

logger.info("normal_event")  # Has environment="production"
```

## 📊 Performance Optimization

### Smart Sampling Strategies

#### Rate-Based Sampling
```python
from farm.utils import LogSampler

# Log 1% of events
sampler = LogSampler(sample_rate=0.01)

for i in range(10000):
    if sampler.should_log():
        logger.debug("high_frequency_event", iteration=i)
```

#### Time-Based Sampling
```python
# Log at most once per 100ms
sampler = LogSampler(min_interval_ms=100)

for i in range(10000):
    if sampler.should_log():
        logger.debug("time_sampled_event", iteration=i)
    time.sleep(0.001)  # 1ms processing
```

#### Combined Sampling
```python
# 10% rate AND minimum 50ms interval
sampler = LogSampler(sample_rate=0.1, min_interval_ms=50)
```

### Conditional Logging

```python
# Only log every Nth iteration
for i in range(1000):
    if i % 100 == 0:  # Every 100th
        logger.info("milestone", iteration=i)

# Only log when condition met
if agent_count > threshold:
    logger.warning("high_agent_count", count=agent_count, threshold=threshold)
```

## 🎨 Advanced Formatting

### Custom Event Structure

Create rich, nested event data:

```python
logger.info(
    "agent_state_snapshot",
    agent_id="agent_001",
    state={
        "position": (10.5, 20.3),
        "health": 85,
        "resources": 42,
        "is_defending": False,
    },
    neighbors=[
        {"id": "agent_002", "distance": 5.2},
        {"id": "agent_003", "distance": 8.7},
    ],
    metrics={
        "total_reward": 125.5,
        "actions_taken": 342,
        "lifetime_steps": 500,
    },
)
```

### Truncation for Large Objects

```python
# Truncate large strings
large_data = "x" * 10000
logger.info("data_processed", data=str(large_data)[:200] + "...")

# Summarize lists
large_list = list(range(1000))
logger.info("items_processed", 
            item_count=len(large_list),
            sample=large_list[:5])
```

## 🔍 Log Analysis Patterns

### Complex Filtering

```python
import json
import pandas as pd

with open("logs/application.json.log") as f:
    df = pd.DataFrame([json.loads(line) for line in f])

# Multi-condition filtering
critical_errors = df[
    (df['level'] == 'error') &
    (df['simulation_id'] == 'sim_001') &
    (df['step'] > 500)
]

# Agent-specific analysis
agent_logs = df[df['agent_id'] == 'agent_123']
agent_timeline = agent_logs.sort_values('timestamp')
```

### Performance Analysis

```python
# Find bottlenecks
if 'duration_ms' in df.columns:
    slow_operations = df[df['duration_ms'] > 100].sort_values('duration_ms', ascending=False)
    
    # Group by operation
    perf_summary = df.groupby('operation')['duration_ms'].agg([
        'count', 'mean', 'median', 'min', 'max', 'std'
    ])
    
    print("Performance Summary:")
    print(perf_summary)

# Time series analysis
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Events per minute
events_per_min = df.resample('1min').size()
```

### Error Pattern Detection

```python
# Error frequency by type
error_df = df[df['level'] == 'error']
error_patterns = error_df.groupby(['error_type', 'event']).size()

# Error timeline
error_timeline = error_df.resample('1min').size()

# Correlation analysis
if 'step' in error_df.columns:
    step_error_correlation = error_df.groupby('step').size()
```

## 🛠️ Production Best Practices

### Structured Error Reporting

```python
class CustomError(Exception):
    def __init__(self, message, context):
        super().__init__(message)
        self.context = context

try:
    raise CustomError("Operation failed", {"user_id": 123, "action": "delete"})
except CustomError as e:
    logger.error(
        "custom_error_occurred",
        error_type=type(e).__name__,
        error_message=str(e),
        error_context=e.context,
        exc_info=True,
    )
```

### Async Logging Support

```python
import asyncio
from farm.utils import get_logger, bind_context

logger = get_logger(__name__)

async def async_operation():
    # Context binding works across async boundaries
    bind_context(correlation_id="req_12345")
    
    logger.info("async_operation_started")
    await asyncio.sleep(0.1)
    logger.info("async_operation_completed")
    # Both logs include correlation_id
```

### Multi-Process Logging

```python
from multiprocessing import Process
from farm.utils import configure_logging, get_logger

def worker_process(worker_id):
    # Each process configures its own logging
    configure_logging(
        environment="production",
        log_level="INFO",
        include_process_info=True,  # Important for multi-process!
    )
    
    logger = get_logger(__name__)
    logger = logger.bind(worker_id=worker_id)
    
    logger.info("worker_started")
    # Do work...
    logger.info("worker_completed")

# Logs will include process_id to distinguish workers
```

## 📈 Advanced Analysis

### Log Aggregation

```python
import pandas as pd
import json

# Load multiple log files
all_logs = []
for log_file in ["logs/sim_001.json", "logs/sim_002.json"]:
    with open(log_file) as f:
        all_logs.extend([json.loads(line) for line in f])

df = pd.DataFrame(all_logs)

# Cross-simulation analysis
simulations = df['simulation_id'].unique()
for sim_id in simulations:
    sim_df = df[df['simulation_id'] == sim_id]
    print(f"\nSimulation {sim_id}:")
    print(f"  Duration: {sim_df['timestamp'].max() - sim_df['timestamp'].min()}")
    print(f"  Events: {len(sim_df)}")
    print(f"  Errors: {len(sim_df[sim_df['level'] == 'error'])}")
```

### Statistical Analysis

```python
# Event frequency distribution
from scipy import stats

event_counts = df['event'].value_counts()
print(f"Mean events per type: {event_counts.mean()}")
print(f"Std dev: {event_counts.std()}")

# Performance distribution
if 'duration_ms' in df.columns:
    duration_stats = df['duration_ms'].describe()
    print("\nDuration statistics:")
    print(duration_stats)
    
    # Identify outliers (> 2 std dev)
    mean = df['duration_ms'].mean()
    std = df['duration_ms'].std()
    outliers = df[df['duration_ms'] > mean + 2*std]
```

## 🔐 Enhanced Security

### Custom Sensitive Fields

Add your own sensitive field patterns:

```python
# In logging_config.py, customize censor_sensitive_data:
def censor_sensitive_data(logger, method_name, event_dict):
    sensitive_keys = {
        "password", "token", "secret", "api_key", "auth",
        "credit_card", "ssn", "private_key",  # Add more
    }
    
    for key in list(event_dict.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            event_dict[key] = "***REDACTED***"
    
    return event_dict
```

## 🎯 Production Deployment

### Log Rotation

```python
import logging.handlers
from farm.utils import configure_logging

# After basic configuration, add rotation
configure_logging(environment="production", log_dir="logs")

# Add rotating file handler
handler = logging.handlers.RotatingFileHandler(
    "logs/application.log",
    maxBytes=100*1024*1024,  # 100MB
    backupCount=10,
)
logging.root.addHandler(handler)
```

### Centralized Logging Services

#### ELK Stack Integration
```python
# Configure JSON logs for Logstash
configure_logging(
    environment="production",
    log_dir="/var/log/agentfarm",
    json_logs=True,
)

# Logstash can tail application.json.log and forward to Elasticsearch
```

#### Datadog Integration
```python
# Use JSON logs with Datadog agent
configure_logging(
    environment="production",
    log_dir="/var/log/agentfarm",
    json_logs=True,
)

# Datadog agent config:
# logs:
#   - type: file
#     path: /var/log/agentfarm/application.json.log
#     service: agentfarm
#     source: python
```

## 💡 Tips & Tricks

### Dynamic Log Levels

```python
import logging
from farm.utils import get_logger

logger = get_logger(__name__)

# Dynamically adjust log level
if debug_mode:
    logging.root.setLevel(logging.DEBUG)
else:
    logging.root.setLevel(logging.INFO)
```

### Correlation IDs

```python
import uuid
from farm.utils import bind_context, get_logger

# Generate correlation ID for request tracking
correlation_id = str(uuid.uuid4())
bind_context(correlation_id=correlation_id)

# All logs in this execution will include correlation_id
```

### Metrics Aggregation

```python
from collections import Counter
from farm.utils import get_logger

logger = get_logger(__name__)

# Track metrics in code
metrics = Counter()

for agent in agents:
    action = agent.decide()
    metrics[action] += 1

# Log aggregated metrics
logger.info(
    "action_distribution",
    total_actions=sum(metrics.values()),
    distribution=dict(metrics),
)
```

## 🧪 Testing with Structured Logs

### Capturing Logs in Tests

```python
import structlog
from io import StringIO

# Capture logs during tests
output = StringIO()
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=lambda: structlog.PrintLoggerFactory(file=output),
)

# Run test
my_function()

# Parse captured logs
import json
logs = [json.loads(line) for line in output.getvalue().split('\n') if line]
assert any(log['event'] == 'expected_event' for log in logs)
```

### Mock Logging

```python
from unittest.mock import Mock, patch
from farm.utils import get_logger

def test_logging():
    with patch('farm.utils.logging_config.get_logger') as mock_logger:
        mock_logger.return_value.info = Mock()
        
        # Run code
        my_function()
        
        # Verify logging
        mock_logger.return_value.info.assert_called_with(
            "expected_event",
            key="value"
        )
```

## 📊 Real-Time Monitoring

### Log Tailing and Analysis

```bash
# Tail logs in real-time
tail -f logs/application.json.log | jq

# Monitor errors
tail -f logs/application.json.log | jq 'select(.level == "error")'

# Track specific simulation
tail -f logs/application.json.log | jq 'select(.simulation_id == "sim_001")'

# Performance monitoring
tail -f logs/application.json.log | jq 'select(.duration_ms > 100)'
```

### Live Dashboard

```python
import json
from collections import deque

# Simple live stats
stats = {
    'events': deque(maxlen=1000),
    'errors': 0,
    'warnings': 0,
}

with open("logs/application.json.log") as f:
    for line in f:
        log = json.loads(line)
        stats['events'].append(log)
        
        if log['level'] == 'error':
            stats['errors'] += 1
        elif log['level'] == 'warning':
            stats['warnings'] += 1

print(f"Recent events: {len(stats['events'])}")
print(f"Total errors: {stats['errors']}")
print(f"Total warnings: {stats['warnings']}")
```

## 🎓 Best Practices Summary

### ✅ DO

1. **Use DatabaseLogger** for database operations
2. **Use PerformanceMonitor** for complex operations with multiple stages
3. **Enable process_info** for parallel execution
4. **Sample high-frequency logs** (>1000/sec)
5. **Bind context early** and clear when done
6. **Include correlation IDs** for request tracing
7. **Log aggregated metrics** rather than individual events
8. **Use appropriate log levels** (DEBUG for verbose, INFO for important)
9. **Include units** in field names (duration_ms, size_bytes)
10. **Structure nested data** with dicts and lists

### ❌ DON'T

1. Don't log every iteration of tight loops without sampling
2. Don't include sensitive data without censoring
3. Don't log entire objects (truncate/summarize)
4. Don't use f-strings in log calls
5. Don't forget to unbind context when done
6. Don't mix logging styles within same module
7. Don't log at ERROR level for expected conditions
8. Don't forget exc_info=True for exceptions
9. Don't create too many logger instances (cache them)
10. Don't skip adding error_type and error_message

## 🚀 Production Checklist

- [ ] Configure JSON logs
- [ ] Enable log rotation
- [ ] Set appropriate log levels (INFO or WARNING)
- [ ] Enable process_info for parallel execution
- [ ] Configure sensitive data censoring
- [ ] Set up log aggregation (ELK, Datadog, etc.)
- [ ] Create log monitoring alerts
- [ ] Document custom events
- [ ] Test log volume under load
- [ ] Set up log retention policy

---

**Your structured logging system is now enterprise-grade!** 🎉
