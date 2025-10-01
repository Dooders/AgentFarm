# Structured Logging Best Practices for AgentFarm

This document provides battle-tested best practices for using structured logging effectively in AgentFarm.

## 🎯 Core Principles

### 1. Events, Not Messages

**❌ Bad:**
```python
logger.info("The simulation has started")
logger.info("Agent moved")
```

**✅ Good:**
```python
logger.info("simulation_started", num_agents=100, num_steps=1000)
logger.info("agent_moved", agent_id="agent_123", from_pos=(0,0), to_pos=(10,20))
```

**Why:** Event names are searchable, filterable, and enable aggregation.

### 2. Rich Context

**❌ Bad:**
```python
logger.error("Failed")
logger.info("Done")
```

**✅ Good:**
```python
logger.error(
    "reproduction_failed",
    agent_id=self.agent_id,
    parent_resources=self.resource_level,
    required_resources=50,
    error_type="InsufficientResources",
)
logger.info(
    "simulation_completed",
    simulation_id=sim_id,
    final_agent_count=len(agents),
    duration_seconds=elapsed,
)
```

**Why:** Context enables debugging without code diving.

### 3. Bind Context Early

**❌ Bad:**
```python
logger.info("event1", simulation_id=sim_id, experiment_id=exp_id)
logger.info("event2", simulation_id=sim_id, experiment_id=exp_id)
logger.info("event3", simulation_id=sim_id, experiment_id=exp_id)
```

**✅ Good:**
```python
bind_context(simulation_id=sim_id, experiment_id=exp_id)
logger.info("event1")
logger.info("event2")
logger.info("event3")
```

**Why:** DRY principle, performance, and consistency.

## 📏 Naming Conventions

### Event Names

**Pattern:** `{component}_{action}_{status?}`

**Examples:**
```python
# Successes
"simulation_started"
"agent_moved"
"database_connected"
"algorithm_initialized"

# Failures
"simulation_failed"
"agent_died"
"database_connection_failed"
"algorithm_initialization_failed"

# States
"simulation_paused"
"agent_defending"
"database_transaction_pending"
```

### Field Names

**Use descriptive, consistent names:**

```python
# Good field names
logger.info("event",
    agent_id="agent_123",           # Not: id, agent, a_id
    duration_ms=123.45,             # Not: time, duration, dur
    simulation_id="sim_001",        # Not: sim, s_id, simulation
    num_agents=100,                 # Not: agents, count, total
    error_type="ValueError",        # Not: err, error, type
)
```

**Include units in names:**
- `duration_ms`, `duration_seconds`
- `size_bytes`, `size_mb`
- `distance_meters`, `distance_pixels`
- `rate_per_second`, `rate_per_minute`

## 🔤 Log Levels Guide

### DEBUG
**Use for:** Detailed diagnostic information, temporary debugging

```python
logger.debug("spatial_index_rebuilt", num_items=5000, duration_ms=45.2)
logger.debug("cache_hit", key="config_production", hit_count=15)
logger.debug("agent_state", agent_id="agent_001", position=(10,20), health=85)
```

### INFO
**Use for:** General informational events, important milestones

```python
logger.info("simulation_started", num_agents=100, num_steps=1000)
logger.info("experiment_completed", experiment_id="exp_001", duration_seconds=120)
logger.info("database_persisted", rows_copied=5000, duration_seconds=2.5)
```

### WARNING
**Use for:** Warning events that might need attention

```python
logger.warning("resource_low", resource_id="res_001", level=5, threshold=10)
logger.warning("performance_degraded", fps=15, expected_fps=30)
logger.warning("algorithm_unavailable", algorithm="ppo", using="fallback")
```

### ERROR
**Use for:** Error events that need investigation

```python
logger.error(
    "database_transaction_failed",
    error_type="IntegrityError",
    error_message="Duplicate key",
    table="agents",
    exc_info=True,
)
```

### CRITICAL
**Use for:** Critical failures requiring immediate action

```python
logger.critical(
    "system_out_of_memory",
    available_mb=50,
    required_mb=500,
    process_id=os.getpid(),
)
```

## 🎨 Context Management

### When to Use What

**Global Context** (`bind_context`):
```python
# Use for: Long-lived context that spans entire execution
bind_context(experiment_id="exp_001", user_id="user_123")
# All logs now include experiment_id and user_id
```

**Scoped Context** (`log_context`):
```python
# Use for: Temporary context for specific section
with log_context(simulation_id="sim_001"):
    run_simulation()
# Context auto-cleaned up after block
```

**Context Managers** (`log_simulation`, `log_step`):
```python
# Use for: Standard patterns with automatic timing
with log_simulation(simulation_id="sim_001", num_agents=100):
    # Auto-logs start/end with timing
    run_simulation()
```

**Logger Binding**:
```python
# Use for: Permanent context for logger instance
agent_logger = logger.bind(agent_id="agent_001", agent_type="system")
# This logger always includes agent_id and agent_type
```

## ⚡ Performance Optimization

### Sampling Strategies

**High-Frequency Logs (>1000/sec):**
```python
sampler = LogSampler(sample_rate=0.01)  # 1%
if sampler.should_log():
    logger.debug("per_agent_per_step_event")
```

**Medium-Frequency Logs (100-1000/sec):**
```python
sampler = LogSampler(sample_rate=0.1)  # 10%
```

**Time-Based Limiting:**
```python
# At most once per 100ms
sampler = LogSampler(min_interval_ms=100)
```

### Conditional Logging

```python
# Log milestones only
if step % 100 == 0:
    logger.info("milestone", step=step)

# Log only significant changes
if abs(new_value - old_value) > threshold:
    logger.info("significant_change", old=old_value, new=new_value)
```

### Lazy Evaluation

```python
# Don't compute expensive values if not logging
if logger.isEnabledFor(logging.DEBUG):
    expensive_data = compute_expensive_summary()
    logger.debug("detailed_state", data=expensive_data)
```

## 🐛 Error Logging

### Complete Error Context

**Always include:**
1. `error_type` - Exception class name
2. `error_message` - Exception message
3. `exc_info=True` - Full traceback
4. Relevant context (IDs, values, state)

```python
try:
    risky_operation()
except Exception as e:
    logger.error(
        "operation_failed",
        operation="risky_operation",
        agent_id=agent.id,
        simulation_id=sim_id,
        step=current_step,
        error_type=type(e).__name__,
        error_message=str(e),
        exc_info=True,  # Include full traceback
    )
    raise
```

### Error Recovery Logging

```python
try:
    primary_operation()
except PrimaryError as e:
    logger.warning(
        "primary_operation_failed_attempting_fallback",
        error_type=type(e).__name__,
        error_message=str(e),
    )
    try:
        fallback_operation()
        logger.info("fallback_operation_successful")
    except FallbackError as e2:
        logger.error(
            "fallback_operation_failed",
            primary_error=str(e),
            fallback_error=str(e2),
            exc_info=True,
        )
        raise
```

## 📊 Data Structures in Logs

### Complex Objects

**❌ Bad:**
```python
logger.info("state", state=entire_agent_object)  # Too large
```

**✅ Good:**
```python
logger.info(
    "agent_state_snapshot",
    agent_id=agent.id,
    position=agent.position,
    health=agent.health,
    resources=agent.resources,
    # Extract only relevant fields
)
```

### Lists and Arrays

**❌ Bad:**
```python
logger.info("agents", agents=list_of_1000_agents)  # Too verbose
```

**✅ Good:**
```python
logger.info(
    "agents_processed",
    agent_count=len(agents),
    agent_types=dict(Counter(a.type for a in agents)),
    sample_ids=[ a.id for a in agents[:5]],  # First 5 only
)
```

### Nested Data

**✅ Good - Well Structured:**
```python
logger.info(
    "combat_resolution",
    attacker={
        "id": attacker.id,
        "type": attacker.type,
        "damage": damage_dealt,
    },
    defender={
        "id": defender.id,
        "type": defender.type,
        "damage_taken": damage_taken,
        "health_remaining": defender.health,
    },
    outcome="defender_defeated",
)
```

## 🔄 Migration Patterns

### Replacing String Formatting

```python
# Old patterns → New patterns

# f-strings
logger.info(f"Agent {id} has {res} resources")
→ logger.info("agent_resources", agent_id=id, resources=res)

# % formatting
logger.info("Agent %s moved to %s", id, pos)
→ logger.info("agent_moved", agent_id=id, position=pos)

# .format()
logger.info("Agent {} died at step {}".format(id, step))
→ logger.info("agent_died", agent_id=id, step=step)
```

### Replacing Conditionals

```python
# Old
if agent.health < 20:
    logger.warning(f"Agent {agent.id} health critical: {agent.health}")

# New
if agent.health < 20:
    logger.warning(
        "agent_health_critical",
        agent_id=agent.id,
        health=agent.health,
        threshold=20,
    )
```

## 🏗️ Architecture Patterns

### Module-Level Logger

```python
# At module top
from farm.utils import get_logger

logger = get_logger(__name__)

# In functions/methods, use the module logger
def my_function():
    logger.info("function_called", arg=value)
```

### Class-Level Logger

```python
class MyClass:
    def __init__(self, id: str):
        self.logger = get_logger(__name__).bind(
            instance_id=id,
            class_name=self.__class__.__name__,
        )
    
    def method(self):
        # Automatically includes instance_id and class_name
        self.logger.info("method_called")
```

### Specialized Loggers

```python
# Use specialized loggers for specific domains
from farm.utils import AgentLogger, DatabaseLogger

# Agent operations
agent_logger = AgentLogger("agent_001", "system")
agent_logger.log_action("move", success=True, reward=0.5)

# Database operations
db_logger = DatabaseLogger("/path/to/db", "sim_001")
db_logger.log_query("select", "agents", duration_ms=12.3, rows=100)
```

## 📈 Observability Tips

### Service-Level Metrics

```python
# Track SLIs (Service Level Indicators)
logger.info(
    "simulation_metrics",
    simulation_id=sim_id,
    duration_seconds=elapsed,
    agent_survival_rate=survivors/total,
    average_steps_per_second=steps/elapsed,
    memory_usage_mb=memory_mb,
)
```

### Distributed Tracing

```python
# Use correlation IDs across services
import uuid

correlation_id = str(uuid.uuid4())
bind_context(correlation_id=correlation_id)

# Pass correlation_id to other services
api_call(correlation_id=correlation_id)

# All logs will be traceable
```

### Health Checks

```python
def health_check():
    """Log system health periodically."""
    import psutil
    
    logger.info(
        "health_check",
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        disk_percent=psutil.disk_usage('/').percent,
        active_agents=len(agents),
        simulation_step=current_step,
    )
```

## 🎯 Common Use Cases

### 1. Long-Running Operations

```python
with PerformanceMonitor("long_operation") as monitor:
    data = load_large_dataset()
    monitor.checkpoint("data_loaded")
    
    processed = process_data(data)
    monitor.checkpoint("data_processed")
    
    save_results(processed)
    monitor.checkpoint("results_saved")
```

### 2. Retry Logic

```python
for attempt in range(max_retries):
    try:
        result = unstable_operation()
        logger.info("operation_succeeded", attempt=attempt+1)
        break
    except Exception as e:
        if attempt < max_retries - 1:
            logger.warning(
                "operation_retry",
                attempt=attempt+1,
                max_retries=max_retries,
                error_type=type(e).__name__,
            )
        else:
            logger.error(
                "operation_failed_all_retries",
                attempts=max_retries,
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            raise
```

### 3. State Transitions

```python
# Log significant state changes
old_state = agent.state
agent.state = new_state

logger.info(
    "agent_state_changed",
    agent_id=agent.id,
    old_state=old_state,
    new_state=new_state,
    trigger="health_threshold",
)
```

### 4. Batch Operations

```python
# Log batch operations with summaries
results = process_batch(items)

logger.info(
    "batch_processed",
    total_items=len(items),
    successful=sum(1 for r in results if r.success),
    failed=sum(1 for r in results if not r.success),
    duration_ms=elapsed_ms,
)
```

## 🔐 Security Best Practices

### Never Log Sensitive Data

**❌ Never:**
```python
logger.info("user_login", username=username, password=password)
logger.info("api_call", api_key=api_key)
logger.info("payment", credit_card=cc_number)
```

**✅ Always:**
```python
logger.info("user_login", username=username)  # password auto-censored
logger.info("api_call")  # Don't log credentials
logger.info("payment", amount=amount, currency=currency)  # Not card details
```

### Sanitize User Input

```python
# Sanitize before logging
user_input = request.get("search_term")
sanitized = user_input[:100].replace('\n', ' ').replace('\r', '')

logger.info("search_performed", query=sanitized, results_count=count)
```

## 🧪 Testing Recommendations

### Test Logging in Unit Tests

```python
import json
from io import StringIO

def test_simulation_logging(caplog):
    """Test that simulation logs expected events."""
    from farm.utils import configure_logging, get_logger
    
    configure_logging(environment="testing", log_level="INFO")
    
    run_simulation(steps=10)
    
    # Check for expected log events
    logs = [json.loads(line) for line in caplog.text.split('\n') if line]
    assert any(log['event'] == 'simulation_started' for log in logs)
    assert any(log['event'] == 'simulation_completed' for log in logs)
```

### Mock Logging for Isolated Tests

```python
from unittest.mock import patch

def test_without_logging():
    with patch('farm.utils.get_logger'):
        # Test code without actual logging
        my_function()
```

## 📋 Checklist for New Code

When adding logging to new code:

- [ ] Import get_logger, not logging
- [ ] Use event names, not messages
- [ ] Include rich context (IDs, values, state)
- [ ] Add error_type and error_message for errors
- [ ] Include exc_info=True for exceptions
- [ ] Bind context at appropriate level
- [ ] Use appropriate log level
- [ ] Sample if high-frequency (>100/sec)
- [ ] Avoid logging sensitive data
- [ ] Use specialized loggers where appropriate
- [ ] Add units to numeric field names
- [ ] Test that logs appear correctly

## 🚀 Production Deployment

### Configuration

```python
# Production configuration
from farm.utils import configure_logging

configure_logging(
    environment="production",
    log_dir="/var/log/agentfarm",
    log_level="INFO",  # or "WARNING" for less noise
    json_logs=True,    # For log aggregation
    include_caller_info=True,
    include_process_info=True,  # For parallel execution
)
```

### Monitoring

Set up alerts for:
```python
# Error rate spike
errors_per_minute > threshold

# Slow operations
duration_seconds > 60

# System resources
memory_percent > 90 OR cpu_percent > 90

# Agent anomalies
agent_count == 0 (early termination)
```

### Log Retention

```bash
# Rotate logs daily, keep 30 days
# In logrotate.d/agentfarm:
/var/log/agentfarm/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 agentfarm agentfarm
}
```

## 📊 Analysis Patterns

### Standard Queries

```bash
# Error summary
cat logs/application.json.log | jq -r 'select(.level=="error") | .error_type' | sort | uniq -c

# Top events
cat logs/application.json.log | jq -r '.event' | sort | uniq -c | sort -rn | head -20

# Slowest operations
cat logs/application.json.log | jq 'select(.duration_ms) | {event, duration_ms}' | sort_by(.duration_ms) | reverse | .[0:10]

# Events timeline
cat logs/application.json.log | jq '{timestamp, event, level}'
```

### Pandas Analysis

```python
import pandas as pd
import json

with open("logs/application.json.log") as f:
    df = pd.DataFrame([json.loads(line) for line in f])

# Top errors
errors = df[df['level'] == 'error']
error_summary = errors.groupby(['event', 'error_type']).size().sort_values(ascending=False)

# Performance analysis
perf = df[df['duration_ms'].notna()]
perf_summary = perf.groupby('event')['duration_ms'].agg(['count', 'mean', 'median', 'max'])

# Time series
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
events_per_minute = df.resample('1min').size()
```

## ✨ Summary

**Golden Rules:**
1. 🎯 Use event names, not messages
2. 📊 Include rich context
3. 🔗 Bind context early
4. ⚡ Sample high-frequency logs
5. 🔐 Never log sensitive data
6. 🐛 Always include error context
7. 📏 Use consistent naming
8. 🎨 Use appropriate log levels
9. 🧪 Test your logging
10. 📈 Monitor production logs

**Follow these practices and your logs will be:**
- Searchable
- Analyzable
- Debuggable
- Secure
- Performant
- Valuable

Happy logging! 🚀
