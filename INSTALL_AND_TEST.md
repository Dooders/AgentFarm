# Installation and Testing Guide - Structured Logging

Quick guide to install and test the new structured logging system.

## 1. Install Dependencies

```bash
pip install structlog>=24.1.0 python-json-logger>=2.0.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## 2. Verify Installation

```bash
python3 -c "import structlog; print(f'✅ structlog {structlog.__version__} installed')"
python3 -c "from pythonjsonlogger import jsonlogger; print('✅ python-json-logger installed')"
```

## 3. Run Examples

### Basic Examples

```bash
python examples/logging_examples.py
```

This will demonstrate:
- Basic structured logging
- Context binding
- Context managers
- Performance decorators
- Agent logger
- Log sampling
- Error logging

Expected output (colored in console):
```
=== Example 1: Basic Logging ===

2025-10-01T12:34:56Z [info     ] simulation_started    num_agents=100 num_steps=1000
2025-10-01T12:34:56Z [debug    ] spatial_index_built   num_items=250 build_time_ms=45.2
2025-10-01T12:34:56Z [warning  ] resource_depleted     resource_id=res_001 remaining=0
2025-10-01T12:34:56Z [error    ] agent_collision       agent_a=agent_123 agent_b=agent_456

=== Example 2: Context Binding ===
...
```

## 4. Test with Simulation

### Development Mode (Colored Console)

```bash
python run_simulation.py --steps 100 --log-level DEBUG
```

Expected: Colored, human-readable console output with rich context.

### Production Mode (JSON Logs)

```bash
python run_simulation.py --environment production --json-logs --steps 100
```

Expected: JSON formatted logs in `logs/application.json.log`.

### View JSON Logs

```bash
# Pretty print JSON logs
cat logs/application.json.log | jq

# Filter by level
cat logs/application.json.log | jq 'select(.level == "error")'

# Get unique event types
cat logs/application.json.log | jq -r '.event' | sort -u

# Find slow operations
cat logs/application.json.log | jq 'select(.duration_ms > 100)'
```

## 5. Test GUI Application

```bash
python main.py
```

Expected: Structured logs to console and `logs/` directory.

## 6. Analyze Logs with Pandas

```python
import json
import pandas as pd

# Load JSON logs
with open("logs/application.json.log") as f:
    logs = [json.loads(line) for line in f]
    df = pd.DataFrame(logs)

# Display summary
print(f"Total logs: {len(df)}")
print(f"\nLogs by level:")
print(df['level'].value_counts())

print(f"\nTop events:")
print(df['event'].value_counts().head(10))

# Find slow operations
if 'duration_ms' in df.columns:
    slow_ops = df[df['duration_ms'] > 100]
    print(f"\nSlow operations (>100ms): {len(slow_ops)}")
    if len(slow_ops) > 0:
        print(slow_ops[['event', 'duration_ms']].sort_values('duration_ms', ascending=False))

# Errors by type
if 'error_type' in df.columns:
    errors = df[df['level'] == 'error']
    if len(errors) > 0:
        print(f"\nErrors by type:")
        print(errors['error_type'].value_counts())
```

## 7. Test Different Configurations

### Debug Level with Colors

```bash
python run_simulation.py --log-level DEBUG --steps 50
```

### Info Level, No Colors

```bash
python run_simulation.py --log-level INFO --steps 50
```

### JSON Output

```bash
python run_simulation.py --json-logs --steps 50
```

### Production Environment

```bash
python run_simulation.py --environment production --log-level WARNING --steps 50
```

## 8. Verify Log Files

Check that log files are created:

```bash
ls -lh logs/
```

Expected files:
- `application.log` - Plain text logs
- `application.json.log` - JSON logs (if `--json-logs` enabled)

## 9. Test Context Propagation

Create a test script:

```python
# test_context.py
from farm.utils import configure_logging, get_logger, bind_context, log_step

configure_logging(environment="development", log_level="DEBUG")
logger = get_logger(__name__)

# Test global context
bind_context(simulation_id="test_sim_001")
logger.info("test_event_1")  # Should include simulation_id

# Test step context
with log_step(step_number=42, simulation_id="test_sim_001"):
    logger.info("test_event_2")  # Should include step and simulation_id
    logger.debug("test_event_3")  # Should include step and simulation_id

logger.info("test_event_4")  # Should include simulation_id but not step

print("\n✅ Context propagation test complete")
```

Run it:
```bash
python test_context.py
```

Expected output:
```
2025-10-01T12:34:56Z [info     ] test_event_1         simulation_id=test_sim_001
2025-10-01T12:34:56Z [debug    ] step_started         step=42 simulation_id=test_sim_001
2025-10-01T12:34:56Z [info     ] test_event_2         step=42 simulation_id=test_sim_001
2025-10-01T12:34:56Z [debug    ] test_event_3         step=42 simulation_id=test_sim_001
2025-10-01T12:34:56Z [debug    ] step_completed       step=42 duration_ms=0.12 simulation_id=test_sim_001
2025-10-01T12:34:56Z [info     ] test_event_4         simulation_id=test_sim_001

✅ Context propagation test complete
```

## 10. Performance Test

Test that logging doesn't significantly impact performance:

```python
# test_performance.py
import time
from farm.utils import configure_logging, get_logger, LogSampler

configure_logging(environment="production", log_level="INFO")
logger = get_logger(__name__)

# Test 1: Without sampling (should be fast, INFO level filters DEBUG)
start = time.time()
for i in range(10000):
    logger.debug("high_frequency_event", iteration=i)
duration1 = time.time() - start
print(f"10k DEBUG logs (filtered): {duration1:.3f}s")

# Test 2: With sampling
sampler = LogSampler(sample_rate=0.01)  # 1%
start = time.time()
for i in range(10000):
    if sampler.should_log():
        logger.info("sampled_event", iteration=i)
duration2 = time.time() - start
print(f"10k sampled logs (1%): {duration2:.3f}s")

# Test 3: Full logging (INFO level)
start = time.time()
for i in range(1000):  # Fewer iterations
    logger.info("full_log_event", iteration=i)
duration3 = time.time() - start
print(f"1k full INFO logs: {duration3:.3f}s")

print(f"\n✅ Performance test complete")
```

## 11. Test Error Logging

```python
# test_errors.py
from farm.utils import configure_logging, get_logger

configure_logging(environment="development", log_level="DEBUG")
logger = get_logger(__name__)

# Test error logging with context
try:
    result = 10 / 0
except ZeroDivisionError as e:
    logger.error(
        "calculation_failed",
        operation="division",
        numerator=10,
        denominator=0,
        error_type=type(e).__name__,
        error_message=str(e),
        exc_info=True,
    )

try:
    data = {"key": "value"}
    value = data["missing_key"]
except KeyError as e:
    logger.error(
        "data_access_failed",
        available_keys=list(data.keys()),
        requested_key="missing_key",
        error_type=type(e).__name__,
        exc_info=True,
    )

print("\n✅ Error logging test complete")
```

## 12. Troubleshooting

### Issue: ModuleNotFoundError: No module named 'structlog'

**Solution:**
```bash
pip install structlog python-json-logger
```

### Issue: Logs not appearing

**Solution:** Ensure `configure_logging()` is called before any logging:
```python
from farm.utils import configure_logging
configure_logging()  # Must be first!
```

### Issue: Context not persisting

**Solution:** Use `bind_context()` or context managers:
```python
from farm.utils import bind_context
bind_context(simulation_id="sim_001")  # Persists globally
```

### Issue: Too many logs

**Solution:** Use sampling or increase log level:
```python
from farm.utils import LogSampler
sampler = LogSampler(sample_rate=0.1)  # Log 10%
```

### Issue: Performance impact

**Solution:** 
- Use appropriate log levels (DEBUG only in development)
- Sample high-frequency logs
- Use JSON logs in production (faster than colored console)

## 13. Validation Checklist

After installation and testing, verify:

- [ ] `structlog` and `python-json-logger` installed
- [ ] Examples run without errors
- [ ] Console output is colored and formatted correctly
- [ ] JSON logs are created in `logs/` directory
- [ ] Context binding works correctly
- [ ] Performance decorators log timing
- [ ] Error logging includes tracebacks
- [ ] Log sampling reduces volume
- [ ] Different log levels work correctly
- [ ] Different environments produce different output

## 14. Next Steps

Once everything is working:

1. **Review the documentation:**
   - [Logging Guide](docs/logging_guide.md)
   - [Quick Reference](docs/LOGGING_QUICK_REFERENCE.md)
   - [Migration Checklist](docs/LOGGING_MIGRATION.md)

2. **Start migrating modules:**
   - Begin with [Phase 2 of the migration](docs/LOGGING_MIGRATION.md#phase-2-core-modules-next-steps)
   - Follow the migration guidelines for each file

3. **Monitor and adjust:**
   - Watch for performance issues
   - Adjust log levels as needed
   - Fine-tune sampling rates

## Success Criteria

✅ Installation is successful when:
- All dependencies are installed
- Examples run without errors
- Logs appear in console and files
- JSON logs are valid and parseable
- Context propagation works correctly
- Performance is acceptable

---

For questions or issues, refer to:
- [Logging Guide](docs/logging_guide.md)
- [Quick Reference](docs/LOGGING_QUICK_REFERENCE.md)
- [Structlog Documentation](https://www.structlog.org/)
