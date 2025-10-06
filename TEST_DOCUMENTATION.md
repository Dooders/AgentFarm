# Unit Tests for Learning Experience Logging

This document describes the comprehensive unit tests added to verify the fix for Issue #481.

## Test Files

### 1. `tests/decision/test_decision_module.py`

Enhanced existing tests with 9 new test methods focused on learning experience logging:

#### Core Functionality Tests

**`test_update_logs_learning_experience`**
- Verifies that `DecisionModule.update()` correctly logs learning experiences to the database
- Tests all required parameters are passed correctly:
  - `step_number` - from agent's time service
  - `agent_id` - agent's unique identifier
  - `module_type` - algorithm type (ppo, sac, dqn, etc.)
  - `module_id` - unique module identifier
  - `action_taken` - numeric action index
  - `action_taken_mapped` - human-readable action name
  - `reward` - reward value

#### Error Handling Tests

**`test_update_without_database_does_not_crash`**
- Ensures system works gracefully when database is not available
- Verifies no exceptions are raised when `agent.environment` is None

**`test_update_without_time_service_skips_logging`**
- Tests behavior when time service is unavailable
- Verifies logging is skipped but update continues normally

**`test_update_without_actions_skips_logging`**
- Tests behavior when actions list is empty or unavailable
- Ensures logging is skipped without crashing

**`test_update_logging_exception_does_not_crash`**
- Verifies that exceptions during logging don't crash the simulation
- Tests graceful error handling with warning logs

#### Data Accuracy Tests

**`test_update_logs_correct_algorithm_type`**
- Tests all supported algorithm types: ppo, sac, dqn, a2c, ddpg
- Uses `subTest` to verify each algorithm type independently
- Ensures correct `module_type` is logged for each algorithm

**`test_update_logs_different_rewards`**
- Tests various reward values: negative, zero, positive, large
- Verifies rewards are stored with correct precision
- Tests edge cases: -10.0, -1.5, 0.0, 1.5, 10.0, 100.5

**`test_update_with_curriculum_logs_correct_action`**
- Tests curriculum learning with restricted action sets
- Verifies action indices are mapped correctly from enabled actions to full action space
- Example: action index 1 in enabled_actions [0, 2, 4] maps to action 2 in full space

### 2. `tests/decision/test_learning_experience_logging.py`

New integration test file with comprehensive end-to-end tests:

#### Integration Tests

**`test_learning_experiences_logged_to_database`**
- Full end-to-end test with real SQLite database
- Simulates 10 steps of agent learning
- Verifies data is actually written to database table
- Validates data structure and values

**`test_multiple_agents_logging`**
- Tests 3 agents logging independently
- Verifies each agent's experiences are tracked separately
- Total: 3 agents × 5 steps = 15 experiences

**`test_different_algorithm_types_logged`**
- Tests all algorithm types in one simulation
- Verifies each algorithm type appears in database
- Uses DISTINCT query to validate variety

**`test_reward_values_persisted_correctly`**
- Tests edge cases for reward values
- Range: -100.5 to +100.5 including zero
- Verifies floating-point precision

**`test_curriculum_action_mapping`**
- Integration test for curriculum learning
- Tests action mapping with restricted action set
- Verifies correct action name is stored

**`test_database_unavailable_does_not_crash`**
- Integration test for missing database
- Ensures system degrades gracefully

#### Performance Tests

**`test_bulk_logging_performance`**
- Tests performance with 1000 learning experiences
- Validates buffered writes are efficient
- Asserts completion in < 5 seconds
- Verifies all data is persisted correctly

## Test Coverage

### What's Covered ✅

1. **Core Functionality**
   - Learning experiences are logged with correct parameters
   - All algorithm types are supported
   - Action mapping works correctly
   - Reward values are persisted accurately

2. **Error Handling**
   - Missing database doesn't crash
   - Missing time service is handled gracefully
   - Missing actions list is handled gracefully
   - Logging exceptions are caught and logged

3. **Edge Cases**
   - Curriculum learning with restricted actions
   - Negative, zero, and positive rewards
   - Multiple agents logging simultaneously
   - Large volumes of data (performance test)

4. **Integration**
   - Real database interactions
   - Multiple components working together
   - Buffered writes and flushing

### What's NOT Covered ⚠️

1. Concurrent writes from multiple threads (if applicable)
2. Database transaction rollback scenarios
3. Memory pressure scenarios
4. Network database connections (uses local SQLite)
5. Schema migrations

## Running the Tests

### Run All Learning Experience Logging Tests

```bash
# Run unit tests
python -m pytest tests/decision/test_decision_module.py::TestDecisionModule::test_update_logs_learning_experience -v

# Run all new unit tests
python -m pytest tests/decision/test_decision_module.py -k "logging" -v

# Run integration tests
python -m pytest tests/decision/test_learning_experience_logging.py -v

# Run specific integration test
python -m pytest tests/decision/test_learning_experience_logging.py::TestLearningExperienceLoggingIntegration::test_learning_experiences_logged_to_database -v

# Run performance test
python -m pytest tests/decision/test_learning_experience_logging.py::TestLearningExperienceLoggingPerformance -v
```

### Run All Decision Module Tests

```bash
python -m pytest tests/decision/test_decision_module.py -v
```

### Run with Coverage

```bash
python -m pytest tests/decision/ --cov=farm.core.decision --cov-report=html
```

## Test Structure

### Unit Tests Pattern

```python
def test_specific_functionality(self):
    """Test description."""
    # 1. Setup - Create mocks and fixtures
    mock_db = Mock()
    mock_logger = Mock()
    # ... setup code
    
    # 2. Execute - Call the method under test
    module.update(state, action, reward, next_state, done)
    
    # 3. Verify - Assert expected behavior
    mock_logger.log_learning_experience.assert_called_once()
    self.assertEqual(call_args.kwargs['reward'], expected_reward)
```

### Integration Tests Pattern

```python
def test_full_workflow(self):
    """Test description."""
    # 1. Setup - Create real database and components
    db = SimulationDatabase(db_path, simulation_id="test")
    db.init_db()
    
    # 2. Execute - Run realistic scenario
    for step in range(10):
        module.update(state, action, reward, next_state, done)
    
    # 3. Verify - Query database to verify persistence
    cursor.execute("SELECT COUNT(*) FROM learning_experiences")
    count = cursor.fetchone()[0]
    self.assertEqual(count, 10)
```

## Mock Objects Used

### Agent Mock
- `agent_id`: Unique identifier
- `environment`: Reference to environment (which has database)
- `time_service`: Provides current simulation step
- `actions`: List of available actions with names

### Environment Mock
- `db`: Database instance
- `action_space`: Gymnasium Discrete space

### Database Mock
- `logger`: DataLogger instance for buffered writes

### Time Service Mock
- `current_time()`: Returns current simulation step

## Assertions Used

### Standard Assertions
- `assertEqual()` - Exact value matching
- `assertAlmostEqual()` - Floating-point comparison with precision
- `assertTrue()` / `assertFalse()` - Boolean checks
- `assertIsInstance()` - Type checking
- `assertIn()` - Membership testing
- `assertLess()` - Numeric comparison (performance)

### Mock Assertions
- `assert_called_once()` - Verify method was called exactly once
- `assert_not_called()` - Verify method was never called
- `call_args.kwargs[key]` - Access keyword arguments from mock call

## Test Maintenance

### When to Update Tests

1. **New Algorithm Types**: Add to `test_update_logs_correct_algorithm_type`
2. **New Required Parameters**: Update `test_update_logs_learning_experience`
3. **Changed Error Handling**: Update exception handling tests
4. **Performance Changes**: Adjust timeout in `test_bulk_logging_performance`

### Adding New Tests

Follow this checklist:
1. Add descriptive docstring
2. Use clear test method name: `test_<feature>_<scenario>`
3. Follow AAA pattern: Arrange, Act, Assert
4. Clean up resources in `tearDown()` if needed
5. Use `subTest()` for parametric tests
6. Add to this documentation

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

- No external dependencies (uses SQLite)
- Self-contained (creates temp databases)
- Fast execution (< 30 seconds total)
- Clear pass/fail criteria
- Detailed error messages

## Related Documentation

- `ISSUE_481_FIX_SUMMARY.md` - Overview of the fix
- `farm/core/decision/decision.py` - Implementation
- `farm/database/data_logging.py` - Logging infrastructure
- `farm/database/models.py` - Database schema
