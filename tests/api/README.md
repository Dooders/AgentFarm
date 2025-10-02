# API Module Tests

This directory contains comprehensive unit and integration tests for the AgentFarm API module.

## Test Structure

### Unit Tests

- **`test_models.py`** - Tests for data models and schemas
  - Enumeration classes (SessionStatus, SimulationStatus, etc.)
  - Data classes (SessionInfo, SimulationResults, etc.)
  - Serialization methods (to_dict)

- **`test_session_manager.py`** - Tests for session management
  - Session creation, retrieval, and deletion
  - Session persistence and loading
  - Session statistics and metadata

- **`test_config_templates.py`** - Tests for configuration management
  - Template creation and validation
  - Configuration conversion and validation
  - Template examples and field requirements

- **`test_unified_controller.py`** - Tests for the main API controller
  - Session management operations
  - Simulation and experiment control
  - Configuration and analysis operations

- **`test_unified_adapter.py`** - Tests for the unified adapter
  - Simulation and experiment lifecycle management
  - Event system and subscriptions
  - Analysis and comparison operations

### Integration Tests

- **`test_integration.py`** - End-to-end workflow tests
  - Complete simulation workflows
  - Complete experiment workflows
  - Session management workflows
  - Error handling and edge cases
  - Concurrent operations
  - Data persistence

## Test Fixtures

The `conftest.py` file provides common test fixtures:

- **`temp_workspace`** - Temporary directory for test data
- **`sample_session_info`** - Sample session data
- **`sample_config_template`** - Sample configuration template
- **`sample_simulation_config`** - Sample simulation configuration
- **`sample_experiment_config`** - Sample experiment configuration
- **Mock objects** - Mock controllers, databases, and environments

## Running Tests

### Run All API Tests

```bash
# From the project root
python -m pytest tests/api -v

# Or use the test runner
python tests/api/run_tests.py
```

### Run Specific Test Files

```bash
# Run only model tests
python -m pytest tests/api/test_models.py -v

# Run only integration tests
python -m pytest tests/api/test_integration.py -v
```

### Run with Coverage

```bash
python -m pytest tests/api --cov=farm.api --cov-report=html
```

## Test Coverage

The tests aim to achieve comprehensive coverage of:

- ✅ All public methods and properties
- ✅ Error handling and edge cases
- ✅ Data validation and serialization
- ✅ Thread safety and concurrent operations
- ✅ Resource cleanup and context management
- ✅ Integration between components

## Test Patterns

### Mocking Strategy

- **External Dependencies**: Mock database connections, file I/O, and external services
- **Internal Components**: Mock controllers and adapters when testing higher-level components
- **Time-dependent Operations**: Use fixed timestamps and mock datetime.now()

### Assertion Patterns

- **Data Validation**: Test both valid and invalid inputs
- **State Changes**: Verify object state before and after operations
- **Error Conditions**: Test exception handling and error messages
- **Side Effects**: Verify file creation, logging, and event emission

### Test Organization

- **Arrange-Act-Assert**: Clear separation of test setup, execution, and verification
- **Descriptive Names**: Test method names describe the scenario being tested
- **Single Responsibility**: Each test focuses on one specific behavior
- **Independent Tests**: Tests don't depend on each other and can run in any order

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

- **Fast Execution**: Tests complete quickly with mocked dependencies
- **Deterministic**: Tests produce consistent results across environments
- **Isolated**: Tests don't interfere with each other or the system
- **Comprehensive**: Cover both happy path and error scenarios

## Debugging Tests

### Verbose Output

```bash
python -m pytest tests/api -v -s
```

### Run Single Test

```bash
python -m pytest tests/api/test_models.py::TestSessionInfo::test_session_info_creation -v
```

### Debug Mode

```bash
python -m pytest tests/api --pdb
```

## Contributing

When adding new tests:

1. **Follow Naming Conventions**: Use descriptive test method names
2. **Add Docstrings**: Document what each test verifies
3. **Use Fixtures**: Leverage existing fixtures for common setup
4. **Mock Appropriately**: Mock external dependencies, not internal logic
5. **Test Edge Cases**: Include tests for error conditions and boundary values
6. **Update This README**: Document new test patterns or fixtures
