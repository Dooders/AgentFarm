# Benchmarks Module Unit Tests

This directory contains comprehensive unit tests for the AgentFarm benchmarks module, covering all core components and functionality.

## Test Coverage

### Core Modules

#### `test_experiments.py`
- **ExperimentContext**: Tests dataclass initialization, serialization, and field handling
- **Experiment**: Tests abstract base class, lifecycle methods (setup/execute_once/teardown), parameter handling
- **Concrete Experiment Implementations**: Tests real experiment behavior and parameter schemas

#### `test_results.py`
- **RunResult**: Tests result aggregation, iteration tracking, artifact management, timing statistics
- **IterationResult**: Tests individual iteration result handling
- **Artifact**: Tests artifact creation and serialization
- **Utility Functions**: Tests environment capture, VCS capture, and file operations

#### `test_registry.py`
- **ExperimentRegistry**: Tests experiment registration, discovery, creation, and parameter validation
- **ExperimentInfo**: Tests metadata container functionality
- **register_experiment Decorator**: Tests decorator registration and validation
- **Global Registry**: Tests singleton behavior and persistence

#### `test_spec.py`
- **RunSpec**: Tests specification dataclass and validation
- **load_spec**: Tests YAML/JSON loading, type conversion, and error handling
- **SPEC_DEFAULTS**: Tests default configuration values

#### `test_runner.py`
- **Runner**: Tests complete benchmark orchestration, lifecycle management, instrumentation
- **Random Run ID**: Tests unique identifier generation
- **Instrumentation Integration**: Tests timing, cProfile, and psutil integration
- **Result Aggregation**: Tests timing statistics calculation and artifact collection

#### `test_instrumentation.py`
- **Timing**: Tests basic timing instrumentation with context managers
- **cProfile**: Tests profiling capture, file generation, and summary creation
- **psutil**: Tests system resource monitoring and sampling
- **Error Handling**: Tests graceful failure handling for all instruments

#### `test_utils.py`
- **config_helper**: Tests configuration utilities for performance optimization
- **get_recommended_config**: Tests recommended configuration generation
- **print_config_recommendations**: Tests configuration documentation output

#### `test_integration.py`
- **End-to-End Workflows**: Tests complete benchmark execution from spec to results
- **Component Integration**: Tests interaction between all major components
- **Registry Integration**: Tests experiment discovery and instantiation
- **Error Handling**: Tests graceful error handling across the system

## Test Structure

### Fixtures and Utilities
- **conftest.py**: Provides common fixtures for temporary directories, mock experiments, and instrumentation
- **Mock Classes**: Custom experiment implementations for testing
- **Test Data**: Consistent test data and parameters across all tests

### Test Patterns
- **Unit Tests**: Individual component testing with mocked dependencies
- **Integration Tests**: End-to-end workflow testing
- **Error Handling**: Comprehensive error condition testing
- **Edge Cases**: Boundary condition and edge case testing

## Running Tests

### Run All Benchmarks Tests
```bash
python -m pytest tests/benchmarks/ -v
```

### Run Specific Test Modules
```bash
python -m pytest tests/benchmarks/test_experiments.py -v
python -m pytest tests/benchmarks/test_runner.py -v
python -m pytest tests/benchmarks/test_integration.py -v
```

### Run with Coverage
```bash
python -m pytest tests/benchmarks/ --cov=benchmarks --cov-report=html
```

## Test Quality

### Coverage
- **159 total tests** covering all major functionality
- **Comprehensive error handling** for all failure modes
- **Edge case testing** for boundary conditions
- **Integration testing** for end-to-end workflows

### Best Practices
- **Isolated Tests**: Each test is independent and can run in any order
- **Mock Usage**: External dependencies are properly mocked
- **Clear Assertions**: Tests have clear, specific assertions
- **Documentation**: All test methods are well-documented
- **Error Scenarios**: Both success and failure paths are tested

### Maintenance
- **Consistent Structure**: All tests follow the same patterns and conventions
- **Reusable Fixtures**: Common test setup is shared via fixtures
- **Clear Naming**: Test names clearly describe what is being tested
- **Comprehensive Coverage**: All public APIs and major code paths are tested

## Dependencies

The tests require the following packages:
- `pytest` - Test framework
- `unittest.mock` - Mocking framework
- `tempfile` - Temporary file handling
- `json` - JSON serialization testing
- `os` - File system operations

## Notes

- Tests use temporary directories for file operations to avoid conflicts
- Mock objects are used extensively to isolate components under test
- Timing-sensitive tests use appropriate tolerances for system performance variations
- Error handling tests verify graceful failure modes
- Integration tests verify complete workflows from specification to results
