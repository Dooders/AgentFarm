# FastAPI Server Unit Tests

This directory contains comprehensive unit tests for the FastAPI server that replaced the original Flask implementation.

## Test Files

### Core Server Tests
- **`test_server.py`** - Main FastAPI server endpoint tests
  - HTTP endpoint testing (GET, POST)
  - Request/response validation with Pydantic models
  - Error handling and edge cases
  - Background task integration
  - API documentation endpoint verification

### WebSocket Tests
- **`test_websocket.py`** - WebSocket functionality tests
  - Connection establishment and cleanup
  - Message handling (JSON parsing, validation)
  - Simulation subscription functionality
  - Error handling for invalid messages
  - Concurrent connection testing

### Background Task Tests
- **`test_background_tasks.py`** - Background simulation task tests
  - Simulation execution in background threads
  - Status updates during execution
  - Error handling and logging
  - Thread safety verification
  - Configuration parameter handling

### Test Infrastructure
- **`conftest.py`** - Updated with FastAPI-specific fixtures
  - FastAPI test client fixtures
  - Mock objects for database, services, and WebSockets
  - Sample request/response data
  - Temporary workspace management

- **`run_server_tests.py`** - Custom test runner script
  - Run all server tests or specific test files
  - Coverage reporting with HTML output
  - Command-line interface for different test scenarios

## Test Coverage

The test suite covers:

### ✅ HTTP Endpoints
- `POST /api/simulation/new` - Create new simulation
- `GET /api/simulation/{sim_id}/step/{step}` - Get simulation step data
- `GET /api/simulation/{sim_id}/analysis` - Get simulation analysis
- `POST /api/analysis/{module_name}` - Run analysis module
- `GET /api/simulations` - List active simulations
- `GET /api/simulation/{sim_id}/export` - Export simulation data
- `GET /api/simulation/{sim_id}/status` - Get simulation status

### ✅ WebSocket Functionality
- `WebSocket /ws/{client_id}` - Real-time communication
- Connection management and cleanup
- Message parsing and validation
- Simulation subscription system
- Error handling and recovery

### ✅ Background Tasks
- Simulation execution in background threads
- Status tracking and updates
- Error handling and logging
- Thread safety and concurrent operations

### ✅ Error Handling
- Invalid request handling
- Missing resource errors (404)
- Server errors (500)
- WebSocket connection errors
- Background task failures

### ✅ Data Validation
- Pydantic model validation
- Request/response serialization
- Type checking and conversion
- Field validation and constraints

## Running Tests

### Run All Server Tests
```bash
# Using pytest directly
python -m pytest tests/api/test_server.py tests/api/test_websocket.py tests/api/test_background_tasks.py -v

# Using the custom test runner
python tests/api/run_server_tests.py server
```

### Run Specific Test Files
```bash
# Server endpoints only
python tests/api/run_server_tests.py server

# WebSocket tests only
python tests/api/run_server_tests.py websocket

# Background task tests only
python tests/api/run_server_tests.py background
```

### Run with Coverage
```bash
python tests/api/run_server_tests.py coverage
```

### Run Individual Tests
```bash
# Specific test method
python -m pytest tests/api/test_server.py::TestFastAPIServer::test_create_simulation_success -v

# Specific test class
python -m pytest tests/api/test_websocket.py::TestWebSocketFunctionality -v
```

## Test Features

### Mocking Strategy
- **External Dependencies**: Database connections, file I/O, external services
- **Internal Components**: Simulation controllers, analysis services
- **Async Operations**: WebSocket connections, background tasks
- **Time-dependent Operations**: Timestamps, simulation durations

### Assertion Patterns
- **Status Codes**: HTTP response codes (200, 404, 500)
- **Response Structure**: JSON schema validation
- **State Changes**: Simulation status updates
- **Error Messages**: Detailed error information
- **Side Effects**: File creation, logging, event emission

### Test Organization
- **Arrange-Act-Assert**: Clear test structure
- **Descriptive Names**: Self-documenting test methods
- **Single Responsibility**: Each test focuses on one behavior
- **Independent Tests**: No dependencies between tests
- **Proper Cleanup**: Resources cleaned up after each test

## FastAPI-Specific Features

### Automatic API Documentation
Tests verify that FastAPI's automatic documentation endpoints work:
- `/docs` - Swagger UI
- `/redoc` - ReDoc documentation
- `/openapi.json` - OpenAPI schema

### Type Safety
- Pydantic model validation
- Request/response type checking
- Automatic serialization/deserialization

### Async Support
- WebSocket testing with async/await
- Background task testing
- Concurrent operation verification

### Performance
- FastAPI's superior performance compared to Flask
- Efficient request handling
- Minimal overhead for API operations

## Integration with Existing Tests

The new FastAPI tests are designed to work alongside existing API tests:
- Shared fixtures in `conftest.py`
- Compatible test patterns and naming conventions
- Integration with existing test infrastructure
- Coverage reporting integration

## Continuous Integration

These tests are designed for CI/CD pipelines:
- **Fast Execution**: Quick test completion with mocked dependencies
- **Deterministic**: Consistent results across environments
- **Isolated**: No interference between tests or with system state
- **Comprehensive**: Full coverage of happy path and error scenarios

## Debugging Tests

### Verbose Output
```bash
python -m pytest tests/api/test_server.py -v -s
```

### Debug Mode
```bash
python -m pytest tests/api/test_server.py --pdb
```

### Single Test Debugging
```bash
python -m pytest tests/api/test_server.py::TestFastAPIServer::test_create_simulation_success -v -s --pdb
```

## Contributing

When adding new tests:

1. **Follow Naming Conventions**: Use descriptive test method names
2. **Add Docstrings**: Document what each test verifies
3. **Use Fixtures**: Leverage existing fixtures for common setup
4. **Mock Appropriately**: Mock external dependencies, not internal logic
5. **Test Edge Cases**: Include tests for error conditions and boundary values
6. **Update Documentation**: Keep this README current with new test patterns

## Migration from Flask Tests

The FastAPI tests replace Flask-specific testing patterns:

### Before (Flask)
```python
def test_endpoint():
    response = client.post('/api/endpoint', json=data)
    assert response.status_code == 200
    assert response.json()['status'] == 'success'
```

### After (FastAPI)
```python
def test_endpoint():
    response = client.post('/api/endpoint', json=data)
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'success'
    # Additional Pydantic validation
    assert isinstance(data, ExpectedResponseModel)
```

### Key Improvements
- **Type Safety**: Pydantic models provide automatic validation
- **Better Error Messages**: FastAPI provides detailed error information
- **Async Support**: Native async/await testing capabilities
- **Performance**: Faster test execution with FastAPI's optimized request handling
- **Documentation**: Automatic API documentation generation and testing
