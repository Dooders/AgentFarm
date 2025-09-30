# MCP Server Implementation Plan - Step-by-Step

## Project Structure

```
mcp/                                 # New top-level directory (separate from farm/)
├── README.md                        # Project overview and quick start
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Dependencies
├── setup.py                         # Installation script
├── .env.example                     # Environment variable template
├── config.example.yaml              # Configuration template
│
├── mcp_server/                      # Main package
│   ├── __init__.py
│   ├── __main__.py                  # Entry point for python -m mcp_server
│   ├── server.py                    # Main MCP server
│   ├── config.py                    # Configuration management
│   ├── cli.py                       # Command-line interface
│   │
│   ├── services/                    # Service layer
│   │   ├── __init__.py
│   │   ├── database_service.py      # Database connection & queries
│   │   ├── cache_service.py         # Caching layer
│   │   └── validation_service.py    # Input validation
│   │
│   ├── tools/                       # MCP tools
│   │   ├── __init__.py
│   │   ├── base.py                  # Base tool class
│   │   ├── metadata_tools.py        # Simulation/experiment metadata
│   │   ├── query_tools.py           # Data retrieval tools
│   │   ├── analysis_tools.py        # Analysis tools
│   │   ├── comparison_tools.py      # Multi-simulation comparison
│   │   └── advanced_tools.py        # Advanced analysis tools
│   │
│   ├── formatters/                  # Output formatters
│   │   ├── __init__.py
│   │   ├── base_formatter.py        # Base formatter interface
│   │   ├── json_formatter.py        # JSON output
│   │   ├── markdown_formatter.py    # Markdown tables
│   │   └── chart_formatter.py       # ASCII/Unicode charts
│   │
│   ├── models/                      # Data models
│   │   ├── __init__.py
│   │   ├── schemas.py               # Pydantic schemas
│   │   └── responses.py             # Response models
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── exceptions.py            # Custom exceptions
│       ├── logging.py               # Logging setup
│       └── helpers.py               # Helper functions
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── test_config.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── test_database_service.py
│   │   ├── test_cache_service.py
│   │   └── test_validation_service.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_metadata_tools.py
│   │   ├── test_query_tools.py
│   │   ├── test_analysis_tools.py
│   │   ├── test_comparison_tools.py
│   │   └── test_advanced_tools.py
│   │
│   ├── formatters/
│   │   ├── __init__.py
│   │   └── test_formatters.py
│   │
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_end_to_end.py
│   │   └── test_performance.py
│   │
│   └── test_server.py
│
├── scripts/                         # Utility scripts
│   ├── setup_env.sh                 # Environment setup
│   ├── run_tests.sh                 # Test runner
│   └── generate_tool_docs.py        # Auto-generate tool documentation
│
├── docs/                            # Documentation
│   ├── index.md
│   ├── installation.md
│   ├── configuration.md
│   ├── tools/                       # Tool documentation
│   │   ├── metadata.md
│   │   ├── query.md
│   │   ├── analysis.md
│   │   └── comparison.md
│   └── examples/                    # Usage examples
│       ├── basic_usage.md
│       ├── llm_integration.md
│       └── advanced_queries.md
│
└── examples/                        # Example configurations
    ├── simple_config.yaml
    ├── production_config.yaml
    └── claude_desktop_config.json
```

## Implementation Steps

### Step 1: Project Setup (Day 1, Morning - 2 hours)

#### 1.1 Create Directory Structure

```bash
# From workspace root
mkdir -p mcp/{mcp_server,tests,scripts,docs,examples}
cd mcp

# Create package structure
mkdir -p mcp_server/{services,tools,formatters,models,utils}
mkdir -p tests/{services,tools,formatters,integration}
mkdir -p docs/{tools,examples}

# Create all __init__.py files
touch mcp_server/__init__.py
touch mcp_server/{services,tools,formatters,models,utils}/__init__.py
touch tests/__init__.py
touch tests/{services,tools,formatters,integration}/__init__.py
```

#### 1.2 Create Package Configuration Files

**File: `mcp/pyproject.toml`**
```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mcp-simulation-server"
version = "0.1.0"
description = "FastMCP server for simulation database analysis"
readme = "README.md"
requires-python = ">=3.8"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "fastmcp>=0.1.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.0.292",
    "mypy>=1.5.0",
]

[project.scripts]
mcp-server = "mcp_server.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["mcp_server*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=mcp_server --cov-report=html --cov-report=term"

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.ruff]
line-length = 100
target-version = "py38"
```

**File: `mcp/requirements.txt`**
```txt
# Core dependencies
fastmcp>=0.1.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
pyyaml>=6.0

# Development dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
black>=23.0.0
ruff>=0.0.292
mypy>=1.5.0
```

**File: `mcp/setup.py`**
```python
from setuptools import setup, find_packages

setup(
    name="mcp-simulation-server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastmcp>=0.1.0",
        "sqlalchemy>=2.0.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "mcp-server=mcp_server.cli:main",
        ],
    },
)
```

**File: `mcp/.env.example`**
```bash
# Database Configuration
DB_PATH=/path/to/simulation.db
DB_POOL_SIZE=5
DB_QUERY_TIMEOUT=30
DB_READ_ONLY=true

# Cache Configuration
CACHE_ENABLED=true
CACHE_MAX_SIZE=100
CACHE_TTL_SECONDS=300

# Server Configuration
LOG_LEVEL=INFO
MAX_RESULT_SIZE=10000
DEFAULT_LIMIT=100
```

**File: `mcp/config.example.yaml`**
```yaml
database:
  path: "/path/to/simulation.db"
  pool_size: 5
  query_timeout: 30
  read_only: true

cache:
  enabled: true
  max_size: 100
  ttl_seconds: 300

server:
  max_result_size: 10000
  default_limit: 100
  log_level: "INFO"
```

**File: `mcp/README.md`**
```markdown
# MCP Simulation Analysis Server

FastMCP server for querying and analyzing simulation databases with LLM agents.

## Quick Start

```bash
# Install
pip install -e .

# Run server
mcp-server --db-path /path/to/simulation.db

# With config file
mcp-server --config config.yaml
```

See [docs/](docs/) for full documentation.
```

#### 1.3 Cursor Tasks for Step 1
- [ ] Create all directories
- [ ] Create all configuration files
- [ ] Create README.md
- [ ] Commit: "Initial project structure"

---

### Step 2: Core Infrastructure (Day 1, Afternoon - 4 hours)

#### 2.1 Configuration System

**File: `mcp/mcp_server/config.py`**

**Cursor Prompt**: 
```
Create a configuration system with Pydantic models for:
- DatabaseConfig (path, pool_size, query_timeout, read_only)
- CacheConfig (enabled, max_size, ttl_seconds)
- ServerConfig (max_result_size, default_limit, log_level)
- MCPConfig (main config combining above)

Include:
- Validators for critical fields
- from_db_path() class method
- from_yaml() class method for loading from file
- from_env() class method for environment variables
```

**Implementation**: See design doc section 2.7

**Test File: `mcp/tests/test_config.py`**

**Cursor Prompt**:
```
Generate comprehensive pytest tests for the config module including:
- Test config creation from database path
- Test config loading from YAML
- Test config loading from environment variables
- Test validation errors for invalid values
- Test default values
```

#### 2.2 Custom Exceptions

**File: `mcp/mcp_server/utils/exceptions.py`**

**Cursor Prompt**:
```
Create custom exception classes:
- MCPException (base)
- DatabaseError
- ValidationError
- QueryTimeoutError
- SimulationNotFoundError
- ResultTooLargeError
- CacheError
- ToolNotFoundError

Each should have helpful error messages and optional details.
```

**Implementation**: See design doc section 2.3

#### 2.3 Logging Setup

**File: `mcp/mcp_server/utils/logging.py`**

**Cursor Prompt**:
```
Create logging configuration with:
- Configurable log levels
- Structured logging format
- File and console handlers
- Log rotation
- setup_logging(log_level, log_file) function
```

```python
"""Logging configuration for MCP server."""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_str: Optional custom format string
    """
    if format_str is None:
        format_str = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_str,
        handlers=handlers
    )
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
```

#### 2.4 Cursor Tasks for Step 2
- [ ] Implement config.py
- [ ] Write tests for config
- [ ] Implement exceptions.py
- [ ] Implement logging.py
- [ ] Run tests: `pytest tests/test_config.py -v`
- [ ] Commit: "Add configuration system and utilities"

---

### Step 3: Service Layer (Day 2, Morning - 4 hours)

#### 3.1 Database Service

**File: `mcp/mcp_server/services/database_service.py`**

**Cursor Prompt**:
```
Implement DatabaseService class with:
- __init__(config: DatabaseConfig) - setup engine and session factory
- get_session() context manager - provide transactional sessions
- execute_query(query_func, timeout) - execute queries with error handling
- validate_simulation_exists(simulation_id) - check if simulation exists
- get_simulation(simulation_id) - retrieve simulation by ID
- close() - cleanup resources

Use SQLAlchemy with connection pooling and read-only mode.
Import models from: from farm.database.models import ...
```

**Implementation**: See design doc section 2.2 and implementation guide section 2.4

**Note**: This service will import from the main `farm` package:
```python
from farm.database.models import (
    Simulation,
    AgentModel,
    AgentStateModel,
    ActionModel,
    ResourceModel,
    SimulationStepModel,
    InteractionModel,
    # ... etc
)
```

**Test File: `mcp/tests/services/test_database_service.py`**

**Cursor Prompt**:
```
Generate comprehensive tests for DatabaseService including:
- Test initialization
- Test session management
- Test query execution
- Test validation methods
- Test error handling
- Test read-only enforcement
- Use fixtures for test database with sample data
```

#### 3.2 Cache Service

**File: `mcp/mcp_server/services/cache_service.py`**

**Cursor Prompt**:
```
Implement CacheService class with:
- __init__(config: CacheConfig)
- get(key) - retrieve from cache with TTL check
- set(key, value) - store in cache with LRU eviction
- clear() - clear all cache
- get_stats() - return cache statistics (hits, misses, hit_rate)
- generate_key(tool_name, params) - static method for key generation

Use OrderedDict for LRU, track timestamps for TTL.
```

**Implementation**: See design doc section 2.6 and implementation guide section 2.5

**Test File: `mcp/tests/services/test_cache_service.py`**

**Cursor Prompt**:
```
Generate tests for CacheService:
- Test basic get/set
- Test TTL expiration
- Test LRU eviction
- Test cache disabled mode
- Test statistics tracking
- Test key generation
```

#### 3.3 Validation Service (Optional, can defer)

**File: `mcp/mcp_server/services/validation_service.py`**

```python
"""Validation service for input validation."""

from typing import Any, Dict, List, Optional
from pydantic import ValidationError as PydanticValidationError

from mcp_server.utils.exceptions import ValidationError

class ValidationService:
    """Service for validating inputs and parameters."""
    
    @staticmethod
    def validate_simulation_id(simulation_id: str) -> str:
        """Validate simulation ID format."""
        if not simulation_id or not isinstance(simulation_id, str):
            raise ValidationError("simulation_id must be a non-empty string")
        return simulation_id
    
    @staticmethod
    def validate_step_range(start: Optional[int], end: Optional[int]) -> tuple:
        """Validate step range."""
        if start is not None and start < 0:
            raise ValidationError("start_step must be >= 0")
        if end is not None and end < 0:
            raise ValidationError("end_step must be >= 0")
        if start is not None and end is not None and start > end:
            raise ValidationError("start_step must be <= end_step")
        return start, end
    
    @staticmethod
    def validate_limit(limit: int, max_limit: int = 10000) -> int:
        """Validate limit parameter."""
        if limit < 1:
            raise ValidationError("limit must be >= 1")
        if limit > max_limit:
            raise ValidationError(f"limit must be <= {max_limit}")
        return limit
```

#### 3.4 Cursor Tasks for Step 3
- [ ] Implement database_service.py
- [ ] Write tests for database service
- [ ] Implement cache_service.py
- [ ] Write tests for cache service
- [ ] Implement validation_service.py (optional)
- [ ] Run tests: `pytest tests/services/ -v`
- [ ] Commit: "Add service layer (database, cache, validation)"

---

### Step 4: Base Tool Infrastructure (Day 2, Afternoon - 3 hours)

#### 4.1 Response Models

**File: `mcp/mcp_server/models/responses.py`**

```python
"""Response models for MCP tools."""

from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ToolMetadata(BaseModel):
    """Metadata included in tool responses."""
    tool: str = Field(..., description="Tool name")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    from_cache: bool = Field(False, description="Whether result came from cache")
    execution_time_ms: float = Field(0.0, description="Execution time in milliseconds")

class ToolError(BaseModel):
    """Error information."""
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ToolResponse(BaseModel):
    """Standard tool response format."""
    success: bool = Field(..., description="Whether operation succeeded")
    data: Optional[Any] = Field(None, description="Response data")
    metadata: ToolMetadata = Field(..., description="Response metadata")
    error: Optional[ToolError] = Field(None, description="Error information if failed")
```

#### 4.2 Base Tool Class

**File: `mcp/mcp_server/tools/base.py`**

**Cursor Prompt**:
```
Implement ToolBase abstract class following the design:

Abstract properties:
- name: str
- description: str
- parameters_schema: type[BaseModel]

Abstract method:
- execute(**params) -> Any

Concrete methods:
- __call__(**params) -> Dict - validate, cache check, execute, format response
- _format_response(data, from_cache, execution_time) -> Dict
- _format_error(error_type, message, details) -> Dict
- _get_cache_key(params) -> str
- get_schema() -> Dict - for MCP registration

Include comprehensive error handling and logging.
```

**Implementation**: See design doc section 2.3 and implementation guide section 2.6

**Test File: `mcp/tests/tools/test_base.py`**

**Cursor Prompt**:
```
Create tests for ToolBase using a concrete test implementation:
- Test successful execution
- Test parameter validation
- Test error handling
- Test caching behavior
- Test response formatting
- Test schema generation
```

#### 4.3 Cursor Tasks for Step 4
- [ ] Implement response models
- [ ] Implement ToolBase class
- [ ] Write tests for ToolBase
- [ ] Run tests: `pytest tests/tools/test_base.py -v`
- [ ] Commit: "Add base tool infrastructure"

---

### Step 5: First Working Tools (Day 3, Full Day - 6 hours)

#### 5.1 Metadata Tools (2 hours)

**File: `mcp/mcp_server/tools/metadata_tools.py`**

Implement these tools one at a time:

**Tool 1: GetSimulationInfoTool**

**Cursor Prompt**:
```
Implement GetSimulationInfoTool:
- Parameter: simulation_id (str)
- Query Simulation table for details
- Return: simulation_id, status, parameters, start_time, end_time, db_path
- Include error handling for not found
```

**Tool 2: ListSimulationsTool**

**Cursor Prompt**:
```
Implement ListSimulationsTool:
- Parameters: status (optional filter), limit, offset
- Query Simulation table with filters
- Return: list of simulations with pagination info
- Include total count
```

**Tool 3: GetExperimentInfoTool**

**Cursor Prompt**:
```
Implement GetExperimentInfoTool:
- Parameter: experiment_id (str)
- Query ExperimentModel for details
- Return: experiment metadata, variables, status, simulations count
```

**Tool 4: ListExperimentsTool**

**Cursor Prompt**:
```
Implement ListExperimentsTool:
- Parameters: status filter, limit, offset
- Query ExperimentModel
- Return: list of experiments with pagination
```

**Test File: `mcp/tests/tools/test_metadata_tools.py`**

**Cursor Prompt**:
```
Generate tests for all metadata tools:
- Test each tool with valid inputs
- Test filtering and pagination
- Test error cases (not found, invalid params)
- Use fixtures with sample data
```

#### 5.2 Query Tools - Part 1 (4 hours)

**File: `mcp/mcp_server/tools/query_tools.py`**

**Tool 1: QueryAgentsTool** (Priority #1)

**Cursor Prompt**:
```
Implement QueryAgentsTool following the design doc example:
- Parameters: simulation_id, agent_type (optional), generation (optional), 
  alive_only (bool), limit, offset
- Query AgentModel with filters
- Return: agents list with all attributes, pagination info
- Include proper error handling
```

**Tool 2: GetSimulationMetricsTool** (Priority #2)

**Cursor Prompt**:
```
Implement GetSimulationMetricsTool:
- Parameters: simulation_id, start_step (optional), end_step (optional)
- Query SimulationStepModel for metrics
- Return: time-series data of simulation metrics
- Include step_number, total_agents, births, deaths, resources, etc.
```

**Tool 3: QueryActionsTool**

**Cursor Prompt**:
```
Implement QueryActionsTool:
- Parameters: simulation_id, agent_id (optional), action_type (optional),
  start_step, end_step, limit, offset
- Query ActionModel with filters
- Return: actions with details, rewards, state changes
```

**Test File: `mcp/tests/tools/test_query_tools.py`**

**Cursor Prompt**:
```
Generate comprehensive tests for query tools:
- Test each tool with various filter combinations
- Test pagination
- Test edge cases (no results, invalid filters)
- Test performance with larger datasets
```

#### 5.3 Cursor Tasks for Step 5
- [ ] Implement GetSimulationInfoTool
- [ ] Implement ListSimulationsTool  
- [ ] Implement GetExperimentInfoTool
- [ ] Implement ListExperimentsTool
- [ ] Write tests for metadata tools
- [ ] Implement QueryAgentsTool
- [ ] Implement GetSimulationMetricsTool
- [ ] Implement QueryActionsTool
- [ ] Write tests for query tools
- [ ] Run all tests: `pytest tests/tools/ -v`
- [ ] Commit: "Add metadata and query tools"

---

### Step 6: Minimal Server (Day 4, Morning - 3 hours)

#### 6.1 Server Implementation

**File: `mcp/mcp_server/server.py`**

**Cursor Prompt**:
```
Implement SimulationMCPServer class:
- __init__(config: MCPConfig) - initialize services and FastMCP
- _register_tools() - register all implemented tools
- _create_tool_wrapper(tool) - create FastMCP-compatible wrapper
- get_tool(name) - retrieve tool by name
- list_tools() - list all registered tools
- get_cache_stats() - get cache statistics
- clear_cache() - clear cache
- run(**kwargs) - start the server
- close() - cleanup

Start with just the tools implemented so far (7-8 tools).
```

**Implementation**: See design doc section 2.1 and implementation guide section 6.1

#### 6.2 CLI Interface

**File: `mcp/mcp_server/cli.py`**

**Cursor Prompt**:
```
Implement CLI with argparse:
- --db-path (required) - database path
- --config (optional) - YAML config file
- --log-level (optional) - logging level
- --log-file (optional) - log file path
- --no-cache - disable caching
- --list-tools - list available tools and exit

Setup logging, load config, create server, run with error handling.
```

**Implementation**: See design doc section 6.2

#### 6.3 Main Entry Point

**File: `mcp/mcp_server/__main__.py`**

```python
"""Entry point for python -m mcp_server."""

from mcp_server.cli import main

if __name__ == "__main__":
    main()
```

#### 6.4 Package Init

**File: `mcp/mcp_server/__init__.py`**

```python
"""MCP Server for Simulation Analysis."""

from mcp_server.server import SimulationMCPServer
from mcp_server.config import MCPConfig

__version__ = "0.1.0"
__all__ = ["SimulationMCPServer", "MCPConfig"]
```

#### 6.5 Test Server

**File: `mcp/tests/test_server.py`**

**Cursor Prompt**:
```
Generate tests for SimulationMCPServer:
- Test initialization
- Test tool registration
- Test get_tool() method
- Test list_tools() method
- Test cache management
- Integration test: create server, call multiple tools
```

#### 6.6 Cursor Tasks for Step 6
- [ ] Implement server.py
- [ ] Implement cli.py
- [ ] Implement __main__.py
- [ ] Update __init__.py
- [ ] Write server tests
- [ ] Test CLI manually: `python -m mcp_server --help`
- [ ] Test with real DB: `python -m mcp_server --db-path /path/to/sim.db --list-tools`
- [ ] Commit: "Add server and CLI implementation"

---

### Step 7: Integration Testing (Day 4, Afternoon - 3 hours)

#### 7.1 Shared Test Fixtures

**File: `mcp/tests/conftest.py`**

**Cursor Prompt**:
```
Create comprehensive pytest fixtures:
- test_db_path - temporary database file
- test_db_with_data - populated test database with:
  * 5 simulations (varied status)
  * 2 experiments
  * Agents, states, actions for first simulation (100 steps)
  * Resources and interactions
- db_config - DatabaseConfig for test DB
- cache_config - CacheConfig for testing
- mcp_config - Complete MCPConfig
- db_service - DatabaseService instance
- cache_service - CacheService instance
- services - tuple of (db_service, cache_service)

Use SQLAlchemy to create and populate test data.
```

**Implementation**: See implementation guide section 5.1

#### 7.2 End-to-End Tests

**File: `mcp/tests/integration/test_end_to_end.py`**

**Cursor Prompt**:
```
Create end-to-end integration tests:
1. test_full_analysis_workflow - 
   - List simulations
   - Get simulation info
   - Analyze population
   - Query agents
   - Verify all succeed

2. test_error_handling_workflow -
   - Test with invalid simulation_id
   - Test with invalid parameters
   - Verify graceful error responses

3. test_caching_workflow -
   - Call same tool twice
   - Verify cache hit on second call
   - Call with different params
   - Verify cache miss

4. test_multi_tool_workflow -
   - Chain multiple tool calls
   - Use output from one tool as input to another
```

**Implementation**: See implementation guide section 5.2

#### 7.3 Performance Tests

**File: `mcp/tests/integration/test_performance.py`**

**Cursor Prompt**:
```
Create performance tests:
1. test_query_performance - queries complete in <2s
2. test_concurrent_queries - handle 10 concurrent requests
3. test_cache_performance - cached queries are 10x faster
4. test_large_result_sets - handle up to max result size
```

#### 7.4 Test Scripts

**File: `mcp/scripts/run_tests.sh`**

```bash
#!/bin/bash
# Run all tests with coverage

set -e

echo "Running tests with coverage..."
pytest tests/ \
    -v \
    --cov=mcp_server \
    --cov-report=html \
    --cov-report=term \
    --cov-report=xml

echo "Coverage report generated in htmlcov/index.html"
```

#### 7.5 Cursor Tasks for Step 7
- [ ] Implement conftest.py with fixtures
- [ ] Write end-to-end tests
- [ ] Write performance tests
- [ ] Create test runner script
- [ ] Run full test suite: `./scripts/run_tests.sh`
- [ ] Verify >80% coverage
- [ ] Commit: "Add integration and performance tests"

---

### Step 8: Remaining Query Tools (Day 5, Morning - 3 hours)

#### 8.1 Complete Query Tools

**File: `mcp/mcp_server/tools/query_tools.py`** (continue)

**Tool 4: QueryStatesTool**

**Cursor Prompt**:
```
Implement QueryStatesTool:
- Parameters: simulation_id, agent_id (optional), start_step, end_step, limit, offset
- Query AgentStateModel
- Return: agent states with position, health, resources over time
```

**Tool 5: QueryResourcesTool**

**Cursor Prompt**:
```
Implement QueryResourcesTool:
- Parameters: simulation_id, step_number (optional), limit, offset
- Query ResourceModel
- Return: resource positions and amounts
```

**Tool 6: QueryInteractionsTool**

**Cursor Prompt**:
```
Implement QueryInteractionsTool:
- Parameters: simulation_id, interaction_type (optional), start_step, end_step
- Query InteractionModel
- Return: interaction events with source, target, type, details
```

#### 8.2 Cursor Tasks for Step 8
- [ ] Implement QueryStatesTool
- [ ] Implement QueryResourcesTool
- [ ] Implement QueryInteractionsTool
- [ ] Add tests for new tools
- [ ] Update server to register new tools
- [ ] Run tests: `pytest tests/tools/test_query_tools.py -v`
- [ ] Commit: "Complete query tools"

---

### Step 9: Analysis Tools (Day 5-6, 8 hours total)

#### 9.1 Population Analysis Tool

**File: `mcp/mcp_server/tools/analysis_tools.py`**

**Tool 1: AnalyzePopulationDynamicsTool**

**Cursor Prompt**:
```
Implement AnalyzePopulationDynamicsTool following the design doc example:
- Parameters: simulation_id, start_step, end_step, include_chart
- Query SimulationStepModel for population data
- Calculate statistics (peak, average, growth rate)
- Break down by agent type
- Optionally generate ASCII chart
- Return comprehensive population analysis
```

**Implementation**: See design doc section 2.5

#### 9.2 Survival Analysis Tool

**Tool 2: AnalyzeSurvivalRatesTool**

**Cursor Prompt**:
```
Implement AnalyzeSurvivalRatesTool:
- Parameters: simulation_id, group_by ('generation' | 'agent_type')
- Query AgentModel for birth/death times
- Calculate survival rates by cohort
- Calculate average lifespan
- Return survival statistics and rates
```

#### 9.3 Resource Efficiency Tool

**Tool 3: AnalyzeResourceEfficiencyTool**

**Cursor Prompt**:
```
Implement AnalyzeResourceEfficiencyTool:
- Parameters: simulation_id, start_step, end_step
- Query SimulationStepModel for resource metrics
- Calculate efficiency trends
- Identify resource bottlenecks
- Return efficiency analysis
```

#### 9.4 Integrate Existing Analyzers

**Tool 4+: Leverage farm/database/analyzers/**

**Cursor Prompt**:
```
Create adapter tools that wrap existing analyzers from farm.database.analyzers:
- AnalyzeAgentPerformanceTool -> use farm.database.analyzers.agent_analyzer
- IdentifyCriticalEventsTool -> use temporal_pattern_analyzer
- AnalyzeSocialPatternsTool -> use behavior_clustering_analyzer

Each adapter should:
1. Take MCP parameters
2. Convert to analyzer format
3. Call existing analyzer
4. Format results for MCP response
```

Example adapter:
```python
from farm.database.analyzers.population_analyzer import PopulationAnalyzer

class AnalyzePopulationDynamicsToolV2(ToolBase):
    def __init__(self, db_service, cache_service):
        super().__init__(db_service, cache_service)
        self.analyzer = PopulationAnalyzer()
    
    def execute(self, **params):
        with self.db.get_session() as session:
            results = self.analyzer.analyze(
                session=session,
                simulation_id=params["simulation_id"]
            )
            return self._format_results(results)
```

#### 9.5 Chart Formatter

**File: `mcp/mcp_server/formatters/chart_formatter.py`**

**Cursor Prompt**:
```
Implement ChartFormatter for ASCII/Unicode charts:
- create_line_chart(data, x_labels, title, width, height) - line chart
- create_bar_chart(data, labels, title) - horizontal bar chart
- create_sparkline(data) - compact sparkline

Use simple ASCII characters for compatibility.
```

#### 9.6 Cursor Tasks for Step 9
- [ ] Implement AnalyzePopulationDynamicsTool
- [ ] Implement AnalyzeSurvivalRatesTool
- [ ] Implement AnalyzeResourceEfficiencyTool
- [ ] Implement AnalyzeAgentPerformanceTool (adapter)
- [ ] Implement IdentifyCriticalEventsTool
- [ ] Implement AnalyzeSocialPatternsTool
- [ ] Implement AnalyzeReproductionTool
- [ ] Implement ChartFormatter
- [ ] Write tests for all analysis tools
- [ ] Update server to register new tools
- [ ] Run tests: `pytest tests/tools/test_analysis_tools.py -v`
- [ ] Commit: "Add analysis tools and chart formatter"

---

### Step 10: Comparison Tools (Day 7, 4 hours)

#### 10.1 Comparison Tools

**File: `mcp/mcp_server/tools/comparison_tools.py`**

**Tool 1: CompareSimulationsTool**

**Cursor Prompt**:
```
Implement CompareSimulationsTool:
- Parameters: simulation_ids (list, 2-10), metrics (optional list)
- Query SimulationStepModel for each simulation
- Calculate statistics for each metric (mean, std, min, max)
- Calculate pairwise differences
- Return comparative analysis
```

**Tool 2: CompareParametersTool**

**Cursor Prompt**:
```
Implement CompareParametersTool:
- Parameters: parameter_name, values (optional filter)
- Find simulations with different values of parameter
- Group simulations by parameter value
- Compare outcomes across groups
- Return parameter impact analysis
```

**Tool 3: RankConfigurationsTool**

**Cursor Prompt**:
```
Implement RankConfigurationsTool:
- Parameters: metric_name, limit
- Query all simulations
- Rank by specified metric (e.g., final_population, average_reward)
- Return ranked list with parameters
```

#### 10.2 Cursor Tasks for Step 10
- [ ] Implement CompareSimulationsTool
- [ ] Implement CompareParametersTool
- [ ] Implement RankConfigurationsTool
- [ ] Write tests for comparison tools
- [ ] Update server to register tools
- [ ] Run tests: `pytest tests/tools/test_comparison_tools.py -v`
- [ ] Commit: "Add comparison tools"

---

### Step 11: Advanced Tools (Day 8, 4 hours)

#### 11.1 Advanced Analysis Tools

**File: `mcp/mcp_server/tools/advanced_tools.py`**

**Tool 1: BuildAgentLineageTool**

**Cursor Prompt**:
```
Implement BuildAgentLineageTool:
- Parameters: simulation_id, agent_id, depth (how many generations)
- Query ReproductionEventModel for parent-child relationships
- Build family tree structure
- Include agent attributes at each node
- Return lineage tree
```

**Tool 2: AnalyzeSpatialDistributionTool**

**Cursor Prompt**:
```
Implement AnalyzeSpatialDistributionTool:
- Parameters: simulation_id, step_number (or range)
- Query AgentStateModel for positions
- Calculate clustering metrics
- Identify spatial patterns
- Return spatial analysis
```

**Tool 3: GetAgentLifecycleTool**

**Cursor Prompt**:
```
Implement GetAgentLifecycleTool:
- Parameters: simulation_id, agent_id
- Query AgentModel for basic info
- Query AgentStateModel for all states
- Query ActionModel for all actions
- Query HealthIncident for health events
- Return complete agent history
```

#### 11.2 Cursor Tasks for Step 11
- [ ] Implement BuildAgentLineageTool
- [ ] Implement AnalyzeSpatialDistributionTool
- [ ] Implement GetAgentLifecycleTool
- [ ] Write tests for advanced tools
- [ ] Update server to register tools
- [ ] Run tests: `pytest tests/tools/test_advanced_tools.py -v`
- [ ] Commit: "Add advanced analysis tools"

---

### Step 12: Formatters & Output (Day 8, 2 hours)

#### 12.1 Output Formatters

**File: `mcp/mcp_server/formatters/markdown_formatter.py`**

**Cursor Prompt**:
```
Implement MarkdownFormatter:
- format_table(data, headers) - markdown table
- format_summary(data) - structured summary
- format_list(items) - markdown list
```

**File: `mcp/mcp_server/formatters/json_formatter.py`**

```python
"""JSON formatter for tool outputs."""

import json
from typing import Any

class JSONFormatter:
    """Format data as JSON."""
    
    @staticmethod
    def format(data: Any, indent: int = 2) -> str:
        """Format data as JSON string."""
        return json.dumps(data, indent=indent, default=str)
    
    @staticmethod
    def pretty_print(data: Any) -> str:
        """Pretty print JSON."""
        return json.dumps(data, indent=2, sort_keys=True, default=str)
```

#### 12.2 Cursor Tasks for Step 12
- [ ] Implement markdown_formatter.py
- [ ] Implement json_formatter.py
- [ ] Complete chart_formatter.py (if not done)
- [ ] Write formatter tests
- [ ] Run tests: `pytest tests/formatters/ -v`
- [ ] Commit: "Add output formatters"

---

### Step 13: Documentation (Day 9, Full Day - 6 hours)

#### 13.1 Main Documentation

**File: `mcp/README.md`** (expand)

**Cursor Prompt**:
```
Create comprehensive README with:
- Project overview
- Features list
- Quick start guide
- Installation instructions
- Basic usage examples
- Configuration guide
- Links to detailed docs
```

#### 13.2 Tool Documentation

**File: `mcp/docs/tools/metadata.md`**

**Cursor Prompt**: "Document all metadata tools with parameters, examples, and use cases"

**File: `mcp/docs/tools/query.md`**

**Cursor Prompt**: "Document all query tools with parameters, examples, and use cases"

**File: `mcp/docs/tools/analysis.md`**

**Cursor Prompt**: "Document all analysis tools with parameters, examples, and use cases"

**File: `mcp/docs/tools/comparison.md`**

**Cursor Prompt**: "Document all comparison tools with parameters, examples, and use cases"

#### 13.3 Usage Examples

**File: `mcp/docs/examples/basic_usage.md`**

```markdown
# Basic Usage Examples

## Starting the Server

```bash
mcp-server --db-path /path/to/simulation.db
```

## Example Queries

### List All Simulations
```python
tool = server.get_tool("list_simulations")
result = tool(limit=10)
```

### Analyze Population
```python
tool = server.get_tool("analyze_population_dynamics")
result = tool(simulation_id="sim_001")
```

[More examples...]
```

**File: `mcp/docs/examples/llm_integration.md`**

```markdown
# LLM Integration Guide

## Claude Desktop Configuration

Add to Claude Desktop config:

```json
{
  "mcpServers": {
    "simulation-analysis": {
      "command": "python",
      "args": ["-m", "mcp_server", "--db-path", "/path/to/simulation.db"]
    }
  }
}
```

[More examples...]
```

#### 13.4 API Documentation

**File: `mcp/docs/api/services.md`**
**File: `mcp/docs/api/tools.md`**
**File: `mcp/docs/api/formatters.md`**

Use docstring generation:

**File: `mcp/scripts/generate_tool_docs.py`**

```python
#!/usr/bin/env python
"""Generate tool documentation from code."""

import importlib
import inspect
from pathlib import Path

def generate_tool_docs():
    """Generate markdown docs for all tools."""
    # Import all tool modules
    # Extract tool classes
    # Generate markdown from docstrings and schemas
    pass

if __name__ == "__main__":
    generate_tool_docs()
```

#### 13.5 Cursor Tasks for Step 13
- [ ] Expand README.md
- [ ] Write tool documentation
- [ ] Write usage examples
- [ ] Write LLM integration guide
- [ ] Write API documentation
- [ ] Create doc generation script
- [ ] Commit: "Add comprehensive documentation"

---

### Step 14: Testing & QA (Day 10, Full Day - 6 hours)

#### 14.1 Complete Test Coverage

**Cursor Prompt for each component**:
```
Review test coverage for [component] and add tests for:
- Uncovered code paths
- Edge cases
- Error conditions
- Performance scenarios
```

Run coverage report:
```bash
pytest tests/ --cov=mcp_server --cov-report=html
open htmlcov/index.html
```

Target: >80% coverage overall, >95% for critical paths

#### 14.2 Manual Testing

Create test checklist:

```markdown
## Manual Test Checklist

### Server Startup
- [ ] Server starts with --db-path
- [ ] Server starts with --config
- [ ] Server shows helpful error if DB not found
- [ ] --list-tools shows all tools
- [ ] --help shows usage

### Tool Execution
- [ ] Each metadata tool works
- [ ] Each query tool works
- [ ] Each analysis tool works
- [ ] Each comparison tool works
- [ ] Each advanced tool works

### Error Handling
- [ ] Invalid simulation_id returns clear error
- [ ] Invalid parameters return validation errors
- [ ] Database errors handled gracefully
- [ ] Timeout works for long queries

### Performance
- [ ] Simple queries < 1s
- [ ] Complex queries < 2s
- [ ] Cache improves repeat queries
- [ ] No memory leaks

### Integration
- [ ] Works with Claude Desktop
- [ ] Works with other MCP clients
- [ ] Multiple concurrent requests work
```

#### 14.3 Performance Benchmarking

**File: `mcp/scripts/benchmark.py`**

```python
#!/usr/bin/env python
"""Benchmark tool performance."""

import time
from mcp_server.server import SimulationMCPServer
from mcp_server.config import MCPConfig

def benchmark_tools(db_path: str):
    """Benchmark all tools."""
    config = MCPConfig.from_db_path(db_path)
    server = SimulationMCPServer(config)
    
    results = {}
    
    for tool_name in server.list_tools():
        tool = server.get_tool(tool_name)
        # Run with test parameters
        # Measure execution time
        # Record results
    
    print_results(results)

if __name__ == "__main__":
    import sys
    benchmark_tools(sys.argv[1])
```

#### 14.4 Cursor Tasks for Step 14
- [ ] Review and improve test coverage
- [ ] Run full test suite
- [ ] Perform manual testing
- [ ] Run benchmarks
- [ ] Fix any issues found
- [ ] Document performance characteristics
- [ ] Commit: "Complete testing and QA"

---

### Step 15: Polish & Release (Day 11, Half Day - 3 hours)

#### 15.1 Code Quality

Run linters and formatters:

```bash
# Format code
black mcp_server/ tests/

# Lint
ruff check mcp_server/ tests/

# Type checking (optional)
mypy mcp_server/
```

#### 15.2 Package Metadata

**File: `mcp/MANIFEST.in`**

```
include README.md
include LICENSE
include requirements.txt
include config.example.yaml
include .env.example
recursive-include docs *.md
recursive-include examples *
```

**File: `mcp/LICENSE`**

Choose and add appropriate license.

#### 15.3 Release Checklist

```markdown
## Release Checklist

### Code
- [ ] All tests passing
- [ ] Code formatted (black)
- [ ] No linter warnings
- [ ] Type hints added (optional)
- [ ] Docstrings complete

### Documentation
- [ ] README complete
- [ ] Tool docs complete
- [ ] API docs complete
- [ ] Examples working
- [ ] CHANGELOG.md created

### Package
- [ ] Version number set
- [ ] Dependencies locked
- [ ] License added
- [ ] MANIFEST.in complete

### Testing
- [ ] Unit tests >80% coverage
- [ ] Integration tests passing
- [ ] Performance benchmarks run
- [ ] Manual testing complete
- [ ] LLM integration verified

### Release
- [ ] Git tags added
- [ ] Package built: `python -m build`
- [ ] Package tested: `pip install dist/*.whl`
- [ ] Documentation published
```

#### 15.4 Cursor Tasks for Step 15
- [ ] Format all code
- [ ] Run linters
- [ ] Add type hints where missing
- [ ] Complete all docstrings
- [ ] Create CHANGELOG.md
- [ ] Add LICENSE
- [ ] Create MANIFEST.in
- [ ] Build package
- [ ] Test installation
- [ ] Commit: "Polish for release"
- [ ] Tag release: `git tag v0.1.0`

---

## Summary Timeline

| Day | Phase | Hours | Deliverables |
|-----|-------|-------|--------------|
| 1 AM | Setup | 2 | Project structure, config, dependencies |
| 1 PM | Infrastructure | 4 | Config system, exceptions, logging |
| 2 AM | Services | 4 | Database service, cache service |
| 2 PM | Base Tools | 3 | ToolBase, response models |
| 3 | First Tools | 6 | Metadata tools (4), Query tools (3) |
| 4 AM | Server | 3 | Server, CLI, basic integration |
| 4 PM | Testing | 3 | Integration tests, fixtures |
| 5 AM | Query Tools | 3 | Remaining query tools (3) |
| 5 PM | Analysis | 3 | Analysis tools (3-4) |
| 6 | Analysis | 5 | More analysis tools, formatters |
| 7 | Comparison | 4 | Comparison tools (3) |
| 8 AM | Advanced | 4 | Advanced tools (3) |
| 8 PM | Formatters | 2 | Output formatters |
| 9 | Documentation | 6 | All documentation |
| 10 | QA | 6 | Testing, benchmarking |
| 11 AM | Polish | 3 | Code quality, release prep |

**Total: ~11 days, ~57 hours of focused work**

Can be compressed to 2-3 weeks with parallel work or extended to 4 weeks with more polish.

## Final Deliverables

After completion, you will have:

✅ **Working MCP Server** with 20+ tools
✅ **Comprehensive test suite** (>80% coverage)
✅ **Complete documentation**
✅ **LLM integration** ready
✅ **Performance optimized** (<2s queries)
✅ **Production ready** code

All in the `mcp/` directory, independent from `farm/` but importing its database models.

---

## Quick Command Reference

```bash
# Install in dev mode
cd mcp
pip install -e .

# Run tests
pytest tests/ -v
pytest tests/ --cov=mcp_server --cov-report=html

# Run server
python -m mcp_server --db-path /path/to/simulation.db
mcp-server --db-path /path/to/simulation.db

# Format code
black mcp_server/ tests/

# Lint
ruff check mcp_server/

# Build package
python -m build

# Install package
pip install dist/*.whl
```

## Next Steps

1. **Review this plan** - Adjust timeline based on your availability
2. **Start with Day 1** - Create project structure
3. **Follow sequentially** - Each step builds on previous
4. **Use Cursor AI** - Leverage prompts provided
5. **Test continuously** - Don't skip testing steps
6. **Commit often** - Save progress frequently

**Ready to start? Begin with Step 1! 🚀**