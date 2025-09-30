# FastMCP Server Implementation & Testing Guide

## 1. Implementation Overview

This guide provides a step-by-step approach to implementing and testing the FastMCP server for simulation database analysis in Cursor.

### 1.1 Implementation Phases

```
Phase 1: Foundation (Week 1)
├── Setup project structure
├── Implement configuration system
├── Create database service layer
└── Build base tool class

Phase 2: Core Tools (Week 2)
├── Implement metadata tools
├── Implement query tools
└── Add basic formatters

Phase 3: Analysis Tools (Week 3)
├── Implement analysis tools
├── Implement comparison tools
└── Add advanced formatters (charts)

Phase 4: Testing & Polish (Week 4)
├── Write comprehensive tests
├── Performance optimization
├── Documentation
└── Integration with existing codebase
```

### 1.2 Development Workflow in Cursor

```
1. Create feature branch for each phase
2. Use Cursor's AI assistance for boilerplate generation
3. Write tests before implementation (TDD when possible)
4. Use Cursor's inline suggestions for implementation
5. Run tests continuously with Cursor's terminal
6. Refactor with AI assistance
7. Code review and merge
```

## 2. Phase 1: Foundation

### 2.1 Project Structure Setup

**Task 1.1: Create Directory Structure**

```bash
# Create directory structure
mkdir -p farm/mcp/{tools,services,formatters,models,utils}
touch farm/mcp/__init__.py
touch farm/mcp/{server,config}.py
touch farm/mcp/tools/{__init__.py,base.py}
touch farm/mcp/services/{__init__.py,database_service.py,cache_service.py,validation_service.py}
touch farm/mcp/formatters/{__init__.py,base_formatter.py,json_formatter.py}
touch farm/mcp/models/{__init__.py,schemas.py,responses.py}
touch farm/mcp/utils/{__init__.py,exceptions.py,logging.py,helpers.py}
```

**Cursor Tips**:
- Use Cmd+Shift+P → "New File" to create files quickly
- Use folder tree view to navigate structure
- Use Cursor's AI to generate initial `__init__.py` contents

**Task 1.2: Setup Dependencies**

Update `requirements.txt`:
```txt
# Existing dependencies...

# MCP Server dependencies
fastmcp>=0.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

**Implementation Checklist**:
- [ ] Directory structure created
- [ ] Dependencies added to requirements.txt
- [ ] Virtual environment updated: `pip install -r requirements.txt`
- [ ] All `__init__.py` files created

### 2.2 Configuration System

**File: `farm/mcp/config.py`**

**Implementation Steps**:

1. **Define Pydantic Models** (Use Cursor AI):
   - Prompt: "Create Pydantic configuration models for database, cache, and server settings"
   - Review and adjust generated code

```python
from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Optional

class DatabaseConfig(BaseModel):
    """Database configuration."""
    path: str = Field(..., description="Path to database file")
    pool_size: int = Field(5, ge=1, le=20)
    query_timeout: int = Field(30, ge=5, le=300)
    read_only: bool = Field(True)
    
    @validator('path')
    def validate_path_exists(cls, v):
        """Ensure database file exists."""
        if not Path(v).exists():
            raise ValueError(f"Database file not found: {v}")
        return v

class CacheConfig(BaseModel):
    """Cache configuration."""
    max_size: int = Field(100, ge=0, le=1000)
    ttl_seconds: int = Field(300, ge=0)
    enabled: bool = Field(True)

class ServerConfig(BaseModel):
    """Server configuration."""
    max_result_size: int = Field(10000, ge=100, le=100000)
    default_limit: int = Field(100, ge=10, le=1000)
    log_level: str = Field("INFO")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

class MCPConfig(BaseModel):
    """Main MCP server configuration."""
    database: DatabaseConfig
    cache: CacheConfig = CacheConfig()
    server: ServerConfig = ServerConfig()
    
    @classmethod
    def from_db_path(cls, db_path: str, **kwargs):
        """Create config from database path with optional overrides."""
        return cls(
            database=DatabaseConfig(path=db_path),
            **kwargs
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

**Testing Strategy**:
```python
# tests/mcp/test_config.py
import pytest
from farm.mcp.config import MCPConfig, DatabaseConfig

def test_config_from_db_path(tmp_path):
    """Test creating config from database path."""
    db_file = tmp_path / "test.db"
    db_file.touch()
    
    config = MCPConfig.from_db_path(str(db_file))
    
    assert config.database.path == str(db_file)
    assert config.cache.enabled is True
    assert config.server.max_result_size == 10000

def test_config_validation_invalid_path():
    """Test that config validates database path exists."""
    with pytest.raises(ValueError, match="Database file not found"):
        MCPConfig.from_db_path("/nonexistent/path.db")

def test_config_validation_pool_size():
    """Test that pool size is validated."""
    # This should pass
    config = DatabaseConfig(path="test.db", pool_size=5)
    assert config.pool_size == 5
    
    # This should fail
    with pytest.raises(ValueError):
        DatabaseConfig(path="test.db", pool_size=100)
```

**Cursor Workflow**:
1. Open `farm/mcp/config.py`
2. Use Cursor Chat: "Implement the configuration classes from the design doc"
3. Review generated code
4. Open test file side-by-side (Cmd+\)
5. Use Cursor: "Generate unit tests for this configuration module"
6. Run tests: `pytest tests/mcp/test_config.py -v`

**Implementation Checklist**:
- [ ] Configuration models implemented
- [ ] Validators added for critical fields
- [ ] YAML loading method implemented
- [ ] Unit tests written and passing
- [ ] Documentation added

### 2.3 Custom Exceptions

**File: `farm/mcp/utils/exceptions.py`**

```python
"""Custom exceptions for MCP server."""

class MCPException(Exception):
    """Base exception for MCP server."""
    pass

class DatabaseError(MCPException):
    """Database-related errors."""
    pass

class ValidationError(MCPException):
    """Parameter validation errors."""
    pass

class QueryTimeoutError(DatabaseError):
    """Query exceeded timeout limit."""
    pass

class SimulationNotFoundError(DatabaseError):
    """Requested simulation does not exist."""
    
    def __init__(self, simulation_id: str):
        self.simulation_id = simulation_id
        super().__init__(f"Simulation not found: {simulation_id}")

class ResultTooLargeError(MCPException):
    """Query result exceeds maximum size."""
    
    def __init__(self, size: int, max_size: int):
        self.size = size
        self.max_size = max_size
        super().__init__(
            f"Result size ({size}) exceeds maximum ({max_size})"
        )

class CacheError(MCPException):
    """Cache-related errors."""
    pass

class ToolNotFoundError(MCPException):
    """Requested tool does not exist."""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool not found: {tool_name}")
```

**Cursor Workflow**:
- Use Cursor: "Create custom exception classes following the design"
- Add docstrings to each exception

### 2.4 Database Service Layer

**File: `farm/mcp/services/database_service.py`**

**Implementation Steps**:

1. **Create Session Manager Wrapper**

```python
"""Database service for MCP server."""

import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from farm.database.models import Base, Simulation
from farm.mcp.config import DatabaseConfig
from farm.mcp.utils.exceptions import (
    DatabaseError,
    QueryTimeoutError,
    SimulationNotFoundError
)

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database operations with connection management."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database service.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self._engine = None
        self._SessionFactory = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine and session factory."""
        try:
            # Create engine with connection pooling
            connect_args = {
                "timeout": self.config.query_timeout,
                "check_same_thread": False,
            }
            
            # Add read-only mode if enabled
            if self.config.read_only:
                connect_args["uri"] = True
                db_url = f"file:{self.config.path}?mode=ro"
            else:
                db_url = f"sqlite:///{self.config.path}"
            
            self._engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=2,
                pool_pre_ping=True,  # Verify connections before use
                connect_args=connect_args,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self._SessionFactory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False
            )
            
            logger.info(
                f"Database service initialized: {self.config.path} "
                f"(read_only={self.config.read_only})"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    @contextmanager
    def get_session(self) -> Session:
        """Provide a transactional scope for database operations.
        
        Yields:
            SQLAlchemy Session
            
        Raises:
            DatabaseError: If session operations fail
        """
        session = self._SessionFactory()
        try:
            yield session
            # Read-only mode, no commit needed
            if not self.config.read_only:
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Query execution failed: {e}")
        finally:
            session.close()
    
    def execute_query(
        self, 
        query_func: Callable[[Session], Any],
        timeout: Optional[int] = None
    ) -> Any:
        """Execute a query function with error handling.
        
        Args:
            query_func: Function that takes a session and returns results
            timeout: Optional query timeout override
            
        Returns:
            Query results
            
        Raises:
            DatabaseError: If query execution fails
            QueryTimeoutError: If query exceeds timeout
        """
        # Use provided timeout or default from config
        query_timeout = timeout or self.config.query_timeout
        
        with self.get_session() as session:
            try:
                # Set statement timeout (SQLite doesn't support this natively,
                # but we can use a timer for monitoring)
                import signal
                
                def timeout_handler(signum, frame):
                    raise QueryTimeoutError(
                        f"Query exceeded timeout of {query_timeout}s"
                    )
                
                # Set timeout alarm (Unix-like systems only)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(query_timeout)
                except AttributeError:
                    # Windows doesn't support SIGALRM, skip timeout enforcement
                    pass
                
                # Execute query
                result = query_func(session)
                
                # Cancel alarm
                try:
                    signal.alarm(0)
                except AttributeError:
                    pass
                
                return result
                
            except QueryTimeoutError:
                raise
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise DatabaseError(f"Query failed: {e}")
    
    def validate_simulation_exists(self, simulation_id: str) -> bool:
        """Check if simulation exists in database.
        
        Args:
            simulation_id: Simulation ID to check
            
        Returns:
            True if simulation exists, False otherwise
        """
        def check_exists(session: Session) -> bool:
            return session.query(Simulation).filter_by(
                simulation_id=simulation_id
            ).first() is not None
        
        return self.execute_query(check_exists)
    
    def get_simulation(self, simulation_id: str) -> Simulation:
        """Get simulation by ID.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Simulation model instance
            
        Raises:
            SimulationNotFoundError: If simulation doesn't exist
        """
        def get_sim(session: Session) -> Simulation:
            sim = session.query(Simulation).filter_by(
                simulation_id=simulation_id
            ).first()
            
            if not sim:
                raise SimulationNotFoundError(simulation_id)
            
            return sim
        
        return self.execute_query(get_sim)
    
    def close(self):
        """Close database connections and dispose of engine."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database service closed")
```

**Testing Strategy**:

```python
# tests/mcp/services/test_database_service.py
import pytest
from pathlib import Path
from farm.mcp.services.database_service import DatabaseService
from farm.mcp.config import DatabaseConfig
from farm.mcp.utils.exceptions import (
    SimulationNotFoundError,
    DatabaseError
)
from farm.database.models import Base, Simulation
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def test_db(tmp_path):
    """Create a test database with sample data."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    
    # Add test simulation
    Session = sessionmaker(bind=engine)
    session = Session()
    sim = Simulation(
        simulation_id="test_sim_001",
        status="completed",
        parameters={"agents": 100},
        simulation_db_path=str(db_path)
    )
    session.add(sim)
    session.commit()
    session.close()
    
    return db_path

@pytest.fixture
def db_service(test_db):
    """Create database service instance."""
    config = DatabaseConfig(path=str(test_db))
    return DatabaseService(config)

def test_database_service_initialization(db_service):
    """Test that database service initializes correctly."""
    assert db_service._engine is not None
    assert db_service._SessionFactory is not None

def test_get_session(db_service):
    """Test session context manager."""
    with db_service.get_session() as session:
        assert session is not None
        # Session should be usable
        result = session.query(Simulation).first()
        assert result is not None

def test_validate_simulation_exists(db_service):
    """Test simulation existence check."""
    assert db_service.validate_simulation_exists("test_sim_001") is True
    assert db_service.validate_simulation_exists("nonexistent") is False

def test_get_simulation(db_service):
    """Test retrieving simulation by ID."""
    sim = db_service.get_simulation("test_sim_001")
    assert sim.simulation_id == "test_sim_001"
    assert sim.status == "completed"

def test_get_simulation_not_found(db_service):
    """Test that getting nonexistent simulation raises error."""
    with pytest.raises(SimulationNotFoundError):
        db_service.get_simulation("nonexistent")

def test_execute_query(db_service):
    """Test execute_query method."""
    def query_func(session):
        return session.query(Simulation).count()
    
    count = db_service.execute_query(query_func)
    assert count == 1

def test_execute_query_error_handling(db_service):
    """Test that query errors are handled properly."""
    def failing_query(session):
        raise ValueError("Test error")
    
    with pytest.raises(DatabaseError):
        db_service.execute_query(failing_query)

def test_read_only_mode(tmp_path, test_db):
    """Test that read-only mode is enforced."""
    config = DatabaseConfig(path=str(test_db), read_only=True)
    service = DatabaseService(config)
    
    # Should be able to read
    sim = service.get_simulation("test_sim_001")
    assert sim is not None
    
    # Write operations should fail (if we tried to commit)
    # Note: SQLite's read-only mode is enforced at connection level
```

**Cursor Workflow**:
1. Open `farm/mcp/services/database_service.py`
2. Use Cursor: "Implement DatabaseService class from design doc"
3. Review and refine generated code
4. Split screen with test file
5. Use Cursor: "Generate comprehensive tests for DatabaseService"
6. Run tests: `pytest tests/mcp/services/test_database_service.py -v`
7. Iterate on failures

**Implementation Checklist**:
- [ ] DatabaseService class implemented
- [ ] Session management working
- [ ] Error handling comprehensive
- [ ] Read-only mode enforced
- [ ] Unit tests written (>80% coverage)
- [ ] Integration tests with real database
- [ ] Documentation complete

### 2.5 Cache Service

**File: `farm/mcp/services/cache_service.py`**

**Cursor Workflow**:
1. Use Cursor: "Implement CacheService with LRU eviction and TTL"
2. Reference design document section 2.6
3. Generate tests alongside implementation

```python
"""Cache service for MCP server."""

import time
import hashlib
import json
from typing import Any, Optional
from collections import OrderedDict
import logging

from farm.mcp.config import CacheConfig

logger = logging.getLogger(__name__)

class CacheService:
    """In-memory cache with TTL and LRU eviction."""
    
    def __init__(self, config: CacheConfig):
        """Initialize cache service.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.enabled = config.enabled
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None
        
        if key not in self._cache:
            self._misses += 1
            return None
        
        # Check TTL
        if self.config.ttl_seconds > 0:
            age = time.time() - self._timestamps[key]
            if age > self.config.ttl_seconds:
                self._evict(key)
                self._misses += 1
                return None
        
        # Move to end (LRU)
        self._cache.move_to_end(key)
        self._hits += 1
        
        logger.debug(f"Cache hit: {key}")
        return self._cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.enabled:
            return
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.config.max_size and key not in self._cache:
            oldest_key = next(iter(self._cache))
            self._evict(oldest_key)
            logger.debug(f"Cache eviction: {oldest_key}")
        
        self._cache[key] = value
        self._timestamps[key] = time.time()
        self._cache.move_to_end(key)
        
        logger.debug(f"Cache set: {key}")
    
    def _evict(self, key: str):
        """Remove key from cache.
        
        Args:
            key: Cache key to evict
        """
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
    
    def clear(self):
        """Clear entire cache."""
        self._cache.clear()
        self._timestamps.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.config.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.config.ttl_seconds
        }
    
    @staticmethod
    def generate_key(tool_name: str, params: dict) -> str:
        """Generate cache key from tool name and parameters.
        
        Args:
            tool_name: Name of the tool
            params: Tool parameters
            
        Returns:
            Cache key string
        """
        # Sort parameters for consistent hashing
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{tool_name}:{param_hash}"
```

**Testing**:

```python
# tests/mcp/services/test_cache_service.py
import pytest
import time
from farm.mcp.services.cache_service import CacheService
from farm.mcp.config import CacheConfig

@pytest.fixture
def cache_service():
    """Create cache service with test config."""
    config = CacheConfig(max_size=3, ttl_seconds=1, enabled=True)
    return CacheService(config)

def test_cache_set_and_get(cache_service):
    """Test basic cache operations."""
    cache_service.set("key1", "value1")
    assert cache_service.get("key1") == "value1"

def test_cache_miss(cache_service):
    """Test cache miss returns None."""
    assert cache_service.get("nonexistent") is None

def test_cache_ttl_expiration(cache_service):
    """Test that cache entries expire after TTL."""
    cache_service.set("key1", "value1")
    assert cache_service.get("key1") == "value1"
    
    # Wait for TTL to expire
    time.sleep(1.5)
    assert cache_service.get("key1") is None

def test_cache_lru_eviction(cache_service):
    """Test LRU eviction when cache is full."""
    # Fill cache to max size (3)
    cache_service.set("key1", "value1")
    cache_service.set("key2", "value2")
    cache_service.set("key3", "value3")
    
    # Add one more, should evict oldest (key1)
    cache_service.set("key4", "value4")
    
    assert cache_service.get("key1") is None
    assert cache_service.get("key2") == "value2"
    assert cache_service.get("key3") == "value3"
    assert cache_service.get("key4") == "value4"

def test_cache_disabled():
    """Test that cache can be disabled."""
    config = CacheConfig(enabled=False)
    cache = CacheService(config)
    
    cache.set("key1", "value1")
    assert cache.get("key1") is None

def test_cache_stats(cache_service):
    """Test cache statistics."""
    cache_service.set("key1", "value1")
    cache_service.get("key1")  # hit
    cache_service.get("key2")  # miss
    
    stats = cache_service.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["size"] == 1

def test_generate_key():
    """Test cache key generation."""
    key1 = CacheService.generate_key("tool1", {"a": 1, "b": 2})
    key2 = CacheService.generate_key("tool1", {"b": 2, "a": 1})
    key3 = CacheService.generate_key("tool1", {"a": 1, "b": 3})
    
    # Same params in different order should produce same key
    assert key1 == key2
    # Different params should produce different key
    assert key1 != key3
```

**Implementation Checklist**:
- [ ] CacheService implemented
- [ ] TTL expiration working
- [ ] LRU eviction working
- [ ] Statistics tracking implemented
- [ ] Tests passing
- [ ] Documentation complete

### 2.6 Base Tool Class

**File: `farm/mcp/tools/base.py`**

This is a critical component. Use Cursor extensively here.

**Cursor Workflow**:
1. Open design doc section 2.3 in one pane
2. Open `farm/mcp/tools/base.py` in another
3. Cursor Chat: "Implement ToolBase abstract class from this design"
4. Review and refine

```python
"""Base class for all MCP tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import logging

from pydantic import BaseModel, ValidationError

from farm.mcp.services.database_service import DatabaseService
from farm.mcp.services.cache_service import CacheService
from farm.mcp.utils.exceptions import MCPException, DatabaseError

logger = logging.getLogger(__name__)

class ToolBase(ABC):
    """Base class for all MCP tools.
    
    Provides common functionality:
    - Parameter validation via Pydantic
    - Response formatting
    - Error handling
    - Caching integration
    - Logging
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        cache_service: CacheService,
    ):
        """Initialize tool with required services.
        
        Args:
            db_service: Database service instance
            cache_service: Cache service instance
        """
        self.db = db_service
        self.cache = cache_service
    
    # Abstract properties that subclasses must implement
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for registration and logging."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description for LLM consumption."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> type[BaseModel]:
        """Pydantic schema for parameter validation."""
        pass
    
    @abstractmethod
    def execute(self, **params) -> Any:
        """Execute the tool with validated parameters.
        
        Args:
            **params: Validated parameters from schema
            
        Returns:
            Tool execution result
        """
        pass
    
    # Concrete methods
    
    def __call__(self, **params) -> Dict[str, Any]:
        """Validate and execute tool with error handling.
        
        This is the main entry point for tool execution.
        
        Args:
            **params: Raw parameters from MCP request
            
        Returns:
            Structured response dictionary
        """
        start_time = datetime.now()
        
        try:
            # Validate parameters using Pydantic schema
            validated_params = self.parameters_schema(**params)
            
            # Check cache if enabled
            cache_key = self._get_cache_key(validated_params)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                logger.info(f"Tool {self.name}: Cache hit")
                return self._format_response(
                    data=cached_result,
                    from_cache=True,
                    execution_time_ms=0
                )
            
            # Execute tool
            logger.info(f"Tool {self.name}: Executing with params: {params}")
            result = self.execute(**validated_params.dict())
            
            # Cache result
            self.cache.set(cache_key, result)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return self._format_response(
                data=result,
                from_cache=False,
                execution_time_ms=execution_time
            )
            
        except ValidationError as e:
            logger.warning(f"Tool {self.name}: Validation error: {e}")
            return self._format_error("ValidationError", str(e), e.errors())
        
        except DatabaseError as e:
            logger.error(f"Tool {self.name}: Database error: {e}")
            return self._format_error("DatabaseError", str(e))
        
        except MCPException as e:
            logger.error(f"Tool {self.name}: MCP error: {e}")
            return self._format_error(type(e).__name__, str(e))
        
        except Exception as e:
            logger.exception(f"Tool {self.name}: Unexpected error: {e}")
            return self._format_error("UnknownError", str(e))
    
    def _format_response(
        self,
        data: Any,
        from_cache: bool = False,
        execution_time_ms: float = 0
    ) -> Dict[str, Any]:
        """Format successful response.
        
        Args:
            data: Result data
            from_cache: Whether result came from cache
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            Formatted response dictionary
        """
        return {
            "success": True,
            "data": data,
            "metadata": {
                "tool": self.name,
                "timestamp": datetime.now().isoformat(),
                "from_cache": from_cache,
                "execution_time_ms": execution_time_ms,
            },
            "error": None
        }
    
    def _format_error(
        self,
        error_type: str,
        message: str,
        details: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Format error response.
        
        Args:
            error_type: Type of error
            message: Error message
            details: Optional additional error details
            
        Returns:
            Formatted error response
        """
        error_dict = {
            "type": error_type,
            "message": message,
        }
        
        if details:
            error_dict["details"] = details
        
        return {
            "success": False,
            "data": None,
            "metadata": {
                "tool": self.name,
                "timestamp": datetime.now().isoformat(),
            },
            "error": error_dict
        }
    
    def _get_cache_key(self, params: BaseModel) -> str:
        """Generate cache key from parameters.
        
        Args:
            params: Validated parameters
            
        Returns:
            Cache key string
        """
        return CacheService.generate_key(self.name, params.dict())
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for MCP registration.
        
        Returns:
            Tool schema dictionary for FastMCP
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema.schema()
        }
```

**Testing**:

```python
# tests/mcp/tools/test_base.py
import pytest
from pydantic import BaseModel, Field
from farm.mcp.tools.base import ToolBase
from farm.mcp.services.database_service import DatabaseService
from farm.mcp.services.cache_service import CacheService
from farm.mcp.config import DatabaseConfig, CacheConfig

# Create a concrete implementation for testing
class TestToolParams(BaseModel):
    """Test parameter schema."""
    value: int = Field(..., ge=0, le=100)
    name: str = Field(..., min_length=1)

class TestTool(ToolBase):
    """Concrete tool for testing."""
    
    @property
    def name(self) -> str:
        return "test_tool"
    
    @property
    def description(self) -> str:
        return "A tool for testing"
    
    @property
    def parameters_schema(self):
        return TestToolParams
    
    def execute(self, **params):
        return {"result": f"Executed with {params}"}

@pytest.fixture
def services(test_db):
    """Create service instances."""
    db_config = DatabaseConfig(path=str(test_db))
    cache_config = CacheConfig(enabled=True)
    
    db_service = DatabaseService(db_config)
    cache_service = CacheService(cache_config)
    
    return db_service, cache_service

@pytest.fixture
def test_tool(services):
    """Create test tool instance."""
    db_service, cache_service = services
    return TestTool(db_service, cache_service)

def test_tool_successful_execution(test_tool):
    """Test successful tool execution."""
    result = test_tool(value=42, name="test")
    
    assert result["success"] is True
    assert "result" in result["data"]
    assert result["error"] is None
    assert result["metadata"]["tool"] == "test_tool"

def test_tool_validation_error(test_tool):
    """Test that validation errors are handled."""
    result = test_tool(value=200, name="test")  # value out of range
    
    assert result["success"] is False
    assert result["error"]["type"] == "ValidationError"

def test_tool_caching(test_tool):
    """Test that results are cached."""
    # First call
    result1 = test_tool(value=42, name="test")
    assert result1["metadata"]["from_cache"] is False
    
    # Second call with same params should hit cache
    result2 = test_tool(value=42, name="test")
    assert result2["metadata"]["from_cache"] is True

def test_tool_schema_generation(test_tool):
    """Test tool schema generation."""
    schema = test_tool.get_schema()
    
    assert schema["name"] == "test_tool"
    assert "description" in schema
    assert "parameters" in schema
```

**Implementation Checklist**:
- [ ] ToolBase class implemented
- [ ] Abstract methods defined
- [ ] Error handling comprehensive
- [ ] Caching integration working
- [ ] Response formatting standardized
- [ ] Tests passing
- [ ] Documentation complete

## 3. Phase 2: Core Tools Implementation

### 3.1 Metadata Tools

**File: `farm/mcp/tools/metadata_tools.py`**

Implement these tools:
1. `ListSimulationsTool`
2. `GetSimulationInfoTool`
3. `ListExperimentsTool`
4. `GetExperimentInfoTool`

**Cursor Workflow for Each Tool**:
1. Use Cursor: "Implement ListSimulationsTool based on the design and ToolBase"
2. Review generated code
3. Add docstrings and type hints
4. Generate tests: "Create comprehensive tests for ListSimulationsTool"
5. Run tests and iterate

**Example: ListSimulationsTool**

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from farm.mcp.tools.base import ToolBase
from farm.database.models import Simulation

class ListSimulationsParams(BaseModel):
    """Parameters for list_simulations tool."""
    status: Optional[str] = Field(None, description="Filter by status")
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)

class ListSimulationsTool(ToolBase):
    """List all simulations with optional filtering."""
    
    @property
    def name(self) -> str:
        return "list_simulations"
    
    @property
    def description(self) -> str:
        return """
        List all simulations in the database with optional filtering.
        
        Returns simulation metadata including:
        - Simulation ID
        - Status (completed, running, failed)
        - Start and end times
        - Parameter summary
        
        Use this to:
        - Get overview of all simulations
        - Find simulations by status
        - Navigate available simulation data
        """
    
    @property
    def parameters_schema(self):
        return ListSimulationsParams
    
    def execute(self, **params):
        """Execute simulation listing."""
        def query_func(session):
            # Build query
            query = session.query(Simulation)
            
            # Apply filters
            if params.get("status"):
                query = query.filter(Simulation.status == params["status"])
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            query = query.limit(params["limit"]).offset(params["offset"])
            
            # Execute
            simulations = query.all()
            
            # Serialize
            results = [
                {
                    "simulation_id": s.simulation_id,
                    "status": s.status,
                    "start_time": s.start_time.isoformat() if s.start_time else None,
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "parameters_summary": {
                        k: v for k, v in list(s.parameters.items())[:5]
                    } if s.parameters else {},
                    "db_path": s.simulation_db_path
                }
                for s in simulations
            ]
            
            return {
                "simulations": results,
                "total_count": total,
                "returned_count": len(results),
                "limit": params["limit"],
                "offset": params["offset"]
            }
        
        return self.db.execute_query(query_func)
```

**Testing Pattern**:

```python
# tests/mcp/tools/test_metadata_tools.py
def test_list_simulations_tool(services, test_db_with_data):
    """Test listing simulations."""
    db_service, cache_service = services
    tool = ListSimulationsTool(db_service, cache_service)
    
    result = tool()
    
    assert result["success"] is True
    assert "simulations" in result["data"]
    assert len(result["data"]["simulations"]) > 0

def test_list_simulations_with_filter(services, test_db_with_data):
    """Test filtering simulations by status."""
    db_service, cache_service = services
    tool = ListSimulationsTool(db_service, cache_service)
    
    result = tool(status="completed")
    
    assert result["success"] is True
    # Verify all returned simulations have correct status
    for sim in result["data"]["simulations"]:
        assert sim["status"] == "completed"
```

**Implementation Checklist**:
- [ ] ListSimulationsTool implemented and tested
- [ ] GetSimulationInfoTool implemented and tested
- [ ] ListExperimentsTool implemented and tested
- [ ] GetExperimentInfoTool implemented and tested
- [ ] All tests passing (>80% coverage)

### 3.2 Query Tools

**File: `farm/mcp/tools/query_tools.py`**

Implement these tools (reference design section 2.4):
1. `QueryAgentsTool`
2. `QueryActionsTool`
3. `QueryStatesTool`
4. `QueryResourcesTool`
5. `QueryInteractionsTool`
6. `GetSimulationMetricsTool`

**Cursor Workflow**:
- Use existing example from design (QueryAgentsTool)
- Generate similar tools: "Create QueryActionsTool following the same pattern as QueryAgentsTool"
- Batch test generation: "Generate tests for all query tools"

**Implementation Checklist**:
- [ ] All 6 query tools implemented
- [ ] Parameter schemas validated
- [ ] Pagination working correctly
- [ ] Filters working correctly
- [ ] Tests passing for all tools
- [ ] Performance acceptable (<2s for typical queries)

## 4. Phase 3: Analysis Tools

### 4.1 Analysis Tools Implementation

**File: `farm/mcp/tools/analysis_tools.py`**

Tools to implement:
1. `AnalyzePopulationDynamicsTool` (example in design section 2.5)
2. `AnalyzeSurvivalRatesTool`
3. `AnalyzeResourceEfficiencyTool`
4. `AnalyzeAgentPerformanceTool`
5. `IdentifyCriticalEventsTool`
6. `AnalyzeSocialPatternsTool`
7. `AnalyzeReproductionTool`

**Integration with Existing Analyzers**:

```python
# Example: Leverage existing analyzer
from farm.database.analyzers.population_analyzer import PopulationAnalyzer

class AnalyzePopulationDynamicsToolV2(ToolBase):
    """Use existing analyzer with MCP interface."""
    
    def __init__(self, db_service, cache_service):
        super().__init__(db_service, cache_service)
        # Reuse existing analyzer
        self.analyzer = PopulationAnalyzer()
    
    def execute(self, **params):
        """Execute using existing analyzer."""
        with self.db.get_session() as session:
            # Use existing analyzer methods
            results = self.analyzer.analyze_population_trends(
                session=session,
                simulation_id=params["simulation_id"],
                start_step=params.get("start_step"),
                end_step=params.get("end_step")
            )
            
            # Format for MCP response
            return self._format_analyzer_results(results)
    
    def _format_analyzer_results(self, results):
        """Convert analyzer output to MCP format."""
        # Transform results as needed
        return results
```

**Cursor Workflow**:
1. Review existing analyzers in `farm/database/analyzers/`
2. Use Cursor: "Create MCP tool wrapper for PopulationAnalyzer"
3. Test integration: "Generate integration tests for analyzer tools"

**Implementation Checklist**:
- [ ] All analysis tools implemented
- [ ] Integration with existing analyzers where applicable
- [ ] New analysis logic for tools without existing analyzers
- [ ] Chart formatting for visual tools
- [ ] Statistical calculations validated
- [ ] Tests passing

### 4.2 Comparison Tools

**File: `farm/mcp/tools/comparison_tools.py`**

Tools to implement:
1. `CompareSimulationsTool`
2. `CompareParametersTool`
3. `RankConfigurationsTool`

**Example: CompareSimulationsTool**

```python
class CompareSimulationsParams(BaseModel):
    """Parameters for comparing simulations."""
    simulation_ids: List[str] = Field(..., min_items=2, max_items=10)
    metrics: Optional[List[str]] = Field(
        None,
        description="Metrics to compare (e.g., 'total_agents', 'average_reward')"
    )

class CompareSimulationsTool(ToolBase):
    """Compare metrics across multiple simulations."""
    
    # ... implementation
    
    def execute(self, **params):
        """Execute comparison."""
        def query_func(session):
            results = {}
            
            for sim_id in params["simulation_ids"]:
                # Validate simulation exists
                if not self.db.validate_simulation_exists(sim_id):
                    raise SimulationNotFoundError(sim_id)
                
                # Get simulation metrics
                steps = session.query(SimulationStepModel).filter(
                    SimulationStepModel.simulation_id == sim_id
                ).all()
                
                # Calculate statistics
                import numpy as np
                metrics_to_compare = params.get("metrics") or [
                    "total_agents", "average_agent_health", "average_reward"
                ]
                
                sim_stats = {}
                for metric in metrics_to_compare:
                    values = [getattr(s, metric) for s in steps if getattr(s, metric) is not None]
                    if values:
                        sim_stats[metric] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                        }
                
                results[sim_id] = sim_stats
            
            # Calculate differences
            comparison = self._calculate_differences(results, params["simulation_ids"])
            
            return {
                "simulations": results,
                "comparison": comparison,
                "metric_names": params.get("metrics") or metrics_to_compare
            }
        
        return self.db.execute_query(query_func)
    
    def _calculate_differences(self, results, sim_ids):
        """Calculate pairwise differences."""
        # Implementation...
        pass
```

**Implementation Checklist**:
- [ ] Comparison tools implemented
- [ ] Statistical comparisons accurate
- [ ] Support for multiple simulations
- [ ] Clear result formatting
- [ ] Tests passing

## 5. Phase 4: Testing & Integration

### 5.1 Comprehensive Test Suite

**Test Structure**:

```
tests/mcp/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_config.py
├── services/
│   ├── test_database_service.py
│   ├── test_cache_service.py
│   └── test_validation_service.py
├── tools/
│   ├── test_base.py
│   ├── test_metadata_tools.py
│   ├── test_query_tools.py
│   ├── test_analysis_tools.py
│   └── test_comparison_tools.py
├── integration/
│   ├── test_end_to_end.py
│   └── test_performance.py
└── test_server.py
```

**Shared Fixtures** (`conftest.py`):

```python
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from farm.database.models import Base, Simulation, AgentModel, SimulationStepModel
from farm.mcp.config import MCPConfig, DatabaseConfig, CacheConfig
from farm.mcp.services import DatabaseService, CacheService

@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """Create a temporary database for testing."""
    db_path = tmp_path_factory.mktemp("data") / "test.db"
    return db_path

@pytest.fixture(scope="session")
def test_db_with_data(test_db_path):
    """Create and populate test database."""
    engine = create_engine(f"sqlite:///{test_db_path}")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create test simulations
    for i in range(5):
        sim = Simulation(
            simulation_id=f"test_sim_{i:03d}",
            status="completed" if i % 2 == 0 else "running",
            parameters={"agents": 100 + i * 10, "resources": 1000},
            simulation_db_path=str(test_db_path)
        )
        session.add(sim)
        
        # Add agents and steps for first simulation
        if i == 0:
            # Add agents
            for j in range(10):
                agent = AgentModel(
                    simulation_id=sim.simulation_id,
                    agent_id=f"agent_{j:03d}",
                    agent_type="system" if j % 3 == 0 else "independent",
                    birth_time=0,
                    death_time=100 if j % 2 == 0 else None,
                    generation=j // 3,
                    position_x=float(j),
                    position_y=float(j),
                    initial_resources=50.0,
                    starting_health=100.0,
                    starvation_counter=0,
                    genome_id=f"genome_{j % 3}"
                )
                session.add(agent)
            
            # Add simulation steps
            for step in range(100):
                step_data = SimulationStepModel(
                    simulation_id=sim.simulation_id,
                    step_number=step,
                    total_agents=10 - (step // 20),
                    system_agents=3,
                    independent_agents=7 - (step // 20),
                    control_agents=0,
                    total_resources=1000.0,
                    average_agent_resources=50.0,
                    births=1 if step % 10 == 0 else 0,
                    deaths=1 if step % 20 == 0 else 0,
                    average_agent_health=90.0,
                    average_reward=10.0,
                    combat_encounters=0,
                    successful_attacks=0,
                    resources_shared=5.0,
                )
                session.add(step_data)
    
    session.commit()
    session.close()
    
    return test_db_path

@pytest.fixture
def db_config(test_db_with_data):
    """Create database config for tests."""
    return DatabaseConfig(path=str(test_db_with_data))

@pytest.fixture
def cache_config():
    """Create cache config for tests."""
    return CacheConfig(max_size=10, ttl_seconds=60, enabled=True)

@pytest.fixture
def mcp_config(db_config, cache_config):
    """Create MCP config for tests."""
    return MCPConfig(database=db_config, cache=cache_config)

@pytest.fixture
def db_service(db_config):
    """Create database service for tests."""
    return DatabaseService(db_config)

@pytest.fixture
def cache_service(cache_config):
    """Create cache service for tests."""
    return CacheService(cache_config)

@pytest.fixture
def services(db_service, cache_service):
    """Provide both services as a tuple."""
    return db_service, cache_service
```

### 5.2 Integration Tests

**File: `tests/mcp/integration/test_end_to_end.py`**

```python
"""End-to-end integration tests."""

import pytest
from farm.mcp.server import SimulationMCPServer
from farm.mcp.config import MCPConfig

def test_full_analysis_workflow(mcp_config):
    """Test complete analysis workflow."""
    # Initialize server
    server = SimulationMCPServer(mcp_config)
    
    # 1. List simulations
    list_tool = server.get_tool("list_simulations")
    result = list_tool()
    assert result["success"] is True
    assert len(result["data"]["simulations"]) > 0
    
    # 2. Get simulation info
    sim_id = result["data"]["simulations"][0]["simulation_id"]
    info_tool = server.get_tool("get_simulation_info")
    result = info_tool(simulation_id=sim_id)
    assert result["success"] is True
    
    # 3. Analyze population
    pop_tool = server.get_tool("analyze_population_dynamics")
    result = pop_tool(simulation_id=sim_id)
    assert result["success"] is True
    assert "population_summary" in result["data"]
    
    # 4. Query agents
    agent_tool = server.get_tool("query_agents")
    result = agent_tool(simulation_id=sim_id, limit=10)
    assert result["success"] is True
    assert len(result["data"]["agents"]) > 0

def test_error_handling_workflow(mcp_config):
    """Test that errors are handled gracefully."""
    server = SimulationMCPServer(mcp_config)
    
    # Try to get info for nonexistent simulation
    info_tool = server.get_tool("get_simulation_info")
    result = info_tool(simulation_id="nonexistent")
    
    assert result["success"] is False
    assert "SimulationNotFoundError" in result["error"]["type"]

def test_caching_workflow(mcp_config):
    """Test that caching works across multiple calls."""
    server = SimulationMCPServer(mcp_config)
    
    tool = server.get_tool("query_agents")
    
    # First call
    result1 = tool(simulation_id="test_sim_000", limit=5)
    assert result1["metadata"]["from_cache"] is False
    
    # Second call should hit cache
    result2 = tool(simulation_id="test_sim_000", limit=5)
    assert result2["metadata"]["from_cache"] is True
    
    # Different params should not hit cache
    result3 = tool(simulation_id="test_sim_000", limit=10)
    assert result3["metadata"]["from_cache"] is False
```

### 5.3 Performance Tests

**File: `tests/mcp/integration/test_performance.py`**

```python
"""Performance tests for MCP server."""

import pytest
import time
from farm.mcp.server import SimulationMCPServer

def test_query_performance(mcp_config):
    """Test that queries complete within acceptable time."""
    server = SimulationMCPServer(mcp_config)
    
    tool = server.get_tool("query_agents")
    
    start = time.time()
    result = tool(simulation_id="test_sim_000", limit=100)
    elapsed = time.time() - start
    
    assert result["success"] is True
    assert elapsed < 2.0  # Should complete in <2 seconds

def test_concurrent_queries(mcp_config):
    """Test handling concurrent requests."""
    import concurrent.futures
    
    server = SimulationMCPServer(mcp_config)
    tool = server.get_tool("query_agents")
    
    def query():
        return tool(simulation_id="test_sim_000", limit=10)
    
    # Execute 10 concurrent queries
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(query) for _ in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # All should succeed
    assert all(r["success"] for r in results)

def test_cache_performance(mcp_config):
    """Test that caching improves performance."""
    server = SimulationMCPServer(mcp_config)
    tool = server.get_tool("analyze_population_dynamics")
    
    # First call (uncached)
    start = time.time()
    result1 = tool(simulation_id="test_sim_000")
    uncached_time = time.time() - start
    
    # Second call (cached)
    start = time.time()
    result2 = tool(simulation_id="test_sim_000")
    cached_time = time.time() - start
    
    # Cached should be significantly faster
    assert cached_time < uncached_time / 10
    assert result2["metadata"]["from_cache"] is True
```

### 5.4 Test Running in Cursor

**Cursor Workflow**:

1. **Run all tests**:
   ```bash
   pytest tests/mcp/ -v
   ```

2. **Run with coverage**:
   ```bash
   pytest tests/mcp/ --cov=farm.mcp --cov-report=html
   ```

3. **Run specific test file**:
   ```bash
   pytest tests/mcp/tools/test_query_tools.py -v
   ```

4. **Run in watch mode** (using pytest-watch):
   ```bash
   ptw tests/mcp/ -- -v
   ```

5. **Use Cursor's terminal**:
   - Open terminal in Cursor (Ctrl+`)
   - Run tests continuously while developing
   - View results in integrated terminal

**Implementation Checklist**:
- [ ] All unit tests written
- [ ] Integration tests complete
- [ ] Performance tests passing
- [ ] Test coverage >80%
- [ ] All tests passing in CI

## 6. Server Implementation

### 6.1 Main Server Module

**File: `farm/mcp/server.py`**

```python
"""Main MCP server implementation."""

import logging
from typing import Dict, Optional
from fastmcp import FastMCP

from farm.mcp.config import MCPConfig
from farm.mcp.services import DatabaseService, CacheService
from farm.mcp.tools import (
    # Metadata tools
    ListSimulationsTool,
    GetSimulationInfoTool,
    ListExperimentsTool,
    GetExperimentInfoTool,
    # Query tools
    QueryAgentsTool,
    QueryActionsTool,
    QueryStatesTool,
    QueryResourcesTool,
    QueryInteractionsTool,
    GetSimulationMetricsTool,
    # Analysis tools
    AnalyzePopulationDynamicsTool,
    AnalyzeSurvivalRatesTool,
    AnalyzeResourceEfficiencyTool,
    AnalyzeAgentPerformanceTool,
    IdentifyCriticalEventsTool,
    AnalyzeSocialPatternsTool,
    AnalyzeReproductionTool,
    # Comparison tools
    CompareSimulationsTool,
    CompareParametersTool,
    RankConfigurationsTool,
)

logger = logging.getLogger(__name__)

class SimulationMCPServer:
    """Main MCP server for simulation database analysis."""
    
    def __init__(self, config: MCPConfig):
        """Initialize MCP server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        
        # Initialize services
        self.db_service = DatabaseService(config.database)
        self.cache_service = CacheService(config.cache)
        
        # Initialize FastMCP
        self.mcp = FastMCP("simulation-analysis")
        
        # Tool registry
        self._tools: Dict[str, ToolBase] = {}
        
        # Register all tools
        self._register_tools()
        
        logger.info("MCP server initialized")
    
    def _register_tools(self):
        """Register all MCP tools."""
        # Define all tools
        tool_classes = [
            # Metadata
            ListSimulationsTool,
            GetSimulationInfoTool,
            ListExperimentsTool,
            GetExperimentInfoTool,
            # Query
            QueryAgentsTool,
            QueryActionsTool,
            QueryStatesTool,
            QueryResourcesTool,
            QueryInteractionsTool,
            GetSimulationMetricsTool,
            # Analysis
            AnalyzePopulationDynamicsTool,
            AnalyzeSurvivalRatesTool,
            AnalyzeResourceEfficiencyTool,
            AnalyzeAgentPerformanceTool,
            IdentifyCriticalEventsTool,
            AnalyzeSocialPatternsTool,
            AnalyzeReproductionTool,
            # Comparison
            CompareSimulationsTool,
            CompareParametersTool,
            RankConfigurationsTool,
        ]
        
        # Instantiate and register each tool
        for tool_class in tool_classes:
            tool = tool_class(self.db_service, self.cache_service)
            self._tools[tool.name] = tool
            
            # Register with FastMCP
            self.mcp.tool()(self._create_tool_wrapper(tool))
            
            logger.info(f"Registered tool: {tool.name}")
    
    def _create_tool_wrapper(self, tool: ToolBase):
        """Create a wrapper function for FastMCP registration.
        
        Args:
            tool: Tool instance
            
        Returns:
            Wrapped function for FastMCP
        """
        def wrapper(**kwargs):
            """Tool wrapper for FastMCP."""
            return tool(**kwargs)
        
        # Set function metadata for FastMCP
        wrapper.__name__ = tool.name
        wrapper.__doc__ = tool.description
        
        return wrapper
    
    def get_tool(self, name: str) -> Optional[ToolBase]:
        """Get tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None
        """
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return self.cache_service.get_stats()
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache_service.clear()
        logger.info("Cache cleared")
    
    def run(self, **kwargs):
        """Start the MCP server.
        
        Args:
            **kwargs: Additional arguments for FastMCP.run()
        """
        logger.info(f"Starting MCP server with {len(self._tools)} tools")
        self.mcp.run(**kwargs)
    
    def close(self):
        """Shutdown server and cleanup resources."""
        self.db_service.close()
        logger.info("MCP server shutdown complete")
```

**Cursor Workflow**:
1. Use Cursor: "Implement SimulationMCPServer with FastMCP integration"
2. Review tool registration logic
3. Test server initialization

### 6.2 CLI Interface

**File: `farm/mcp/cli.py`**

```python
"""Command-line interface for MCP server."""

import argparse
import logging
from pathlib import Path

from farm.mcp.server import SimulationMCPServer
from farm.mcp.config import MCPConfig

def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FastMCP Server for Simulation Database Analysis"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to simulation database"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file (optional)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    if args.config:
        config = MCPConfig.from_yaml(args.config)
    else:
        config = MCPConfig.from_db_path(args.db_path)
    
    # Override cache setting if requested
    if args.no_cache:
        config.cache.enabled = False
    
    # Create and run server
    server = SimulationMCPServer(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.close()

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Run with database path
python -m farm.mcp.cli --db-path /path/to/simulation.db

# Run with config file
python -m farm.mcp.cli --db-path /path/to/simulation.db --config config.yaml

# Run with debugging
python -m farm.mcp.cli --db-path /path/to/simulation.db --log-level DEBUG
```

## 7. Documentation

### 7.1 API Documentation

Create comprehensive documentation using Cursor:

**File: `farm/mcp/README.md`**

Use Cursor: "Generate comprehensive README for the MCP server with usage examples"

Should include:
- Overview
- Installation
- Configuration
- Tool reference
- Examples
- Troubleshooting

### 7.2 Docstrings

Ensure all modules have comprehensive docstrings:
- Module-level docstrings
- Class docstrings
- Method docstrings with Args/Returns/Raises

Use Cursor: "Add comprehensive docstrings to this module"

## 8. Deployment & Usage

### 8.1 Package Installation

```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### 8.2 Running the Server

```bash
# As CLI
python -m farm.mcp.cli --db-path simulation.db

# Programmatically
from farm.mcp.server import SimulationMCPServer
from farm.mcp.config import MCPConfig

config = MCPConfig.from_db_path("simulation.db")
server = SimulationMCPServer(config)
server.run()
```

### 8.3 Integration with LLM

Configure LLM client to connect to MCP server:

```json
{
  "mcpServers": {
    "simulation-analysis": {
      "command": "python",
      "args": ["-m", "farm.mcp.cli", "--db-path", "/path/to/simulation.db"]
    }
  }
}
```

## 9. Final Checklist

### Phase 1 ✓
- [ ] Project structure created
- [ ] Configuration system implemented
- [ ] Database service layer complete
- [ ] Cache service implemented
- [ ] Base tool class complete
- [ ] Exception handling in place
- [ ] All Phase 1 tests passing

### Phase 2 ✓
- [ ] All metadata tools implemented
- [ ] All query tools implemented
- [ ] Basic formatters complete
- [ ] All Phase 2 tests passing

### Phase 3 ✓
- [ ] All analysis tools implemented
- [ ] All comparison tools implemented
- [ ] Advanced formatters (charts) implemented
- [ ] All Phase 3 tests passing

### Phase 4 ✓
- [ ] Comprehensive test suite complete
- [ ] Integration tests passing
- [ ] Performance tests passing
- [ ] Test coverage >80%
- [ ] Server implementation complete
- [ ] CLI interface implemented
- [ ] Documentation complete
- [ ] Ready for production use

## 10. Tips for Using Cursor

### 10.1 Effective Prompts

**Good prompts**:
- "Implement QueryAgentsTool following the ToolBase pattern"
- "Generate unit tests for DatabaseService with 80% coverage"
- "Refactor this method to be more readable"
- "Add comprehensive docstrings to this class"

**Less effective prompts**:
- "Make it better" (too vague)
- "Fix bugs" (not specific)

### 10.2 Cursor Features to Use

1. **Cmd+K**: Inline code generation
2. **Cmd+L**: Chat with context
3. **Cmd+Shift+L**: Chat with full codebase context
4. **Cmd+I**: Ask about selected code
5. **Split editor**: View design doc + implementation side-by-side

### 10.3 Workflow Tips

1. **Start with tests**: Generate test stubs first, then implement
2. **Use examples**: Reference existing code when asking for similar code
3. **Iterate**: Don't expect perfect code first time, refine iteratively
4. **Review carefully**: Always review AI-generated code
5. **Commit often**: Commit working code frequently

### 10.4 Debugging in Cursor

1. Use integrated terminal for test output
2. Set breakpoints with debugger
3. Use Cursor to explain error messages
4. Ask Cursor to suggest fixes for test failures

## 11. Estimated Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1: Foundation | 3-5 days | Core infrastructure |
| Phase 2: Core Tools | 4-6 days | Metadata + Query tools |
| Phase 3: Analysis | 5-7 days | Analysis + Comparison tools |
| Phase 4: Testing | 3-5 days | Comprehensive testing |
| **Total** | **15-23 days** | ~3-4 weeks |

Adjust based on:
- Familiarity with FastMCP
- Existing analyzer reuse
- Testing thoroughness
- Documentation depth