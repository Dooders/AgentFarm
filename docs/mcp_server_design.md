# FastMCP Server for Simulation Database Analysis - Design Document

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM Agent                            │
│                  (Claude, GPT, etc.)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ MCP Protocol
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    FastMCP Server                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Tool Registry & Router                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌────────────┬────────────┬────────────┬────────────┐     │
│  │  Metadata  │   Query    │  Analysis  │ Comparison │     │
│  │   Tools    │   Tools    │   Tools    │   Tools    │     │
│  └────────────┴────────────┴────────────┴────────────┘     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Query Builder & Validator                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Database Service Layer                  │   │
│  │  ┌────────────┬─────────────┬──────────────────┐    │   │
│  │  │  Session   │   Cache     │   Connection     │    │   │
│  │  │  Manager   │   Manager   │      Pool        │    │   │
│  │  └────────────┴─────────────┴──────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Response Formatter & Serializer           │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ SQLAlchemy ORM
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              Simulation Database (SQLite/PostgreSQL)        │
│  ┌──────────┬──────────┬───────────┬────────────────────┐   │
│  │  Agents  │  States  │  Actions  │  Simulations...    │   │
│  └──────────┴──────────┴───────────┴────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Module Structure

```
farm/mcp/
├── __init__.py
├── server.py                    # Main FastMCP server entry point
├── config.py                    # Configuration management
├── tools/
│   ├── __init__.py
│   ├── base.py                  # Base tool class
│   ├── metadata_tools.py        # Simulation/experiment metadata tools
│   ├── query_tools.py           # Basic data query tools
│   ├── analysis_tools.py        # Analysis and aggregation tools
│   └── comparison_tools.py      # Multi-simulation comparison tools
├── services/
│   ├── __init__.py
│   ├── database_service.py      # Database connection & session management
│   ├── query_builder.py         # Query construction helpers
│   ├── cache_service.py         # Query result caching
│   └── validation_service.py    # Input validation
├── formatters/
│   ├── __init__.py
│   ├── base_formatter.py        # Base formatter interface
│   ├── json_formatter.py        # JSON output formatting
│   ├── markdown_formatter.py    # Markdown tables and summaries
│   └── chart_formatter.py       # ASCII/Unicode charts
├── models/
│   ├── __init__.py
│   ├── schemas.py               # Pydantic models for validation
│   └── responses.py             # Response data structures
└── utils/
    ├── __init__.py
    ├── exceptions.py            # Custom exceptions
    ├── logging.py               # Logging configuration
    └── helpers.py               # Utility functions
```

## 2. Component Design

### 2.1 FastMCP Server (`server.py`)

**Responsibilities**:
- Initialize FastMCP server instance
- Register all tools
- Handle MCP protocol communication
- Manage server lifecycle

**Key Classes**:
```python
class SimulationMCPServer:
    """Main MCP server for simulation database analysis."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.db_service = DatabaseService(config.db_path)
        self.cache = CacheService(config.cache_config)
        self.mcp = FastMCP("simulation-analysis")
        self._register_tools()
    
    def _register_tools(self):
        """Register all MCP tools."""
        # Register metadata tools
        # Register query tools
        # Register analysis tools
        # Register comparison tools
    
    def run(self):
        """Start the MCP server."""
        self.mcp.run()
```

### 2.2 Database Service Layer (`services/database_service.py`)

**Responsibilities**:
- Manage database connections
- Provide session management
- Handle connection pooling
- Execute queries with error handling

**Key Classes**:
```python
class DatabaseService:
    """Service for database operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.session_manager = SessionManager()
    
    @contextmanager
    def get_session(self):
        """Provide a transactional scope for database operations."""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_query(self, query_func, *args, **kwargs):
        """Execute a query with error handling."""
        with self.get_session() as session:
            return query_func(session, *args, **kwargs)
    
    def validate_simulation_exists(self, simulation_id: str) -> bool:
        """Check if simulation exists."""
        with self.get_session() as session:
            return session.query(Simulation).filter_by(
                simulation_id=simulation_id
            ).first() is not None
```

### 2.3 Tool Base Class (`tools/base.py`)

**Responsibilities**:
- Provide common tool functionality
- Handle parameter validation
- Format responses consistently
- Manage error handling

**Key Classes**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel

class ToolBase(ABC):
    """Base class for all MCP tools."""
    
    def __init__(
        self, 
        db_service: DatabaseService,
        cache_service: CacheService,
        validator: ValidationService
    ):
        self.db = db_service
        self.cache = cache_service
        self.validator = validator
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> type[BaseModel]:
        """Pydantic schema for parameters."""
        pass
    
    @abstractmethod
    def execute(self, **params) -> Dict[str, Any]:
        """Execute the tool with validated parameters."""
        pass
    
    def __call__(self, **params) -> Dict[str, Any]:
        """Validate and execute tool."""
        try:
            # Validate parameters
            validated_params = self.parameters_schema(**params)
            
            # Check cache
            cache_key = self._get_cache_key(validated_params)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return self._format_response(cached_result, from_cache=True)
            
            # Execute query
            result = self.execute(**validated_params.dict())
            
            # Cache result
            self.cache.set(cache_key, result)
            
            return self._format_response(result)
            
        except ValidationError as e:
            return self._format_error("ValidationError", str(e))
        except DatabaseError as e:
            return self._format_error("DatabaseError", str(e))
        except Exception as e:
            return self._format_error("UnknownError", str(e))
    
    def _format_response(
        self, 
        data: Any, 
        from_cache: bool = False
    ) -> Dict[str, Any]:
        """Format successful response."""
        return {
            "success": True,
            "data": data,
            "metadata": {
                "tool": self.name,
                "from_cache": from_cache,
                "timestamp": datetime.now().isoformat()
            },
            "error": None
        }
    
    def _format_error(self, error_type: str, message: str) -> Dict[str, Any]:
        """Format error response."""
        return {
            "success": False,
            "data": None,
            "metadata": {"tool": self.name},
            "error": {
                "type": error_type,
                "message": message
            }
        }
    
    def _get_cache_key(self, params: BaseModel) -> str:
        """Generate cache key from parameters."""
        import hashlib
        import json
        param_str = json.dumps(params.dict(), sort_keys=True)
        return f"{self.name}:{hashlib.md5(param_str.encode()).hexdigest()}"
```

### 2.4 Example Tool: Query Agents (`tools/query_tools.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from .base import ToolBase

class QueryAgentsParams(BaseModel):
    """Parameters for query_agents tool."""
    simulation_id: str = Field(..., description="Simulation ID to query")
    agent_type: Optional[str] = Field(None, description="Filter by agent type")
    generation: Optional[int] = Field(None, description="Filter by generation")
    alive_only: bool = Field(False, description="Return only living agents")
    limit: int = Field(100, description="Maximum results to return", le=1000)
    offset: int = Field(0, description="Pagination offset", ge=0)

class QueryAgentsTool(ToolBase):
    """Query agents with filtering options."""
    
    @property
    def name(self) -> str:
        return "query_agents"
    
    @property
    def description(self) -> str:
        return """
        Query agents from a simulation with flexible filtering options.
        
        Returns agent data including:
        - Agent ID, type, and generation
        - Birth and death times
        - Initial resources and health
        - Position and genome information
        
        Use this tool to:
        - List all agents in a simulation
        - Find agents of specific types or generations
        - Identify living vs. dead agents
        - Get agent details for further analysis
        """
    
    @property
    def parameters_schema(self) -> type[BaseModel]:
        return QueryAgentsParams
    
    def execute(self, **params) -> Dict[str, Any]:
        """Execute agent query."""
        # Validate simulation exists
        if not self.db.validate_simulation_exists(params["simulation_id"]):
            raise ValueError(f"Simulation {params['simulation_id']} not found")
        
        def query_func(session):
            # Build query
            query = session.query(AgentModel).filter(
                AgentModel.simulation_id == params["simulation_id"]
            )
            
            # Apply filters
            if params.get("agent_type"):
                query = query.filter(AgentModel.agent_type == params["agent_type"])
            
            if params.get("generation") is not None:
                query = query.filter(AgentModel.generation == params["generation"])
            
            if params.get("alive_only"):
                query = query.filter(AgentModel.death_time.is_(None))
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            query = query.limit(params["limit"]).offset(params["offset"])
            
            # Execute and serialize
            agents = query.all()
            results = [
                {
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "generation": a.generation,
                    "birth_time": a.birth_time,
                    "death_time": a.death_time,
                    "position": {"x": a.position_x, "y": a.position_y},
                    "initial_resources": a.initial_resources,
                    "starting_health": a.starting_health,
                    "genome_id": a.genome_id,
                }
                for a in agents
            ]
            
            return {
                "agents": results,
                "total_count": total,
                "returned_count": len(results),
                "limit": params["limit"],
                "offset": params["offset"]
            }
        
        return self.db.execute_query(query_func)
```

### 2.5 Analysis Tool Example (`tools/analysis_tools.py`)

```python
class AnalyzePopulationDynamicsParams(BaseModel):
    """Parameters for population dynamics analysis."""
    simulation_id: str = Field(..., description="Simulation ID to analyze")
    start_step: Optional[int] = Field(None, description="Start step (inclusive)")
    end_step: Optional[int] = Field(None, description="End step (inclusive)")
    include_chart: bool = Field(True, description="Include ASCII chart in output")

class AnalyzePopulationDynamicsTool(ToolBase):
    """Analyze population dynamics over time."""
    
    @property
    def name(self) -> str:
        return "analyze_population_dynamics"
    
    @property
    def description(self) -> str:
        return """
        Analyze how agent populations evolve over simulation time.
        
        Returns:
        - Total agents per step
        - Breakdown by agent type
        - Birth and death rates
        - Population growth rate
        - Optional ASCII chart visualization
        
        Use this to:
        - Understand population trends
        - Identify population crashes or booms
        - Compare different agent types' success
        - Detect critical events
        """
    
    @property
    def parameters_schema(self) -> type[BaseModel]:
        return AnalyzePopulationDynamicsParams
    
    def execute(self, **params) -> Dict[str, Any]:
        """Execute population analysis."""
        def query_func(session):
            # Build query for simulation steps
            query = session.query(SimulationStepModel).filter(
                SimulationStepModel.simulation_id == params["simulation_id"]
            )
            
            # Apply step range filters
            if params.get("start_step") is not None:
                query = query.filter(
                    SimulationStepModel.step_number >= params["start_step"]
                )
            if params.get("end_step") is not None:
                query = query.filter(
                    SimulationStepModel.step_number <= params["end_step"]
                )
            
            query = query.order_by(SimulationStepModel.step_number)
            steps = query.all()
            
            if not steps:
                return {"error": "No data found for specified range"}
            
            # Extract data
            step_numbers = [s.step_number for s in steps]
            total_agents = [s.total_agents for s in steps]
            system_agents = [s.system_agents for s in steps]
            independent_agents = [s.independent_agents for s in steps]
            control_agents = [s.control_agents for s in steps]
            births = [s.births for s in steps]
            deaths = [s.deaths for s in steps]
            
            # Calculate statistics
            import numpy as np
            
            result = {
                "step_range": {
                    "start": step_numbers[0],
                    "end": step_numbers[-1],
                    "count": len(steps)
                },
                "population_summary": {
                    "peak_population": max(total_agents),
                    "peak_step": step_numbers[total_agents.index(max(total_agents))],
                    "final_population": total_agents[-1],
                    "average_population": np.mean(total_agents),
                    "total_births": sum(births),
                    "total_deaths": sum(deaths),
                },
                "by_type": {
                    "system": {
                        "peak": max(system_agents),
                        "average": np.mean(system_agents),
                        "final": system_agents[-1]
                    },
                    "independent": {
                        "peak": max(independent_agents),
                        "average": np.mean(independent_agents),
                        "final": independent_agents[-1]
                    },
                    "control": {
                        "peak": max(control_agents),
                        "average": np.mean(control_agents),
                        "final": control_agents[-1]
                    }
                },
                "time_series": {
                    "steps": step_numbers,
                    "total_agents": total_agents,
                    "system_agents": system_agents,
                    "independent_agents": independent_agents,
                    "control_agents": control_agents,
                    "births": births,
                    "deaths": deaths
                }
            }
            
            # Add chart if requested
            if params.get("include_chart"):
                from farm.mcp.formatters.chart_formatter import ChartFormatter
                chart = ChartFormatter.create_line_chart(
                    data={
                        "Total": total_agents,
                        "System": system_agents,
                        "Independent": independent_agents,
                        "Control": control_agents
                    },
                    x_labels=step_numbers,
                    title="Population Over Time",
                    width=60,
                    height=20
                )
                result["chart"] = chart
            
            return result
        
        return self.db.execute_query(query_func)
```

### 2.6 Cache Service (`services/cache_service.py`)

```python
from typing import Any, Optional
import time
from collections import OrderedDict

class CacheService:
    """Simple in-memory cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if valid."""
        if key not in self._cache:
            return None
        
        # Check TTL
        if time.time() - self._timestamps[key] > self.ttl_seconds:
            self._evict(key)
            return None
        
        # Move to end (LRU)
        self._cache.move_to_end(key)
        return self._cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = next(iter(self._cache))
            self._evict(oldest_key)
        
        self._cache[key] = value
        self._timestamps[key] = time.time()
        self._cache.move_to_end(key)
    
    def _evict(self, key: str):
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
    
    def clear(self):
        """Clear entire cache."""
        self._cache.clear()
        self._timestamps.clear()
```

### 2.7 Configuration (`config.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional

class CacheConfig(BaseModel):
    """Cache configuration."""
    max_size: int = Field(100, description="Maximum cache entries")
    ttl_seconds: int = Field(300, description="Time to live in seconds")
    enabled: bool = Field(True, description="Enable caching")

class DatabaseConfig(BaseModel):
    """Database configuration."""
    path: str = Field(..., description="Path to database file")
    pool_size: int = Field(5, description="Connection pool size")
    query_timeout: int = Field(30, description="Query timeout in seconds")
    read_only: bool = Field(True, description="Read-only access")

class ServerConfig(BaseModel):
    """Server configuration."""
    max_result_size: int = Field(10000, description="Max results per query")
    default_limit: int = Field(100, description="Default pagination limit")
    log_level: str = Field("INFO", description="Logging level")

class MCPConfig(BaseModel):
    """Main MCP server configuration."""
    database: DatabaseConfig
    cache: CacheConfig = CacheConfig()
    server: ServerConfig = ServerConfig()
    
    @classmethod
    def from_db_path(cls, db_path: str):
        """Create config from database path."""
        return cls(database=DatabaseConfig(path=db_path))
```

## 3. Tool Registry

### 3.1 Complete Tool List

| Category | Tool Name | Description |
|----------|-----------|-------------|
| **Metadata** | `list_simulations` | List all simulations with filters |
| | `get_simulation_info` | Get detailed simulation metadata |
| | `list_experiments` | List research experiments |
| | `get_experiment_info` | Get experiment details |
| **Query** | `query_agents` | Query agents with filters |
| | `query_actions` | Retrieve action logs |
| | `query_states` | Get agent states over time |
| | `query_resources` | Fetch resource states |
| | `query_interactions` | Retrieve interaction data |
| | `get_simulation_metrics` | Get step-level metrics |
| **Analysis** | `analyze_population_dynamics` | Population trends |
| | `analyze_survival_rates` | Survival analysis |
| | `analyze_resource_efficiency` | Resource utilization |
| | `analyze_agent_performance` | Individual agent analysis |
| | `identify_critical_events` | Detect significant events |
| | `analyze_social_patterns` | Social interaction patterns |
| | `analyze_reproduction` | Reproduction success rates |
| **Comparison** | `compare_simulations` | Multi-simulation comparison |
| | `compare_parameters` | Parameter impact analysis |
| | `rank_configurations` | Performance ranking |
| **Advanced** | `build_agent_lineage` | Construct family trees |
| | `analyze_spatial_distribution` | Spatial statistics |
| | `get_agent_lifecycle` | Complete agent history |

## 4. Data Flow

### 4.1 Request Flow

```
1. LLM Agent sends tool request via MCP
   ↓
2. FastMCP server receives request
   ↓
3. Tool router identifies appropriate tool
   ↓
4. Tool validates parameters (Pydantic)
   ↓
5. Cache service checks for cached result
   ↓ (cache miss)
6. Database service executes query
   ↓
7. Results formatted and serialized
   ↓
8. Response cached (if cacheable)
   ↓
9. Response returned to LLM agent
```

### 4.2 Error Handling Flow

```
Exception occurs
   ↓
Caught by tool __call__ method
   ↓
Exception type determined
   ↓
Error formatted with context
   ↓
Logged for debugging
   ↓
Returned as structured error response
```

## 5. Security Design

### 5.1 Read-Only Enforcement

- Database connection opened in read-only mode
- No write operations in any tool
- SQLAlchemy configured for read-only queries
- File system permissions enforced

### 5.2 Input Validation

- All parameters validated with Pydantic schemas
- SQL injection prevention through parameterized queries
- Simulation ID existence checks before queries
- Numeric range validation
- String length limits

### 5.3 Resource Limits

- Query timeout (30 seconds default)
- Maximum result set size (10,000 records)
- Cache size limits
- Connection pool limits

## 6. Performance Optimizations

### 6.1 Query Optimization

- Use `joinedload` for related entities
- Limit result sets with pagination
- Index-aware query construction
- Select only required columns

### 6.2 Caching Strategy

- Cache query results by parameter hash
- TTL-based expiration (5 minutes default)
- LRU eviction policy
- Separate cache for metadata vs. data queries

### 6.3 Connection Management

- Connection pooling (5 connections default)
- Lazy connection initialization
- Proper session cleanup
- Connection recycling

## 7. Extension Points

### 7.1 Custom Tools

```python
# Developer can add custom tools by:
1. Subclassing ToolBase
2. Implementing required methods
3. Registering in server.py

# Example:
class MyCustomTool(ToolBase):
    @property
    def name(self) -> str:
        return "my_custom_analysis"
    
    # Implement other abstract methods...
```

### 7.2 Custom Formatters

```python
# Add new output formats:
class CustomFormatter(BaseFormatter):
    def format(self, data: Any) -> str:
        # Custom formatting logic
        pass
```

### 7.3 Analyzer Integration

```python
# Integrate existing analyzers:
from farm.database.analyzers import PopulationAnalyzer

class AnalysisToolAdapter(ToolBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = PopulationAnalyzer()
    
    def execute(self, **params):
        # Use existing analyzer
        return self.analyzer.analyze(...)
```

## 8. Testing Strategy

### 8.1 Unit Tests

- Test each tool independently
- Mock database service
- Validate parameter schemas
- Test error handling

### 8.2 Integration Tests

- Test against real database
- Verify query correctness
- Test caching behavior
- Validate response formats

### 8.3 Performance Tests

- Query execution time
- Cache hit rates
- Memory usage
- Concurrent request handling

## 9. Deployment

### 9.1 Installation

```bash
pip install fastmcp sqlalchemy pandas numpy pydantic
```

### 9.2 Configuration

```python
# config.yaml
database:
  path: "/path/to/simulation.db"
  pool_size: 5
  query_timeout: 30
  read_only: true

cache:
  max_size: 100
  ttl_seconds: 300
  enabled: true

server:
  max_result_size: 10000
  default_limit: 100
  log_level: "INFO"
```

### 9.3 Running the Server

```bash
# As a standalone server
python -m farm.mcp.server --config config.yaml

# Programmatically
from farm.mcp.server import SimulationMCPServer
from farm.mcp.config import MCPConfig

config = MCPConfig.from_db_path("simulation.db")
server = SimulationMCPServer(config)
server.run()
```

## 10. Monitoring & Observability

### 10.1 Logging

- Query execution logs
- Cache hit/miss metrics
- Error logs with stack traces
- Performance metrics (query time)

### 10.2 Metrics

- Queries per second
- Average query duration
- Cache hit rate
- Error rate
- Active connections

## 11. Design Principles Applied

### SOLID Principles

- **Single Responsibility**: Each tool handles one specific query/analysis
- **Open/Closed**: Easy to add new tools without modifying existing code
- **Liskov Substitution**: All tools implement ToolBase interface
- **Interface Segregation**: Tools only depend on interfaces they use
- **Dependency Inversion**: Tools depend on abstractions (DatabaseService, CacheService)

### DRY

- Common functionality in ToolBase
- Shared query builders
- Reusable formatters

### KISS

- Simple, focused tools
- Clear parameter schemas
- Straightforward error handling

### Composition Over Inheritance

- Tools composed of services (DB, cache, validation)
- Formatters plugged in as needed
- Analyzers integrated as dependencies