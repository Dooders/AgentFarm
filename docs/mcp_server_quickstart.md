# FastMCP Server Quick Start Guide

## TL;DR - Start Here

This is a **read-only MCP server** that lets LLM agents query and analyze your simulation database through natural language.

### What You're Building

A Model Context Protocol (MCP) server that exposes ~20 tools for:
- Listing simulations and experiments
- Querying agents, actions, resources
- Analyzing population dynamics, survival rates, resource efficiency
- Comparing simulations and parameters
- Building agent lineages and spatial analysis

### Prerequisites

```bash
# Install dependencies
pip install fastmcp sqlalchemy pydantic pandas numpy
```

## Day 1: Get Something Working

### Step 1: Create Basic Structure (15 min)

```bash
# Create directories
mkdir -p farm/mcp/{tools,services,formatters,models,utils}
mkdir -p tests/mcp/{tools,services,integration}

# Create __init__ files
find farm/mcp tests/mcp -type d -exec touch {}/__init__.py \;
```

### Step 2: Config (30 min)

**Create `farm/mcp/config.py`**:

```python
from pydantic import BaseModel, Field
from pathlib import Path

class DatabaseConfig(BaseModel):
    path: str
    pool_size: int = 5
    query_timeout: int = 30
    read_only: bool = True

class CacheConfig(BaseModel):
    max_size: int = 100
    ttl_seconds: int = 300
    enabled: bool = True

class MCPConfig(BaseModel):
    database: DatabaseConfig
    cache: CacheConfig = CacheConfig()
    
    @classmethod
    def from_db_path(cls, db_path: str):
        return cls(database=DatabaseConfig(path=db_path))
```

**Test it immediately**:

```python
# tests/mcp/test_config.py
from farm.mcp.config import MCPConfig

def test_config():
    config = MCPConfig.from_db_path("test.db")
    assert config.database.path == "test.db"
```

```bash
pytest tests/mcp/test_config.py
```

### Step 3: Database Service (45 min)

**Create `farm/mcp/services/database_service.py`**:

```python
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from farm.mcp.config import DatabaseConfig
from farm.database.models import Simulation

class DatabaseService:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine = create_engine(
            f"sqlite:///{config.path}",
            pool_size=config.pool_size
        )
        self._SessionFactory = sessionmaker(bind=self._engine)
    
    @contextmanager
    def get_session(self):
        session = self._SessionFactory()
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, query_func):
        with self.get_session() as session:
            return query_func(session)
    
    def get_simulation(self, simulation_id: str):
        def get_sim(session):
            return session.query(Simulation).filter_by(
                simulation_id=simulation_id
            ).first()
        return self.execute_query(get_sim)
```

### Step 4: First Working Tool (1 hour)

**Create `farm/mcp/tools/base.py`** (simplified version):

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class ToolBase(ABC):
    def __init__(self, db_service, cache_service=None):
        self.db = db_service
        self.cache = cache_service
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> type[BaseModel]:
        pass
    
    @abstractmethod
    def execute(self, **params):
        pass
    
    def __call__(self, **params):
        try:
            validated = self.parameters_schema(**params)
            result = self.execute(**validated.dict())
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

**Create first tool `farm/mcp/tools/metadata_tools.py`**:

```python
from pydantic import BaseModel, Field
from farm.mcp.tools.base import ToolBase
from farm.database.models import Simulation

class GetSimulationInfoParams(BaseModel):
    simulation_id: str = Field(..., description="Simulation ID")

class GetSimulationInfoTool(ToolBase):
    @property
    def name(self) -> str:
        return "get_simulation_info"
    
    @property
    def description(self) -> str:
        return "Get detailed information about a simulation"
    
    @property
    def parameters_schema(self):
        return GetSimulationInfoParams
    
    def execute(self, **params):
        def query(session):
            sim = session.query(Simulation).filter_by(
                simulation_id=params["simulation_id"]
            ).first()
            
            if not sim:
                raise ValueError(f"Simulation not found: {params['simulation_id']}")
            
            return {
                "simulation_id": sim.simulation_id,
                "status": sim.status,
                "parameters": sim.parameters,
                "start_time": sim.start_time.isoformat() if sim.start_time else None
            }
        
        return self.db.execute_query(query)
```

**Test it**:

```python
# tests/mcp/tools/test_first_tool.py
from farm.mcp.services.database_service import DatabaseService
from farm.mcp.config import DatabaseConfig
from farm.mcp.tools.metadata_tools import GetSimulationInfoTool

def test_get_simulation_info(test_db_path):  # Use existing test DB
    config = DatabaseConfig(path=str(test_db_path))
    db_service = DatabaseService(config)
    tool = GetSimulationInfoTool(db_service)
    
    result = tool(simulation_id="test_sim_001")
    assert result["success"] is True
    assert result["data"]["simulation_id"] == "test_sim_001"
```

### Step 5: Minimal Server (30 min)

**Create `farm/mcp/server.py`**:

```python
from farm.mcp.config import MCPConfig
from farm.mcp.services.database_service import DatabaseService
from farm.mcp.tools.metadata_tools import GetSimulationInfoTool

class SimulationMCPServer:
    def __init__(self, config: MCPConfig):
        self.config = config
        self.db_service = DatabaseService(config.database)
        self._tools = {}
        self._register_tools()
    
    def _register_tools(self):
        # Register first tool
        tool = GetSimulationInfoTool(self.db_service)
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str):
        return self._tools.get(name)

# Test it
if __name__ == "__main__":
    config = MCPConfig.from_db_path("path/to/your/simulation.db")
    server = SimulationMCPServer(config)
    tool = server.get_tool("get_simulation_info")
    result = tool(simulation_id="your_sim_id")
    print(result)
```

**Run it**:
```bash
python farm/mcp/server.py
```

## ✅ End of Day 1

You now have:
- ✅ Configuration system
- ✅ Database service
- ✅ Base tool class
- ✅ One working tool
- ✅ Basic server
- ✅ Tests passing

## Day 2-3: Add Core Tools

### Tool Template (Copy-Paste This)

```python
# Add to farm/mcp/tools/query_tools.py

from pydantic import BaseModel, Field
from typing import Optional
from farm.mcp.tools.base import ToolBase
from farm.database.models import AgentModel

class QueryAgentsParams(BaseModel):
    simulation_id: str
    agent_type: Optional[str] = None
    limit: int = Field(100, le=1000)

class QueryAgentsTool(ToolBase):
    @property
    def name(self) -> str:
        return "query_agents"
    
    @property
    def description(self) -> str:
        return "Query agents from a simulation with filters"
    
    @property
    def parameters_schema(self):
        return QueryAgentsParams
    
    def execute(self, **params):
        def query(session):
            q = session.query(AgentModel).filter(
                AgentModel.simulation_id == params["simulation_id"]
            )
            
            if params.get("agent_type"):
                q = q.filter(AgentModel.agent_type == params["agent_type"])
            
            q = q.limit(params["limit"])
            agents = q.all()
            
            return {
                "agents": [
                    {
                        "agent_id": a.agent_id,
                        "agent_type": a.agent_type,
                        "generation": a.generation,
                    }
                    for a in agents
                ]
            }
        
        return self.db.execute_query(query)
```

### Add to Server

```python
# In farm/mcp/server.py
from farm.mcp.tools.query_tools import QueryAgentsTool

def _register_tools(self):
    tools = [
        GetSimulationInfoTool(self.db_service),
        QueryAgentsTool(self.db_service),
        # Add more...
    ]
    
    for tool in tools:
        self._tools[tool.name] = tool
```

### Tools to Implement (Priority Order)

**Day 2** (High Priority):
1. ✅ GetSimulationInfoTool (done)
2. QueryAgentsTool
3. QueryActionsTool  
4. GetSimulationMetricsTool

**Day 3** (Medium Priority):
5. AnalyzePopulationDynamicsTool
6. QueryStatesTool
7. ListSimulationsTool

**Later** (Lower Priority):
8. CompareSimulationsTool
9. AnalyzeSurvivalRatesTool
10. etc...

## Day 4-5: Add FastMCP Integration

### Install FastMCP

```bash
pip install fastmcp
```

### Update Server with FastMCP

```python
# farm/mcp/server.py
from fastmcp import FastMCP

class SimulationMCPServer:
    def __init__(self, config: MCPConfig):
        self.config = config
        self.db_service = DatabaseService(config.database)
        self.mcp = FastMCP("simulation-analysis")
        self._tools = {}
        self._register_tools()
    
    def _register_tools(self):
        tools = [
            GetSimulationInfoTool(self.db_service),
            QueryAgentsTool(self.db_service),
        ]
        
        for tool in tools:
            self._tools[tool.name] = tool
            # Register with FastMCP
            self.mcp.tool(name=tool.name, description=tool.description)(
                self._create_wrapper(tool)
            )
    
    def _create_wrapper(self, tool):
        def wrapper(**kwargs):
            return tool(**kwargs)
        return wrapper
    
    def run(self):
        self.mcp.run()
```

### Create CLI

```python
# farm/mcp/cli.py
import argparse
from farm.mcp.server import SimulationMCPServer
from farm.mcp.config import MCPConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    args = parser.parse_args()
    
    config = MCPConfig.from_db_path(args.db_path)
    server = SimulationMCPServer(config)
    server.run()

if __name__ == "__main__":
    main()
```

### Run It

```bash
python -m farm.mcp.cli --db-path /path/to/simulation.db
```

## Testing Strategy

### Quick Test Fixture

```python
# tests/mcp/conftest.py
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from farm.database.models import Base, Simulation

@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Add test data
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
```

### Test Each Tool

```python
def test_tool_name(test_db):
    from farm.mcp.config import DatabaseConfig
    from farm.mcp.services.database_service import DatabaseService
    from farm.mcp.tools.your_tool import YourTool
    
    config = DatabaseConfig(path=str(test_db))
    db_service = DatabaseService(config)
    tool = YourTool(db_service)
    
    result = tool(param1="value1")
    
    assert result["success"] is True
    assert "expected_key" in result["data"]
```

## Cursor Tips for Fast Development

### 1. Generate Tool Boilerplate

**Cursor Prompt**:
```
Create a new MCP tool called QueryResourcesTool that:
- Queries ResourceModel from the database
- Filters by simulation_id and step_number
- Returns resource positions and amounts
- Follows the ToolBase pattern
```

### 2. Generate Tests

**Cursor Prompt**:
```
Generate pytest tests for QueryResourcesTool including:
- Test successful query
- Test with filters
- Test empty results
- Test error handling
```

### 3. Debug Errors

**Cursor Prompt**:
```
This test is failing with [error message]. 
Here's the tool code: [paste code]
What's wrong and how do I fix it?
```

### 4. Refactor Code

**Cursor Prompt**:
```
Refactor this execute method to be more readable and follow the DRY principle
```

## Common Issues & Solutions

### Issue 1: Import Errors

```python
# Fix: Update __init__.py files
# farm/mcp/__init__.py
from farm.mcp.server import SimulationMCPServer
from farm.mcp.config import MCPConfig

# farm/mcp/tools/__init__.py
from farm.mcp.tools.metadata_tools import GetSimulationInfoTool
from farm.mcp.tools.query_tools import QueryAgentsTool
```

### Issue 2: Database Connection Fails

```python
# Check: File exists
from pathlib import Path
assert Path(db_path).exists()

# Check: Can connect
from sqlalchemy import create_engine
engine = create_engine(f"sqlite:///{db_path}")
conn = engine.connect()
conn.close()
```

### Issue 3: Validation Errors

```python
# Debug with:
from pydantic import ValidationError

try:
    params = YourParamsSchema(**raw_params)
except ValidationError as e:
    print(e.errors())
```

## Next Steps

After getting the basics working:

1. **Add caching** (CacheService from design doc)
2. **Add more tools** (use template above)
3. **Add formatters** (JSON, Markdown, Charts)
4. **Performance optimization**
5. **Comprehensive testing**
6. **Documentation**

## Reference

- **Full Requirements**: `docs/mcp_server_requirements.md`
- **Full Design**: `docs/mcp_server_design.md`
- **Implementation Guide**: `docs/mcp_server_implementation_guide.md`
- **This Quick Start**: `docs/mcp_server_quickstart.md`

## Getting Help in Cursor

**Good prompts**:
- "Show me how to implement [specific tool] following the ToolBase pattern"
- "Generate tests for this tool with edge cases"
- "Why is this query failing? [paste error]"
- "Refactor this to follow SOLID principles"

**Use Cursor features**:
- `Cmd+K`: Generate code inline
- `Cmd+L`: Chat about code
- `Cmd+I`: Explain selected code
- Split editor: View design doc + implementation side-by-side

## Success Metrics

You'll know it's working when:
- ✅ Server starts without errors
- ✅ At least 3 tools working
- ✅ Tests passing
- ✅ Can query real simulation database
- ✅ LLM agent can use tools successfully

## Estimated Time

- **Minimal MVP (3 tools)**: 1-2 days
- **Core tools (10 tools)**: 3-5 days  
- **Full implementation (20+ tools)**: 2-3 weeks
- **Production ready with tests**: 3-4 weeks

Start simple, iterate, and expand! 🚀