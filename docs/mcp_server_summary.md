# FastMCP Server for Simulation Analysis - Project Summary

## 📋 Document Index

1. **[Requirements](mcp_server_requirements.md)** - What we're building and why
2. **[Design](mcp_server_design.md)** - How it's architected
3. **[Implementation Guide](mcp_server_implementation_guide.md)** - Step-by-step development process
4. **[Quick Start](mcp_server_quickstart.md)** - Get started in hours, not days

## 🎯 Project Overview

### What Is This?

A **read-only** Model Context Protocol (MCP) server that enables LLM agents (like Claude, GPT-4) to query and analyze your SQLAlchemy-based simulation database through natural language.

### Key Features

- **~20 specialized tools** for simulation analysis
- **Zero database modification risk** (read-only enforcement)
- **Smart caching** for performance
- **Comprehensive error handling**
- **Built-in validation** via Pydantic schemas
- **FastMCP integration** for easy LLM connectivity

### Architecture at a Glance

```
LLM Agent (Claude/GPT)
        ↓
    MCP Protocol
        ↓
   FastMCP Server (20+ Tools)
        ↓
   SQLAlchemy ORM
        ↓
  Simulation Database (SQLite)
```

## 🗂️ Database Schema (Current)

Your simulation database contains:

### Core Tables
- **Simulation** - Simulation metadata and configuration
- **ExperimentModel** - Research experiment grouping
- **AgentModel** - Agent entities and attributes
- **AgentStateModel** - Time-series agent states
- **ActionModel** - Agent actions/behaviors
- **ResourceModel** - Resource states and positions
- **SimulationStepModel** - Per-step simulation metrics

### Event Tables
- **InteractionModel** - Agent-to-agent/resource interactions
- **ReproductionEventModel** - Reproduction attempts
- **SocialInteractionModel** - Social behaviors
- **LearningExperienceModel** - Agent learning
- **HealthIncident** - Health events

### Existing Analysis Capabilities
Your codebase already has analyzers in `farm/database/analyzers/`:
- Population analysis
- Resource analysis  
- Behavior clustering
- Temporal patterns
- Spatial analysis
- Causal analysis
- Decision patterns

**The MCP server will expose these through LLM-friendly tools!**

## 🛠️ Tools to Implement

### 1. Metadata Tools (4 tools)
Query simulation and experiment metadata

| Tool | Purpose |
|------|---------|
| `list_simulations` | List all simulations with filters |
| `get_simulation_info` | Get detailed simulation data |
| `list_experiments` | List research experiments |
| `get_experiment_info` | Get experiment details |

### 2. Query Tools (6 tools)
Flexible data retrieval

| Tool | Purpose |
|------|---------|
| `query_agents` | Query agents with filters (type, generation, alive) |
| `query_actions` | Retrieve action logs |
| `query_states` | Get agent states over time |
| `query_resources` | Fetch resource states |
| `query_interactions` | Retrieve interaction data |
| `get_simulation_metrics` | Get step-level metrics |

### 3. Analysis Tools (7 tools)
Advanced analytics and insights

| Tool | Purpose |
|------|---------|
| `analyze_population_dynamics` | Population trends over time |
| `analyze_survival_rates` | Survival analysis by cohort |
| `analyze_resource_efficiency` | Resource utilization metrics |
| `analyze_agent_performance` | Individual agent analysis |
| `identify_critical_events` | Detect significant events |
| `analyze_social_patterns` | Social interaction patterns |
| `analyze_reproduction` | Reproduction success rates |

### 4. Comparison Tools (3 tools)
Multi-simulation analysis

| Tool | Purpose |
|------|---------|
| `compare_simulations` | Compare metrics across simulations |
| `compare_parameters` | Parameter impact analysis |
| `rank_configurations` | Performance ranking |

### 5. Advanced Tools (3 tools)
Specialized analysis

| Tool | Purpose |
|------|---------|
| `build_agent_lineage` | Construct family trees |
| `analyze_spatial_distribution` | Spatial statistics |
| `get_agent_lifecycle` | Complete agent history |

**Total: 23 tools** (start with 5-10, expand later)

## 🏗️ Architecture Components

### 1. Configuration System (`config.py`)
- Pydantic models for type safety
- Database, cache, and server config
- YAML file support
- Validation with sensible defaults

### 2. Services Layer

**DatabaseService** (`services/database_service.py`)
- Connection pooling
- Session management
- Query execution with timeouts
- Read-only enforcement

**CacheService** (`services/cache_service.py`)
- In-memory LRU cache
- TTL-based expiration
- Hit/miss statistics
- Configurable size limits

**ValidationService** (`services/validation_service.py`)
- Parameter validation
- Simulation existence checks
- Type coercion

### 3. Tool System

**ToolBase** (`tools/base.py`) - Abstract base class
- Parameter validation (Pydantic)
- Response formatting
- Error handling
- Caching integration
- Logging

**Tool Categories** (`tools/*.py`)
- `metadata_tools.py` - Simulation/experiment metadata
- `query_tools.py` - Data retrieval
- `analysis_tools.py` - Analytics
- `comparison_tools.py` - Multi-sim analysis

### 4. Server (`server.py`)
- FastMCP integration
- Tool registration
- Request routing
- Lifecycle management

### 5. Utilities
- Custom exceptions (`utils/exceptions.py`)
- Logging configuration (`utils/logging.py`)
- Helper functions (`utils/helpers.py`)

## 📊 Response Format

All tools return structured responses:

### Success Response
```json
{
  "success": true,
  "data": {
    "agents": [...],
    "total_count": 150
  },
  "metadata": {
    "tool": "query_agents",
    "timestamp": "2024-01-15T10:30:00",
    "from_cache": false,
    "execution_time_ms": 145
  },
  "error": null
}
```

### Error Response
```json
{
  "success": false,
  "data": null,
  "metadata": {
    "tool": "query_agents",
    "timestamp": "2024-01-15T10:30:00"
  },
  "error": {
    "type": "ValidationError",
    "message": "Invalid simulation_id",
    "details": {...}
  }
}
```

## 🔒 Security & Safety

### Read-Only Guarantees
1. Database opened in read-only mode
2. No write operations in any tool
3. SQLAlchemy configured for SELECT only
4. File system permissions enforced

### Input Validation
1. Pydantic schemas for all parameters
2. SQL injection prevention (parameterized queries)
3. Simulation ID existence verification
4. Numeric range validation
5. String length limits

### Resource Limits
1. Query timeout: 30s default
2. Result set size: 10,000 records max
3. Cache size: 100 entries max
4. Connection pool: 5-10 connections

## ⚡ Performance

### Optimization Strategies

1. **Query Optimization**
   - Use `joinedload` for relationships
   - Limit result sets with pagination
   - Index-aware query construction
   - Select only required columns

2. **Caching**
   - LRU cache with TTL
   - Parameter-based cache keys
   - Automatic invalidation
   - Hit rate monitoring

3. **Connection Management**
   - Connection pooling
   - Lazy initialization
   - Proper cleanup
   - Pool size tuning

### Performance Targets
- Query response: <2s for 95% of queries
- Cache hit rate: >50% for repeated queries
- Concurrent queries: Support 10+ simultaneous
- Memory usage: <200MB for typical workloads

## 🧪 Testing Strategy

### Test Pyramid

```
        /\
       /E2E\         Integration Tests (10%)
      /------\       
     /  Unit  \      Unit Tests (80%)
    /----------\     
   / Fixtures   \    Test Data & Mocks (10%)
  /--------------\   
```

### Test Categories

1. **Unit Tests** (`tests/mcp/`)
   - Config validation
   - Service layer logic
   - Tool execution
   - Error handling

2. **Integration Tests** (`tests/mcp/integration/`)
   - End-to-end workflows
   - Database queries
   - Cache behavior
   - Tool chaining

3. **Performance Tests** (`tests/mcp/integration/`)
   - Query speed
   - Concurrent requests
   - Cache effectiveness
   - Memory usage

### Coverage Target
- Overall: >80%
- Critical paths: >95%
- Error paths: >70%

## 📅 Implementation Phases

### Phase 1: Foundation (Week 1)
- ✅ Project structure
- ✅ Configuration system
- ✅ Database service
- ✅ Cache service
- ✅ Base tool class
- ✅ Exception handling

### Phase 2: Core Tools (Week 2)
- ✅ Metadata tools (4)
- ✅ Query tools (6)
- ✅ Basic formatters

### Phase 3: Analysis (Week 3)
- ✅ Analysis tools (7)
- ✅ Comparison tools (3)
- ✅ Advanced formatters

### Phase 4: Polish (Week 4)
- ✅ Comprehensive tests
- ✅ Performance optimization
- ✅ Documentation
- ✅ Integration

**Total: 3-4 weeks**

## 🚀 Quick Start Path

### Minimum Viable Product (2 days)

Day 1:
1. Setup structure (30 min)
2. Config system (1 hr)
3. Database service (2 hrs)
4. Base tool class (2 hrs)
5. First tool: `get_simulation_info` (2 hrs)

Day 2:
1. Add 2-3 more tools (4 hrs)
2. Basic server (2 hrs)
3. Testing (2 hrs)

**Result**: Working MCP server with 3-4 tools

### Full Implementation (3-4 weeks)

Week 1: Foundation
Week 2: Core tools (10 tools)
Week 3: Analysis tools (10 tools)
Week 4: Testing, docs, polish

## 🔧 Development Tools

### Required
- Python 3.8+
- FastMCP
- SQLAlchemy
- Pydantic
- Pandas/NumPy

### Development
- pytest (testing)
- pytest-cov (coverage)
- black (formatting)
- ruff (linting)
- Cursor AI (coding assistant)

### Optional
- pytest-watch (continuous testing)
- ipython (debugging)
- memory_profiler (performance)

## 📚 Usage Examples

### Starting the Server

```bash
# CLI
python -m farm.mcp.cli --db-path simulation.db

# With config
python -m farm.mcp.cli --db-path simulation.db --config config.yaml

# Programmatically
from farm.mcp.server import SimulationMCPServer
from farm.mcp.config import MCPConfig

config = MCPConfig.from_db_path("simulation.db")
server = SimulationMCPServer(config)
server.run()
```

### LLM Integration

Configure Claude Desktop (or other MCP client):

```json
{
  "mcpServers": {
    "simulation-analysis": {
      "command": "python",
      "args": [
        "-m",
        "farm.mcp.cli",
        "--db-path",
        "/absolute/path/to/simulation.db"
      ]
    }
  }
}
```

### Example LLM Interactions

**User**: "Show me the population dynamics for simulation sim_001"

**LLM** → Calls `analyze_population_dynamics(simulation_id="sim_001")`

**Server** → Returns population trends, charts, statistics

**LLM** → Synthesizes natural language response

---

**User**: "Compare the survival rates between simulations with 100 vs 200 agents"

**LLM** → Calls:
1. `list_simulations()` to find relevant sims
2. `compare_simulations()` with filtered list
3. `analyze_survival_rates()` for each

**Server** → Returns comparative analysis

**LLM** → Explains differences and insights

## 🎓 Learning Resources

### Understanding MCP
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

### Understanding Your Database
- Review `farm/database/models.py`
- Explore existing analyzers in `farm/database/analyzers/`
- Check `analysis/simulation_analysis.py` for patterns

### Design Patterns Used
- Repository Pattern (database access)
- Strategy Pattern (tool execution)
- Template Method (ToolBase)
- Dependency Injection (services)
- Factory Pattern (tool creation)

## 🐛 Troubleshooting

### Common Issues

1. **Database not found**
   - Check path is absolute
   - Verify file exists
   - Check permissions

2. **Import errors**
   - Ensure `__init__.py` files exist
   - Check PYTHONPATH
   - Verify package installation

3. **Slow queries**
   - Enable query logging
   - Check indexes
   - Review query patterns
   - Adjust cache settings

4. **Memory issues**
   - Reduce cache size
   - Limit result sets
   - Check for memory leaks
   - Use connection pooling

## ✅ Success Criteria

### Technical
- [ ] All tools return correct data
- [ ] 95% of queries complete <2s
- [ ] Test coverage >80%
- [ ] Zero database modifications
- [ ] Graceful error handling

### Functional
- [ ] LLM can answer complex questions
- [ ] Multi-step analysis workflows work
- [ ] Comparisons across simulations accurate
- [ ] Spatial and temporal queries functional

### Quality
- [ ] Comprehensive documentation
- [ ] Code follows SOLID principles
- [ ] Logging and monitoring in place
- [ ] Performance acceptable

## 🔮 Future Enhancements

### Phase 2 Features
- [ ] Streaming large result sets
- [ ] Real-time simulation monitoring
- [ ] Visualization generation (plots/charts)
- [ ] ML model integration for predictions
- [ ] Natural language query translation
- [ ] Multi-database federation
- [ ] Write operations (with safeguards)
- [ ] Query optimization recommendations

### Potential Integrations
- Jupyter notebook support
- Web dashboard
- Slack/Discord bot
- API gateway
- GraphQL interface

## 📞 Getting Help

### In Cursor

**Use these prompts**:
- "Implement [tool name] following the ToolBase pattern from design doc"
- "Generate comprehensive tests for [component]"
- "Explain this error: [error message]"
- "Refactor this code to follow SOLID principles"

**Cursor Features**:
- `Cmd+K`: Inline code generation
- `Cmd+L`: Chat with context
- `Cmd+I`: Explain code
- Split view: Design doc + implementation

### Resources
1. Review design documents
2. Check existing analyzers for patterns
3. Look at similar tools for examples
4. Use Cursor AI for specific questions

## 📈 Progress Tracking

### Milestones

- [ ] **M1: Foundation Complete** (Week 1)
  - Config, services, base classes working
  - First tool implemented and tested

- [ ] **M2: Core Tools Complete** (Week 2)
  - 10 essential tools working
  - Server running with FastMCP

- [ ] **M3: Analysis Complete** (Week 3)
  - All analysis tools implemented
  - Advanced features working

- [ ] **M4: Production Ready** (Week 4)
  - All tests passing
  - Documentation complete
  - Performance validated
  - LLM integration verified

## 🎯 Next Steps

1. **Read the documents in order**:
   - Start with Requirements (understand what)
   - Review Design (understand how)
   - Follow Implementation Guide (understand process)
   - Use Quick Start (get hands-on)

2. **Set up your environment**:
   - Install dependencies
   - Create project structure
   - Set up testing

3. **Build incrementally**:
   - Start with 1 tool
   - Add tests
   - Expand gradually
   - Iterate and refine

4. **Use Cursor effectively**:
   - Keep design docs open
   - Use AI for boilerplate
   - Review generated code carefully
   - Test continuously

## 📋 Implementation Checklist

Copy this to track your progress:

```markdown
## Foundation
- [ ] Project structure created
- [ ] Config system implemented
- [ ] Database service working
- [ ] Cache service implemented
- [ ] Base tool class complete
- [ ] Exception handling in place
- [ ] Logging configured

## Core Tools (Priority)
- [ ] get_simulation_info
- [ ] list_simulations
- [ ] query_agents
- [ ] query_actions
- [ ] get_simulation_metrics

## Analysis Tools
- [ ] analyze_population_dynamics
- [ ] analyze_survival_rates
- [ ] analyze_resource_efficiency
- [ ] identify_critical_events

## Comparison Tools
- [ ] compare_simulations
- [ ] compare_parameters

## Server & Integration
- [ ] FastMCP server working
- [ ] CLI implemented
- [ ] LLM integration tested
- [ ] Documentation complete

## Testing
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] Performance tests
- [ ] All tests passing

## Production
- [ ] Performance validated
- [ ] Security reviewed
- [ ] Documentation complete
- [ ] Ready for use
```

---

**You're all set! Start with the Quick Start guide and build something amazing! 🚀**