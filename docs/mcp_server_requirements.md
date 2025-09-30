# FastMCP Server for Simulation Database Analysis - Requirements

## 1. Overview

Design a custom Model Context Protocol (MCP) server using FastMCP that enables LLM agents to interact with the SQLAlchemy-based simulation database for information retrieval and analysis of agent-based simulation results.

## 2. System Context

### Current Database Schema
The simulation database contains:
- **AgentModel**: Agent entities with lifecycle data, genetics, and attributes
- **AgentStateModel**: Time-series agent state data (position, health, resources)
- **ActionModel**: Agent behavior and actions
- **ResourceModel**: Resource states and positions
- **SimulationStepModel**: Per-step simulation metrics
- **InteractionModel**: Agent-to-agent and agent-to-resource interactions
- **ReproductionEventModel**: Reproduction attempts and outcomes
- **SocialInteractionModel**: Social behaviors between agents
- **ExperimentModel**: Research experiment metadata
- **Simulation**: Individual simulation metadata
- **LearningExperienceModel**: Agent learning data
- **HealthIncident**: Health-related events

### Existing Analysis Capabilities
- Population analyzers
- Resource analyzers
- Action statistics
- Behavior clustering
- Temporal pattern analysis
- Spatial analysis
- Causal analysis
- Decision pattern analysis

## 3. Functional Requirements

### 3.1 Database Query Capabilities

**FR-1: Basic Data Retrieval**
- Retrieve simulation metadata (ID, status, parameters, duration)
- Query agent data by ID, type, generation, or lifecycle stage
- Fetch simulation step metrics within time ranges
- Access action logs with filtering by type, agent, or step

**FR-2: Aggregated Analytics**
- Calculate population statistics over time ranges
- Compute resource distribution metrics
- Analyze agent survival rates by generation/type
- Generate interaction frequency matrices
- Calculate reproduction success rates

**FR-3: Temporal Analysis**
- Identify critical simulation events (population spikes/crashes)
- Track metric evolution over time windows
- Compare simulation phases (early/mid/late game)
- Detect anomalies and trends

**FR-4: Comparative Analysis**
- Compare metrics across multiple simulations
- Analyze parameter impact on outcomes
- Identify best-performing configurations
- Generate experiment-level summaries

**FR-5: Agent-Focused Queries**
- Retrieve complete agent lifecycle (birth to death)
- Analyze individual agent decision patterns
- Track agent lineages and family trees
- Examine agent learning progression

**FR-6: Spatial Analysis**
- Query agents/resources by spatial region
- Analyze movement patterns
- Identify resource hotspots
- Examine spatial clustering behaviors

### 3.2 Tool Definitions

**FR-7: MCP Tools**
The server must expose the following tool categories:

1. **Metadata Tools**
   - `list_simulations`: List all simulations with filtering
   - `get_simulation_info`: Get detailed simulation metadata
   - `list_experiments`: List research experiments
   - `get_experiment_info`: Get experiment details

2. **Query Tools**
   - `query_agents`: Query agents with flexible filtering
   - `query_actions`: Retrieve action logs
   - `query_states`: Get agent states over time
   - `query_resources`: Fetch resource states
   - `query_interactions`: Retrieve interaction data
   - `get_simulation_metrics`: Get step-level metrics

3. **Analysis Tools**
   - `analyze_population_dynamics`: Population trends over time
   - `analyze_survival_rates`: Survival analysis by cohort
   - `analyze_resource_efficiency`: Resource utilization metrics
   - `analyze_agent_performance`: Individual agent analysis
   - `identify_critical_events`: Detect significant events
   - `analyze_social_patterns`: Social interaction patterns

4. **Comparison Tools**
   - `compare_simulations`: Multi-simulation comparison
   - `compare_parameters`: Parameter impact analysis
   - `rank_configurations`: Performance ranking

5. **Advanced Tools**
   - `build_agent_lineage`: Construct family trees
   - `analyze_spatial_distribution`: Spatial statistics
   - `detect_behavioral_clusters`: Cluster similar agents
   - `predict_outcomes`: ML-based predictions

### 3.3 Response Formatting

**FR-8: Structured Outputs**
- Return data in JSON format for programmatic access
- Support markdown-formatted summaries for readability
- Include metadata (query time, result count, data source)
- Provide pagination for large result sets

**FR-9: Visualization Support**
- Generate text-based charts (ASCII/Unicode)
- Provide data in formats suitable for external plotting
- Support table formatting for tabular data

## 4. Non-Functional Requirements

### 4.1 Performance

**NFR-1: Query Performance**
- Queries should complete in < 2 seconds for typical operations
- Support connection pooling to reduce overhead
- Implement query result caching for repeated requests
- Use efficient SQLAlchemy query patterns (joinedload, etc.)

**NFR-2: Scalability**
- Support concurrent queries from multiple LLM agents
- Handle databases with millions of records
- Implement pagination for large result sets (max 1000 rows per query)

### 4.2 Security

**NFR-3: Data Protection**
- Read-only database access (no modifications through MCP)
- SQL injection prevention through parameterized queries
- Input validation for all tool parameters
- Configurable database path with access restrictions

**NFR-4: Error Handling**
- Graceful handling of database connection failures
- Clear error messages for invalid queries
- Timeout mechanisms for long-running queries
- Logging of all queries for audit purposes

### 4.3 Usability

**NFR-5: LLM-Friendly Design**
- Tool descriptions should be comprehensive and clear
- Parameter schemas should include examples
- Support natural language-like parameters (e.g., "last 100 steps" → convert to step range)
- Provide helpful error messages when parameters are invalid

**NFR-6: Documentation**
- Each tool has detailed description
- Parameters include type, description, and constraints
- Examples provided for common use cases
- Schema documentation for database models

### 4.4 Maintainability

**NFR-7: Code Quality**
- Follow SOLID principles
- Modular design with clear separation of concerns
- Comprehensive docstrings
- Type hints for all functions

**NFR-8: Extensibility**
- Easy to add new tools
- Pluggable analyzer modules
- Configuration-driven behavior
- Version compatibility management

## 5. Technical Requirements

### 5.1 Technology Stack

**TR-1: Core Dependencies**
- FastMCP for MCP server implementation
- SQLAlchemy for database ORM
- Pandas for data manipulation
- NumPy for numerical operations
- Python 3.8+ compatibility

**TR-2: Database Support**
- SQLite (primary)
- PostgreSQL (optional future support)
- Support for both file-based and in-memory databases

### 5.2 Configuration

**TR-3: Server Configuration**
- Database path/connection string
- Query timeout settings
- Cache configuration
- Logging level
- Maximum result set size

**TR-4: Tool Configuration**
- Enable/disable specific tools
- Custom analyzer integration
- Default parameter values
- Result format preferences

## 6. Data Requirements

### 6.1 Input Validation

**DR-1: Parameter Validation**
- Simulation IDs must exist in database
- Step ranges must be valid (start <= end)
- Agent IDs must be strings
- Numerical ranges must be sensible
- Enum values must be from allowed sets

**DR-2: Type Safety**
- All inputs validated against schema
- Type coercion where appropriate
- Clear error messages for type mismatches

### 6.2 Output Specifications

**DR-3: Result Structure**
```json
{
  "success": true,
  "data": {...},
  "metadata": {
    "query_time_ms": 150,
    "result_count": 42,
    "truncated": false,
    "simulation_id": "sim_001"
  },
  "error": null
}
```

**DR-4: Error Structure**
```json
{
  "success": false,
  "data": null,
  "metadata": {...},
  "error": {
    "type": "ValidationError",
    "message": "Invalid simulation_id: 'xyz' does not exist",
    "details": {...}
  }
}
```

## 7. Use Cases

### UC-1: Population Analysis
**Actor**: LLM Agent
**Goal**: Understand population dynamics over simulation
**Flow**:
1. Agent requests population metrics for simulation
2. Server queries SimulationStepModel for all steps
3. Returns time-series data with births, deaths, totals
4. Agent can request drill-down by agent type or generation

### UC-2: Agent Lifecycle Investigation
**Actor**: LLM Agent
**Goal**: Investigate why specific agents died
**Flow**:
1. Agent queries for agents that died early (death_time < 100)
2. Server retrieves agent records with filters
3. Agent requests action history for suspicious agents
4. Server retrieves ActionModel records
5. Agent requests health incidents
6. Server provides HealthIncident records
7. Agent synthesizes findings

### UC-3: Experiment Comparison
**Actor**: LLM Agent
**Goal**: Compare different parameter configurations
**Flow**:
1. Agent requests list of experiments
2. Server provides experiment metadata
3. Agent selects simulations to compare
4. Server computes comparative metrics
5. Agent analyzes which parameters drove success

### UC-4: Spatial Pattern Detection
**Actor**: LLM Agent
**Goal**: Identify spatial clustering patterns
**Flow**:
1. Agent requests spatial distribution at specific steps
2. Server queries agent positions and resource locations
3. Computes clustering metrics
4. Returns spatial statistics and patterns

### UC-5: Behavioral Analysis
**Actor**: LLM Agent
**Goal**: Understand agent decision-making patterns
**Flow**:
1. Agent requests action distribution by agent type
2. Server aggregates ActionModel by type and agent_type
3. Agent requests contextual information (resources, health)
4. Server joins state data with actions
5. Agent identifies behavioral patterns

## 8. Constraints

### C-1: Read-Only Operations
- The MCP server must NOT support database modifications
- All tools are read-only queries and analyses

### C-2: Resource Limits
- Query timeout: 30 seconds maximum
- Result set size: 10,000 records maximum (with pagination)
- Cache size: 100 MB maximum
- Concurrent queries: 10 maximum

### C-3: Compatibility
- Must work with existing database schema
- No schema migrations required
- Support both SimulationDatabase and InMemorySimulationDatabase
- Compatible with existing analysis modules

## 9. Success Criteria

**SC-1**: LLM agent can answer complex questions about simulations without human intervention
**SC-2**: Query response time < 2 seconds for 95% of queries
**SC-3**: Zero data corruption or modification through MCP server
**SC-4**: Comprehensive error handling with 100% query safety
**SC-5**: All major analysis use cases covered by available tools
**SC-6**: Easy integration with existing codebase (< 500 LOC for integration)

## 10. Future Considerations

**F-1**: Support for streaming large result sets
**F-2**: Integration with visualization tools (generate plots)
**F-3**: Machine learning model integration for predictions
**F-4**: Real-time simulation monitoring
**F-5**: Write operations (with proper safeguards)
**F-6**: Multi-database federation (query across multiple simulation DBs)
**F-7**: Natural language query translation (text → SQL)
**F-8**: Caching and query optimization recommendations