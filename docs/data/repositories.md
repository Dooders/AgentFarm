# Repository Documentation

This document provides comprehensive documentation for all repository classes in the AgentFarm data layer. Repositories act as data access layers that encapsulate database query logic and provide structured interfaces for accessing simulation data.

## Overview

Repositories follow the Repository pattern to provide a clean abstraction over the database layer. Each repository is responsible for a specific domain of data and provides methods to query and retrieve data in a structured way.

### Common Interface

All repositories inherit from `BaseRepository` and provide consistent patterns:

```python
class BaseRepository[T]:
    def __init__(self, session_manager: SessionManager)
    def _execute_in_transaction(self, query_func)
```

### Session Management

Repositories use `SessionManager` for database operations, which provides:
- Automatic session management
- Transaction safety with retry logic
- Connection pooling
- Error handling

## Repository Classes

### 1. ActionRepository

**Purpose**: Manages agent action records and provides filtering capabilities.

**Location**: `farm/database/repositories/action_repository.py`

**Key Methods**:

```python
class ActionRepository(BaseRepository[ActionModel]):
    def get_actions_by_scope(
        self,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[AgentActionData]:
```

**Features**:
- Scope-based filtering (episode, experiment, simulation)
- Agent-specific filtering
- Step-based filtering
- Range-based filtering
- Joins with agent data for complete context

**Usage Example**:
```python
from farm.database.repositories.action_repository import ActionRepository
from farm.database.session_manager import SessionManager

session_manager = SessionManager("simulation.db")
action_repo = ActionRepository(session_manager)

# Get all actions for a specific agent
actions = action_repo.get_actions_by_scope(
    scope="simulation",
    agent_id="agent_001"
)

# Get actions for a specific time range
actions = action_repo.get_actions_by_scope(
    scope="episode",
    step_range=(100, 200)
)
```

### 2. AgentRepository

**Purpose**: Comprehensive agent data access including states, actions, and lifecycle information.

**Location**: `farm/database/repositories/agent_repository.py`

**Key Methods**:

```python
class AgentRepository(BaseRepository[AgentModel]):
    def get_agent_by_id(self, agent_id: str) -> Optional[AgentModel]
    def get_actions_by_agent_id(self, agent_id: str) -> List[ActionModel]
    def get_states_by_agent_id(self, agent_id: str) -> List[AgentStateModel]
    def get_health_incidents_by_agent_id(self, agent_id: str) -> List[HealthIncidentData]
    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]
    def get_agent_current_stats(self, agent_id: str) -> Dict[str, Any]
    def get_agent_performance_metrics(self, agent_id: str) -> Dict[str, Any]
    def get_agent_state_history(self, agent_id: str) -> List[AgentStateModel]
    def get_agent_children(self, agent_id: str) -> List[AgentModel]
    def get_random_agent_id(self) -> Optional[str]
```

**Features**:
- Complete agent lifecycle data
- Performance metrics calculation
- Genealogical relationships
- Health incident tracking
- State history analysis

**Usage Example**:
```python
from farm.database.repositories.agent_repository import AgentRepository

agent_repo = AgentRepository(session_manager)

# Get complete agent information
agent = agent_repo.get_agent_by_id("agent_001")

# Get agent's action history
actions = agent_repo.get_actions_by_agent_id("agent_001")

# Get agent's performance metrics
metrics = agent_repo.get_agent_performance_metrics("agent_001")

# Get agent's children (offspring)
children = agent_repo.get_agent_children("agent_001")
```

### 3. PopulationRepository

**Purpose**: Population-level statistics and dynamics analysis.

**Location**: `farm/database/repositories/population_repository.py`

**Key Methods**:

```python
class PopulationRepository(BaseRepository[SimulationStepModel]):
    def get_population_data(self, session, scope: str, ...) -> List[Population]
    def get_agent_type_distribution(self, session, scope: str, ...) -> AgentDistribution
    def get_states(self, scope: str, ...) -> List[AgentStates]
    def evolution(self, session, scope: str, ...) -> AgentEvolutionMetrics
    def get_all_agents(self) -> List[AgentModel]
```

**Features**:
- Population dynamics tracking
- Agent type distribution analysis
- Evolution metrics calculation
- Resource consumption analysis
- Survival rate calculations

**Usage Example**:
```python
from farm.database.repositories.population_repository import PopulationRepository

pop_repo = PopulationRepository(session_manager)

# Get population data over time
population_data = pop_repo.get_population_data(
    session, 
    scope="simulation"
)

# Get agent type distribution
distribution = pop_repo.get_agent_type_distribution(
    session,
    scope="episode"
)

# Get evolution metrics
evolution = pop_repo.evolution(
    session,
    scope="simulation",
    generation=2
)
```

### 4. ResourceRepository

**Purpose**: Resource distribution, consumption, and efficiency analysis.

**Location**: `farm/database/repositories/resource_repository.py`

**Key Methods**:

```python
class ResourceRepository(BaseRepository[ResourceModel]):
    def resource_distribution(self, session) -> List[ResourceDistributionStep]
    def consumption_patterns(self, session) -> ConsumptionStats
    def resource_hotspots(self, session) -> List[ResourceHotspot]
    def efficiency_metrics(self, session) -> ResourceEfficiencyMetrics
    def execute(self, session) -> ResourceAnalysis
```

**Features**:
- Resource distribution analysis
- Consumption pattern identification
- Hotspot detection
- Efficiency metrics calculation
- Comprehensive resource analysis

**Usage Example**:
```python
from farm.database.repositories.resource_repository import ResourceRepository

resource_repo = ResourceRepository(session_manager)

# Get resource distribution over time
distribution = resource_repo.resource_distribution(session)

# Get consumption patterns
consumption = resource_repo.consumption_patterns(session)

# Get resource hotspots
hotspots = resource_repo.resource_hotspots(session)

# Get comprehensive analysis
analysis = resource_repo.execute(session)
```

### 5. LearningRepository

**Purpose**: Learning experience and module performance analysis.

**Location**: `farm/database/repositories/learning_repository.py`

**Key Methods**:

```python
class LearningRepository(BaseRepository[LearningExperienceModel]):
    def get_learning_progress(self, session, scope: str, ...) -> List[LearningProgress]
    def get_module_performance(self, session, scope: str, ...) -> Dict[str, ModulePerformance]
    def get_agent_learning_stats(self, session, agent_id: Optional[int], ...) -> Dict[str, AgentLearningStats]
    def get_learning_experiences(self, session, scope: str, ...) -> List[LearningExperienceModel]
```

**Features**:
- Learning progress tracking
- Module performance analysis
- Agent-specific learning statistics
- Experience aggregation
- Reward progression analysis

**Usage Example**:
```python
from farm.database.repositories.learning_repository import LearningRepository

learning_repo = LearningRepository(session_manager)

# Get learning progress over time
progress = learning_repo.get_learning_progress(
    session,
    scope="simulation"
)

# Get module performance
module_performance = learning_repo.get_module_performance(
    session,
    scope="episode"
)

# Get agent learning stats
agent_stats = learning_repo.get_agent_learning_stats(
    session,
    agent_id="agent_001"
)
```

### 6. SimulationRepository

**Purpose**: Simulation state and step-based data retrieval.

**Location**: `farm/database/repositories/simulation_repository.py`

**Key Methods**:

```python
class SimulationRepository(BaseRepository[SimulationStepModel]):
    def agent_states(self, step_number: Optional[int] = None) -> List[AgentStates]
    def resource_states(self, step_number: int) -> List[ResourceStates]
    def simulation_state(self, step_number: int) -> SimulationState
    def execute(self, step_number: int) -> SimulationResults
```

**Features**:
- Step-based state retrieval
- Complete simulation state access
- Agent and resource state aggregation
- Simulation results compilation

**Usage Example**:
```python
from farm.database.repositories.simulation_repository import SimulationRepository

sim_repo = SimulationRepository(session_manager)

# Get agent states for a specific step
agent_states = sim_repo.agent_states(step_number=100)

# Get resource states for a specific step
resource_states = sim_repo.resource_states(step_number=100)

# Get complete simulation state
simulation_state = sim_repo.simulation_state(step_number=100)

# Get complete simulation results
results = sim_repo.execute(step_number=100)
```

### 7. GUIRepository

**Purpose**: GUI-specific data queries for visualization and analysis.

**Location**: `farm/database/repositories/gui_repository.py`

**Key Methods**:

```python
class GUIRepository(BaseRepository[SimulationStepModel]):
    def get_historical_data(self, agent_id: Optional[int], step_range: Optional[Tuple[int, int]]) -> Dict
    def get_metrics_summary(self) -> Dict
    def get_step_data(self, step_number: int) -> Dict
    def get_simulation_data(self, step_number: int) -> Dict
```

**Features**:
- Historical data retrieval for plotting
- Metrics summarization
- Step-specific data access
- Simulation data aggregation for GUI

**Usage Example**:
```python
from farm.database.repositories.gui_repository import GUIRepository

gui_repo = GUIRepository(session_manager)

# Get historical data for plotting
historical_data = gui_repo.get_historical_data(
    agent_id="agent_001",
    step_range=(0, 1000)
)

# Get metrics summary
summary = gui_repo.get_metrics_summary()

# Get data for specific step
step_data = gui_repo.get_step_data(step_number=500)

# Get simulation data
sim_data = gui_repo.get_simulation_data(step_number=500)
```

## Common Patterns

### Scope Filtering

Most repositories support scope-based filtering:

```python
# Available scopes
scopes = ["simulation", "episode", "experiment", "agent"]

# Example usage
actions = action_repo.get_actions_by_scope(scope="simulation")
```

### Agent Filtering

Repositories support agent-specific filtering:

```python
# Filter by specific agent
data = repo.get_data(agent_id="agent_001")

# Filter by agent type
data = repo.get_data(agent_type="system")
```

### Step Filtering

Time-based filtering is supported:

```python
# Single step
data = repo.get_data(step=100)

# Step range
data = repo.get_data(step_range=(100, 200))
```

### Error Handling

Repositories use `SessionManager` for robust error handling:

```python
try:
    data = repo.get_data()
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    # Handle error appropriately
```

## Performance Considerations

### Query Optimization

- Repositories use appropriate indexes
- Joins are optimized for common queries
- Lazy loading is used where appropriate

### Caching

- Session-level caching is provided by `SessionManager`
- Repository-level caching can be implemented as needed
- Consider data freshness requirements

### Batch Operations

- Use step ranges for large datasets
- Consider pagination for very large result sets
- Use appropriate scopes to limit data processing

## Best Practices

### 1. Use Appropriate Repositories

```python
# For action analysis
action_repo = ActionRepository(session_manager)

# For agent analysis
agent_repo = AgentRepository(session_manager)

# For population analysis
pop_repo = PopulationRepository(session_manager)
```

### 2. Leverage Scope Filtering

```python
# Use specific scopes to limit data processing
data = repo.get_data(scope="episode")  # Instead of "simulation"
```

### 3. Handle Errors Gracefully

```python
try:
    data = repo.get_data()
except Exception as e:
    logger.error(f"Repository error: {e}")
    # Provide fallback or error handling
```

### 4. Use Type Hints

```python
from typing import List, Optional
from farm.database.data_types import AgentActionData

def get_actions() -> List[AgentActionData]:
    return action_repo.get_actions_by_scope(scope="simulation")
```

## Integration with Services

Repositories are typically used by services that coordinate multiple repositories:

```python
from farm.database.services.actions_service import ActionsService
from farm.database.repositories.action_repository import ActionRepository

action_repo = ActionRepository(session_manager)
actions_service = ActionsService(action_repo)

# Service coordinates multiple repositories and analyzers
results = actions_service.analyze_actions(
    scope="simulation",
    analysis_types=['stats', 'behavior', 'causal']
)
```

## Cross-References

For related documentation:

- **Database Schema**: [Database Schema](database_schema.md)
- **Data API**: [Data API Overview](data_api.md)
- **Services**: [Services Documentation](data_services.md)
- **Analysis**: [Analysis Overview](analysis/Analysis.md)

## Notes

- All repositories support multi-simulation databases via `simulation_id` filtering
- Repositories use `SessionManager` for transaction safety
- Type hints are provided for all public methods
- Error handling is consistent across all repositories
- Performance optimizations are built into common query patterns 