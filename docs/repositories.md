# Repository Documentation

This document provides a comprehensive overview of all available repositories and the data they can provide. Each repository is designed to handle specific types of data queries and analysis.

## Overview

The repository system provides modular data access layers for different aspects of the simulation. Each repository inherits from `BaseRepository` and uses a `SessionManager` for database operations.

## Available Repositories

### 1. ActionRepository
Handles queries related to agent actions and their outcomes.

```python
from farm.database.repositories import ActionRepository

action_repo = ActionRepository(session_manager)

# Get actions filtered by scope
actions = action_repo.get_actions_by_scope(
    scope="episode",
    agent_id=1,  # optional
    step=100,    # optional
    step_range=(0, 100)  # optional
)
```

**Available Data:**
- Action history by agent
- Action types and frequencies
- Action outcomes and rewards
- State transitions
- Action targets and impacts

### 2. AgentRepository
Manages agent-related data including states, actions, and lifecycle events.

```python
from farm.database.repositories import AgentRepository

agent_repo = AgentRepository(session_manager)

# Get agent by ID
agent = agent_repo.get_agent_by_id("agent_123")

# Get agent states
states = agent_repo.get_states_by_agent_id("agent_123")

# Get health incidents
health_history = agent_repo.get_health_incidents_by_agent_id("agent_123")

# Get current stats
stats = agent_repo.get_agent_current_stats("agent_123")

# Get state history
history = agent_repo.get_agent_state_history("agent_123")

# Get agent children
children = agent_repo.get_agent_children("agent_123")
```

**Available Data:**
- Agent lifecycle information
- State history
- Health incidents
- Current statistics
- Evolutionary relationships
- Action statistics
- Performance metrics

### 3. GUIRepository
Provides data access methods optimized for GUI visualization and analysis.

```python
from farm.database.repositories import GUIRepository

gui_repo = GUIRepository(session_manager)

# Get historical data
history = gui_repo.get_historical_data(
    agent_id=1,  # optional
    step_range=(0, 100)  # optional
)

# Get metrics summary
summary = gui_repo.get_metrics_summary()

# Get step data
step_data = gui_repo.get_step_data(step_number=100)
```

**Available Data:**
- Time series data
- Population metrics
- Resource distribution
- Agent positions
- System statistics
- Performance summaries

### 4. LearningRepository
Handles learning-related data and analysis.

```python
from farm.database.repositories import LearningRepository
from farm.database.session_manager import SessionManager

learning_repo = LearningRepository(session_manager)

# Get learning progress
progress = learning_repo.get_learning_progress(
    session,
    scope="episode",
    agent_id=1,  # optional
    step=100,    # optional
    step_range=(0, 100)  # optional
)

# Get module performance
performance = learning_repo.get_module_performance(
    session,
    scope="episode"
)

# Get agent learning stats (note: agent_id is optional, scope defaults to "simulation")
stats = learning_repo.get_agent_learning_stats(
    session,
    agent_id=1,  # optional
    scope="simulation"  # optional, defaults to "simulation"
)
```

**Available Data:**
- Learning progress metrics
- Module performance statistics
- Agent learning history
- Training metrics
- Experience data
- Reward history

### 5. PopulationRepository
Manages population-level data and demographics.

```python
from farm.database.repositories import PopulationRepository
from farm.database.session_manager import SessionManager

pop_repo = PopulationRepository(session_manager)

# Get population data
pop_data = pop_repo.get_population_data(
    session,
    scope="episode"
)

# Get agent type distribution
distribution = pop_repo.get_agent_type_distribution(
    session,
    scope="episode"
)

# Get agent states (note: requires session parameter)
states = pop_repo.get_states(
    session,  # required session parameter
    scope="episode",
    agent_id=1  # optional
)

# Get evolution metrics
evolution = pop_repo.evolution(
    session,
    scope="episode",
    generation=1  # optional
)
```

**Available Data:**
- Population statistics
- Demographic data
- Agent type distributions
- State snapshots
- Evolution metrics
- Survival statistics

### 6. ResourceRepository
Handles resource-related queries and analysis.

```python
from farm.database.repositories import ResourceRepository
from farm.database.session_manager import SessionManager

resource_repo = ResourceRepository(session_manager)

# Note: ResourceRepository methods use @execute_query decorator and require session parameter
# Get resource distribution
distribution = resource_repo.resource_distribution(session)

# Get consumption patterns
consumption = resource_repo.consumption_patterns(session)

# Get resource hotspots
hotspots = resource_repo.resource_hotspots(session)

# Get efficiency metrics
efficiency = resource_repo.efficiency_metrics(session)

# Get comprehensive analysis
analysis = resource_repo.execute(session)
```

**Available Data:**
- Resource distribution
- Consumption patterns
- Resource hotspots
- Efficiency metrics
- Distribution entropy
- Resource dynamics

### 7. SimulationRepository
Provides access to overall simulation state and metrics.

```python
from farm.database.repositories import SimulationRepository

sim_repo = SimulationRepository(db)  # Note: uses db parameter, not session_manager

# Get agent states
agent_states = sim_repo.agent_states(step_number=100)  # optional

# Get resource states
resource_states = sim_repo.resource_states(step_number=100)

# Get simulation state
sim_state = sim_repo.simulation_state(step_number=100)

# Get complete results
results = sim_repo.execute(step_number=100)
```

**Available Data:**
- Simulation state
- Agent states
- Resource states
- System metrics
- Performance data
- Complete results

## Common Usage Patterns

### 1. Getting Agent History
```python
# Get complete agent history
agent_id = "agent_123"
agent = agent_repo.get_agent_by_id(agent_id)
states = agent_repo.get_states_by_agent_id(agent_id)
actions = action_repo.get_actions_by_scope("simulation", agent_id=agent_id)
learning = learning_repo.get_agent_learning_stats(session, agent_id=agent_id)
```

### 2. Analyzing Population Trends
```python
# Get population analysis
pop_data = pop_repo.get_population_data(session, scope="simulation")
distribution = pop_repo.get_agent_type_distribution(session, scope="simulation")
evolution = pop_repo.evolution(session, scope="simulation")
```

### 3. Resource Analysis
```python
# Get resource analysis (note: requires session parameter)
analysis = resource_repo.execute(session)
hotspots = resource_repo.resource_hotspots(session)
efficiency = resource_repo.efficiency_metrics(session)
```

## Best Practices

1. **Use Session Management**
   - Always use repositories with a proper SessionManager
   - Let repositories handle transaction management

2. **Scope Filtering**
   - Use appropriate scopes to filter data
   - Combine with step ranges for temporal analysis

3. **Performance**
   - Use specific queries rather than fetching all data
   - Take advantage of built-in filtering
   - Consider pagination for large datasets

4. **Error Handling**
   - Repositories include built-in error handling
   - Check return values for None/empty results

## Common Queries

Here are some common query patterns:

1. **Agent Performance**
```python
# Get comprehensive agent performance data
agent_data = {
    "info": agent_repo.get_agent_by_id(agent_id),
    "current_stats": agent_repo.get_agent_current_stats(agent_id),
    "history": agent_repo.get_agent_state_history(agent_id),
    "actions": action_repo.get_actions_by_scope("simulation", agent_id=agent_id),
    "learning": learning_repo.get_agent_learning_stats(session, agent_id=agent_id)
}
```

2. **Population Analysis**
```python
# Get population analysis data
population_analysis = {
    "data": pop_repo.get_population_data(session, scope="simulation"),
    "distribution": pop_repo.get_agent_type_distribution(session, scope="simulation"),
    "evolution": pop_repo.evolution(session, scope="simulation"),
    "resources": resource_repo.execute(session)  # Note: requires session parameter
}
```

3. **Simulation State**
```python
# Get complete simulation state
step_state = {
    "simulation": sim_repo.simulation_state(step_number),
    "agents": sim_repo.agent_states(step_number),
    "resources": sim_repo.resource_states(step_number)
}
```

## Error Handling

All repositories include built-in error handling through the SessionManager and will:
- Use `execute_with_retry()` for automatic retry of failed queries
- Handle transaction rollbacks automatically
- Return None for missing data
- Raise appropriate SQLAlchemy exceptions for critical errors
- Use `@execute_query` decorator (ResourceRepository) for consistent error handling
- Provide proper session management through SessionManager

## Dependencies

- SQLAlchemy: Database ORM
- Pandas: Data processing
- NumPy: Numerical operations
- PyTorch: Tensor operations (for some repositories)

## Further Reading

- [Database Schema Documentation](database_schema.md)
- [Data API Documentation](data_api.md)
- [Metrics Documentation](metrics.md) 