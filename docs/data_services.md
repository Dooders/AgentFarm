# Services Documentation

This document provides a comprehensive overview of all available services and their capabilities. Services act as high-level coordinators that orchestrate complex operations between different components of the system.

## Overview

The service system provides high-level interfaces for complex operations by coordinating between multiple components. Each service focuses on a specific domain area and follows key design principles:

1. **Separation of Concerns**
   - Services coordinate between different components
   - Each service focuses on a specific domain area

2. **Dependency Injection**
   - Services receive dependencies through constructor injection
   - Makes services testable and loosely coupled

3. **High-Level Interface**
   - Simple, intuitive interfaces for complex operations
   - Hide implementation details

4. **Stateless Operation**
   - Services don't maintain state between operations
   - Each method call is independent

## Available Services

### 1. ActionsService
Orchestrates analysis of agent actions using various analyzers.

```python
from farm.database.services import ActionsService
from farm.database.repositories import ActionRepository

# Initialize
action_repo = ActionRepository(session_manager)
actions_service = ActionsService(action_repo)

# Comprehensive analysis
results = actions_service.analyze_actions(
    scope="EPISODE",
    agent_id=123,
    step=100,    # optional
    step_range=(0, 100),  # optional
    analysis_types=['stats', 'behavior', 'causal']
)

# Get action summary
summary = actions_service.get_action_summary(
    scope="SIMULATION",
    agent_id=123
)
```

**Available Analysis Types:**
- `stats`: Basic action statistics and metrics
- `behavior`: Behavioral patterns and clustering
- `causal`: Causal relationships
- `decision`: Decision patterns
- `resource`: Resource impacts
- `sequence`: Action sequences
- `temporal`: Temporal patterns

**Available Data:**
- Action statistics
- Behavior clusters
- Causal analysis
- Decision patterns
- Resource impacts
- Sequence patterns
- Temporal patterns
- Action summaries

### 2. PopulationService
Manages population-level analysis and statistics.

```python
from farm.database.services import PopulationService

pop_service = PopulationService(session_manager)

# Get comprehensive statistics
stats = pop_service.execute()

# Get basic statistics
basic_stats = pop_service.basic_population_statistics()
```

**Available Data:**
- Population metrics
  - Total agents
  - System agents
  - Independent agents
  - Control agents
- Population variance
  - Variance
  - Standard deviation
  - Coefficient of variation
- Basic statistics
  - Average population
  - Death step
  - Peak population
  - Resources consumed
  - Resources available
  - Statistical measures

### 3. AgentService
Manages individual agent analysis and tracking.

```python
from farm.database.services import AgentService

agent_service = AgentService(session_manager)

# Get agent analysis
analysis = agent_service.analyze_agent(agent_id=123)

# Get agent history
history = agent_service.get_agent_history(agent_id=123)
```

**Available Data:**
- Agent metrics
- Performance history
- Behavior analysis
- Learning progress
- Resource usage
- Interaction patterns

## Common Usage Patterns

### 1. Comprehensive Agent Analysis
```python
# Get complete agent analysis
agent_id = 123
analysis = {
    "actions": actions_service.analyze_actions(
        agent_id=agent_id,
        analysis_types=['stats', 'behavior', 'causal']
    ),
    "summary": actions_service.get_action_summary(agent_id=agent_id),
    "history": agent_service.get_agent_history(agent_id=agent_id)
}
```

### 2. Population Analysis
```python
# Get population analysis
population_analysis = {
    "stats": pop_service.execute(),
    "basic_stats": pop_service.basic_population_statistics()
}
```

## Best Practices

1. **Service Initialization**
   - Initialize services with proper repositories
   - Use dependency injection
   - Keep services stateless

2. **Analysis Scoping**
   - Use appropriate scopes for analysis
   - Combine with step ranges for temporal analysis
   - Filter by agent ID when needed

3. **Performance**
   - Request only needed analysis types
   - Use specific queries rather than full analysis
   - Consider data volume when requesting histories

4. **Error Handling**
   - Services include built-in error handling
   - Check return values for None/empty results
   - Handle exceptions appropriately

## Dependencies

- SQLAlchemy: Database ORM
- Pandas: Data processing
- NumPy: Numerical operations
- PyTorch: Machine learning operations (for some services)

## Service Architecture

Services follow a layered architecture:
1. **Service Layer**: High-level coordination
2. **Repository Layer**: Data access
3. **Analyzer Layer**: Specific analysis implementations
4. **Data Layer**: Raw data storage

## Error Handling

All services include built-in error handling and will:
- Log errors appropriately
- Return meaningful error messages
- Handle missing data gracefully
- Maintain data consistency

## Further Reading

- [Repository Documentation](repositories.md)
- [Database Schema Documentation](database_schema.md)
- [Analysis Documentation](analysis.md) 