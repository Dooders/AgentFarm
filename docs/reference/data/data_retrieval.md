# Data Retrieval Module

This module provides a comprehensive interface for querying and analyzing simulation data through the `DataRetriever` class. It handles all database operations with optimized queries and efficient data aggregation methods.

## Features

- Agent statistics and lifecycle analysis
- Population dynamics and demographics
- Resource distribution and consumption patterns
- Learning and adaptation metrics
- Behavioral pattern analysis
- Historical trend analysis
- Performance and efficiency metrics

## Key Methods

### Population Statistics

#### `get_population_statistics()`

Returns comprehensive population statistics including:

- Basic stats (average population, peak population, death step)
- Resource metrics (utilization, consumption)
- Population variance (standard deviation, coefficient of variation)
- Agent distribution (by type)
- Survival metrics (survival rate, average lifespan)

```python
{
    "basic_stats": {
        "average_population": float,
        "peak_population": int,
        "death_step": int,
        "total_steps": int
    },
    "resource_metrics": {
        "resource_utilization": float,
        "resources_consumed": float,
        "resources_available": float,
        "utilization_per_agent": float
    },
    "population_variance": {
        "variance": float,
        "standard_deviation": float,
        "coefficient_variation": float
    },
    "agent_distribution": {
        "system_agents": float,
        "independent_agents": float,
        "control_agents": float
    },
    "survival_metrics": {
        "survival_rate": float,
        "average_lifespan": float
    }
}
```

### Advanced Statistics

#### `get_advanced_statistics()`

Provides detailed analysis of simulation behavior including:

- Population metrics
- Interaction patterns
- Resource utilization
- Agent type distribution
- Survival metrics

```python
{
    "population_metrics": {
        "peak_population": int,
        "average_population": float,
        "minimum_population": int,
        "total_steps": int,
        "average_health": float,
        "population_diversity": float
    },
    "interaction_metrics": {
        "total_actions": int,
        "conflict_rate": float,
        "cooperation_rate": float,
        "reproduction_rate": float,
        "conflict_cooperation_ratio": float
    },
    "resource_metrics": {
        "average_efficiency": float,
        "average_total_resources": float,
        "average_agent_resources": float,
        "resource_utilization": float
    },
    "agent_distribution": {
        "system_ratio": float,
        "independent_ratio": float,
        "control_ratio": float,
        "type_entropy": float
    },
    "survival_metrics": {
        "population_stability": float,
        "health_maintenance": float,
        "interaction_rate": float
    }
}
```

### Simulation Data

#### `get_simulation_data(step_number: int)`

Retrieves complete simulation state for a specific step:

- Agent states
- Resource states
- Simulation metrics

```python
{
    "agent_states": List[Tuple],  # (agent_id, type, pos_x, pos_y, resources, health, defending)
    "resource_states": List[Tuple],  # (resource_id, amount, pos_x, pos_y)
    "metrics": Dict  # Current step metrics
}
```

### Agent Analysis

#### `get_agent_decisions(agent_id: Optional[int], start_step: Optional[int], end_step: Optional[int])`

Analyzes agent decision-making patterns with optional filtering:

- Decision patterns by action type
- Reward statistics
- Action frequencies
- Temporal patterns

## Query Optimization

The module implements several optimization strategies:

- Efficient subqueries for population data
- Batch processing for large datasets
- Index utilization for common queries
- Aggregation at database level where possible

## Dependencies

- sqlalchemy: Database ORM and query building
- pandas: Data processing and statistical analysis
- json: JSON data handling
- logging: Error and operation logging

## Usage Example

```python
retriever = DataRetriever(database)

# Get population statistics
pop_stats = retriever.get_population_statistics()

# Get data for specific simulation step
step_data = retriever.get_simulation_data(step_number=100)

# Analyze agent decisions
agent_decisions = retriever.get_agent_decisions(
    agent_id=1,
    start_step=0,
    end_step=100
)
```

## Notes

- All database operations are executed within transactions
- Query optimization is implemented for large datasets
- Results are cached where appropriate
- Error handling and logging is implemented throughout
- Type hints are provided for all methods
