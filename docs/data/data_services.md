# Services Documentation

This document provides comprehensive documentation for all service classes in the AgentFarm data layer. Services act as high-level orchestrators that coordinate complex operations between repositories, analyzers, and other components.

## Overview

Services provide a clean abstraction layer between application logic and underlying implementation details. They coordinate multiple components to perform complex operations while maintaining separation of concerns.

### Service Design Principles

1. **Separation of Concerns**
   - Services coordinate between different components but don't implement core logic
   - Each service focuses on a specific domain area (e.g., actions, population)

2. **Dependency Injection**
   - Services receive their dependencies through constructor injection
   - Makes services more testable and loosely coupled

3. **High-Level Interface**
   - Services provide simple, intuitive interfaces for complex operations
   - Hide implementation details and coordinate between multiple components

4. **Stateless Operation**
   - Services generally don't maintain state between operations
   - Each method call is independent and self-contained

## Service Classes

### 1. ActionsService

**Purpose**: High-level service for analyzing agent actions using various analyzers.

**Location**: `farm/database/services/actions_service.py`

**Key Features**:
- Orchestrates different types of analysis on agent actions
- Coordinates multiple analyzers for comprehensive analysis
- Provides unified interface for action analysis
- Supports selective analysis types

**Available Analysis Types**:
- `stats`: Basic action statistics and metrics
- `behavior`: Behavioral patterns and clustering
- `causal`: Causal relationships between actions
- `decision`: Decision patterns and trends
- `resource`: Resource impacts of actions
- `sequence`: Action sequence analysis
- `temporal`: Temporal patterns and trends

**Key Methods**:

```python
class ActionsService:
    def __init__(self, action_repository: ActionRepository)
    
    def analyze_actions(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
        analysis_types: Optional[List[str]] = None,
    ) -> Dict[str, Union[List[ActionMetrics], BehaviorClustering, ...]]
    
    def get_action_summary(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]
```

**Usage Example**:
```python
from farm.database.services.actions_service import ActionsService
from farm.database.repositories.action_repository import ActionRepository

# Initialize repository and service
action_repo = ActionRepository(session_manager)
actions_service = ActionsService(action_repo)

# Perform comprehensive analysis
results = actions_service.analyze_actions(
    scope="SIMULATION",
    agent_id=123,
    analysis_types=['stats', 'behavior', 'causal']
)

# Get high-level summary
summary = actions_service.get_action_summary(
    scope="SIMULATION",
    agent_id=123
)

# Access specific analysis results
action_stats = results['action_stats']
behavior_clusters = results['behavior_clusters']
causal_analysis = results['causal_analysis']
```

**Analysis Results Structure**:

```python
{
    'action_stats': List[ActionMetrics],           # Basic statistics
    'behavior_clusters': BehaviorClustering,       # Behavioral patterns
    'causal_analysis': List[CausalAnalysis],      # Causal relationships
    'decision_patterns': DecisionPatterns,         # Decision patterns
    'resource_impacts': List[ResourceImpact],     # Resource effects
    'sequence_patterns': List[SequencePattern],   # Action sequences
    'temporal_patterns': List[TimePattern]        # Temporal patterns
}
```

**Action Summary Structure**:

```python
{
    'move': {
        'success_rate': 0.85,        # Percentage of successful actions
        'avg_reward': 0.5,           # Average reward per action
        'frequency': 0.3,            # Relative frequency
        'resource_efficiency': 0.8   # Resource gain/loss efficiency
    },
    'gather': {
        'success_rate': 0.92,
        'avg_reward': 1.2,
        'frequency': 0.25,
        'resource_efficiency': 0.9
    }
    # ... other action types
}
```

### 2. PopulationService

**Purpose**: Population-level analysis and statistics coordination.

**Location**: `farm/database/services/population_service.py`

**Key Features**:
- Comprehensive population statistics calculation
- Resource consumption analysis
- Population variance and distribution metrics
- Agent type distribution analysis
- Survival metrics calculation

**Key Methods**:

```python
class PopulationService:
    def execute(self, session) -> PopulationStatistics
    
    def basic_population_statistics(
        self, 
        session, 
        pop_data: Optional[List[Population]] = None
    ) -> BasicPopulationStatistics
```

**Usage Example**:
```python
from farm.database.services.population_service import PopulationService
from farm.database.repositories.population_repository import PopulationRepository

# Initialize repository and service
pop_repo = PopulationRepository(session_manager)
pop_service = PopulationService(pop_repo)

# Get comprehensive population statistics
stats = pop_service.execute(session)

# Access specific metrics
population_metrics = stats.population_metrics
population_variance = stats.population_variance

print(f"Total agents: {population_metrics.total_agents}")
print(f"Peak population: {population_metrics.peak_population}")
print(f"Population variance: {population_variance.variance}")
```

**Population Statistics Structure**:

```python
PopulationStatistics(
    population_metrics=PopulationMetrics(
        total_agents=150,
        system_agents=50,
        independent_agents=60,
        control_agents=40
    ),
    population_variance=PopulationVariance(
        variance=25.5,
        standard_deviation=5.05,
        coefficient_variation=0.34
    )
)
```

**Basic Population Statistics Structure**:

```python
BasicPopulationStatistics(
    avg_population=125.5,      # Average population across steps
    death_step=1000,           # Final step with agents
    peak_population=200,       # Maximum population reached
    resources_consumed=5000.0, # Total resources consumed
    resources_available=8000.0, # Total resources available
    sum_squared=15750.0,       # Sum of squared populations
    step_count=800             # Number of active steps
)
```

## Service Architecture

### Component Coordination

Services coordinate between multiple components:

```
┌─────────────────┐
│   Services      │  ← High-level orchestration
├─────────────────┤
│  Repositories   │  ← Data access
├─────────────────┤
│   Analyzers     │  ← Analysis logic
├─────────────────┤
│   Database      │  ← Data storage
└─────────────────┘
```

### Data Flow

1. **Service Initialization**
   ```python
   # Services receive repositories through dependency injection
   service = ActionsService(action_repository)
   ```

2. **Method Execution**
   ```python
   # Services coordinate multiple analyzers
   results = service.analyze_actions(scope="SIMULATION")
   ```

3. **Result Aggregation**
   ```python
   # Services aggregate results from multiple components
   summary = service.get_action_summary(scope="SIMULATION")
   ```

## Analysis Scopes

Services support different analysis scopes:

- **SIMULATION**: Complete simulation data
- **EPISODE**: Specific episode or time period
- **AGENT**: Individual agent analysis
- **STEP**: Single simulation step

## Filtering Options

Services support various filtering mechanisms:

- **agent_id**: Filter for specific agent
- **step**: Analyze specific timestep
- **step_range**: Analyze range of timesteps
- **analysis_types**: Select specific analysis types

## Error Handling

Services provide robust error handling:

```python
try:
    results = actions_service.analyze_actions(
        scope="SIMULATION",
        analysis_types=['stats', 'behavior']
    )
except Exception as e:
    logger.error(f"Analysis failed: {e}")
    # Handle error appropriately
```

## Performance Considerations

### Analysis Type Selection

```python
# Only perform needed analysis types for performance
results = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats', 'behavior']  # Skip other types
)
```

### Scope Optimization

```python
# Use appropriate scopes to limit data processing
results = actions_service.analyze_actions(
    scope="EPISODE",  # Instead of "SIMULATION"
    step_range=(100, 200)  # Limit time range
)
```

### Caching

```python
# Services can cache results for repeated queries
# Consider data freshness requirements
```

## Best Practices

### 1. Use Services for High-Level Operations

```python
# Prefer services over direct analyzer usage
actions_service = ActionsService(action_repo)
results = actions_service.analyze_actions(scope="SIMULATION")

# Instead of coordinating analyzers manually
```

### 2. Select Appropriate Analysis Types

```python
# Only request needed analysis types
results = actions_service.analyze_actions(
    analysis_types=['stats', 'behavior']  # Skip unused types
)
```

### 3. Handle Results Appropriately

```python
# Check for expected result types
if 'action_stats' in results:
    stats = results['action_stats']
    for stat in stats:
        print(f"{stat.action_type}: {stat.avg_reward}")
```

### 4. Use Proper Error Handling

```python
try:
    summary = actions_service.get_action_summary(scope="SIMULATION")
except Exception as e:
    logger.error(f"Failed to get action summary: {e}")
    # Provide fallback or error handling
```

## Integration Examples

### Comprehensive Action Analysis

```python
from farm.database.services.actions_service import ActionsService
from farm.database.repositories.action_repository import ActionRepository

# Setup
action_repo = ActionRepository(session_manager)
actions_service = ActionsService(action_repo)

# Perform comprehensive analysis
results = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats', 'behavior', 'causal', 'resource']
)

# Process results
for action_type, metrics in results['action_stats'].items():
    print(f"{action_type}: {metrics.avg_reward:.2f} avg reward")

# Get behavioral insights
clusters = results['behavior_clusters']
print(f"Found {len(clusters.clusters)} behavioral clusters")

# Analyze resource impacts
for impact in results['resource_impacts']:
    print(f"{impact.action_type}: {impact.resource_efficiency:.2f} efficiency")
```

### Population Analysis

```python
from farm.database.services.population_service import PopulationService
from farm.database.repositories.population_repository import PopulationRepository

# Setup
pop_repo = PopulationRepository(session_manager)
pop_service = PopulationService(pop_repo)

# Get population statistics
stats = pop_service.execute(session)

# Analyze population dynamics
print(f"Total agents: {stats.population_metrics.total_agents}")
print(f"Peak population: {stats.population_metrics.peak_population}")
print(f"Population variance: {stats.population_variance.variance:.2f}")
print(f"Coefficient of variation: {stats.population_variance.coefficient_variation:.2f}")
```

## Cross-References

For related documentation:

- **Repositories**: [Repository Documentation](repositories.md)
- **Analyzers**: [Analysis Overview](analysis/Analysis.md)
- **Data API**: [Data API Overview](data_api.md)
- **Database Schema**: [Database Schema](database_schema.md)

## Notes

- Services use dependency injection for testability
- All services support multi-simulation databases
- Services provide high-level interfaces for complex operations
- Error handling is consistent across all services
- Performance optimizations are built into service patterns
- Services coordinate between repositories and analyzers
- Type hints are provided for all public methods 