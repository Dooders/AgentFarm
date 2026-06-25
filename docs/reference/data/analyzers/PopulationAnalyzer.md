# Population Analyzer

The `PopulationAnalyzer` provides comprehensive analysis of population dynamics, resource utilization, and agent distributions across simulation steps. This analyzer calculates statistics about agent populations, resource consumption, and survival metrics.

## Overview

The analyzer examines:
- Population dynamics and growth patterns
- Agent distribution and type analysis
- Resource consumption and availability
- Survival rates and mortality patterns
- Population variance and stability metrics

## Key Features

### 1. Basic Population Statistics
```python
stats = analyzer.analyze_basic_statistics()
```
Returns `BasicPopulationStatistics` containing:
- Average population and peak population
- Death step and resource consumption
- Population variance and stability metrics

### 2. Agent Distribution Analysis
```python
distribution = analyzer.analyze_agent_distribution()
```
Returns `AgentDistribution` containing:
- Agent type distributions
- Population composition analysis
- Type-specific metrics

### 3. Population Momentum Analysis
```python
momentum = analyzer.analyze_population_momentum()
```
Returns momentum metrics for population growth trends.

### 4. Comprehensive Statistics
```python
comprehensive = analyzer.analyze_comprehensive_statistics()
```
Returns `PopulationStatistics` with all population metrics.

## Detailed Metrics

### Basic Population Statistics

Each `BasicPopulationStatistics` object contains:

- **Population Metrics**
  - `avg_population`: Mean population across all steps
  - `peak_population`: Maximum population reached
  - `lowest_population`: Minimum population reached
  - `initial_population`: Population count at first step

- **Temporal Metrics**
  - `death_step`: Final step number where agents existed
  - `total_steps`: Total number of simulation steps

- **Resource Metrics**
  - `resources_consumed`: Total resources used by agents
  - `resources_available`: Total resources available
  - `resource_efficiency`: Resource utilization ratio

- **Statistical Metrics**
  - `sum_squared`: Sum of squared population counts (for variance)
  - `population_variance`: Variance in population over time

### Agent Distribution Analysis

Each `AgentDistribution` object contains:

- **Type Distributions**
  - `system_agents`: Number of system-type agents
  - `independent_agents`: Number of independent-type agents
  - `control_agents`: Number of control-type agents

- **Distribution Metrics**
  - `type_ratios`: Proportions of each agent type
  - `dominant_type`: Most common agent type
  - `diversity_index`: Measure of type diversity

### Population Momentum

The momentum analysis provides:
- Growth rate calculations
- Population stability metrics
- Trend analysis over time

## Usage Examples

### Basic Population Analysis
```python
from farm.database.analyzers.population_analyzer import PopulationAnalyzer
from farm.database.repositories.population_repository import PopulationRepository

repository = PopulationRepository(session)
analyzer = PopulationAnalyzer(repository)

# Get comprehensive population statistics
stats = analyzer.analyze_comprehensive_statistics()

# Print key metrics
print(f"Peak Population: {stats.population_metrics.total_agents}")
print(f"Average Population: {stats.basic_stats.avg_population:.1f}")
print(f"Resource Efficiency: {stats.basic_stats.resource_efficiency:.2%}")
```

### Agent Distribution Analysis
```python
# Analyze agent type distribution
distribution = analyzer.analyze_agent_distribution()

print(f"System Agents: {distribution.system_agents}")
print(f"Independent Agents: {distribution.independent_agents}")
print(f"Control Agents: {distribution.control_agents}")
print(f"Dominant Type: {distribution.dominant_type}")
```

### Population Momentum Analysis
```python
# Analyze population growth momentum
momentum = analyzer.analyze_population_momentum()

if momentum > 0:
    print(f"Population growing (momentum: {momentum:.2f})")
elif momentum < 0:
    print(f"Population declining (momentum: {momentum:.2f})")
else:
    print("Population stable")
```

## Analysis Parameters

Most analysis methods accept common parameters:

```python
def analyze_basic_statistics(
    self,
    scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
    agent_id: Optional[int] = None,
    step: Optional[int] = None,
    step_range: Optional[Tuple[int, int]] = None
) -> BasicPopulationStatistics
```

- `scope`: Analysis scope (SIMULATION, EPISODE, etc.)
- `agent_id`: Filter for specific agent (optional)
- `step`: Analyze specific timestep (optional)
- `step_range`: Analyze range of timesteps (optional)

## Return Data Structures

### Basic Population Statistics
```python
BasicPopulationStatistics(
    avg_population=45.2,           # Average population
    death_step=1000,               # Final step with agents
    peak_population=75,            # Maximum population
    lowest_population=10,          # Minimum population
    resources_consumed=2500.5,     # Total resources used
    resources_available=3000.0,    # Total resources available
    sum_squared=102500,            # Sum of squared counts
    initial_population=20          # Starting population
)
```

### Agent Distribution
```python
AgentDistribution(
    system_agents=25,              # Number of system agents
    independent_agents=30,         # Number of independent agents
    control_agents=20,             # Number of control agents
    type_ratios={                  # Proportions of each type
        'system': 0.33,
        'independent': 0.40,
        'control': 0.27
    },
    dominant_type='independent',   # Most common type
    diversity_index=0.85          # Type diversity measure
)
```

## Statistical Notes

1. **Population Calculations**
   - Based on step-by-step population data
   - Handles population changes over time
   - Accounts for births and deaths

2. **Resource Metrics**
   - Calculates total resource consumption
   - Compares against available resources
   - Provides efficiency ratios

3. **Distribution Analysis**
   - Analyzes agent type proportions
   - Calculates diversity metrics
   - Identifies dominant types

## Integration Points

The PopulationAnalyzer integrates with:
- Action analysis for behavioral context
- Resource analysis for consumption patterns
- Learning analysis for population adaptation
- Temporal analysis for growth trends

## Example Workflows

### Population Growth Analysis
```python
# Get population statistics
stats = analyzer.analyze_comprehensive_statistics()

# Analyze growth patterns
if stats.basic_stats.peak_population > stats.basic_stats.initial_population * 2:
    print("Strong population growth")
elif stats.basic_stats.peak_population < stats.basic_stats.initial_population:
    print("Population decline")
else:
    print("Stable population")

# Check resource efficiency
if stats.basic_stats.resource_efficiency > 0.8:
    print("High resource efficiency")
elif stats.basic_stats.resource_efficiency < 0.5:
    print("Low resource efficiency")
else:
    print("Moderate resource efficiency")
```

### Agent Type Analysis
```python
# Analyze agent distribution
distribution = analyzer.analyze_agent_distribution()

# Find dominant agent type
print(f"Dominant type: {distribution.dominant_type}")

# Check type diversity
if distribution.diversity_index > 0.7:
    print("High type diversity")
else:
    print("Low type diversity")

# Compare type populations
type_counts = {
    'system': distribution.system_agents,
    'independent': distribution.independent_agents,
    'control': distribution.control_agents
}

for agent_type, count in type_counts.items():
    print(f"{agent_type}: {count} agents")
```

### Population Stability Analysis
```python
# Analyze population variance
variance = analyzer.analyze_population_variance()

# Check population stability
if variance.population_variance < 10:
    print("Stable population")
elif variance.population_variance > 50:
    print("Volatile population")
else:
    print("Moderate population stability")

# Analyze momentum
momentum = analyzer.analyze_population_momentum()
if momentum > 0.1:
    print("Strong growth momentum")
elif momentum < -0.1:
    print("Declining momentum")
else:
    print("Stable momentum")
```

## Performance Considerations

- Processes all population data in specified scope
- Memory usage scales with simulation length
- Consider using step_range for large datasets
- Caches population calculations internally

## Best Practices

1. **Analysis Scope**
   - Use appropriate scopes for your use case
   - Consider data volume when selecting ranges
   - Use comprehensive analysis for detailed insights

2. **Performance Optimization**
   - Use step ranges for large datasets
   - Cache frequent analyses
   - Filter unnecessary metrics

3. **Data Interpretation**
   - Consider simulation parameters
   - Account for environmental factors
   - Compare against expected patterns

## Error Handling

The analyzer handles common issues:
- Missing population data
- Incomplete simulation histories
- Invalid time ranges
- Null values in population metrics

## Future Considerations

The analyzer is designed for extension:
- Additional population metrics
- Custom stability measures
- Enhanced growth analysis
- Real-time population monitoring 