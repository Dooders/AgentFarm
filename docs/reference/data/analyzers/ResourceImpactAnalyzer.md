# Resource Impact Analyzer

The `ResourceImpactAnalyzer` analyzes the resource impact of agent actions in a simulation. This analyzer processes agent actions to calculate various resource-related metrics such as average resource changes and efficiency for different action types.

## Overview

The analyzer examines:
- Resource changes before and after actions
- Average resource efficiency per action type
- Resource utilization patterns
- Action-specific resource impacts

## Key Features

### 1. Resource Change Analysis
```python
impacts = analyzer.analyze(scope="SIMULATION")
```
Returns `List[ResourceImpact]` containing:
- Action type and resource metrics
- Average resources before actions
- Average resource changes
- Resource efficiency calculations

### 2. Action-Specific Analysis
```python
# Analyze specific action types
gather_impacts = analyzer.analyze(
    scope="SIMULATION",
    step_range=(100, 200)
)
```

## Detailed Metrics

### Resource Impact Structure

Each `ResourceImpact` object contains:

- **Basic Information**
  - `action_type`: Type of action (e.g., "gather", "move", "attack")
  - `avg_resources_before`: Average resource level before the action
  - `avg_resource_change`: Average change in resources due to the action
  - `resource_efficiency`: Efficiency metric for resource utilization

### Metric Calculations

- **Average Resources Before**: Mean resource level across all instances of the action
- **Average Resource Change**: Mean difference between resources after and before
- **Resource Efficiency**: Normalized measure of resource utilization effectiveness

## Usage Examples

### Basic Resource Analysis
```python
from farm.database.analyzers.resource_impact_analyzer import ResourceImpactAnalyzer
from farm.database.repositories.action_repository import ActionRepository

repository = ActionRepository(session)
analyzer = ResourceImpactAnalyzer(repository)

# Get comprehensive resource impact statistics
impacts = analyzer.analyze(scope="SIMULATION")

# Print results
for impact in impacts:
    print(f"Action: {impact.action_type}")
    print(f"  Avg Resources Before: {impact.avg_resources_before:.2f}")
    print(f"  Avg Resource Change: {impact.avg_resource_change:.2f}")
    print(f"  Resource Efficiency: {impact.resource_efficiency:.2f}")
```

### Agent-Specific Resource Analysis
```python
# Analyze specific agent's resource usage
agent_impacts = analyzer.analyze(
    scope="SIMULATION",
    agent_id="agent_001"
)

# Compare resource efficiency
for impact in agent_impacts:
    print(f"Agent 1 {impact.action_type}: {impact.resource_efficiency:.2f} efficiency")
```

### Time-Based Resource Analysis
```python
# Analyze resource patterns over time
period_impacts = analyzer.analyze(
    scope="EPISODE",
    step_range=(100, 200)
)

# Find most efficient actions
efficient_actions = sorted(period_impacts, key=lambda x: x.resource_efficiency, reverse=True)
for impact in efficient_actions[:3]:
    print(f"Most efficient: {impact.action_type} ({impact.resource_efficiency:.2f})")
```

## Analysis Parameters

The `analyze()` method accepts several parameters for filtering and scoping the analysis:

```python
def analyze(
    self,
    scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
    agent_id: Optional[int] = None,
    step: Optional[int] = None,
    step_range: Optional[Tuple[int, int]] = None
) -> List[ResourceImpact]
```

- `scope`: Analysis scope (SIMULATION, EPISODE, etc.)
- `agent_id`: Filter for specific agent
- `step`: Analyze specific timestep
- `step_range`: Analyze range of timesteps

## Return Data Structure

The analyzer returns a list of `ResourceImpact` objects containing:

```python
ResourceImpact(
    action_type="gather",
    avg_resources_before=25.5,      # Average resources before action
    avg_resource_change=+5.2,        # Average change in resources
    resource_efficiency=0.8          # Resource utilization efficiency
)
```

## Statistical Notes

1. **Resource Calculations**
   - Based on resources_before and resources_after from action data
   - Handles null resource values appropriately
   - Calculates averages across all instances of each action type

2. **Efficiency Metrics**
   - Normalized for comparison across action types
   - Considers both positive and negative resource changes
   - Accounts for action frequency and resource availability

3. **Data Quality**
   - Requires valid resource data for accurate analysis
   - Handles missing resource information gracefully
   - Provides meaningful metrics even with limited data

## Integration Points

The ResourceImpactAnalyzer integrates with:
- Action analysis for context
- Population analysis for resource availability
- Learning analysis for efficiency trends
- Temporal analysis for resource patterns over time

## Example Workflows

### Resource Efficiency Analysis
```python
# Get resource impact data
impacts = analyzer.analyze()

# Find most efficient actions
efficient_actions = [imp for imp in impacts if imp.resource_efficiency > 0.7]
for action in efficient_actions:
    print(f"Efficient: {action.action_type} ({action.resource_efficiency:.2f})")

# Find resource-intensive actions
intensive_actions = [imp for imp in impacts if imp.avg_resource_change < -2.0]
for action in intensive_actions:
    print(f"Costly: {action.action_type} ({action.avg_resource_change:.2f})")
```

### Comparative Analysis
```python
# Compare resource usage across agents
agent1_impacts = analyzer.analyze(agent_id="agent_001")
agent2_impacts = analyzer.analyze(agent_id="agent_002")

# Compare efficiency
for a1, a2 in zip(agent1_impacts, agent2_impacts):
    if a1.action_type == a2.action_type:
        efficiency_diff = a1.resource_efficiency - a2.resource_efficiency
        print(f"{a1.action_type}: {efficiency_diff:+.2f} efficiency difference")
```

### Time Period Analysis
```python
# Analyze resource patterns over different time periods
early_impacts = analyzer.analyze(step_range=(0, 100))
late_impacts = analyzer.analyze(step_range=(900, 1000))

# Compare efficiency changes
for early, late in zip(early_impacts, late_impacts):
    if early.action_type == late.action_type:
        efficiency_change = late.resource_efficiency - early.resource_efficiency
        print(f"{early.action_type}: {efficiency_change:+.2f} efficiency change")
```

## Performance Considerations

- Processes all actions in specified scope
- Memory usage scales with action count
- Consider using step_range for large datasets
- Caches resource calculations internally

## Best Practices

1. **Analysis Scope**
   - Use appropriate scopes for your use case
   - Consider data volume when selecting ranges
   - Use agent-specific analysis for detailed insights

2. **Performance Optimization**
   - Use step ranges for large datasets
   - Cache frequent analyses
   - Filter unnecessary metrics

3. **Data Interpretation**
   - Consider resource availability context
   - Account for environmental factors
   - Compare against population averages

## Error Handling

The analyzer handles common issues:
- Missing resource data
- Incomplete action histories
- Invalid time ranges
- Null values in resource metrics

## Future Considerations

The analyzer is designed for extension:
- New resource efficiency metrics
- Custom scoring systems
- Additional resource pattern recognition
- Enhanced visualization support 