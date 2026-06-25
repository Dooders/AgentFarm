# Action Statistics Analyzer

The `ActionStatsAnalyzer` provides comprehensive analysis of agent actions and their outcomes in the Farm simulation. It processes action data to generate detailed metrics about frequency, rewards, interaction patterns, and various behavioral statistics.

## Overview

The analyzer examines:
- Action frequencies and distributions
- Reward statistics and patterns
- Interaction rates between agents
- Performance metrics for different action types
- Temporal and decision-making patterns

## Key Features

### 1. Basic Action Metrics
- Action counts and frequencies
- Average rewards per action type
- Success rates and failure patterns
- Resource impact analysis

### 2. Detailed Reward Statistics
- Mean, median, and mode rewards
- Variance and standard deviation
- Quartile analysis
- Confidence intervals
- Min/max reward values

### 3. Interaction Analysis
- Interaction rates between agents
- Solo vs collaborative performance
- Target selection patterns
- Success rates for interactive actions

### 4. Performance Metrics
- Solo performance metrics
- Interaction performance metrics
- Resource efficiency
- Action effectiveness scores

## Usage

```python
from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer
from farm.database.repositories.action_repository import ActionRepository

# Initialize analyzer
repository = ActionRepository(session)
analyzer = ActionStatsAnalyzer(repository)

# Get comprehensive action statistics
action_metrics = analyzer.analyze(scope="SIMULATION")

# Example: Print average rewards for each action type
for metric in action_metrics:
    print(f"Action: {metric.action_type}")
    print(f"  Average Reward: {metric.avg_reward:.2f}")
    print(f"  Frequency: {metric.frequency:.2%}")
    print(f"  Interaction Rate: {metric.interaction_rate:.2%}")
```

## Return Data Structure

The analyzer returns a list of `ActionMetrics` objects containing:

```python
ActionMetrics(
    action_type="gather",
    count=100,                    # Total occurrences
    frequency=0.4,                # 40% of all actions
    avg_reward=2.5,               # Average reward
    min_reward=0.0,               # Minimum reward
    max_reward=5.0,               # Maximum reward
    variance_reward=0.2,          # Reward variance
    std_dev_reward=0.447,         # Standard deviation
    median_reward=2.5,            # Median reward
    quartiles_reward=[2.0, 3.0],  # First and third quartiles
    confidence_interval=0.087,     # 95% confidence interval
    interaction_rate=0.1,         # 10% involved other agents
    solo_performance=2.7,         # Average solo reward
    interaction_performance=1.2,   # Average interactive reward
    temporal_patterns=[...],      # Time-based patterns
    resource_impacts=[...],       # Resource effects
    decision_patterns=[...]       # Decision-making patterns
)
```

## Analysis Parameters

The `analyze()` method accepts several parameters for filtering and scoping the analysis:

```python
def analyze(
    self,
    scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
    agent_id: Optional[str] = None,
    step: Optional[int] = None,
    step_range: Optional[Tuple[int, int]] = None
) -> List[ActionMetrics]
```

- `scope`: Analysis scope (SIMULATION, EPISODE, etc.)
- `agent_id`: Filter for specific agent
- `step`: Analyze specific timestep
- `step_range`: Analyze range of timesteps

## Statistical Notes

1. **Frequency Calculation**
   - Frequencies are normalized between 0 and 1
   - Represents proportion of total actions

2. **Reward Statistics**
   - Requires at least 2 data points
   - Confidence intervals at 95% level
   - Handles null rewards appropriately

3. **Interaction Metrics**
   - Based on presence of target_id
   - Separate performance tracking for solo/interactive

4. **Performance Metrics**
   - Normalized for comparison
   - Accounts for action frequency
   - Considers resource efficiency

## Integration Points

The analyzer integrates with several other analyzers for comprehensive analysis:

- `DecisionPatternAnalyzer`: For decision-making patterns
- `ResourceImpactAnalyzer`: For resource utilization effects
- `TemporalPatternAnalyzer`: For timing and sequence patterns

## Example Workflows

### Basic Action Analysis
```python
# Get basic action statistics
metrics = analyzer.analyze()
for m in metrics:
    print(f"{m.action_type}: {m.count} occurrences, {m.avg_reward:.2f} avg reward")
```

### Focused Agent Analysis
```python
# Analyze specific agent's actions
agent_metrics = analyzer.analyze(agent_id="agent_1")
for m in agent_metrics:
    print(f"Agent 1 {m.action_type}: {m.frequency:.2%} frequency")
```

### Time Period Analysis
```python
# Analyze specific time period
period_metrics = analyzer.analyze(step_range=(100, 200))
for m in period_metrics:
    print(f"Steps 100-200 {m.action_type}: {m.interaction_rate:.2%} interaction rate")
```

## Performance Considerations

- Processes all actions in specified scope
- Memory usage scales with action count
- Consider using step_range for large datasets
- Caches common calculations internally