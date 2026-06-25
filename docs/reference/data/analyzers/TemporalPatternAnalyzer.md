# Temporal Pattern Analyzer

The `TemporalPatternAnalyzer` analyzes temporal patterns in agent actions over time, including rolling averages and event segmentation. This analyzer processes agent actions to identify patterns in their occurrence and associated rewards across different time periods.

## Overview

The analyzer examines:
- Action frequencies over time
- Reward progression patterns
- Rolling averages and trends
- Temporal segmentation of behaviors
- Time-based pattern analysis

## Key Features

### 1. Temporal Pattern Analysis
```python
patterns = analyzer.analyze(scope="SIMULATION")
```
Returns `List[TimePattern]` containing:
- Action type and temporal metrics
- Time distribution of actions
- Reward progression over time
- Rolling averages

### 2. Event Segmentation
```python
segments = analyzer.segment_events(event_steps=[100, 200, 300])
```
Returns `List[EventSegment]` containing:
- Event-based time segments
- Segment-specific metrics
- Temporal clustering

## Detailed Metrics

### Time Pattern Structure

Each `TimePattern` object contains:

- **Basic Information**
  - `action_type`: Type of action analyzed
  - `time_distribution`: Action frequencies per time period
  - `reward_progression`: Average rewards per time period
  - `rolling_average_rewards`: Smoothed reward progression
  - `rolling_average_counts`: Smoothed action frequencies

### Event Segment Structure

Each `EventSegment` object contains:

- **Segment Information**
  - `start_step`: Beginning of the segment
  - `end_step`: End of the segment
  - `duration`: Length of the segment
  - `action_counts`: Action frequencies in the segment
  - `average_rewards`: Average rewards in the segment

### Metric Calculations

- **Time Distribution**: Action frequencies divided into time periods
- **Reward Progression**: Average rewards calculated per time period
- **Rolling Averages**: Smoothed trends using specified window sizes
- **Event Segmentation**: Time periods based on significant events

## Usage Examples

### Basic Temporal Analysis
```python
from farm.database.analyzers.temporal_pattern_analyzer import TemporalPatternAnalyzer
from farm.database.repositories.action_repository import ActionRepository

repository = ActionRepository(session)
analyzer = TemporalPatternAnalyzer(repository)

# Get comprehensive temporal pattern statistics
patterns = analyzer.analyze(
    scope="SIMULATION",
    time_period_size=100,
    rolling_window_size=10
)

# Print results
for pattern in patterns:
    print(f"Action: {pattern.action_type}")
    print(f"  Time Distribution: {len(pattern.time_distribution)} periods")
    print(f"  Avg Reward: {sum(pattern.reward_progression) / len(pattern.reward_progression):.2f}")
```

### Agent-Specific Temporal Analysis
```python
# Analyze specific agent's temporal patterns
agent_patterns = analyzer.analyze(
    scope="SIMULATION",
    agent_id="agent_001",
    time_period_size=50
)

# Find patterns with increasing rewards
for pattern in agent_patterns:
    if len(pattern.reward_progression) > 1:
        trend = pattern.reward_progression[-1] - pattern.reward_progression[0]
        if trend > 0:
            print(f"Agent 1 {pattern.action_type}: +{trend:.2f} reward trend")
```

### Event-Based Segmentation
```python
# Segment analysis based on significant events
event_steps = [100, 200, 300, 400]  # Significant simulation steps
segments = analyzer.segment_events(
    event_steps=event_steps,
    scope="SIMULATION"
)

# Analyze each segment
for segment in segments:
    print(f"Segment {segment.start_step}-{segment.end_step}:")
    print(f"  Duration: {segment.duration} steps")
    print(f"  Avg Reward: {segment.average_rewards:.2f}")
```

## Analysis Parameters

The `analyze()` method accepts several parameters for filtering and scoping the analysis:

```python
def analyze(
    self,
    scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
    agent_id: Optional[int] = None,
    step: Optional[int] = None,
    step_range: Optional[Tuple[int, int]] = None,
    time_period_size: int = 100,
    rolling_window_size: int = 10
) -> List[TimePattern]
```

- `scope`: Analysis scope (SIMULATION, EPISODE, etc.)
- `agent_id`: Filter for specific agent
- `step`: Analyze specific timestep
- `step_range`: Analyze range of timesteps
- `time_period_size`: Size of each time period in steps
- `rolling_window_size`: Size of window for rolling averages

## Return Data Structure

The analyzer returns a list of `TimePattern` objects containing:

```python
TimePattern(
    action_type="gather",
    time_distribution=[10, 15, 12, 8],           # Actions per time period
    reward_progression=[2.1, 2.3, 2.0, 2.5],     # Average rewards per period
    rolling_average_rewards=[2.1, 2.2, 2.1, 2.2], # Smoothed rewards
    rolling_average_counts=[10, 12.5, 12.3, 11.3] # Smoothed counts
)
```

## Statistical Notes

1. **Time Period Calculations**
   - Divides simulation timeline into equal periods
   - Calculates metrics for each period independently
   - Handles incomplete periods appropriately

2. **Rolling Averages**
   - Uses specified window size for smoothing
   - Handles edge cases at beginning and end
   - Provides trend analysis capabilities

3. **Event Segmentation**
   - Based on significant simulation events
   - Calculates segment-specific metrics
   - Supports temporal clustering analysis

## Integration Points

The TemporalPatternAnalyzer integrates with:
- Action analysis for context
- Sequence analysis for temporal patterns
- Learning analysis for progression trends
- Population analysis for temporal dynamics

## Example Workflows

### Trend Analysis
```python
# Get temporal patterns
patterns = analyzer.analyze(time_period_size=50)

# Find actions with increasing trends
for pattern in patterns:
    if len(pattern.reward_progression) > 1:
        start_reward = pattern.reward_progression[0]
        end_reward = pattern.reward_progression[-1]
        trend = end_reward - start_reward
        
        if trend > 0.5:
            print(f"Increasing trend: {pattern.action_type} (+{trend:.2f})")
        elif trend < -0.5:
            print(f"Decreasing trend: {pattern.action_type} ({trend:.2f})")
```

### Comparative Temporal Analysis
```python
# Compare temporal patterns across agents
agent1_patterns = analyzer.analyze(agent_id="agent_001")
agent2_patterns = analyzer.analyze(agent_id="agent_002")

# Compare reward progression
for a1, a2 in zip(agent1_patterns, agent2_patterns):
    if a1.action_type == a2.action_type:
        a1_avg = sum(a1.reward_progression) / len(a1.reward_progression)
        a2_avg = sum(a2.reward_progression) / len(a2.reward_progression)
        diff = a1_avg - a2_avg
        print(f"{a1.action_type}: {diff:+.2f} average difference")
```

### Event-Based Analysis
```python
# Define significant events
event_steps = [100, 200, 300, 400, 500]
segments = analyzer.segment_events(event_steps)

# Analyze segment characteristics
for i, segment in enumerate(segments):
    print(f"Segment {i+1} ({segment.start_step}-{segment.end_step}):")
    print(f"  Duration: {segment.duration} steps")
    print(f"  Total Actions: {sum(segment.action_counts.values())}")
    print(f"  Average Reward: {segment.average_rewards:.2f}")
```

## Performance Considerations

- Processes all actions in specified scope
- Memory usage scales with action count and time periods
- Consider using step_range for large datasets
- Rolling averages require additional computation

## Best Practices

1. **Time Period Selection**
   - Choose appropriate time_period_size for your analysis
   - Consider simulation length and data granularity
   - Balance detail vs. computational cost

2. **Rolling Window Size**
   - Larger windows provide smoother trends
   - Smaller windows preserve short-term variations
   - Consider the nature of your temporal patterns

3. **Event Segmentation**
   - Choose meaningful event steps
   - Consider simulation context and significance
   - Validate segment boundaries

## Error Handling

The analyzer handles common issues:
- Missing temporal data
- Incomplete time periods
- Invalid time ranges
- Null values in temporal metrics

## Future Considerations

The analyzer is designed for extension:
- Additional temporal metrics
- Custom segmentation algorithms
- Enhanced trend analysis
- Real-time temporal monitoring 