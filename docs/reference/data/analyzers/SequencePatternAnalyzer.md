# Sequence Pattern Analyzer

The `SequencePatternAnalyzer` analyzes sequences of agent actions to identify patterns and their probabilities. This analyzer looks for consecutive pairs of actions performed by the same agent and calculates the frequency and probability of these action sequences occurring.

## Overview

The analyzer examines:
- Consecutive action pairs by the same agent
- Sequence frequencies and probabilities
- Action transition patterns
- Behavioral sequence analysis

## Key Features

### 1. Sequence Pattern Detection
```python
patterns = analyzer.analyze(scope="SIMULATION")
```
Returns `List[SequencePattern]` containing:
- Action sequences (e.g., "MOVE->ATTACK")
- Frequency counts
- Transition probabilities

### 2. Agent-Specific Sequences
```python
# Analyze sequences for specific agent
agent_patterns = analyzer.analyze(
    scope="SIMULATION",
    agent_id="agent_001"
)
```

## Detailed Metrics

### Sequence Pattern Structure

Each `SequencePattern` object contains:

- **Basic Information**
  - `sequence`: The action sequence (e.g., "MOVE->ATTACK")
  - `count`: Number of times this sequence occurred
  - `probability`: Probability of the second action following the first action

### Metric Calculations

- **Sequence Detection**: Identifies consecutive action pairs by the same agent
- **Frequency Count**: Counts occurrences of each sequence
- **Transition Probability**: Calculates P(second_action | first_action)

## Usage Examples

### Basic Sequence Analysis
```python
from farm.database.analyzers.sequence_pattern_analyzer import SequencePatternAnalyzer
from farm.database.repositories.action_repository import ActionRepository

repository = ActionRepository(session)
analyzer = SequencePatternAnalyzer(repository)

# Get comprehensive sequence pattern statistics
patterns = analyzer.analyze(scope="SIMULATION")

# Print results
for pattern in patterns:
    print(f"Sequence: {pattern.sequence}")
    print(f"  Count: {pattern.count}")
    print(f"  Probability: {pattern.probability:.2%}")
```

### Agent-Specific Sequence Analysis
```python
# Analyze specific agent's action sequences
agent_patterns = analyzer.analyze(
    scope="SIMULATION",
    agent_id="agent_001"
)

# Find most common sequences
common_sequences = sorted(agent_patterns, key=lambda x: x.count, reverse=True)
for pattern in common_sequences[:5]:
    print(f"Agent 1 common: {pattern.sequence} ({pattern.count} times)")
```

### Time-Based Sequence Analysis
```python
# Analyze sequences over time period
period_patterns = analyzer.analyze(
    scope="EPISODE",
    step_range=(100, 200)
)

# Find high-probability transitions
high_prob_sequences = [p for p in period_patterns if p.probability > 0.5]
for pattern in high_prob_sequences:
    print(f"High probability: {pattern.sequence} ({pattern.probability:.1%})")
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
) -> List[SequencePattern]
```

- `scope`: Analysis scope (SIMULATION, EPISODE, etc.)
- `agent_id`: Filter for specific agent
- `step`: Analyze specific timestep
- `step_range`: Analyze range of timesteps

## Return Data Structure

The analyzer returns a list of `SequencePattern` objects containing:

```python
SequencePattern(
    sequence="MOVE->ATTACK",     # Action sequence
    count=15,                    # Number of occurrences
    probability=0.25             # 25% probability of ATTACK following MOVE
)
```

## Statistical Notes

1. **Sequence Detection**
   - Only considers consecutive actions by the same agent
   - Requires at least 2 actions for sequence analysis
   - Handles action gaps and interruptions appropriately

2. **Probability Calculations**
   - P(second_action | first_action) = count(sequence) / count(first_action)
   - Based on observed frequencies in the data
   - Requires sufficient data for meaningful probabilities

3. **Data Quality**
   - Requires ordered action sequences
   - Handles missing or incomplete action data
   - Provides meaningful patterns even with limited data

## Integration Points

The SequencePatternAnalyzer integrates with:
- Action analysis for context
- Behavioral clustering for pattern classification
- Temporal analysis for sequence timing
- Learning analysis for sequence evolution

## Example Workflows

### Common Sequence Analysis
```python
# Get sequence patterns
patterns = analyzer.analyze()

# Find most common sequences
common_sequences = sorted(patterns, key=lambda x: x.count, reverse=True)
for pattern in common_sequences[:10]:
    print(f"Common: {pattern.sequence} ({pattern.count} times)")

# Find high-probability sequences
high_prob = [p for p in patterns if p.probability > 0.7]
for pattern in high_prob:
    print(f"High probability: {pattern.sequence} ({pattern.probability:.1%})")
```

### Comparative Analysis
```python
# Compare sequences across agents
agent1_patterns = analyzer.analyze(agent_id="agent_001")
agent2_patterns = analyzer.analyze(agent_id="agent_002")

# Find unique sequences for each agent
agent1_sequences = {p.sequence for p in agent1_patterns}
agent2_sequences = {p.sequence for p in agent2_patterns}

unique_to_agent1 = agent1_sequences - agent2_sequences
unique_to_agent2 = agent2_sequences - agent1_sequences

print(f"Unique to Agent 1: {len(unique_to_agent1)} sequences")
print(f"Unique to Agent 2: {len(unique_to_agent2)} sequences")
```

### Time Period Analysis
```python
# Analyze sequence patterns over different time periods
early_patterns = analyzer.analyze(step_range=(0, 100))
late_patterns = analyzer.analyze(step_range=(900, 1000))

# Compare sequence evolution
early_sequences = {p.sequence for p in early_patterns}
late_sequences = {p.sequence for p in late_patterns}

new_sequences = late_sequences - early_sequences
lost_sequences = early_sequences - late_sequences

print(f"New sequences: {len(new_sequences)}")
print(f"Lost sequences: {len(lost_sequences)}")
```

## Performance Considerations

- Processes all actions in specified scope
- Memory usage scales with action count
- Consider using step_range for large datasets
- Caches sequence calculations internally

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
   - Consider action context and timing
   - Account for environmental factors
   - Compare against population averages

## Error Handling

The analyzer handles common issues:
- Missing action data
- Incomplete action histories
- Invalid time ranges
- Null values in sequence data

## Future Considerations

The analyzer is designed for extension:
- Longer sequence patterns (3+ actions)
- Custom sequence scoring systems
- Additional pattern recognition algorithms
- Enhanced visualization support 