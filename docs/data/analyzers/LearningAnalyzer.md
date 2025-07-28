# Learning Analyzer

The `LearningAnalyzer` analyzes learning-related data and patterns in agent behavior. This analyzer provides comprehensive analysis of learning experiences, module performance, and adaptation patterns throughout the simulation.

## Overview

The analyzer examines:
- Learning progress over time
- Module performance metrics
- Agent learning statistics
- Learning efficiency and adaptation
- Skill development patterns

## Key Features

### 1. Learning Progress Analysis
```python
progress = analyzer.analyze_learning_progress()
```
Returns `List[LearningProgress]` containing:
- Step-by-step learning metrics
- Reward progression
- Action diversity patterns

### 2. Module Performance Analysis
```python
module_performance = analyzer.analyze_module_performance()
```
Returns `Dict[str, ModulePerformance]` containing:
- Performance metrics per module
- Module-specific statistics
- Learning efficiency by module

### 3. Agent Learning Statistics
```python
agent_stats = analyzer.analyze_agent_learning_stats()
```
Returns `Dict[str, AgentLearningStats]` containing:
- Individual agent learning metrics
- Agent-specific performance data
- Learning progression patterns

### 4. Learning Efficiency Analysis
```python
efficiency = analyzer.analyze_learning_efficiency()
```
Returns `LearningEfficiencyMetrics` containing:
- Overall learning efficiency
- Reward efficiency metrics
- Adaptation patterns

## Detailed Metrics

### Learning Progress Structure

Each `LearningProgress` object contains:

- **Basic Information**
  - `step`: Step number in the simulation
  - `reward`: Average reward achieved in this step
  - `action_count`: Total number of actions taken
  - `unique_actions`: Number of distinct actions used

### Module Performance Structure

Each `ModulePerformance` object contains:

- **Module Information**
  - `module_type`: Type of learning module
  - `module_id`: Unique identifier for the module
  - `avg_reward`: Average reward achieved by the module
  - `action_count`: Total actions taken by the module
  - `unique_actions`: Number of distinct actions used

### Agent Learning Statistics

Each `AgentLearningStats` object contains:

- **Agent Metrics**
  - `agent_id`: Unique agent identifier
  - `total_reward`: Cumulative reward earned
  - `action_diversity`: Measure of action variety
  - `learning_rate`: Rate of improvement over time

### Learning Efficiency Metrics

Each `LearningEfficiencyMetrics` object contains:

- **Efficiency Metrics**
  - `reward_efficiency`: Overall reward efficiency
  - `action_efficiency`: Action utilization efficiency
  - `adaptation_rate`: Rate of behavioral adaptation
  - `learning_progression`: Learning curve metrics

## Usage Examples

### Basic Learning Analysis
```python
from farm.database.analyzers.learning_analyzer import LearningAnalyzer
from farm.database.repositories.learning_repository import LearningRepository

repository = LearningRepository(session)
analyzer = LearningAnalyzer(repository)

# Get comprehensive learning statistics
stats = analyzer.analyze_comprehensive_statistics()

# Print key metrics
print(f"Average Reward: {stats.learning_efficiency.reward_efficiency:.2f}")
print(f"Action Diversity: {stats.learning_efficiency.action_efficiency:.2f}")
print(f"Adaptation Rate: {stats.learning_efficiency.adaptation_rate:.2f}")
```

### Learning Progress Analysis
```python
# Analyze learning progress over time
progress = analyzer.analyze_learning_progress()

# Find learning trends
for p in progress[-10:]:  # Last 10 steps
    print(f"Step {p.step}: {p.reward:.2f} reward, {p.unique_actions} actions")

# Calculate overall improvement
if len(progress) > 1:
    early_avg = sum(p.reward for p in progress[:10]) / 10
    late_avg = sum(p.reward for p in progress[-10:]) / 10
    improvement = late_avg - early_avg
    print(f"Learning improvement: {improvement:+.2f}")
```

### Module Performance Analysis
```python
# Analyze performance by module
module_performance = analyzer.analyze_module_performance()

# Find best performing modules
for module_id, performance in module_performance.items():
    print(f"Module {module_id}:")
    print(f"  Type: {performance.module_type}")
    print(f"  Avg Reward: {performance.avg_reward:.2f}")
    print(f"  Actions: {performance.action_count}")
    print(f"  Unique Actions: {performance.unique_actions}")
```

## Analysis Parameters

Most analysis methods accept common parameters:

```python
def analyze_learning_progress(
    self,
    scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
    agent_id: Optional[int] = None,
    step: Optional[int] = None,
    step_range: Optional[tuple[int, int]] = None
) -> List[LearningProgress]
```

- `scope`: Analysis scope (SIMULATION, EPISODE, etc.)
- `agent_id`: Filter for specific agent (optional)
- `step`: Analyze specific timestep (optional)
- `step_range`: Analyze range of timesteps (optional)

## Return Data Structures

### Learning Progress
```python
LearningProgress(
    step=150,                      # Simulation step
    reward=2.5,                    # Average reward
    action_count=25,               # Total actions
    unique_actions=8               # Distinct actions used
)
```

### Module Performance
```python
ModulePerformance(
    module_type="gather",          # Module type
    module_id="gather_001",        # Module identifier
    avg_reward=3.2,                # Average reward
    action_count=150,              # Total actions
    unique_actions=5               # Distinct actions
)
```

### Learning Efficiency Metrics
```python
LearningEfficiencyMetrics(
    reward_efficiency=0.85,        # Reward efficiency
    action_efficiency=0.72,        # Action efficiency
    adaptation_rate=0.15,          # Adaptation rate
    learning_progression=0.68      # Learning progression
)
```

## Statistical Notes

1. **Learning Progress Calculations**
   - Based on step-by-step learning data
   - Tracks reward progression over time
   - Measures action diversity patterns

2. **Module Performance**
   - Analyzes performance per learning module
   - Compares efficiency across modules
   - Identifies best-performing modules

3. **Efficiency Metrics**
   - Normalized for comparison
   - Considers both rewards and actions
   - Accounts for learning progression

## Integration Points

The LearningAnalyzer integrates with:
- Action analysis for behavioral context
- Population analysis for learning trends
- Temporal analysis for progression patterns
- Resource analysis for learning efficiency

## Example Workflows

### Learning Trend Analysis
```python
# Get learning progress
progress = analyzer.analyze_learning_progress()

# Analyze learning trends
if len(progress) > 10:
    early_rewards = [p.reward for p in progress[:10]]
    late_rewards = [p.reward for p in progress[-10:]]
    
    early_avg = sum(early_rewards) / len(early_rewards)
    late_avg = sum(late_rewards) / len(late_rewards)
    
    if late_avg > early_avg * 1.2:
        print("Strong learning improvement")
    elif late_avg < early_avg * 0.8:
        print("Learning decline")
    else:
        print("Stable learning")
```

### Module Comparison
```python
# Compare module performance
module_performance = analyzer.analyze_module_performance()

# Find best and worst modules
modules_by_reward = sorted(
    module_performance.items(),
    key=lambda x: x[1].avg_reward,
    reverse=True
)

print("Best performing modules:")
for module_id, performance in modules_by_reward[:3]:
    print(f"  {module_id}: {performance.avg_reward:.2f} avg reward")

print("Worst performing modules:")
for module_id, performance in modules_by_reward[-3:]:
    print(f"  {module_id}: {performance.avg_reward:.2f} avg reward")
```

### Agent Learning Comparison
```python
# Compare learning across agents
agent_stats = analyzer.analyze_agent_learning_stats()

# Find best learners
best_learners = sorted(
    agent_stats.items(),
    key=lambda x: x[1].total_reward,
    reverse=True
)

print("Top learners:")
for agent_id, stats in best_learners[:5]:
    print(f"  {agent_id}: {stats.total_reward:.2f} total reward")
    print(f"    Action diversity: {stats.action_diversity:.2f}")
    print(f"    Learning rate: {stats.learning_rate:.2f}")
```

## Performance Considerations

- Processes all learning data in specified scope
- Memory usage scales with learning history
- Consider using step_range for large datasets
- Caches learning calculations internally

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
   - Consider learning context and environment
   - Account for module-specific factors
   - Compare against expected learning patterns

## Error Handling

The analyzer handles common issues:
- Missing learning data
- Incomplete learning histories
- Invalid time ranges
- Null values in learning metrics

## Future Considerations

The analyzer is designed for extension:
- Additional learning metrics
- Custom efficiency measures
- Enhanced progression analysis
- Real-time learning monitoring 