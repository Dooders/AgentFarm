# Reproduction Module

## Overview

The reproduction module has been simplified from a DQN-based learning system to a rule-based implementation. This change reduces complexity while maintaining effective reproduction behavior.

## Rule-Based Implementation

### Simple Reproduction Logic

The reproduction action now uses straightforward conditional logic:

```python
def reproduce_action(agent: "BaseAgent") -> None:
    # Simple rule-based reproduction in farm/core/action.py
    if random.random() < 0.5 and agent.resource_level >= agent.config.min_reproduction_resources:
        agent.reproduce()
```

### Reproduction Conditions

The rule-based system checks two main conditions:

1. **Random Chance**: 50% probability of attempting reproduction
2. **Resource Threshold**: Agent must have sufficient resources (≥ `min_reproduction_resources`)

### Configuration

Reproduction behavior is controlled through `SimulationConfig`:

```python
class SimulationConfig:
    min_reproduction_resources: int = 8
    offspring_cost: int = 3
    offspring_initial_resources: int = 5
    max_population: int = 3000
```

## Benefits of Rule-Based Approach

### Reduced Complexity
- No neural network training required
- Simpler debugging and analysis
- Faster execution

### Predictable Behavior
- Clear reproduction conditions
- Deterministic resource requirements
- Easy to tune parameters

### Computational Efficiency
- No DQN overhead
- Minimal memory usage
- Faster agent updates

## Integration with Curriculum Learning

The reproduction module is enabled in the final curriculum phase:

```python
curriculum_phases = [
    {"steps": 100, "enabled_actions": ["move", "gather"]},
    {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
    {"steps": -1, "enabled_actions": ["move", "gather", "share", "attack", "reproduce"]}
]
```

This allows agents to learn basic survival skills before attempting reproduction.

## Usage

The reproduction action is called through the standard action system:

```python
# In agent decision loop
if "reproduce" in enabled_actions:
    reproduce_action(agent)
```

## Migration from DQN

### Previous DQN Implementation

The previous version used a full DQN system with:
- `ReproduceQNetwork` for Q-value approximation
- `ReproduceModule` for learning and training
- Complex state representation and reward calculation

### Current Rule-Based System

The current system uses:
- Simple conditional logic
- Direct resource threshold checking
- Random chance for exploration

## Configuration Parameters

| Parameter                     | Description                      | Default Value |
| ----------------------------- | -------------------------------- | ------------- |
| `min_reproduction_resources`  | Minimum resources required       | 8             |
| `offspring_cost`              | Resources consumed per offspring | 3             |
| `offspring_initial_resources` | Starting resources for offspring | 5             |
| `max_population`              | Maximum population limit         | 3000          |

## Future Enhancements

While the current implementation is rule-based, future versions could include:

1. **Adaptive Probabilities**: Adjust reproduction chance based on population density
2. **Quality-Based Selection**: Consider agent fitness for reproduction
3. **Environmental Factors**: Include resource availability in reproduction decisions
4. **Hybrid Approach**: Combine rule-based logic with simple learning components

## Best Practices

1. **Population Control**: Monitor population growth and adjust thresholds
2. **Resource Balance**: Ensure reproduction costs don't deplete resources too quickly
3. **Curriculum Integration**: Use reproduction only in later training phases
4. **Performance Monitoring**: Track reproduction rates and adjust parameters as needed 