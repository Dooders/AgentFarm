# Gather Action Documentation

The Gather Action is a simple resource gathering system that enables agents to gather resources from the nearest available source. This action uses straightforward rule-based logic for resource collection, making it lightweight and predictable.

---

## Overview

The Gather Action implements a simple rule-based approach:

- **Nearest Resource Selection**: Finds and gathers from the closest available resource
- **Distance-Based Prioritization**: Uses Euclidean distance to identify optimal targets
- **Configurable Gathering**: Respects maximum gathering amounts and range limits
- **Resource Validation**: Only gathers from non-depleted resources
- **Efficient Spatial Queries**: Uses spatial index for fast proximity searches

---

## Simple Implementation

### Core Logic

The gather action uses straightforward Python code:

```python
def gather_action(agent: "BaseAgent") -> None:
    # Find nearby resources using spatial index
    nearby_resources = agent.environment.get_nearby_resources(
        agent.position, gathering_range
    )

    # Filter out depleted resources
    available_resources = [
        r for r in nearby_resources
        if not r.is_depleted() and r.amount > 0
    ]

    if not available_resources:
        return  # No resources available

    # Find closest resource
    closest_resource = min(
        available_resources,
        key=lambda r: math.sqrt(
            (r.position[0] - agent.position[0]) ** 2 +
            (r.position[1] - agent.position[1]) ** 2
        )
    )

    # Gather from the closest resource
    gather_amount = min(max_gather, closest_resource.amount)
    actual_gathered = closest_resource.consume(gather_amount)
    agent.resource_level += actual_gathered

    # Simple reward calculation
    reward = actual_gathered * 0.1
    agent.total_reward += reward
```

### Key Features

- **Spatial Index Integration**: Uses `get_nearby_resources()` for efficient O(log n) queries
- **Distance-Based Selection**: Finds closest resource using Euclidean distance
- **Resource Validation**: Only gathers from non-depleted resources with amount > 0
- **Configurable Limits**: Respects `max_amount` and `gathering_range` parameters
- **Simple Rewards**: Linear reward based on amount gathered (0.1 per unit)

---

## Technical Details

### Resource Selection Algorithm
1. **Spatial Query**: Uses `get_nearby_resources()` within `gathering_range`
2. **Resource Filtering**: Excludes depleted resources and those with amount ≤ 0
3. **Distance Calculation**: Euclidean distance to find closest resource
4. **Amount Limiting**: Respects `max_amount` configuration parameter
5. **Resource Consumption**: Calls `resource.consume()` to gather resources

### Configuration Parameters

The gather action uses the following configuration parameters:

- **`gathering_range`**: Maximum distance to search for resources (default: 30)
- **`max_amount`**: Maximum amount that can be gathered per action (default: 10)

### Reward System

Simple linear reward calculation:
- **Reward per unit**: 0.1 points per resource unit gathered
- **Total reward**: `gathered_amount * 0.1`
- **Added to**: `agent.total_reward`

---

## Usage Example

```python
# Simple gather action usage
from farm.core.action import gather_action

# Agent will automatically find and gather from the nearest resource
gather_action(agent)

# The action will:
# 1. Find nearby resources within gathering_range
# 2. Filter out depleted resources
# 3. Select the closest available resource
# 4. Gather up to max_amount from the resource
# 5. Update agent's resource level and total reward
```

## Integration with Action System

The gather action integrates seamlessly with the action registry:

```python
from farm.core.action import action_registry

# Get the gather action
gather_action_func = action_registry.get("gather")

# Execute gather action
gather_action_func(agent)
```

## Configuration Examples

### Basic Configuration
```python
# Set gathering parameters in agent config
agent.config.gathering_range = 30  # Search radius for resources
agent.config.max_amount = 10       # Maximum amount to gather per action
```

### Environment Integration
```python
# Resources must implement:
class Resource:
    def is_depleted(self) -> bool:
        return self.amount <= 0

    def consume(self, amount: float) -> float:
        # Return actual amount consumed
        actual = min(amount, self.amount)
        self.amount -= actual
        return actual
```

---

## Best Practices

1. **Configuration Tuning**
   - Set appropriate `gathering_range` for your environment size
   - Adjust `max_amount` based on resource regeneration rates
   - Balance range vs. efficiency for optimal gathering

2. **Resource Management**
   - Monitor resource depletion patterns
   - Consider agent density when setting gathering parameters
   - Ensure resources implement proper `is_depleted()` and `consume()` methods

3. **Performance Optimization**
   - Spatial index provides O(log n) query performance
   - Distance calculations are computationally efficient
   - Simple reward system minimizes processing overhead

4. **Environment Integration**
   - Ensure `get_nearby_resources()` returns valid resource objects
   - Validate resource positions and amounts
   - Handle edge cases like no nearby resources gracefully

---

## Migration from DQN

### Previous DQN Implementation

The original implementation used:
- `GatherQNetwork` for Q-value approximation
- `GatherModule` for training and decision making
- Complex state representation (6+ dimensions)
- Experience replay and target networks
- Sophisticated reward calculations

### Current Simple Implementation

The new implementation uses:
- Direct spatial queries for resource finding
- Simple distance-based resource selection
- Linear reward calculation (0.1 per unit)
- No neural networks or training
- Minimal computational overhead

---

## Performance Characteristics

1. **Computational Efficiency**
   - O(log n) spatial index queries for resource finding
   - O(k) distance calculations where k is number of nearby resources
   - Minimal memory usage (no neural networks or replay buffers)
   - No GPU requirements - runs entirely on CPU

2. **Scalability**
   - Performance scales with environment size through spatial indexing
   - Distance calculations are vectorized where possible
   - Simple reward system has constant-time computation
   - No training overhead or iterative learning processes

3. **Reliability**
   - Deterministic behavior based on spatial relationships
   - No neural network training instability
   - Graceful handling of edge cases (no resources, depleted resources)
   - Simple error handling and logging

---

## Future Enhancements

While the current implementation is rule-based, future versions could include:

1. **Resource Prioritization**
   - Quality-based resource selection (regeneration rate, max capacity)
   - Multi-resource gathering strategies
   - Resource competition awareness

2. **Adaptive Behavior**
   - Dynamic range adjustment based on resource density
   - Learning-based reward scaling
   - Agent specialization for different resource types

3. **Environmental Integration**
   - Weather/seasonal effects on gathering efficiency
   - Terrain-based movement cost adjustments
   - Cooperative gathering behaviors

---

## Benefits of Simple Implementation

1. **Reduced Complexity**: No neural networks, training loops, or complex state management
2. **Better Performance**: Instant decisions with minimal computational overhead
3. **Easier Debugging**: Clear, predictable behavior based on simple rules
4. **Lower Resource Usage**: No GPU requirements or large memory allocations
5. **Maintainability**: Simple code that's easy to understand and modify 