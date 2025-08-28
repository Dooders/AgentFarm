# Share Action Documentation

The Share Action is a simple cooperative behavior system that enables agents to share resources with other agents in need. This action uses straightforward rule-based logic for resource distribution, making it lightweight and predictable.

---

## Overview

The Share Action implements a simple rule-based approach:

- **Need-Based Target Selection**: Finds agents with the lowest resource levels
- **Fixed Sharing Amounts**: Shares a configurable amount of resources
- **Resource Validation**: Ensures agents have sufficient resources to share
- **Simple Rewards**: Linear reward based on amount shared
- **Efficient Spatial Queries**: Uses spatial index for fast proximity searches

---

## Simple Implementation

### Core Logic

The share action uses straightforward Python code:

```python
def share_action(agent: "BaseAgent") -> None:
    # Find nearby agents within sharing range
    nearby_agents = agent.environment.get_nearby_agents(
        agent.position, share_range
    )

    # Filter out self and find valid targets
    valid_targets = [
        target for target in nearby_agents
        if target.agent_id != agent.agent_id and target.alive
    ]

    if not valid_targets:
        return  # No one to share with

    # Find agent with lowest resource level (simple need-based selection)
    target = min(valid_targets, key=lambda a: a.resource_level)

    # Check if agent has enough resources to share
    share_amount = getattr(agent.config, "share_amount", 2)
    min_keep = getattr(agent.config, "min_keep_resources", 5)

    if agent.resource_level < min_keep + share_amount:
        return  # Not enough resources

    # Execute sharing
    agent.resource_level -= share_amount
    target.resource_level += share_amount

    # Simple reward calculation
    reward = share_amount * 0.05
    agent.total_reward += reward
```

### Key Features

- **Spatial Index Integration**: Uses `get_nearby_agents()` for efficient O(log n) queries
- **Need-Based Selection**: Targets the agent with lowest resource level
- **Resource Validation**: Ensures sufficient resources before sharing
- **Configurable Parameters**: Adjustable share amounts and minimum retention
- **Simple Rewards**: Linear reward based on amount shared (0.05 per unit)

---

## Configuration Parameters

The share action uses the following configuration parameters:

- **`share_range`**: Maximum distance to search for agents to share with (default: 30)
- **`share_amount`**: Amount of resources to share per action (default: 2)
- **`min_keep_resources`**: Minimum resources agent must keep for itself (default: 5)

### Reward System

Simple linear reward calculation:
- **Reward per unit**: 0.05 points per resource unit shared
- **Total reward**: `shared_amount * 0.05`
- **Added to**: `agent.total_reward`

---

## Target Selection Algorithm

The share action uses a simple algorithm to select sharing targets:

1. **Spatial Query**: Uses `get_nearby_agents()` within `share_range`
2. **Target Filtering**: Excludes self and non-alive agents
3. **Need-Based Selection**: Selects agent with minimum `resource_level`
4. **Resource Validation**: Ensures agent has sufficient resources to share
5. **Amount Calculation**: Uses fixed `share_amount` parameter

---

## Usage Example

```python
# Simple share action usage
from farm.core.action import share_action

# Agent will automatically find the nearest agent in need and share resources
share_action(agent)

# The action will:
# 1. Find nearby agents within share_range
# 2. Filter out self and non-alive agents
# 3. Select the agent with lowest resource level
# 4. Share share_amount resources if agent has enough
# 5. Update agent's resource level and total reward
```

## Integration with Action System

The share action integrates seamlessly with the action registry:

```python
from farm.core.action import action_registry

# Get the share action
share_action_func = action_registry.get("share")

# Execute share action
share_action_func(agent)
```

## Configuration Examples

### Basic Configuration
```python
# Set sharing parameters in agent config
agent.config.share_range = 30      # Search radius for agents to help
agent.config.share_amount = 2      # Amount to share per action
agent.config.min_keep_resources = 5  # Minimum resources to keep
```

---

## Performance Characteristics

1. **Computational Efficiency**
   - O(log n) spatial index queries for agent finding
   - O(k) distance calculations where k is number of nearby agents
   - Simple resource level comparison for target selection
   - Minimal memory usage (no neural networks or complex state)

2. **Scalability**
   - Performance scales with environment size through spatial indexing
   - Linear time complexity for target selection among nearby agents
   - No training overhead or iterative learning processes
   - Constant-time reward calculation

3. **Reliability**
   - Deterministic behavior based on resource levels
   - Graceful handling of edge cases (no nearby agents, insufficient resources)
   - Simple error handling and logging
   - No complex dependencies or external libraries required

---

## Migration from DQN

### Previous DQN Implementation

The original implementation used:
- `ShareQNetwork` for Q-value approximation
- `ShareModule` for training and decision making
- Complex state representation (8+ dimensions)
- Cooperation history tracking and scoring
- Experience replay and target networks

### Current Simple Implementation

The new implementation uses:
- Direct spatial queries for agent finding
- Simple resource-level based target selection
- Fixed sharing amounts with resource validation
- Linear reward calculation (0.05 per unit)
- No neural networks or training

---

## Best Practices

1. **Configuration Tuning**
   - Set appropriate `share_range` for your environment size
   - Adjust `share_amount` based on typical resource levels
   - Balance `min_keep_resources` to prevent agents from depleting themselves

2. **Resource Management**
   - Monitor resource distribution across agents
   - Ensure agents have sufficient resources to share
   - Consider agent density when setting sharing parameters

3. **Performance Optimization**
   - Spatial index provides O(log n) query performance
   - Simple resource comparison is computationally efficient
   - No complex dependencies or GPU requirements

4. **Environment Integration**
   - Ensure `get_nearby_agents()` returns valid agent objects
   - Validate agent positions and resource levels
   - Handle edge cases like no nearby agents gracefully

---

## Benefits of Simple Implementation

1. **Reduced Complexity**: No neural networks, training loops, or complex state management
2. **Better Performance**: Instant decisions with minimal computational overhead
3. **Easier Debugging**: Clear, predictable behavior based on simple rules
4. **Lower Resource Usage**: No GPU requirements or large memory allocations
5. **Maintainability**: Simple code that's easy to understand and modify

---

## Future Enhancements

While the current implementation is rule-based, future versions could include:

1. **Resource Prioritization**
   - Quality-based sharing (consider agent health, specialization)
   - Multi-agent sharing strategies
   - Resource type differentiation

2. **Adaptive Behavior**
   - Dynamic sharing amounts based on resource availability
   - Learning-based reward scaling
   - Environmental context awareness

3. **Social Dynamics**
   - Reciprocal relationship tracking
   - Group formation and cooperation
   - Reputation-based sharing decisions

---

## Implementation Notes

- **No External Dependencies**: The simple implementation uses only standard Python
- **Thread-Safe**: No shared state or complex data structures
- **Memory Efficient**: Minimal memory usage with no neural network overhead
- **Deterministic**: Same inputs produce same outputs (except for random tie-breaking if needed)
- **Observable**: Clear logging and interaction tracking for analysis 