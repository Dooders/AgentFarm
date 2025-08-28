# Move Action Documentation

The Move Action is a simple movement system that enables agents to move randomly in a 2D environment. This action uses straightforward rule-based logic for movement, making it lightweight and predictable.

---

## Overview

The Move Action implements a simple rule-based approach:

- **Random Direction Selection**: Chooses randomly from 4 cardinal directions (up, down, left, right)
- **Configurable Movement Distance**: Uses `max_movement` parameter from agent configuration
- **Boundary Validation**: Ensures agents stay within environment bounds
- **Position Validation**: Checks if new position is valid before moving

---

## Simple Implementation

### Core Logic

The move action uses straightforward Python code:

```python
def move_action(agent: "BaseAgent") -> None:
    # Select random direction (up, down, left, right)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dx, dy = random.choice(directions)

    # Calculate new position
    move_distance = getattr(agent.config, "max_movement", 1)
    new_x = agent.position[0] + dx * move_distance
    new_y = agent.position[1] + dy * move_distance

    # Stay within bounds
    new_x = max(0, min(agent.environment.width - 1, new_x))
    new_y = max(0, min(agent.environment.height - 1, new_y))

    # Move if position is valid
    new_position = (new_x, new_y)
    if agent.environment.is_valid_position(new_position):
        agent.update_position(new_position)
```

### Key Features

- **Random Direction Selection**: Equal probability for all 4 cardinal directions
- **Configurable Distance**: Uses `max_movement` parameter from agent config
- **Boundary Checking**: Prevents agents from moving outside environment bounds
- **Position Validation**: Only moves to valid positions

---

## Technical Details

- **Movement Directions**: 4 cardinal directions (right, left, up, down)
- **Movement Distance**: Configurable via `max_movement` parameter (default: 1)
- **Boundary Constraints**: Agents cannot move outside environment bounds
- **Position Validation**: Uses environment's `is_valid_position()` method

---

## Configuration Parameters

The move action uses the following configuration parameters:

- **`max_movement`**: Distance to move in chosen direction (default: `1`)
- **Environment Bounds**: Uses `environment.width` and `environment.height` for boundary checking

---

## Usage Example

```python
# Simple move action usage
from farm.core.action import move_action

# Agent will randomly move in one of 4 directions
move_action(agent)

# Movement is automatic - no configuration needed beyond agent config
# The action will:
# 1. Choose random direction (up, down, left, right)
# 2. Calculate new position based on max_movement
# 3. Stay within environment bounds
# 4. Move only if position is valid
```

## Integration with Action System

The move action is registered with the action registry:

```python
from farm.core.action import action_registry

# Get the move action
move_action_func = action_registry.get("move")

# Execute move action
move_action_func(agent)
```

---

## Benefits of Simple Implementation

### Reduced Complexity
- No neural network training required
- Simpler debugging and analysis
- Faster execution and lower memory usage

### Predictable Behavior
- Clear movement logic with random direction selection
- Deterministic boundary checking
- Easy to understand and modify

### Better Performance
- No GPU requirements
- Minimal computational overhead
- Instant response without training delays

## Migration from DQN

The original implementation used:
- `MoveQNetwork` for Q-value approximation
- `MoveModule` for training and action selection
- Experience replay and target networks
- Complex state representation

The new implementation uses:
- Simple random direction selection
- Direct boundary validation
- No training or neural networks

---

## Integration with Action System

The move action integrates seamlessly with the action registry system:

1. **Automatic Registration**: The move action is registered at module import time
2. **Action Selection**: Available through `action_registry.get("move")`
3. **Curriculum Learning**: Can be enabled/disabled in different training phases

---

## Best Practices

1. **Configuration**: Set appropriate `max_movement` values for your environment size
2. **Boundary Handling**: Ensure environment provides proper `is_valid_position()` method
3. **Performance**: Random movement provides good exploration in most scenarios
4. **Curriculum**: Enable movement early in training phases for basic exploration
