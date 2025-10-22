# Reward Component Integration

## Overview

The reward component has been implemented as the **default reward system** for all agents in the simulation. This document explains the integration and how it works.

## What Changed

### 1. Reward Component as Default System

The reward component is now automatically included in all agents created through the factory system:

- **Default agents**: `factory.create_default_agent()` includes reward component
- **Learning agents**: `factory.create_learning_agent()` includes reward component
- **Custom agents**: Any agent created with `AgentComponentConfig` includes reward component

### 2. Environment Integration

The environment now uses the reward component for reward calculation:

- **Primary method**: `Environment._get_agent_reward()` checks for reward component first
- **Fallback**: Falls back to original `_calculate_reward()` if no reward component found
- **Backward compatibility**: Old agents without reward component still work

### 3. Configuration System

The reward component uses the standard configuration system:

- **Default config**: `RewardConfig()` with sensible defaults
- **Custom config**: Can be customized per agent via `AgentComponentConfig`
- **Preset configs**: Can be extended with preset configurations

## How It Works

### Agent Creation

```python
# All agents automatically get reward component
agent = factory.create_default_agent(
    agent_id="my_agent",
    position=(0.0, 0.0),
    config=AgentComponentConfig.default()  # Includes RewardConfig
)

# Custom reward configuration
custom_config = AgentComponentConfig(
    reward=RewardConfig(
        resource_reward_scale=2.0,
        survival_bonus=0.5,
        death_penalty=-15.0,
    )
)
agent = factory.create_learning_agent(
    agent_id="learning_agent",
    position=(0.0, 0.0),
    config=custom_config
)
```

### Reward Calculation

The reward component calculates rewards in two phases:

1. **Step Start**: Captures pre-action state for delta calculation
2. **Step End**: Calculates and applies rewards based on state changes

```python
# This happens automatically during simulation
reward_component.on_step_start()  # Capture state
# ... agent actions ...
reward_component.on_step_end()    # Calculate rewards
```

### Environment Integration

The environment automatically uses the reward component:

```python
# Environment gets reward from component
reward = env._get_agent_reward(agent_id, pre_action_state)

# This method:
# 1. Checks if agent has reward component
# 2. Uses component's step_reward if available
# 3. Falls back to original calculation if not
```

## Configuration Options

### Default Configuration

```python
RewardConfig(
    resource_reward_scale=1.0,      # Resource reward multiplier
    health_reward_scale=0.5,        # Health reward multiplier
    survival_bonus=0.1,             # Per-step survival bonus
    death_penalty=-10.0,            # Death penalty
    age_bonus=0.01,                 # Longevity bonus
    combat_success_bonus=2.0,       # Combat success reward
    reproduction_bonus=5.0,         # Reproduction reward
    cooperation_bonus=1.0,          # Cooperation reward
    max_history_length=1000,        # Reward history limit
    recent_window=100,              # Recent reward window
    use_delta_rewards=True,         # Use delta-based calculation
    normalize_rewards=False,        # Normalize to [-1, 1]
)
```

### Custom Configurations

You can create custom reward configurations for different scenarios:

```python
# Aggressive agents (encourage combat)
aggressive_config = RewardConfig(
    resource_reward_scale=1.5,
    health_reward_scale=0.3,
    combat_success_bonus=10.0,
    reproduction_bonus=15.0,
)

# Conservative agents (encourage survival)
conservative_config = RewardConfig(
    resource_reward_scale=0.8,
    health_reward_scale=1.2,
    survival_bonus=0.3,
    death_penalty=-20.0,
    age_bonus=0.05,
)

# Exploration agents (encourage movement)
exploration_config = RewardConfig(
    resource_reward_scale=1.0,
    survival_bonus=0.1,
    age_bonus=0.02,
    # Could add exploration-specific bonuses
)
```

## Accessing Reward Information

### From Agent

```python
# Get reward component
reward_component = None
for comp in agent.components:
    if hasattr(comp, 'cumulative_reward'):
        reward_component = comp
        break

# Access reward information
total_reward = reward_component.total_reward
current_reward = reward_component.current_reward
stats = reward_component.get_reward_stats()

# Add manual rewards
reward_component.add_reward(5.0, "achievement bonus")

# Reset for new episode
reward_component.reset_rewards()
```

### From Environment

The environment automatically uses the reward component, so you don't need to change existing code that uses `env.step()` or similar methods.

## Testing

Comprehensive tests have been added:

- **Unit tests**: `test_reward.py` - Tests component functionality
- **Integration tests**: `test_reward_integration.py` - Tests factory integration
- **Environment tests**: `test_reward_environment_integration.py` - Tests environment integration
- **Default system tests**: `test_reward_default_system.py` - Tests that it's the default system

## Migration Guide

### For Existing Code

**No changes required!** The reward component is backward compatible:

- Existing agents will continue to work
- Environment will use reward component if available
- Falls back to original calculation if not available

### For New Code

Use the standard factory methods - reward component is included automatically:

```python
# This automatically includes reward component
agent = factory.create_default_agent(
    agent_id="my_agent",
    position=(0.0, 0.0),
)

# Customize reward behavior
agent = factory.create_learning_agent(
    agent_id="learning_agent",
    position=(0.0, 0.0),
    config=AgentComponentConfig(
        reward=RewardConfig(
            resource_reward_scale=2.0,
            survival_bonus=0.5,
        )
    )
)
```

## Benefits

1. **Consistent reward system**: All agents use the same reward calculation logic
2. **Configurable**: Easy to customize reward behavior per agent
3. **Extensible**: Easy to add new reward types
4. **Testable**: Comprehensive test coverage
5. **Backward compatible**: Existing code continues to work
6. **Performance**: Efficient reward calculation and tracking
7. **Debugging**: Rich reward statistics and history

## Future Enhancements

The reward component can be easily extended with:

- **Action-specific rewards**: Rewards based on specific actions taken
- **Social rewards**: Rewards for cooperation, competition, etc.
- **Environmental rewards**: Rewards based on environment state
- **Learning rewards**: Rewards that adapt based on agent performance
- **Multi-objective rewards**: Different reward signals for different goals

The component-based architecture makes these extensions straightforward to implement and test.