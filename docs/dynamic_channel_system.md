# Dynamic Channel System for Agent Observations

The AgentFarm observation system has been enhanced with a dynamic channel registry that allows users to define custom observation channels without modifying the core code. This system maintains full backward compatibility while providing powerful extensibility.

## Overview

The dynamic channel system consists of:

- **ChannelHandler**: Abstract base class for implementing custom channel logic
- **ChannelRegistry**: Central registry for managing and organizing channels
- **ChannelBehavior**: Enum defining how channels behave (instant, dynamic, persistent)
- **Core Handlers**: Built-in implementations of all standard channels

## Channel Behaviors

### Instant Channels
- Cleared and overwritten every tick with fresh data
- Examples: `SELF_HP`, `ALLIES_HP`, `ENEMIES_HP`, `RESOURCES`, `OBSTACLES`

### Dynamic Channels  
- Persist across ticks and decay over time using gamma factors
- Examples: `TRAILS`, `DAMAGE_HEAT`, `ALLY_SIGNAL`, `KNOWN_EMPTY`

### Persistent Channels
- Remain unchanged until explicitly cleared
- Useful for long-term memory or permanent environment features

## Creating Custom Channels

### Basic Custom Handler

```python
from farm.core.channels import ChannelHandler, ChannelBehavior, register_channel

class MyCustomHandler(ChannelHandler):
    def __init__(self):
        super().__init__("MY_CUSTOM", ChannelBehavior.INSTANT)
    
    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        # observation is an AgentObservation instance
        # Use sparse storage methods for efficiency when available
        custom_data = kwargs.get("my_custom_data")
        if custom_data is not None:
            # Use sparse methods: observation._store_sparse_point(channel_idx, y, x, value)
            # Or direct tensor access: observation.tensor()[channel_idx] = data
            pass

# Register the channel
custom_idx = register_channel(MyCustomHandler())
```

### Dynamic Channel with Decay

```python
class DecayingChannel(ChannelHandler):
    def __init__(self, gamma=0.9):
        super().__init__("DECAYING", ChannelBehavior.DYNAMIC, gamma=gamma)
    
    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        events = kwargs.get("decay_events", [])
        R = config.R
        ay, ax = agent_world_pos

        for event_y, event_x, intensity in events:
            dy, dx = event_y - ay, event_x - ax
            y, x = R + dy, R + dx
            if 0 <= y < 2*R+1 and 0 <= x < 2*R+1:
                # Use sparse storage if available for efficiency
                if hasattr(observation, '_store_sparse_point'):
                    current_val = observation.tensor()[channel_idx, y, x].item()
                    new_val = max(current_val, float(intensity))
                    observation._store_sparse_point(channel_idx, y, x, new_val)
                else:
                    # Fallback to direct tensor access
                    observation[channel_idx, y, x] = max(
                        observation[channel_idx, y, x].item(),
                        float(intensity)
                    )
```

### World Layer Channel

```python
class WorldLayerChannel(ChannelHandler):
    def __init__(self, name, layer_key):
        super().__init__(name, ChannelBehavior.INSTANT)
        self.layer_key = layer_key
    
    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        world_layers = kwargs.get("world_layers", {})
        if self.layer_key in world_layers:
            from farm.core.observations import crop_local
            R = config.R
            crop = crop_local(world_layers[self.layer_key], agent_world_pos, R)
            # Use sparse storage for dense grids if available
            if hasattr(observation, '_store_sparse_grid'):
                observation._store_sparse_grid(channel_idx, crop)
            else:
                # Fallback to direct tensor access
                observation[channel_idx].copy_(crop)

# Register for elevation data
elevation_handler = WorldLayerChannel("ELEVATION", "elevation_map")
register_channel(elevation_handler)
```

## Using Custom Channels

Once registered, custom channels work seamlessly with the observation system:

```python
from farm.core.observations import AgentObservation, ObservationConfig

# Create observation system
config = ObservationConfig(R=6, fov_radius=5)
agent_obs = AgentObservation(config)

# Use with custom data
agent_obs.perceive_world(
    world_layers={"RESOURCES": resource_grid},
    agent_world_pos=(50, 50),
    self_hp01=0.8,
    allies=[],
    enemies=[],
    goal_world_pos=None,
    # Custom channel data
    my_custom_data=custom_data,
    decay_events=[(45, 47, 0.8), (52, 53, 0.6)],
    elevation_map=elevation_grid
)

# Access observation tensor (includes custom channels)
obs_tensor = agent_obs.tensor()
```

## Channel Registry API

### Registration
```python
from farm.core.channels import register_channel, get_channel_registry

# Register with automatic index assignment
index = register_channel(my_handler)

# Register with specific index (for compatibility)
index = register_channel(my_handler, index=20)

# Access registry directly
registry = get_channel_registry()
all_handlers = registry.get_all_handlers()
```

### Lookup
```python
# Get channel information
handler = registry.get_handler("MY_CUSTOM")
index = registry.get_index("MY_CUSTOM") 
name = registry.get_name(index)
num_channels = registry.num_channels
```

## Advanced Examples

See `farm/core/custom_channel_example.py` for comprehensive examples including:

- **WeatherHandler**: Environmental weather effects
- **SoundHandler**: Audio perception with distance attenuation
- **SmellHandler**: Chemical detection with diffusion
- **TemperatureHandler**: Temperature sensing with normalization
- **CommunicationHandler**: Advanced inter-agent messaging

## Backward Compatibility

The system maintains full backward compatibility:

- Original `Channel` enum still works
- `NUM_CHANNELS` reflects total channel count
- All existing code continues to work unchanged
- Core channel indices remain fixed (0-11)

## Best Practices

### Naming Conventions
- Use descriptive, uppercase names: `WEATHER`, `SOUND`, `TEMPERATURE`
- Avoid conflicts with existing channels
- Consider prefixing for organization: `SENSOR_TEMPERATURE`, `ENV_WEATHER`

### Performance Considerations
- Use appropriate channel behavior for your use case
- Minimize unnecessary computation in `process()` method
- Consider caching expensive calculations
- Use efficient tensor operations

### Error Handling
```python
def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
    try:
        data = kwargs.get("my_data")
        if data is None:
            return  # Gracefully handle missing data

        # Process data using observation (AgentObservation instance)...
        # Use sparse methods: observation._store_sparse_point(...)
        # Or tensor access: observation.tensor()[channel_idx] = ...

    except Exception as e:
        # Log error but don't crash observation system
        print(f"Warning: {self.name} channel failed: {e}")
```

### Testing Custom Channels
```python
import pytest
from farm.core.observations import AgentObservation, ObservationConfig

def test_my_custom_channel():
    config = ObservationConfig(R=3)
    agent_obs = AgentObservation(config)
    
    # Test channel registration
    initial_channels = agent_obs.registry.num_channels
    register_channel(MyCustomHandler())
    assert agent_obs.registry.num_channels == initial_channels + 1
    
    # Test channel processing
    agent_obs.perceive_world(
        world_layers={}, 
        agent_world_pos=(50, 50),
        self_hp01=1.0,
        allies=[], enemies=[], goal_world_pos=None,
        my_custom_data=test_data
    )
    
    # Verify results
    obs = agent_obs.tensor()
    custom_idx = agent_obs.registry.get_index("MY_CUSTOM")
    assert obs[custom_idx].sum() > 0  # Verify data was written
```

## Migration Guide

For users upgrading from the hardcoded channel system:

1. **No immediate changes required** - existing code continues to work
2. **Optional: Use new channel system** for custom channels
3. **Consider refactoring** hardcoded channel access to use registry lookups for future flexibility

### Before (Hardcoded)
```python
# Direct channel access
obs[Channel.TRAILS] *= 0.9
obs[Channel.DAMAGE_HEAT].zero_()
```

### After (Registry-based, optional)
```python
# Registry-based access (more flexible)
registry = get_channel_registry()
trails_idx = registry.get_index("TRAILS")
damage_idx = registry.get_index("DAMAGE_HEAT")
obs[trails_idx] *= 0.9
obs[damage_idx].zero_()
```

The dynamic channel system provides a powerful, extensible foundation for custom observation capabilities while maintaining the simplicity and performance of the original system.
