# AgentFarm Observation Channels System

## Executive Summary

The AgentFarm observation channels system provides a flexible, extensible framework for defining multi-channel environmental observations. The system supports instant, dynamic, and persistent channel behaviors, enabling agents to perceive their environment through multiple complementary data streams with different temporal characteristics.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Channel Behaviors](#channel-behaviors)
3. [Core Channel Types](#core-channel-types)
4. [Custom Channel Development](#custom-channel-development)
5. [Channel Registry & Management](#channel-registry--management)
6. [Configuration](#configuration)
7. [Performance Optimization](#performance-optimization)
8. [Integration & Usage](#integration--usage)
9. [Examples & Use Cases](#examples--use-cases)
10. [References & Technical Details](#references--technical-details)

---

## System Overview

### Mission & Requirements

The channel system enables:

- **Multi-channel environmental perception** with different data types and temporal behaviors
- **Extensible observation framework** supporting custom channel development
- **Efficient storage and processing** through sparse/dense optimization strategies
- **Temporal awareness** with decay and persistence mechanisms

### Core Architecture

**Key Components:**

- **ChannelHandler**: Abstract base class for channel implementations
- **ChannelRegistry**: Central registry managing channel registration and lookup
- **ChannelBehavior**: Enum defining temporal behavior patterns
- **Observation Pipeline**: Processing pipeline with sparse/dense optimization

---

## Channel Behaviors

The system supports three fundamental channel behaviors that determine how information persists and decays over time:

### 1. Instant Channels

**Behavior**: Completely overwritten each simulation step
**Use Case**: Current state information that changes rapidly
**Storage**: Can use sparse or dense storage depending on data patterns
**Examples**: Health values, resource locations, obstacle positions

**Characteristics:**

- **Immediate Updates**: Fresh data every tick
- **No Persistence**: Previous values are cleared
- **Performance**: Fast updates, minimal memory overhead
- **Best For**: Dynamic environment state, current agent status

### 2. Dynamic Channels

**Behavior**: Persist across ticks with exponential decay
**Use Case**: Temporal information that fades over time
**Storage**: Sparse storage with decay mechanisms
**Examples**: Movement trails, damage heat, communication signals

**Characteristics:**

- **Temporal Decay**: Values decrease by gamma factor each step
- **Memory Building**: Accumulates information over time
- **Self-Cleaning**: Automatic cleanup of decayed values
- **Best For**: Trail following, heat maps, fading memories

### 3. Persistent Channels

**Behavior**: Remain unchanged until explicitly cleared
**Use Case**: Long-term memory or permanent environment features
**Storage**: Sparse accumulation with manual cleanup
**Examples**: Goal waypoints, landmark locations, learned behaviors

**Characteristics:**

- **Indefinite Persistence**: Values remain until cleared
- **Accumulation**: New information adds to existing data
- **Manual Management**: Requires explicit clearing when needed
- **Best For**: Navigation waypoints, permanent landmarks, learned patterns

---

## Core Channel Types

### Entity Channels (Sparse Storage)

**SELF_HP**: Agent's current health status

- **Storage**: Single center pixel
- **Behavior**: Instant
- **Purpose**: Agent self-awareness

**ALLIES_HP**: Visible allied agents' health positions

- **Storage**: Coordinate-value pairs with accumulation
- **Behavior**: Instant
- **Purpose**: Team awareness and coordination

**ENEMIES_HP**: Visible enemy agents' health positions

- **Storage**: Coordinate-value pairs with accumulation
- **Behavior**: Instant
- **Purpose**: Threat assessment and combat planning

### Environmental Channels (Dense Storage)

**RESOURCES**: Resource distribution with bilinear interpolation

- **Storage**: Full continuous grid
- **Behavior**: Instant
- **Purpose**: Resource discovery and navigation

**OBSTACLES**: Obstacle/passability map

- **Storage**: Full discrete grid
- **Behavior**: Instant
- **Purpose**: Path planning and collision avoidance

**TERRAIN_COST**: Movement cost terrain

- **Storage**: Full continuous grid
- **Behavior**: Instant
- **Purpose**: Optimal path finding

**VISIBILITY**: Field-of-view mask for partial observability

- **Storage**: Full disk mask
- **Behavior**: Instant
- **Purpose**: Realistic perception constraints

### Temporal Channels (Sparse with Decay)

**DAMAGE_HEAT**: Recent combat damage locations

- **Storage**: Sparse points with exponential decay
- **Behavior**: Dynamic (gamma typically 0.90-0.95)
- **Purpose**: Combat area identification and avoidance

**TRAILS**: Agent movement paths

- **Storage**: Sparse points with exponential decay
- **Behavior**: Dynamic (gamma typically 0.85-0.95)
- **Purpose**: Trail following and path prediction

**ALLY_SIGNAL**: Communication signals from allied agents

- **Storage**: Sparse points with exponential decay
- **Behavior**: Dynamic (gamma typically 0.80-0.90)
- **Purpose**: Team coordination and signaling

**KNOWN_EMPTY**: Previously explored empty areas

- **Storage**: Sparse points with exponential decay
- **Behavior**: Dynamic (gamma typically 0.95-0.98)
- **Purpose**: Exploration optimization and memory

### Navigation Channels

**GOAL**: Target waypoint or objective location

- **Storage**: Single target coordinate
- **Behavior**: Instant
- **Purpose**: Goal-directed navigation

**LANDMARKS**: Permanent reference points

- **Storage**: Accumulating coordinate set
- **Behavior**: Persistent
- **Purpose**: Spatial reference and mapping

---

## Custom Channel Development

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

### Advanced Custom Channels

**Weather Effects Channel:**

```python
class WeatherHandler(ChannelHandler):
    def __init__(self):
        super().__init__("WEATHER", ChannelBehavior.DYNAMIC, gamma=0.95)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        weather_data = kwargs.get("weather_system", {})
        # Implement weather simulation logic
        weather_intensity = self._simulate_weather(agent_world_pos)
        # Apply weather effects to observation
```

**Audio Perception Channel:**

```python
class SoundHandler(ChannelHandler):
    def __init__(self):
        super().__init__("SOUND", ChannelBehavior.DYNAMIC, gamma=0.90)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        sound_sources = kwargs.get("sound_sources", [])
        # Implement distance-based audio attenuation
        for source_pos, intensity, frequency in sound_sources:
            distance = self._calculate_distance(agent_world_pos, source_pos)
            attenuated_intensity = intensity / (1 + distance)
            # Store attenuated sound information
```

---

## Channel Registry & Management

### Registration API

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

### Lookup Operations

```python
# Get channel information
handler = registry.get_handler("MY_CUSTOM")
index = registry.get_index("MY_CUSTOM")
name = registry.get_name(index)
num_channels = registry.num_channels
```

### Registry Architecture

The channel registry provides **dynamic channel management** with efficient indexing:

```python
# Channel registration with automatic indexing
register_channel(SelfHPHandler(), 0)      # Fixed index for compatibility
register_channel(AlliesHPHandler(), 1)    # Fixed index for compatibility
register_channel(WorldLayerHandler("RESOURCES", "RESOURCES"), 3)   # Dynamic channels get auto indices

# Efficient lookup operations
channel_idx = registry.get_index("SELF_HP")     # O(1) dictionary lookup
channel_name = registry.get_name(0)            # O(1) array lookup
handler = registry.get_handler("SELF_HP")      # O(1) dictionary lookup
```

---

## Configuration

### Basic Configuration

```yaml
channels:
  # Core observation settings
  observation_radius: 6             # Base observation radius
  fov_radius: 5                     # Field-of-view radius
  channel_stacking: true            # Stack multiple time steps

  # Channel-specific settings
  decay_factors:                    # Temporal decay rates (0-1)
    trails: 0.95                    # Movement trail decay
    damage_heat: 0.90               # Combat heat decay
    ally_signals: 0.85              # Communication signal decay
    known_empty: 0.98               # Memory of explored areas

  # Channel enablement
  enabled_channels:                 # Which channels to include
    - SELF_HP
    - ALLIES_HP
    - ENEMIES_HP
    - RESOURCES
    - OBSTACLES
    - VISIBILITY
    - KNOWN_EMPTY
    - DAMAGE_HEAT
    - TRAILS

  # Custom channels
  custom_channels:                  # User-defined channels
    - name: "WEATHER"
      type: "dynamic"
      gamma: 0.95
    - name: "RESOURCE_DENSITY"
      type: "instant"

  # Channel processing
  normalization: "layer"            # "none", "layer", "batch", "instance"
  preprocessing:                    # Data preprocessing
    gaussian_blur: false
```

### Advanced Configuration

```yaml
# Performance tuning
sparse_storage_threshold: 0.1       # Sparsity threshold for storage selection
memory_pool_size: 1000              # Tensor reuse pool size
decay_batch_size: 100               # Batch size for decay operations

# Channel-specific optimization
channel_optimization:
  point_channels: "sparse"          # "sparse", "dense", "auto"
  environmental_channels: "dense"   # "sparse", "dense", "auto"
  temporal_channels: "sparse_decay" # "sparse_decay", "dense_decay", "auto"

# Custom channel configuration
custom_channel_settings:
  weather:
    update_rate: 0.1
    diffusion_rate: 0.05
  sound:
    attenuation_model: "inverse_square"
    frequency_response: "broadband"
```

---

## Performance Optimization

### Sparse vs Dense Storage Selection

**Sparse Storage (Recommended for):**

- Point entity data (health, positions)
- Temporal decay channels
- Low-density environmental data
- Memory-constrained scenarios

**Dense Storage (Required for):**

- Full grid environmental data (visibility masks)
- Neural network convolution layers
- High-frequency updates
- Continuous value distributions

### Channel-Specific Optimization

#### Point Entity Channels (Sparse)

- **SELF_HP**: Single center pixel storage
- **ALLIES_HP/ENEMIES_HP**: Coordinate-value pairs with accumulation
- **GOAL**: Single target coordinate (instant behavior)
- **LANDMARKS**: Accumulating coordinate set (persistent behavior)

#### Environmental Channels (Dense)

- **VISIBILITY**: Full disk mask (needed for NN convolution)
- **RESOURCES**: Bilinear distributed (continuous values)
- **OBSTACLES/TERRAIN_COST**: Full grid data

#### Temporal Channels (Optimization)

- **DAMAGE_HEAT/TRAILS/ALLY_SIGNAL**: Sparse points with exponential decay
- **KNOWN_EMPTY**: Sparse known-empty cells with decay

### Memory Performance

| Channel Type | Typical Sparsity | Memory Savings | Best Storage |
|-------------|------------------|----------------|--------------|
| **Point Entities** | 92-100% | 95-99% | Sparse |
| **Environmental** | 70-95% | Limited | Dense (NN compatibility) |
| **Temporal** | 85-95% | 80-95% | Sparse with decay |
| **Navigation** | 95-100% | 95-99% | Sparse |

---

## Integration & Usage

### Basic Usage

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

### Channel Processing Pipeline

```python
def update_observation(self, agent_position: Tuple[int, int], environment: Environment):
    """Update agent's observation from current environment state."""

    # 1. Clear previous instant observations
    self._clear_instant_channels()

    # 2. Extract local view from world state
    world_view = environment.get_world_view()
    local_view = crop_local(world_view, agent_position, self.config.R)

    # 3. Process each channel
    for channel_name, channel_handler in self._channel_handlers.items():
        channel_idx = get_channel_registry().get_index(channel_name)
        channel_handler.process(
            observation=self._observation,
            channel_idx=channel_idx,
            config=self.config,
            agent_world_pos=agent_position,
            environment=environment,
            agent=self._agent
        )

    # 4. Apply field-of-view masking
    self._apply_fov_mask()

    # 5. Apply temporal decay to dynamic channels
    self._apply_decay()
```

### Integration with Reinforcement Learning

```python
# Channel system integrates seamlessly with RL frameworks
def get_observation_tensor(self, agent_id: str) -> torch.Tensor:
    """Get observation tensor for RL agent."""
    agent = self._agents[agent_id]

    # All channels (core + custom) are included in the tensor
    obs_tensor = agent.observation.tensor()

    # Shape: (num_channels, height, width)
    # Includes all registered channels: core + custom
    return obs_tensor
```

---

## Examples & Use Cases

### 1. Environmental Sensing

**Weather Channel Implementation:**
```python
class WeatherHandler(ChannelHandler):
    def __init__(self):
        super().__init__("WEATHER", ChannelBehavior.DYNAMIC, gamma=0.95)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        weather_system = kwargs.get("weather_system", {})
        weather_intensity = self._calculate_weather_intensity(agent_world_pos, weather_system)

        # Apply weather effects to observation visibility
        if hasattr(observation, '_apply_weather_effect'):
            observation._apply_weather_effect(channel_idx, weather_intensity)
```

### 2. Multi-Modal Perception

**Audio-Visual Integration:**
```python
class AudioVisualHandler(ChannelHandler):
    def __init__(self):
        super().__init__("AUDIO_VISUAL", ChannelBehavior.INSTANT)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        visual_data = kwargs.get("visual_channels", {})
        audio_data = kwargs.get("audio_sources", {})

        # Combine visual and audio information
        combined_perception = self._fuse_audio_visual(
            visual_data, audio_data, agent_world_pos
        )

        # Store integrated perception data
        observation[channel_idx] = combined_perception
```

### 3. Social Communication

**Communication Network Channel:**
```python
class CommunicationHandler(ChannelHandler):
    def __init__(self):
        super().__init__("COMMUNICATION", ChannelBehavior.DYNAMIC, gamma=0.90)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        messages = kwargs.get("incoming_messages", [])

        # Process and visualize communication network
        for sender_pos, message_type, urgency in messages:
            distance = self._calculate_distance(agent_world_pos, sender_pos)
            signal_strength = urgency / (1 + distance)

            # Visualize communication signals
            if hasattr(observation, '_store_sparse_point'):
                observation._store_sparse_point(
                    channel_idx,
                    sender_pos[0] - agent_world_pos[0] + config.R,
                    sender_pos[1] - agent_world_pos[1] + config.R,
                    signal_strength
                )
```

### 4. Learning and Adaptation

**Experience-Based Channel:**
```python
class ExperienceHandler(ChannelHandler):
    def __init__(self):
        super().__init__("EXPERIENCE", ChannelBehavior.PERSISTENT)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        learned_patterns = kwargs.get("learned_patterns", {})

        # Visualize learned dangerous/critical areas
        for pattern_pos, danger_level in learned_patterns.items():
            if self._is_near_position(agent_world_pos, pattern_pos, config.R):
                local_pos = self._world_to_local(pattern_pos, agent_world_pos, config.R)
                observation._store_sparse_point(
                    channel_idx, local_pos[0], local_pos[1], danger_level
                )
```

---

## References & Technical Details

### 📖 Detailed Technical Documentation

For in-depth technical information, refer to:

**Core Implementation:**
- [Dynamic Channel System](dynamic_channel_system.md) - Complete channel implementation guide
- [Perception System Design](perception_system_design.md) - Technical channel architecture details
- [Core Architecture](core_architecture.md) - Channel processing pipeline and integration

**Configuration & Usage:**
- [Configuration Guide](configuration_guide.md) - Complete channel configuration reference
- [Usage Examples](usage_examples.md) - Practical channel implementation examples
- [Module Overview](module_overview.md) - High-level channel system overview

### 🔧 Implementation Files

Key implementation modules:
- `farm/core/channels.py` - Channel system implementation
- `farm/core/observations.py` - Observation system integration
- `farm/core/channel_handlers/` - Built-in channel handler implementations
- `farm/core/custom_channel_example.py` - Advanced custom channel examples

### ⚙️ Configuration Examples

See [Configuration Guide](configuration_guide.md) for:
- Complete configuration options
- Performance tuning recommendations
- Custom channel setup examples

### 📊 Performance Analysis

For detailed performance characteristics:
- Channel memory usage patterns
- Processing time benchmarks
- Sparsity optimization analysis
- Scaling performance with channel count

---

## Conclusion

The AgentFarm observation channels system provides a powerful, extensible framework for multi-channel environmental perception. By supporting instant, dynamic, and persistent channel behaviors with efficient sparse/dense storage optimization, the system enables agents to maintain rich, temporally-aware representations of their environment.

The channel system's key strengths include:
- **Behavioral Flexibility**: Support for different temporal patterns (instant/dynamic/persistent)
- **Extensibility**: Easy custom channel development without core modifications
- **Performance Optimization**: Automatic sparse/dense storage selection
- **Integration**: Seamless compatibility with reinforcement learning frameworks
- **Backward Compatibility**: Full compatibility with existing observation systems

This channel architecture enables the creation of sophisticated agent behaviors that can perceive, remember, and adapt to complex environmental patterns across multiple sensory modalities.
