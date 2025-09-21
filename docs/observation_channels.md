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
7. [Performance Characteristics](#performance-characteristics)
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

The observation system is configured using the `ObservationConfig` class:

```python
from farm.core.observations import ObservationConfig

# Basic configuration with defaults
config = ObservationConfig(
    R=6,                    # Observation radius (cells visible in each direction)
    fov_radius=6,           # Field-of-view radius for visibility mask
    gamma_trail=0.90,       # Decay rate for movement trails
    gamma_dmg=0.85,         # Decay rate for damage heat
    gamma_sig=0.92,         # Decay rate for ally signals
    gamma_known=0.98,       # Decay rate for known empty cells
    device="cpu",           # Device for tensor operations
    dtype="float32",        # PyTorch dtype as string
    initialization="zeros"  # Tensor initialization method
)
```

### Advanced Configuration

```python
# Advanced configuration with custom settings
config = ObservationConfig(
    R=8,                    # Larger observation radius
    fov_radius=7,           # Larger field-of-view
    device="cuda",          # Use GPU if available
    dtype="float16",        # Use half precision for memory efficiency
    initialization="random", # Random initialization instead of zeros
    random_min=-0.1,        # Random initialization range
    random_max=0.1,
    # Custom gamma factors for different decay rates
    gamma_trail=0.95,       # Slower decay for movement trails
    gamma_dmg=0.90,         # Moderate decay for damage heat
    gamma_sig=0.85,         # Faster decay for communication signals
    gamma_known=0.98,       # Very slow decay for known empty areas
    # High-frequency channel prebuilding (optional)
    high_frequency_channels=["RESOURCES", "VISIBILITY"]
)
```

### High-Frequency Channels (Prebuilding Optimization)

- Purpose: Mark channels that are frequently overwritten/read each tick to be maintained as prebuilt dense slices for fast copies during dense construction.
- Recommended for: `RESOURCES`, `OBSTACLES`, `VISIBILITY`, or other dense world-layer channels accessed every step.
- Trade-offs:
  - Uses additional memory: one `(2R+1) x (2R+1)` tensor per high-frequency channel per agent
  - Faster dense builds via single copy instead of many Python-loop assignments
- Configuration:

```python
config = ObservationConfig(
    R=6,
    high_frequency_channels=["RESOURCES", "VISIBILITY"]
)
```

- Behavior:
  - Store/clear/decay operate directly on the prebuilt slice for marked channels
  - `_build_dense_tensor()` copies prebuilt slices O(S²) once per channel
  - Other sparse channels are vectorized using masked indexing (no Python loops)

### YAML Configuration

```yaml
# config.yaml - Observation settings
observation:
  R: 6                      # Observation radius
  fov_radius: 6             # Field-of-view radius
  gamma_trail: 0.90         # Trail decay rate
  gamma_dmg: 0.85           # Damage heat decay rate
  gamma_sig: 0.92           # Signal decay rate
  gamma_known: 0.98         # Known empty decay rate
  device: "cpu"             # Device for tensor operations
  dtype: "float32"          # PyTorch dtype
  initialization: "zeros"   # Initialization method
  random_min: 0.0           # Random init minimum
  random_max: 1.0           # Random init maximum
```

All 13 default channels are always included. Custom channels can be added through the dynamic channel registry system.

---

## Performance Characteristics

### Memory Usage

The observation system uses dense tensor storage for all channels. Memory usage scales with:

- **Observation radius (R)**: Memory scales as O(R²) per channel
- **Number of channels**: Currently 13 default channels
- **Tensor precision**: float32 (default) or float16 for memory efficiency

### Channel Storage Patterns

#### Point Entity Channels

- **SELF_HP**: Single center pixel (R, R)
- **ALLIES_HP/ENEMIES_HP**: Coordinate-value pairs at entity positions
- **GOAL**: Single target coordinate (instant behavior)
- **LANDMARKS**: Accumulating coordinate set (persistent behavior)

#### Environmental Channels

- **VISIBILITY**: Full disk mask (needed for neural network convolution)
- **RESOURCES**: Full grid with bilinear interpolation
- **OBSTACLES/TERRAIN_COST**: Full grid data

#### Temporal Channels

- **DAMAGE_HEAT/TRAILS/ALLY_SIGNAL**: Points with exponential decay
- **KNOWN_EMPTY**: Known-empty cells with decay

### Performance Considerations

- **GPU Acceleration**: Use `device="cuda"` for GPU-accelerated operations
- **Memory Efficiency**: Use `dtype="float16"` for reduced memory usage
- **Initialization**: Random initialization can help with training stability
- **High-Frequency Channels**: Use `high_frequency_channels` for channels updated every tick to reduce CPU overhead in dense tensor builds

### Metrics

`AgentObservation.get_metrics()` exposes fields to monitor optimization impact:

- `grid_population_ops`: Number of full-grid copy operations
- `vectorized_point_assign_ops`: Number of vectorized point assignment batches
- `prebuilt_channel_copies`: Number of prebuilt channel copies into the dense cache
- `prebuilt_channels_active`: Count of high-frequency channels configured

These help validate that prebuilt channels are used (copies increase) and Python-loop assignments are avoided (vectorized ops >= 1 for sparse dict paths).

---

## Integration & Usage

### Basic Usage

```python
from farm.core.observations import AgentObservation, ObservationConfig

# Create observation system
config = ObservationConfig(R=6, fov_radius=5)
agent_obs = AgentObservation(config)

# Update observation with world state
agent_obs.perceive_world(
    world_layers={"RESOURCES": resource_grid, "OBSTACLES": obstacle_grid},
    agent_world_pos=(50, 50),
    self_hp01=0.8,  # Required: agent's health normalized to [0,1]
    allies=[(48, 50, 0.9), (52, 50, 0.7)],  # Optional: (y, x, hp) tuples
    enemies=[(45, 45, 0.6)],  # Optional: (y, x, hp) tuples
    goal_world_pos=(60, 60),  # Optional: (y, x) goal position
    recent_damage_world=[(45, 47, 0.8), (52, 53, 0.6)],  # Optional: damage events
    ally_signals_world=[(48, 49, 0.5)],  # Optional: communication signals
    trails_world_points=[(49, 50, 0.3)],  # Optional: movement trails
    # Custom channel data can be passed via **kwargs
    my_custom_data=custom_data,
    elevation_map=elevation_grid
)

# Access observation tensor (includes all registered channels)
obs_tensor = agent_obs.tensor()  # Shape: (num_channels, 2R+1, 2R+1)
```

### Channel Processing Pipeline

The `perceive_world` method automatically handles the complete observation update process:

```python
# The perceive_world method orchestrates the full update sequence:
# 1. Apply decay to dynamic channels (trails, damage heat, signals, known empty)
# 2. Clear all instantaneous channels  
# 3. Process all registered channels using their handlers
# 4. Update known empty cells based on visibility and entity presence

# All channels are processed automatically through the dynamic registry system
# Custom channels receive data via **kwargs in perceive_world()
```

### Integration with Reinforcement Learning

```python
# Channel system integrates seamlessly with RL frameworks
def get_observation_tensor(self, agent_id: str) -> torch.Tensor:
    """Get observation tensor for RL agent."""
    agent = self._agents[agent_id]

    # All channels (core + custom) are included in the tensor
    obs_tensor = agent.observation.tensor()

    # Shape: (num_channels, 2R+1, 2R+1)
    # Includes all registered channels: core + custom
    return obs_tensor
```

---

## API Reference

### ObservationConfig Class

The `ObservationConfig` class configures the observation system:

```python
class ObservationConfig:
    R: int = 6                    # Observation radius (cells visible in each direction)
    fov_radius: int = 6           # Field-of-view radius for visibility mask
    gamma_trail: float = 0.90     # Decay rate for movement trails
    gamma_dmg: float = 0.85       # Decay rate for damage heat
    gamma_sig: float = 0.92       # Decay rate for ally signals
    gamma_known: float = 0.98     # Decay rate for known empty cells
    device: str = "cpu"           # Device for tensor operations
    dtype: str = "float32"        # PyTorch dtype as string
    initialization: str = "zeros" # Tensor initialization method
    random_min: float = 0.0       # Minimum value for random initialization
    random_max: float = 1.0       # Maximum value for random initialization
```

### AgentObservation.perceive_world() Method

```python
def perceive_world(
    self,
    world_layers: Dict[str, torch.Tensor],  # Required: world tensors (H,W) with values [0,1]
    agent_world_pos: Tuple[int, int],       # Required: (y,x) in world coordinates
    self_hp01: float,                       # Required: agent's health normalized [0,1]
    allies: Optional[List[Tuple[int, int, float]]] = None,      # Optional: (y,x,hp) tuples
    enemies: Optional[List[Tuple[int, int, float]]] = None,     # Optional: (y,x,hp) tuples
    goal_world_pos: Optional[Tuple[int, int]] = None,           # Optional: (y,x) goal position
    recent_damage_world: Optional[List[Tuple[int, int, float]]] = None,  # Optional: damage events
    ally_signals_world: Optional[List[Tuple[int, int, float]]] = None,   # Optional: signals
    trails_world_points: Optional[List[Tuple[int, int, float]]] = None,  # Optional: trails
    spatial_index: Optional[SpatialIndex] = None,               # Optional: for auto-computation
    agent_object: Optional[object] = None,                      # Optional: agent reference
    **kwargs  # Additional data for custom channels
) -> None
```

**Required Parameters:**

- `world_layers`: Dictionary with keys like "RESOURCES", "OBSTACLES", "TERRAIN_COST"
- `agent_world_pos`: Agent's position in world coordinates (y, x)
- `self_hp01`: Agent's health normalized to [0, 1]

**Optional Parameters:**

- `allies`: List of (y, x, hp) tuples for visible allies
- `enemies`: List of (y, x, hp) tuples for visible enemies  
- `goal_world_pos`: Goal position in world coordinates (y, x)
- `recent_damage_world`: List of (y, x, intensity) tuples for recent damage events
- `ally_signals_world`: List of (y, x, intensity) tuples for communication signals
- `trails_world_points`: List of (y, x, intensity) tuples for movement trails
- `**kwargs`: Custom channel data passed to registered channel handlers

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

The AgentFarm observation channels system provides a powerful, extensible framework for multi-channel environmental perception. By supporting instant, dynamic, and persistent channel behaviors with efficient tensor-based storage, the system enables agents to maintain rich, temporally-aware representations of their environment.

The channel system's key strengths include:

- **Behavioral Flexibility**: Support for different temporal patterns (instant/dynamic/persistent)
- **Extensibility**: Easy custom channel development without core modifications
- **Performance Optimization**: Efficient tensor-based storage and processing
- **Integration**: Seamless compatibility with reinforcement learning frameworks
- **Backward Compatibility**: Full compatibility with existing observation systems

This channel architecture enables the creation of sophisticated agent behaviors that can perceive, remember, and adapt to complex environmental patterns across multiple sensory modalities.
