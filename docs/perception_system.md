# AgentFarm Perception System Overview

## Executive Summary

The AgentFarm perception system implements a sophisticated multi-agent observation framework that provides agents with **local, agent-centric views** of their environment. The system balances memory efficiency, computational performance, and neural network compatibility while enabling scalable multi-agent reinforcement learning simulations.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Key Principles](#key-principles)
3. [Core Components](#core-components)
4. [Configuration](#configuration)
5. [Performance Characteristics](#performance-characteristics)
6. [Integration & Usage](#integration--usage)
7. [References & Technical Details](#references--technical-details)

---

## System Overview

### Mission & Requirements

The perception system provides:

- **Real-time observation generation** for hundreds to thousands of agents
- **Multi-channel environmental awareness** with temporal persistence
- **Neural network compatibility** with existing RL frameworks
- **Scalable memory usage** that grows linearly with agent count
- **Efficient spatial queries** for proximity-based perception

### Core Innovation: Hybrid Architecture

The system combines:

- **Sparse internal storage** for memory efficiency
- **Lazy dense construction** for computational performance
- **Channel-specific optimization** strategies
- **Spatial indexing** for efficient proximity queries

---

## Key Principles

### Local, Agent-Centric Perception

**Local Perspective Fields**: Each agent receives a small window (configurable) centered on their position, creating **subjective, limited experiences** rather than global knowledge.

**Orientation Alignment**: The agent's facing direction is always "up" in the perception grid, ensuring stable, agent-centric perspectives.

**Limited Field of View**: Agents can only "see" within a certain radius, with areas beyond this masked as unknown, simulating **partial observability**.

### Multi-Channel Observations

The system uses **13 default channels** (indices 0-12) that can be extended:

**Entity Channels** (Sparse storage, INSTANT behavior):

- `SELF_HP` (Index 0) - Agent's current health at center position
- `ALLIES_HP` (Index 1) - Visible allies' health at their positions
- `ENEMIES_HP` (Index 2) - Visible enemies' health at their positions

**Environmental Channels** (Dense storage, INSTANT behavior):

- `RESOURCES` (Index 3) - Resource distribution with bilinear interpolation
- `OBSTACLES` (Index 4) - Obstacle/passability map
- `TERRAIN_COST` (Index 5) - Movement cost terrain
- `VISIBILITY` (Index 6) - Field-of-view visibility mask

**Temporal Channels** (Sparse with decay, DYNAMIC behavior):

- `KNOWN_EMPTY` (Index 7) - Previously observed empty cells (decays over time)
- `DAMAGE_HEAT` (Index 8) - Recent damage events (decays over time)
- `TRAILS` (Index 9) - Agent movement trails (decays over time)
- `ALLY_SIGNAL` (Index 10) - Communication signals (decays over time)

**Navigation Channels**:

- `GOAL` (Index 11) - Target waypoint positions (INSTANT behavior)
- `LANDMARKS` (Index 12) - Permanent reference points (PERSISTENT behavior)

### Spatial Reasoning & Efficiency

**KD-Tree Spatial Indexing**: Uses scipy.spatial.cKDTree for O(log n) proximity queries with smart change detection to minimize rebuilds.

**Key Features**:

- **Multi-level optimization**: Dirty flag check (O(1)) → Count check (O(1)) → Hash verification (O(n))
- **Named index system**: Configurable indices for different entity types (agents, resources, etc.)
- **Automatic filtering**: Built-in support for filtering (e.g., only alive agents)
- **Position validation**: Relaxed bounds checking with 1% margin for edge cases

**Query Types**:

- `get_nearby(position, radius, index_names)` - Radius-based entity queries across multiple indices
- `get_nearest(position, index_names)` - Single nearest entity lookup per index
- Coordinate transformation from world → local observation space

### Channel Behaviors

**INSTANT**: Completely overwritten each simulation step with fresh data. Used for current state information that doesn't need to persist between ticks.

**DYNAMIC**: Persist across ticks and decay over time using configurable gamma factors. Values gradually fade away, simulating the natural decay of transient information.

**PERSISTENT**: Remain unchanged until explicitly cleared. Used for long-term memory and permanent environment features that should accumulate over time.

---

## Core Components

### 1. AgentObservation Class

**Purpose**: Central observation management with hybrid storage strategy

**Key Features**:

- Hybrid sparse/dense tensor management
- Lazy dense construction on-demand
- Channel-specific storage optimization
- Temporal decay with automatic cleanup
- Backward compatibility with existing RL code

**Memory Optimization**:

- Sparse storage: ~2-3 KB per agent average
- Lazy dense: ~6-24 KB built when needed
- **~60% memory reduction** vs pure dense approach

### 2. Channel System

**Purpose**: Extensible multi-channel observation framework

**Architecture**:

- Dynamic channel registration system
- Handler-based processing with inheritance
- Three behavioral types: INSTANT, DYNAMIC, PERSISTENT
- Type-safe channel indexing and lookup

### 3. Spatial Index Integration

**Purpose**: Efficient proximity queries and spatial reasoning with smart change detection

**Technical Implementation**:

- **KD-tree based spatial indexing** using scipy.spatial.cKDTree
- **Smart change detection** with three-level optimization strategy:
  - Dirty flag check (O(1)) for no-change scenarios
  - Count-based check (O(1)) for structural changes  
  - Hash-based verification (O(n)) for position changes
- **Named index system** supporting configurable entity types with custom filters
- **Automatic position validation** with relaxed bounds checking
- **Memory-efficient caching** with position hash tracking

---

## Configuration

### Basic Configuration

```python
from farm.core.observations import ObservationConfig
from farm.core.config import SimulationConfig

# Create observation configuration
obs_config = ObservationConfig(
    R=6,                          # Observation radius (cells visible in each direction)
    fov_radius=6,                 # Field-of-view radius for visibility mask (default: 6)
    gamma_trail=0.90,             # Decay rate for movement trails (default: 0.90)
    gamma_dmg=0.85,               # Decay rate for damage heat (default: 0.85)
    gamma_sig=0.92,               # Decay rate for ally signals (default: 0.92)
    gamma_known=0.98,             # Decay rate for known empty cells (default: 0.98)
    device="cpu",                 # Device for tensor operations (default: "cpu")
    dtype="float32",              # PyTorch dtype as string (default: "float32")
    initialization="zeros"        # Tensor initialization method (default: "zeros")
)

# Create simulation configuration with observation settings
config = SimulationConfig(
    width=100,
    height=100,
    observation=obs_config,        # Include observation configuration
    # ... other simulation parameters
)
```

All 13 default channels are always included. Custom channels can be added through the dynamic channel registry system.

### Advanced Configuration

```python
# Advanced ObservationConfig with custom settings
obs_config = ObservationConfig(
    R=8,                          # Larger observation radius
    fov_radius=7,                 # Larger field-of-view
    device="cuda",                # Use GPU if available
    dtype="float16",              # Use half precision for memory efficiency
    initialization="random",      # Random initialization instead of zeros
    random_min=-0.1,             # Random initialization range
    random_max=0.1,
    # Custom gamma factors for different decay rates
    gamma_trail=0.95,            # Slower decay for movement trails
    gamma_dmg=0.90,              # Moderate decay for damage heat
    gamma_sig=0.85,              # Faster decay for communication signals
    gamma_known=0.98             # Very slow decay for known empty areas
)

# Alternative: Create with defaults and modify
obs_config = ObservationConfig()  # Use all defaults
obs_config.R = 10                # Larger observation radius
obs_config.device = "cuda"       # Switch to GPU
obs_config.gamma_trail = 0.95    # Custom trail decay rate
```

### YAML Configuration

```yaml
# config.yaml - Observation settings
observation:
  R: 6                           # Observation radius
  fov_radius: 6                  # Field-of-view radius
  gamma_trail: 0.90              # Trail decay rate
  gamma_dmg: 0.85                # Damage heat decay rate
  gamma_sig: 0.92                # Signal decay rate
  gamma_known: 0.98              # Known empty decay rate
  device: "cpu"                  # Device for tensor operations
  dtype: "float32"               # PyTorch dtype
  initialization: "zeros"        # Initialization method
  random_min: 0.0                # Random init minimum
  random_max: 1.0                # Random init maximum

# Load from YAML
from farm.core.config import SimulationConfig
config = SimulationConfig.from_yaml("config.yaml")
```

---

## Performance Characteristics

### Memory Performance

| Component | Radius 5 | Radius 8 | Radius 10 |
|-----------|----------|----------|-----------|
| **Observation Tensor** | 6,292 bytes | 15,044 bytes | 23,508 bytes |
| **Channel Overhead** | 256 bytes | 512 bytes | 768 bytes |
| **Spatial Index** | 2,048 bytes | 2,048 bytes | 2,048 bytes |
| **Total per Agent** | **8,636 bytes** | **17,604 bytes** | **26,324 bytes** |

### Computational Performance

| Operation | Dense Complexity | Sparse Complexity | Notes |
|-----------|------------------|------------------|-------|
| **Storage** | O(1) | O(1) | Dictionary lookup |
| **Retrieval** | O(1) | O(1) | Cached construction |
| **Decay** | O(grid_size) | O(active_elements) | Sparse much faster |
| **NN Processing** | O(grid_size) | O(grid_size) | Same dense tensor |

### Scalability Analysis

| Agent Count | Dense Memory | Sparse Memory | Savings |
|-------------|--------------|---------------|---------|
| 100 agents | 864 KB | 1.7 MB | 864 KB (50%) |
| 1,000 agents | 8.6 MB | 17.6 MB | 9 MB (51%) |
| 10,000 agents | 86.4 MB | 176 MB | 89.6 MB (51%) |

### Benchmarking and Metrics (Sample Run)

We added `perception_metrics` to measure perception throughput, memory, cache behavior, and interpolation cost.

- Config: 100 agents, R=5, hybrid storage, bilinear on, CPU.
- Results (3 iterations):
  - Observes/sec: 880–1,199
  - Mean step time: 83–114 ms
  - Dense bytes per agent (R=5): 6,292 bytes
  - Sparse logical bytes (sample): 984 bytes
  - Memory reduction vs dense: 84.36%
  - Cache hit rate: 0.50
  - Dense rebuilds: 4; total rebuild time: ~0.22–0.41 ms
  - Perception profile totals (3 steps):
    - Spatial query time: 0.031–0.051 s
    - Bilinear time: ~1.12–1.34 ms; points: 48
    - Nearest time: ~0 s; points: 0
  - Estimated GFLOPS (dense reconstruction): ~3.7e-5–5.0e-5

Formulas:
- Sparse per-channel memory ≈ `num_active_points * (sizeof(value) + sizeof(y) + sizeof(x))`.
- Dense memory per agent = `channels * (2R+1)^2 * sizeof(dtype)`.
- Cache hit rate = `hits / (hits + misses)` during `tensor()` calls.
- Per-agent update time (ms) = `(total_time / (steps * agents)) * 1000`.

Notes:
- Hybrid vs dense: hybrid cut logical memory by ~84% in this run with negligible rebuild cost.
- Bilinear vs nearest: bilinear adds small overhead but preserves continuous positions; nearest is faster but coarser.

---

## Integration & Usage

### Basic Usage

```python
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig
from farm.core.config import SimulationConfig

# Create observation configuration
obs_config = ObservationConfig(
    R=6,                    # Observation radius
    fov_radius=6,          # Field of view radius (default: 6)
    gamma_trail=0.90,      # Trail decay rate
    gamma_dmg=0.85,        # Damage heat decay rate
    gamma_sig=0.92,        # Signal decay rate
    gamma_known=0.98       # Known empty decay rate
)

# Create simulation configuration
config = SimulationConfig(
    width=100,
    height=100,
    observation=obs_config,  # Include observation settings
    # ... other simulation parameters
)

# Create environment with configuration
env = Environment(
    width=100,
    height=100,
    resource_distribution="uniform",
    config=config           # Pass the configuration
)

# Environment automatically includes spatial indexing and observation system
```

### Direct AgentObservation Usage

```python
from farm.core.observations import AgentObservation, ObservationConfig

# Create observation configuration
config = ObservationConfig(R=6, fov_radius=5)

# Create agent observation instance
agent_obs = AgentObservation(config)

# Update observation with world state
agent_obs.perceive_world(
    world_layers={
        "RESOURCES": resource_grid,
        "OBSTACLES": obstacle_grid,
        "TERRAIN_COST": terrain_grid
    },
    agent_world_pos=(50, 50),
    self_hp01=0.8,
    allies=[(48, 50, 0.9), (52, 50, 0.7)],
    enemies=[(45, 45, 0.6)],
    goal_world_pos=(60, 60),
    spatial_index=env.spatial_index,
    agent_object=agent
)

# Get observation tensor for neural network input
observation_tensor = agent_obs.tensor()  # Shape: (13, 13, 13)
```

### Channel Registration

```python
from farm.core.channels import ChannelHandler, ChannelBehavior, register_channel

class CustomChannel(ChannelHandler):
    def __init__(self):
        super().__init__("CUSTOM_CHANNEL", ChannelBehavior.DYNAMIC, gamma=0.95)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        # Implement your channel processing logic
        pass

# Register the channel
channel_idx = register_channel(CustomChannel())
```

### Spatial Queries

```python
# Get nearby entities using spatial index
nearby_entities = env.spatial_index.get_nearby(
    position=agent.position, 
    radius=5.0, 
    index_names=["agents", "resources"]
)

# Extract specific entity types
nearby_agents = nearby_entities.get("agents", [])
nearby_resources = nearby_entities.get("resources", [])

# Get nearest entities across multiple indices
nearest_entities = env.spatial_index.get_nearest(
    position=agent.position, 
    index_names=["agents", "resources"]
)
nearest_agent = nearest_entities.get("agents")
nearest_resource = nearest_entities.get("resources")

# Process nearby entities (agents are pre-filtered to alive only)
for agent_obj in nearby_agents:
    # Calculate distance for observation processing
    distance = ((agent.position[0] - agent_obj.position[0])**2 + 
               (agent.position[1] - agent_obj.position[1])**2)**0.5
    
    # Update observation channels based on proximity
    if distance <= observation_radius:
        # Process agent for observation update
        pass

# Register custom indices for specialized queries
env.spatial_index.register_index(
    name="enemies",
    data_reference=env.agents,
    position_getter=lambda a: a.position,
    filter_func=lambda a: a.team != agent.team and a.alive
)

# Query custom indices
nearby_enemies = env.spatial_index.get_nearby(
    position=agent.position,
    radius=10.0,
    index_names=["enemies"]
)
```

---

## References & Technical Details

### 📖 Detailed Technical Documentation

For in-depth technical information, refer to:

**Design & Architecture:**

- [Perception System Design](perception_system_design.md) - Complete technical implementation details
- [Core Architecture](core_architecture.md) - System integration and architecture patterns
- [Dynamic Channel System](dynamic_channel_system.md) - Channel system implementation details

**Performance & Optimization:**

- [Observation Footprint Analysis](observation_footprint.md) - Memory usage and performance analysis
- [Configuration Guide](configuration_guide.md) - Complete configuration reference

**Integration & Usage:**

- [API Reference](api_reference.md) - Complete API documentation
- [Usage Examples](usage_examples.md) - Practical implementation examples

### 🔧 Implementation Files

Key implementation modules:

- `farm/core/observations.py` - Core observation system
- `farm/core/channels.py` - Channel system implementation
- `farm/core/spatial_index.py` - Spatial indexing implementation
- `farm/core/environment.py` - Environment integration

### ⚙️ Configuration Examples

See [Configuration Guide](configuration_guide.md) for:

- Complete configuration options
- Performance tuning recommendations
- Environment-specific settings

### 📊 Performance Analysis

For detailed performance characteristics:

- Memory usage patterns
- Computational bottlenecks
- Scaling recommendations
- Optimization strategies

---

## Conclusion

The AgentFarm perception system provides a sophisticated foundation for realistic agent perception in multi-agent simulations. By combining efficient spatial indexing with a flexible channel system and hybrid storage strategies, it enables scalable, memory-efficient observation generation that supports both simple scripted agents and complex reinforcement learning systems.

The system's design emphasizes **local, subjective experiences** over global knowledge, creating more realistic and adaptable agent behavior while maintaining the computational efficiency required for large-scale simulations.
