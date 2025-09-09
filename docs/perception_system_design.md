# AgentFarm Perception & Observation System Design Document

## Executive Summary

The AgentFarm perception system implements a sophisticated multi-agent observation framework that balances memory efficiency, computational performance, and neural network compatibility. This document details the architectural design choices, trade-offs, and optimization strategies that enable scalable multi-agent reinforcement learning simulations.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architectural Components](#core-architectural-components)
3. [Design Philosophy & Trade-offs](#design-philosophy--trade-offs)
4. [Memory vs Computation Optimization](#memory-vs-computation-optimization)
5. [Spatial Index Integration](#spatial-index-integration)
6. [Channel System Architecture](#channel-system-architecture)
7. [Performance Characteristics](#performance-characteristics)
8. [Implementation Details](#implementation-details)
9. [Future Optimizations](#future-optimizations)

---

## System Overview

### Mission & Requirements

The perception system must provide:
- **Real-time observation generation** for hundreds to thousands of agents
- **Multi-channel environmental awareness** with temporal persistence
- **Neural network compatibility** with existing RL frameworks
- **Scalable memory usage** that grows linearly with agent count
- **Efficient spatial queries** for proximity-based perception

### Key Design Constraints

1. **Neural Network Compatibility**: Must work with PyTorch/TensorFlow dense tensor expectations
2. **Real-time Performance**: Sub-100ms observation updates for smooth simulation
3. **Memory Scalability**: Support 10,000+ agents without excessive memory pressure
4. **Spatial Efficiency**: O(log n) proximity queries for entity detection
5. **Temporal Awareness**: Support for persistent and decaying observations

### Core Innovation: Hybrid Sparse/Dense Architecture

The system implements a **hybrid approach** that combines:
- **Sparse internal storage** for memory efficiency
- **Lazy dense construction** for computational performance
- **Channel-specific optimization** strategies

---

## Core Architectural Components

### 1. AgentObservation Class

**Purpose**: Central observation management with hybrid storage strategy

**Key Features**:
- Hybrid sparse/dense tensor management
- Lazy dense construction on-demand
- Channel-specific storage optimization
- Temporal decay with automatic cleanup
- Backward compatibility with existing RL code

**Memory Optimization**:
- Sparse storage: [TBD] bytes per agent
- Lazy dense: [TBD] bytes built when needed
- **[TBD]% memory reduction** vs pure dense approach

### 2. Channel System (ChannelRegistry + ChannelHandler)

**Purpose**: Extensible multi-channel observation framework

**Architecture**:
- Dynamic channel registration system
- Handler-based processing with inheritance
- Three behavioral types: INSTANT, DYNAMIC, PERSISTENT
- Type-safe channel indexing and lookup

**Channel Categories**:
- **Entity Channels**: SELF_HP, ALLIES_HP, ENEMIES_HP (sparse point storage)
- **Environmental Channels**: RESOURCES, OBSTACLES, TERRAIN_COST (dense grid storage)
- **Visibility Channels**: VISIBILITY mask (dense grid storage)
- **Temporal Channels**: DAMAGE_HEAT, TRAILS, ALLY_SIGNAL (sparse with decay)
- **Navigation Channels**: GOAL, LANDMARKS (sparse persistent)

### 3. SpatialIndex Integration

**Purpose**: Efficient proximity queries and spatial reasoning

**Technical Implementation**:
- KD-tree based spatial indexing (scipy.spatial.cKDTree)
- O(log n) query complexity for proximity searches
- Automatic position tracking and cache invalidation
- Bilinear interpolation for continuous position representation

**Query Types**:
- `get_nearby()`: Radius-based entity queries
- `get_nearest()`: Single nearest entity lookup
- Coordinate transformation: World → Local observation space

### 4. Environment Integration

**Purpose**: Seamless integration with simulation environment

**Key Interfaces**:
- `Environment._get_observation()`: Main observation generation pipeline
- Resource distribution with bilinear interpolation
- Position discretization methods (floor, round, ceil)
- Multi-agent coordination and state synchronization

---

## Design Philosophy & Trade-offs

### Core Design Principles

1. **Memory Efficiency First**: Sparse representation where possible
2. **Computational Performance**: Dense processing when needed
3. **Backward Compatibility**: Drop-in replacement for existing systems
4. **Scalability**: Linear growth with agent count
5. **Extensibility**: Plugin architecture for new channel types

### Fundamental Trade-offs

#### Memory vs Computation Trade-off

**Dense Approach (Original)**:
```
✅ PROS:
   - Perfect GPU acceleration ([TBD] GFLOPS)
   - Native neural network compatibility
   - Optimal cache utilization ([TBD]% hit rate)
   - SIMD vectorization (16 elements/instruction)
   - Zero computational overhead

❌ CONS:
   - [TBD] KB per agent (wasteful)
   - [TBD]% memory stores zeros
   - Scales poorly with agent count
   - High memory pressure at scale
```

**Sparse Approach (Theoretical)**:
```
✅ PROS:
   - Minimal memory usage ([TBD] KB per agent)
   - Perfect memory efficiency
   - Scales linearly with information content

❌ CONS:
   - [TBD]x slower computation
   - Complex custom GPU kernels needed
   - Poor cache utilization ([TBD]% hit rate)
   - Neural network incompatibility
```

**Hybrid Approach (Our Solution)**:
```
✅ PROS:
   - [TBD]% memory reduction
   - Same computational performance as dense
   - Neural network compatibility maintained
   - Optimal GPU utilization when processing
   - Scales efficiently with agent count

⚠️  CONS:
   - Implementation complexity
   - Lazy construction overhead
   - Memory fragmentation potential
```

#### Instant vs Persistent Information

**INSTANT Channels**: Overwritten each tick
- **Use Case**: Current state (health, positions, immediate threats)
- **Storage**: Sparse (single points or small sets)
- **Memory**: Minimal, cleared each tick
- **Performance**: Fast updates, no accumulation

**DYNAMIC Channels**: Persist with exponential decay
- **Use Case**: Temporal information (trails, damage heat, signals)
- **Storage**: Sparse with automatic cleanup
- **Memory**: Grows then stabilizes, self-cleaning
- **Performance**: Decay computation + cleanup

**PERSISTENT Channels**: Never cleared
- **Use Case**: Long-term memory (landmarks, learned behaviors)
- **Storage**: Sparse accumulation
- **Memory**: Grows over time, manual cleanup
- **Performance**: Minimal overhead, stable operation

---

## Memory vs Computation Optimization

### Memory Efficiency Analysis

#### Per-Agent Memory Usage

| Component | Dense (Original) | Sparse (Optimized) | Savings |
|-----------|------------------|-------------------|---------|
| **Observation Tensor** | [TBD] bytes | [TBD] bytes | [TBD] bytes |
| **Channel Overhead** | 0 bytes | [TBD] bytes | [TBD] bytes |
| **Spatial Index** | [TBD] bytes | [TBD] bytes | [TBD] bytes |
| **Total per Agent** | **[TBD] bytes** | **[TBD] bytes** | **[TBD] bytes** |

#### Scaling Analysis

| Agent Count | Dense Memory | Sparse Memory | Savings |
|-------------|--------------|---------------|---------|
| 100 agents | [TBD] KB | [TBD] KB | [TBD] KB ([TBD]%) |
| 1,000 agents | [TBD] MB | [TBD] MB | [TBD] MB ([TBD]%) |
| 10,000 agents | [TBD] MB | [TBD] MB | [TBD] MB ([TBD]%) |

### Computational Performance Analysis

#### Operation Complexity

| Operation | Dense Complexity | Sparse Complexity | Notes |
|-----------|------------------|------------------|-------|
| **Storage** | O(1) | O(1) | Dictionary lookup |
| **Retrieval** | O(1) | O(1) | Cached construction |
| **Decay** | O(grid_size) | O(active_elements) | Sparse much faster |
| **NN Processing** | O(grid_size) | O(grid_size) | Same dense tensor |

#### GPU Performance Characteristics

**Dense Operations**:
- Memory bandwidth: [TBD] GB/s (coalesced access)
- Cache hit rate: [TBD]%
- SIMD efficiency: 100%
- Tensor core utilization: Optimal

**Sparse Operations** (if used directly):
- Memory bandwidth: [TBD] GB/s (scattered access)
- Cache hit rate: [TBD]%
- SIMD efficiency: [TBD]%
- Tensor core utilization: Poor

**Hybrid Approach**:
- Memory bandwidth: [TBD] GB/s (when dense)
- Cache hit rate: [TBD]% (when dense)
- SIMD efficiency: [TBD]% (when dense)
- Sparse storage overhead: Minimal

---

## Spatial Index Integration

### Architecture Overview

The spatial index provides **O(log n) proximity queries** that are fundamental to efficient observation generation.

#### KD-Tree Implementation
```python
# Spatial index maintains separate KD-trees for agents and resources
self.agent_kdtree = cKDTree(agent_positions)      # O(n log n) build
self.resource_kdtree = cKDTree(resource_positions) # O(m log m) build

# Query operations
nearby_agents = spatial_index.get_nearby(position, radius, ["agents"])  # O(log n)
nearest_resource = spatial_index.get_nearest(position, ["resources"])   # O(log n)
```

### Integration Points

#### 1. Environment → Spatial Index
```python
# Environment updates spatial index when agents/resources change
self.spatial_index.set_references(self._agent_objects.values(), self.resources)
self.spatial_index.update()  # Rebuilds KD-trees as needed
```

#### 2. Spatial Index → AgentObservation
```python
# AgentObservation queries spatial index for entity detection
nearby = spatial_index.get_nearby(agent.position, config.fov_radius, ["agents"])
computed_allies, computed_enemies = process_nearby_agents(nearby["agents"])
```

#### 3. Coordinate Transformation Pipeline
```python
# World coordinates → Local observation coordinates
world_y, world_x = entity.position
local_y = config.R + (world_y - agent_y)
local_x = config.R + (world_x - agent_x)
```

### Performance Characteristics

#### Query Performance
- **Build Time**: O(n log n) for n entities
- **Query Time**: O(log n) per proximity query
- **Update Frequency**: Only when positions change significantly
- **Memory Usage**: ~[TBD] KB per 100 agents/resources

#### Optimization Strategies
1. **Change Detection**: Only rebuild when positions actually change
2. **Incremental Updates**: Partial tree updates for small changes
3. **Query Batching**: Multiple queries in single tree traversal
4. **Caching**: Cache frequent proximity queries

---

## Channel System Architecture

### Channel Registry Design

The channel registry provides **dynamic channel management** with efficient indexing:

```python
# Channel registration with automatic indexing
register_channel(SelfHPHandler(), 0)      # Fixed index for compatibility
register_channel(AlliesHPHandler(), 1)    # Fixed index for compatibility
register_channel(ResourcesHandler(), 3)   # Dynamic channels get auto indices

# Efficient lookup operations
channel_idx = registry.get_index("SELF_HP")     # O(1) dictionary lookup
channel_name = registry.get_name(0)            # O(1) array lookup
handler = registry.get_handler("SELF_HP")      # O(1) dictionary lookup
```

### Channel Handler Pattern

Each channel implements a **handler pattern** for processing:

```python
class ChannelHandler(ABC):
    def __init__(self, name: str, behavior: ChannelBehavior, gamma: float = None):
        self.name = name
        self.behavior = behavior  # INSTANT, DYNAMIC, PERSISTENT
        self.gamma = gamma        # Decay rate for DYNAMIC channels

    @abstractmethod
    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        """Process world data into observation channel"""
        pass

    def decay(self, observation, channel_idx, config=None):
        """Apply temporal decay if DYNAMIC"""
        if self.behavior == ChannelBehavior.DYNAMIC and self.gamma:
            # Apply decay with sparse awareness
            pass
```

### Channel-Specific Optimization

#### Point Entity Channels (Sparse)
- **SELF_HP**: Single center pixel storage
- **ALLIES_HP/ENEMIES_HP**: Coordinate-value pairs with accumulation
- **GOAL**: Single target coordinate
- **LANDMARKS**: Accumulating coordinate set

#### Environmental Channels (Dense)
- **VISIBILITY**: Full disk mask (needed for NN convolution)
- **RESOURCES**: Bilinear distributed (continuous values)
- **OBSTACLES/TERRAIN_COST**: Full grid data

#### Temporal Channels (Sparse with Decay)
- **DAMAGE_HEAT/TRAILS/ALLY_SIGNAL**: Sparse points with exponential decay
- **KNOWN_EMPTY**: Sparse known-empty cells with decay

---

## Performance Characteristics

### Benchmark Results

#### Memory Performance
- **Memory per Agent**: [TBD] KB (vs [TBD] KB dense)
- **Memory Efficiency**: [TBD]% reduction
- **Scaling Factor**: Linear with agent count
- **Peak Memory**: ~[TBD] MB for 10,000 agents

#### Computational Performance
- **Observation Generation**: < [TBD]ms per agent
- **Spatial Queries**: < [TBD]ms per agent
- **Neural Network Processing**: Same as dense ([TBD] KB tensor)
- **Decay Operations**: O(active_elements) vs O(grid_size)

### Scalability Analysis

#### Agent Count Scaling
```
Agents | Memory (MB) | Generation (ms) | NN Processing (ms)
-------|-------------|----------------|------------------
100    | [TBD]       | [TBD]         | [TBD]
1,000  | [TBD]      | [TBD]         | [TBD]
10,000 | [TBD]      | [TBD]         | [TBD]
```

#### Perception Radius Scaling
```
Radius | Grid Size | Memory (KB) | Query Time (μs)
-------|-----------|-------------|----------------
3      | 7×7      | [TBD]       | [TBD]
6      | 13×13    | [TBD]       | [TBD]
12     | 25×25    | [TBD]       | [TBD]
```

### Bottleneck Analysis

#### Memory Bottlenecks
1. **Dense Tensor Cache**: [TBD] KB per agent when active
2. **Sparse Dictionary Overhead**: Python dict per channel
3. **Channel Registry**: Global registry overhead

#### Computational Bottlenecks
1. **Spatial Index Updates**: O(n log n) rebuilds
2. **Dense Tensor Construction**: O(grid_size) population
3. **Bilinear Interpolation**: O(nearby_resources) computation

---

## Implementation Details

### Sparse Storage Implementation

#### Data Structures
```python
# Sparse channel storage
self.sparse_channels = {
    0: {(6, 6): 0.8},                    # SELF_HP: single point
    1: {(5, 7): 0.9, (7, 5): 0.7},      # ALLIES_HP: multiple points
    6: torch_tensor,                     # VISIBILITY: dense grid
    8: {(4, 4): 0.3, (8, 8): 0.1},      # DAMAGE_HEAT: decaying points
}
```

#### Storage Methods
```python
def _store_sparse_point(self, channel_idx, y, x, value):
    """Store single coordinate-value pair"""
    if channel_idx not in self.sparse_channels:
        self.sparse_channels[channel_idx] = {}
    self.sparse_channels[channel_idx][(y, x)] = value
    self.cache_dirty = True

def _store_sparse_points(self, channel_idx, points):
    """Store multiple points with max accumulation"""
    if channel_idx not in self.sparse_channels:
        self.sparse_channels[channel_idx] = {}
    channel_data = self.sparse_channels[channel_idx]
    for y, x, value in points:
        key = (y, x)
        channel_data[key] = max(channel_data.get(key, 0.0), value)
    self.cache_dirty = True
```

### Lazy Dense Construction

#### Construction Algorithm
```python
def _build_dense_tensor(self):
    if not self.cache_dirty and self.dense_cache is not None:
        return self.dense_cache

    # Allocate dense tensor
    S = 2 * self.config.R + 1
    num_channels = self.registry.num_channels
    self.dense_cache = torch.zeros(num_channels, S, S, ...)

    # Populate from sparse data
    for channel_idx, channel_data in self.sparse_channels.items():
        if isinstance(channel_data, dict):
            # Sparse points
            for (y, x), value in channel_data.items():
                if 0 <= y < S and 0 <= x < S:
                    self.dense_cache[channel_idx, y, x] = value
        else:
            # Dense grids
            self.dense_cache[channel_idx] = channel_data

    self.cache_dirty = False
    return self.dense_cache
```

### Temporal Decay Implementation

#### Sparse-Aware Decay
```python
def _decay_sparse_channel(self, channel_idx, decay_factor):
    if channel_idx in self.sparse_channels:
        channel_data = self.sparse_channels[channel_idx]
        if isinstance(channel_data, dict):
            # Decay sparse points
            keys_to_remove = []
            for pos, value in channel_data.items():
                new_value = value * decay_factor
                if abs(new_value) < 1e-6:
                    keys_to_remove.append(pos)
                else:
                    channel_data[pos] = new_value
            # Remove effectively zero values
            for pos in keys_to_remove:
                del channel_data[pos]
        else:
            # Decay dense grids
            channel_data *= decay_factor
    self.cache_dirty = True
```

---

## Future Optimizations

### 1. Advanced Sparse Backends

#### PyTorch Sparse Tensors
```python
# Future: Direct sparse tensor support
sparse_obs = torch.sparse_coo_tensor(
    indices=sparse_indices,
    values=sparse_values,
    size=(13, 13, 13)
)
```

#### cuSPARSE Integration
```python
# GPU-accelerated sparse operations
sparse_result = cusparse.spmm(sparse_matrix, dense_weights)
```

### 2. Quantization Strategies

#### Channel-Specific Precision
```python
# Binary channels: int8 (0/1 values)
visibility_quantized = visibility_mask.to(torch.int8)

# Continuous channels: float16
resources_half = resources_tensor.to(torch.float16)
```

#### Adaptive Quantization
```python
# Dynamic precision based on channel requirements
precision_map = {
    "SELF_HP": torch.float16,      # Health values
    "VISIBILITY": torch.int8,      # Binary mask
    "RESOURCES": torch.float16,    # Resource amounts
}
```

### 3. Memory Pooling System

#### Tensor Reuse Pool
```python
class DenseTensorPool:
    def __init__(self, max_size=1000):
        self.pool = []
        self.max_size = max_size

    def get_tensor(self, shape, dtype, device):
        # Find reusable tensor or create new one
        for tensor in self.pool:
            if (tensor.shape == shape and
                tensor.dtype == dtype and
                tensor.device == device):
                self.pool.remove(tensor)
                return tensor
        return torch.zeros(shape, dtype=dtype, device=device)

    def return_tensor(self, tensor):
        if len(self.pool) < self.max_size:
            tensor.zero_()  # Clear for reuse
            self.pool.append(tensor)
```

### 4. GPU Sparse Operations

#### Direct Sparse Neural Networks
```python
# Sparse convolution layers
sparse_conv = torch.nn.Conv2d(
    in_channels=13,
    out_channels=32,
    kernel_size=3,
    sparse_input=True  # Future PyTorch feature
)
```

#### Custom CUDA Kernels
```cuda
// Custom sparse observation processing kernel
__global__ void process_sparse_observation(
    const int* indices,
    const float* values,
    const int num_entries,
    float* output
) {
    // Direct sparse processing on GPU
}
```

### 5. Advanced Spatial Indexing

#### R-Tree Integration
```python
# More efficient for 2D spatial queries
from rtree import index
spatial_index = index.Index()
```

#### Approximate Nearest Neighbor
```python
# ANN for faster approximate queries
import nmslib
ann_index = nmslib.init(method='hnsw', space='l2')
```

---

## Conclusion

The AgentFarm perception system represents a sophisticated balance between memory efficiency and computational performance. Through the hybrid sparse/dense architecture, we've achieved:

- **[TBD]% memory reduction** while maintaining full computational performance
- **Linear scalability** with agent count and perception complexity
- **Backward compatibility** with existing neural network architectures
- **Extensible design** supporting new channel types and optimization strategies

The system's design acknowledges the fundamental memory vs computation trade-off in machine learning systems and provides a practical solution that scales to thousands of agents while maintaining real-time performance requirements.

Key success factors:
1. **Hybrid approach** combining sparse storage with dense processing
2. **Lazy construction** minimizing unnecessary computation
3. **Channel-specific optimization** leveraging different data patterns
4. **Spatial index integration** providing efficient proximity queries
5. **Temporal decay** with automatic cleanup maintaining sparsity

This architecture enables AgentFarm to support large-scale multi-agent simulations with sophisticated perception systems while maintaining the computational efficiency required for reinforcement learning training.
