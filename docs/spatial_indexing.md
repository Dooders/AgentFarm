# AgentFarm Spatial Indexing System

## Executive Summary

The AgentFarm spatial indexing system provides efficient proximity queries and spatial reasoning capabilities essential for scalable multi-agent simulations. The system implements multiple spatial indexing strategies — KD-tree, Quadtree, and Spatial Hash Grid — to enable fast proximity queries for entity detection and spatial awareness, optimized for different query patterns.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Spatial Indexing Strategies](#spatial-indexing-strategies)
3. [Performance Characteristics](#performance-characteristics)
4. [Configuration](#configuration)
5. [Integration & Usage](#integration--usage)
6. [Use Cases & Examples](#use-cases--examples)
7. [References & Technical Details](#references--technical-details)

---

## System Overview

### Mission & Requirements

The spatial indexing system enables:

- **Efficient proximity queries** for agent-agent and agent-resource interactions
- **Scalable entity detection** within perception radii
- **Real-time spatial awareness** for hundreds to thousands of agents
- **Multiple indexing strategies** optimized for different use cases
- **Memory-efficient storage** with configurable performance trade-offs

### Core Components

**Spatial Index Implementation:**

- **KD-Tree Indexing**: O(log n) queries using scipy.spatial.cKDTree with optimized change detection
- **Quadtree Indexing**: Hierarchical spatial partitioning for efficient range queries and dynamic updates
- **Spatial Hash Grid**: Uniform bucket grid for near-constant-time neighborhood queries and efficient dynamic updates
- **Multiple Index Support**: Choose index per use case; enable additional indices alongside defaults
- **Dynamic Updates**: Efficient position updates without full index rebuilds

---

## Spatial Indexing Strategies

### 1. KD-Tree Based Indexing

**Purpose**: Provides efficient continuous-space proximity queries using scientific computing libraries.

**Technical Implementation:**

```python
from scipy.spatial import cKDTree
import numpy as np

# Spatial index maintains separate KD-trees for agents and resources
self.agent_kdtree = cKDTree(agent_positions)      # O(n log n) build
self.resource_kdtree = cKDTree(resource_positions) # O(m log m) build

# Query operations
nearby_agents = spatial_index.get_nearby(position, radius, ["agents"])  # O(log n)
nearest_resource = spatial_index.get_nearest(position, ["resources"])   # O(log n)
```

**Key Features:**

- **Continuous Position Support**: Handles floating-point coordinates
- **Bilinear Interpolation**: Smooth position representation for continuous movement
- **Multi-Entity Type Queries**: Separate trees for agents and resources
- **Automatic Cache Invalidation**: Rebuilds when positions change significantly

### 2. Quadtree Based Indexing

**Purpose**: Provides hierarchical spatial partitioning for efficient range queries and dynamic entity updates.

**Technical Implementation:**

```python
from farm.core.spatial_index import Quadtree

# Create quadtree with environment bounds
bounds = (0, 0, width, height)
quadtree = Quadtree(bounds, capacity=4)

# Insert entities
for entity in entities:
    position = entity.position
    quadtree.insert(entity, position)

# Query operations
nearby_entities = quadtree.query_radius(center, radius)  # O(log n) average
range_entities = quadtree.query_range((x, y, w, h))     # O(log n) average
```

**Key Features:**

- **Hierarchical Subdivision**: Automatically divides space into quadrants when capacity exceeded
- **Rectangular Range Queries**: Highly efficient for finding entities within rectangular regions
- **Dynamic Updates**: Efficient position updates without full tree rebuilds
- **Memory Efficient**: Hierarchical structure reduces cache misses for range queries
- **Spatial Locality**: Nearby entities are grouped together in the hierarchy

**When to Use Quadtree Indexing:**

- Rectangular or area-of-effect queries (combat, vision cones, territory analysis)
- High-frequency position updates (agent movement, dynamic environments)
- Range-based spatial operations (crowd density, group formations)
- Scenarios requiring hierarchical spatial reasoning

### 3. Spatial Hash Grid Indexing

**Purpose**: Provides grid-based bucketing for O(1)-ish average neighborhood queries and fast dynamic updates.

**Technical Implementation:**

```python
from farm.core.environment import Environment

env = Environment(width=100, height=100, resource_distribution="uniform")
env.enable_spatial_hash_indices(cell_size=5.0)  # optional cell size override

# Queries
nearby = env.spatial_index.get_nearby((50, 50), 10, ["agents_hash"])  # uses spatial hash
nearest = env.spatial_index.get_nearest((42, 18), ["resources_hash"])  # hash nearest
```

**Key Features:**

- **Uniform Grid Buckets**: Entities stored in integer (ix, iy) buckets
- **Bounded Query Cost**: Only checks buckets overlapping the query region
- **Fast Dynamic Updates**: O(1) remove/insert on moves
- **Hotspot Robustness**: Performs well under non-uniform distributions

**When to Use Spatial Hashing:**

- Large, dynamic populations where rebuild cost is high
- Frequent radius/neighbor queries with moderate radii
- Non-uniform distributions (clusters, crowds) where locality helps

---

## Performance Characteristics

### Query Performance Characteristics

| Strategy | Build Time | Query Time | Memory Usage | Best For |
|----------|------------|------------|--------------|----------|
| **KD-Tree** | O(n log n) | O(log n) | ~200 KB/100 agents | Radial queries, nearest neighbor |
| **Quadtree** | O(n log n) | O(log n) | ~150 KB/100 agents | Range queries, dynamic updates |
| **Spatial Hash** | O(n) | ~O(1) avg | ~120 KB/100 agents | Frequent neighbor queries, dynamic updates |

### Scaling Analysis

**For KD-Tree Implementation:**

- **Build Time**: O(n log n) for n entities
- **Query Time**: O(log n) per proximity query
- **Update Frequency**: Only when positions change significantly
- **Memory Usage**: ~200 KB per 100 agents/resources

**For Quadtree Implementation:**

- **Build Time**: O(n log n) for n entities (with hierarchical subdivision)
- **Query Time**: O(log n) average for range queries, O(log n) for radius queries
- **Update Frequency**: Incremental updates for position changes
- **Memory Usage**: ~150 KB per 100 agents (more efficient for range queries)
- **Subdivision**: Automatic quadrant division based on entity density

### Optimization Strategies

**Change Detection**: Only rebuild when positions actually change
**Incremental Updates**: Partial tree updates for small changes
**Query Batching**: Multiple queries in single tree traversal
**Caching**: Cache frequent proximity queries
**Dual Indexing**: Use KD-trees for radial queries, Quadtrees for range queries


---

## Configuration

### Basic Configuration

```yaml
# Spatial indexing settings
spatial_index_enabled: true       # Enable spatial indexing
spatial_analysis: true            # Analyze spatial patterns
```

### Advanced Configuration

```yaml
# Performance tuning
spatial_update_batch_size: 100     # Batch size for position updates
spatial_query_timeout: 0.01        # Query timeout in seconds

# Memory management
spatial_memory_pool_size: 1000     # Size of tensor reuse pool
spatial_gc_threshold: 0.8          # Memory usage threshold for garbage collection

# Debugging and monitoring
spatial_debug_queries: false       # Log spatial queries
spatial_performance_metrics: true  # Collect performance metrics
```

### When to Use KD-Tree Indexing

**KD-Tree indexing is ideal when:**

- Environment uses continuous coordinates
- Agents have large perception radii
- High precision is required for spatial queries
- Memory budget allows for tree storage
- You need efficient proximity queries for hundreds to thousands of agents

**Note**: The system supports both KD-tree and Quadtree indexing. KD-trees are the default for their superior performance on radial queries. Quadtree indices can be enabled for specific use cases requiring efficient range queries and dynamic updates.

---

## Integration & Usage

### Basic Usage

```python
from farm.core.spatial_index import SpatialIndex
from farm.core.environment import Environment

# Create environment with spatial indexing
env = Environment(
    width=100,
    height=100,
    resource_distribution="uniform",
    spatial_index_enabled=True
)

# Access spatial index
spatial_index = env.spatial_index

# Query nearby agents
nearby_agents = spatial_index.get_nearby(agent.position, radius=5, ["agents"])

# Query nearest resource
nearest_resource = spatial_index.get_nearest(agent.position, ["resources"])
```

### Enabling Quadtree Indices

```python
# Enable Quadtree indices for range queries and dynamic updates
env.enable_quadtree_indices()

# Now you can use Quadtree-optimized queries
nearby_in_range = spatial_index.get_nearby_range(
    (x, y, width, height),  # Rectangular bounds
    ["agents_quadtree"]     # Use Quadtree index
)

# Dynamic position updates are more efficient with Quadtrees
spatial_index.update_entity_position(
    agent, old_position, new_position
)

# Get detailed Quadtree statistics
# Get detailed Quadtree statistics
quadtree_stats = spatial_index.get_quadtree_stats("agents_quadtree")
print(f"Quadtree depth: {quadtree_stats}")
```

### Enabling Spatial Hash Indices

```python
# Enable Spatial Hash indices for fast neighborhood queries and dynamic updates
env.enable_spatial_hash_indices(cell_size=5.0)  # optional; uses heuristic if None

# Use spatial-hash-backed indices
nearby = spatial_index.get_nearby((x, y), radius, ["agents_hash"])  # bucketed radius query
nearest = spatial_index.get_nearest((x, y), ["resources_hash"])     # nearest via grid expansion

# Range queries
in_rect = spatial_index.get_nearby_range((rx, ry, rw, rh), ["agents_hash"])
```

### Advanced Usage with Custom Queries

```python
# Multi-type proximity query
nearby_entities = spatial_index.get_nearby(
    position=agent.position,
    radius=10,
    entity_types=["agents", "resources", "obstacles"]
)

# Filtered queries with custom conditions
nearby_allies = [
    aid for aid in nearby_agents
    if env.get_agent(aid).team == agent.team
]

# Spatial analysis for decision making
def evaluate_position_safety(self, position):
    """Evaluate safety of a position based on nearby threats."""
    threats = spatial_index.get_nearby(position, self.threat_radius, ["enemies"])
    allies = spatial_index.get_nearby(position, self.support_radius, ["allies"])

    threat_score = len(threats) * self.threat_weight
    support_score = len(allies) * self.support_weight

    return support_score - threat_score
```

### Environment Integration

```python
# Environment updates spatial index when agents/resources change
self.spatial_index.set_references(self._agent_objects.values(), self.resources)
self.spatial_index.update()  # Rebuilds KD-trees as needed

# AgentObservation queries spatial index for entity detection
nearby = spatial_index.get_nearby(agent.position, config.fov_radius, ["agents"])
computed_allies, computed_enemies = process_nearby_agents(nearby["agents"])
```

### Coordinate Transformation

```python
# World coordinates → Local observation coordinates
world_y, world_x = entity.position
local_y = config.R + (world_y - agent_y)
local_x = config.R + (world_x - agent_x)

# Handle boundary conditions
local_y = max(0, min(2*config.R, local_y))
local_x = max(0, min(2*config.R, local_x))
```

---

## Use Cases & Examples

### 1. Combat Simulation

**Spatial Index Usage in Combat:**

- **Target Acquisition**: Find nearest enemies within attack range
- **Formation Analysis**: Identify allies within support radius
- **Threat Assessment**: Count enemies within danger zone

```python
# Combat target selection using spatial index
def find_combat_target(self, agent, max_range):
    """Find closest enemy within attack range."""
    nearby_enemies = spatial_index.get_nearby(
        agent.position,
        max_range,
        ["enemies"]
    )

    if nearby_enemies:
        # Get actual distance to closest enemy
        closest_enemy = min(
            nearby_enemies,
            key=lambda eid: self.distance(agent.position, env.get_agent(eid).position)
        )
        return closest_enemy

    return None
```

### 2. Resource Gathering

**Spatial Index Usage in Resource Management:**

- **Resource Discovery**: Find nearest resources of specific types
- **Competition Analysis**: Identify other agents targeting same resources
- **Path Optimization**: Plan routes avoiding contested resources

```python
# Resource gathering with spatial awareness
def find_optimal_resource(self, agent, resource_type):
    """Find best resource considering competition."""
    nearby_resources = spatial_index.get_nearby(
        agent.position,
        self.search_radius,
        [resource_type]
    )

    # Filter out heavily contested resources
    optimal_resources = []
    for resource in nearby_resources:
        competitors = spatial_index.get_nearby(
            resource.position,
            self.competition_radius,
            ["agents"]
        )
        if len(competitors) < self.max_competitors:
            optimal_resources.append(resource)

    return optimal_resources[0] if optimal_resources else None
```

### 3. Social Behavior

**Spatial Index Usage in Social Interactions:**

- **Ally Detection**: Find nearby allies for coordination
- **Group Formation**: Identify agents for flocking behavior
- **Communication Networks**: Establish local communication links

```python
# Social behavior using spatial awareness
def find_social_partners(self, agent):
    """Find nearby allies for social interactions."""
    nearby_allies = spatial_index.get_nearby(
        agent.position,
        self.social_radius,
        ["allies"]
    )

    # Prioritize based on relationship strength and distance
    potential_partners = []
    for ally_id in nearby_allies:
        ally = env.get_agent(ally_id)
        distance = self.distance(agent.position, ally.position)
        relationship = self.get_relationship_strength(agent, ally)

        score = relationship / (1 + distance)  # Closer, stronger relationships preferred
        potential_partners.append((ally_id, score))

    return sorted(potential_partners, key=lambda x: x[1], reverse=True)
```

### 4. Navigation and Pathfinding

**Spatial Index Usage in Navigation:**

- **Obstacle Avoidance**: Identify obstacles in planned paths
- **Goal-directed Movement**: Find waypoints toward objectives
- **Terrain Analysis**: Evaluate movement costs in different areas

```python
# Navigation assistance using spatial index
def plan_navigation_path(self, start_pos, goal_pos):
    """Plan path considering spatial obstacles."""
    # Get obstacles along potential path
    path_obstacles = spatial_index.get_nearby(
        start_pos,
        self.path_lookahead,
        ["obstacles"]
    )

    # Adjust path based on obstacle positions
    adjusted_path = self.adjust_path_for_obstacles(
        start_pos, goal_pos, path_obstacles
    )

    return adjusted_path
```

---

## References & Technical Details

### 📖 Detailed Technical Documentation

For in-depth technical information, refer to:

**Core Implementation:**

- [Perception System Design](perception_system_design.md) - Complete spatial index implementation details
- [Core Architecture](core_architecture.md) - System integration and spatial management patterns

**Performance & Optimization:**

- [Configuration Guide](configuration_guide.md) - Complete spatial indexing configuration reference
- [Observation Footprint](observation_footprint.md) - Memory usage analysis for spatial indexing

**Research & Advanced Topics:**

- [Index Optimization Strategies](research/agent_state_memory/index_optimization_strategies.md) - Advanced indexing optimization techniques
- [Redis Index Schema](research/agent_state_memory/redis_index_schema.md) - Spatial indexing in memory systems

### 🔧 Implementation Files

Key implementation modules:

- `farm/core/spatial_index.py` - Core spatial indexing implementation
- `farm/core/environment.py` - Environment integration
- `farm/core/observations.py` - Observation system integration

### ⚙️ Configuration Examples

See [Configuration Guide](configuration_guide.md) for:

- Complete configuration options
- Performance tuning recommendations
- Environment-specific settings

### 📊 Performance Benchmarks

For detailed performance characteristics:

- Query latency benchmarks
- Memory usage patterns by strategy
- Scaling performance with agent count
- Optimization recommendations

---

## Batch Spatial Updates with Dirty Region Tracking

### Overview

The spatial indexing system now includes **batch spatial updates with dirty region tracking**, a performance optimization feature that significantly improves efficiency in dynamic simulations by only updating regions that have actually changed.

### Key Features

- **Dirty Region Tracking**: Only regions that have changed are marked for updates
- **Batch Processing**: Multiple position updates are collected and processed together
- **Priority-Based Updates**: Higher priority regions are updated first
- **Automatic Cleanup**: Old regions are automatically cleaned up to prevent memory bloat
- **Performance Monitoring**: Detailed statistics about update efficiency

### Benefits

- **Reduced Update Overhead**: Up to 70% reduction in computational overhead for dynamic simulations
- **Improved Scalability**: Performance improvements scale with simulation size
- **Data Integrity**: Ensures all regions reflect current state without stale data
- **Memory Efficiency**: Better memory usage patterns through batched processing

### Configuration

```python
from farm.config.config import SpatialIndexConfig

spatial_config = SpatialIndexConfig(
    enable_batch_updates=True,      # Enable batch updates
    region_size=50.0,               # Size of each region
    max_batch_size=100,             # Maximum updates per batch
    max_regions=1000,               # Maximum regions to track
    performance_monitoring=True     # Enable performance monitoring
)
```

### Usage

```python
# Batch updates are automatically enabled and work transparently
env = Environment(width=200, height=200, resource_distribution="uniform", config=config)

# Monitor performance
stats = env.get_spatial_performance_stats()
print(f"Batch updates processed: {stats['batch_updates']['total_batch_updates']}")
print(f"Average batch size: {stats['batch_updates']['average_batch_size']}")
```

## Conclusion

The AgentFarm spatial indexing system provides the foundation for efficient spatial reasoning in multi-agent simulations. The system now includes advanced batch spatial updates with dirty region tracking, offering optimal performance for different query patterns in continuous-space environments.

The spatial indexing system enables:

- **Efficient proximity queries** for real-time agent interactions using O(log n) queries
- **Dual indexing strategies**: KD-trees for radial queries, Quadtrees for range queries
- **Scalable spatial awareness** supporting thousands of agents with optimized change detection
- **Memory-efficient storage** with intelligent cache invalidation and position tracking
- **Dynamic updates**: Efficient position changes without full index rebuilds
- **Flexible named indices** for custom spatial data sources and filtering
- **Batch spatial updates**: Only update regions that have changed for maximum efficiency
- **Dirty region tracking**: Systematic approach to ensure data integrity and performance

**Choosing the Right Index:**

- **Use KD-trees** for: Nearest neighbor searches, radial proximity queries, static or slowly changing environments
- **Use Quadtrees** for: Rectangular range queries, area-of-effect operations, frequently moving entities, hierarchical spatial analysis
- **Use Spatial Hash** for: Large dynamic populations, frequent neighborhood queries, and hotspot-heavy distributions
- **Use Batch Updates** for: Dynamic simulations with frequent position changes, large-scale environments, performance-critical applications

This enhanced spatial foundation is essential for creating realistic multi-agent behaviors, from combat and resource gathering to social interactions and navigation, making it a critical component of the AgentFarm simulation framework. The batch spatial updates feature ensures that the system can scale efficiently to handle complex, dynamic simulations with thousands of agents while maintaining optimal performance.
