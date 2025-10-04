# Spatial Indexing & Performance

![Feature](https://img.shields.io/badge/feature-spatial%20%26%20performance-brightgreen)

## Overview

AgentFarm's Spatial Indexing & Performance system provides state-of-the-art spatial data structures and optimization techniques for efficient large-scale multi-agent simulations. With multiple indexing strategies, batch update processing, and comprehensive performance monitoring, the system can handle thousands of agents with minimal computational overhead.

### Why Spatial Indexing Matters

Efficient spatial queries are crucial for:
- **Fast Proximity Detection**: Find nearby entities in milliseconds
- **Scalable Simulations**: Support thousands of agents simultaneously
- **Real-Time Performance**: Maintain responsiveness in dynamic environments
- **Accurate Interactions**: Enable precise spatial awareness
- **Resource Efficiency**: Minimize memory and CPU usage

---

## Core Capabilities

### 1. Advanced Spatial Indexing

AgentFarm implements three complementary spatial indexing strategies, each optimized for different query patterns.

#### KD-Tree Indexing

**Best for**: Radial proximity queries, nearest neighbor searches

```python
from farm.core.environment import Environment

# KD-trees are enabled by default
env = Environment(
    width=100,
    height=100,
    spatial_index_enabled=True
)

# Efficient radial queries: O(log n)
nearby_agents = env.spatial_index.get_nearby(
    position=(50, 50),
    radius=10.0,
    entity_types=["agents"]
)

# Nearest neighbor queries: O(log n)
nearest_resource = env.spatial_index.get_nearest(
    position=(25, 25),
    entity_types=["resources"]
)
```

**Key Features:**
- **Sub-linear Query Time**: O(log n) average for both radius and nearest neighbor
- **Continuous Positions**: Supports floating-point coordinates
- **Bilinear Interpolation**: Smooth position representation
- **Automatic Cache Invalidation**: Rebuilds only when positions change
- **Multi-Entity Support**: Separate trees for agents, resources, obstacles

**Performance Characteristics:**
- Build Time: ~1.26ms for 1000 entities
- Query Time: ~4.85μs per query
- Memory: ~0.1MB per 1000 entities
- Scaling: Near-optimal sub-linear scaling

#### Quadtree Indexing

**Best for**: Rectangular range queries, area-of-effect operations, hierarchical spatial analysis

```python
from farm.core.environment import Environment

# Enable Quadtree indices
env = Environment(width=100, height=100)
env.enable_quadtree_indices()

# Rectangular range queries: O(log n) average
agents_in_area = env.spatial_index.get_nearby_range(
    bounds=(25, 25, 20, 20),  # x, y, width, height
    entity_types=["agents_quadtree"]
)

# Radius queries (also supported)
nearby = env.spatial_index.get_nearby(
    position=(50, 50),
    radius=15.0,
    entity_types=["agents_quadtree"]
)

# Efficient dynamic updates
env.spatial_index.update_entity_position(
    entity=agent,
    old_position=(40, 40),
    new_position=(45, 45)
)

# Get Quadtree statistics
stats = env.spatial_index.get_quadtree_stats("agents_quadtree")
print(f"Quadtree depth: {stats['depth']}")
print(f"Node count: {stats['node_count']}")
```

**Key Features:**
- **Hierarchical Subdivision**: Automatic quadrant division based on density
- **Range Queries**: Highly efficient for rectangular regions
- **Dynamic Updates**: Incremental updates without full rebuilds
- **Spatial Locality**: Nearby entities grouped in hierarchy
- **Memory Efficient**: Hierarchical structure reduces cache misses

**Performance Characteristics:**
- Build Time: ~8.84ms for 1000 entities
- Query Time: ~6.76μs per query
- Memory: ~0.3MB per 1000 entities
- Scaling: Good O(log n) for range queries

**When to Use Quadtree:**
- Area-of-effect abilities (explosions, spells)
- Vision cone queries
- Territory analysis
- Group formation detection
- Crowd density mapping

#### Spatial Hash Grid Indexing

**Best for**: Frequent neighbor queries, large dynamic populations, hotspot-heavy distributions

```python
from farm.core.environment import Environment

# Enable Spatial Hash indices
env = Environment(width=100, height=100)
env.enable_spatial_hash_indices(cell_size=5.0)  # Optional cell size

# Fast neighborhood queries: O(1) average
nearby = env.spatial_index.get_nearby(
    position=(50, 50),
    radius=10.0,
    entity_types=["agents_hash"]
)

# Nearest neighbor (via grid expansion)
nearest = env.spatial_index.get_nearest(
    position=(42, 18),
    entity_types=["resources_hash"]
)

# Range queries also supported
in_rect = env.spatial_index.get_nearby_range(
    bounds=(30, 30, 20, 20),
    entity_types=["agents_hash"]
)
```

**Key Features:**
- **Uniform Grid Buckets**: Entities stored in integer (ix, iy) buckets
- **Bounded Query Cost**: Only checks buckets overlapping query region
- **Fast Dynamic Updates**: O(1) remove/insert on position changes
- **Hotspot Robust**: Performs well under non-uniform distributions
- **Simple Implementation**: Easy to debug and understand

**Performance Characteristics:**
- Build Time: ~3.12ms for 1000 entities
- Query Time: ~3.61μs per query
- Memory: ~0.2MB per 1000 entities
- Scaling: Near-constant time for local queries

**When to Use Spatial Hash:**
- Large, frequently changing populations
- Non-uniform entity distributions (clusters, crowds)
- Moderate-radius neighbor queries
- Scenarios where rebuild cost is high

---

### 2. Batch Spatial Updates

Dirty region tracking system that provides modest performance improvements for spatial updates.

#### How Batch Updates Work

Instead of rebuilding spatial indices every time an entity moves, batch updates:
1. **Track Changes**: Mark regions as "dirty" when entities move
2. **Queue Updates**: Collect multiple position changes
3. **Batch Process**: Update all dirty regions in one operation
4. **Priority-Based**: Process high-priority regions first

```python
from farm.config import SpatialIndexConfig, SimulationConfig

# Configure batch updates
spatial_config = SpatialIndexConfig(
    enable_batch_updates=True,
    region_size=50.0,                  # Size of each dirty region
    max_batch_size=100,                # Max updates per batch
    max_regions=1000,                  # Max regions to track
    dirty_region_batch_size=10,        # Regions per batch
    performance_monitoring=True
)

config = SimulationConfig(
    width=200,
    height=200,
    spatial_index_config=spatial_config
)

# Create environment with batch updates
env = Environment(width=200, height=200, config=config)

# Position updates are automatically batched
for agent in agents:
    agent.move_to(new_position)  # Queued for batch processing

# Batch is automatically processed when:
# 1. Flush interval expires (default: 1 second)
# 2. Max pending updates reached (default: 50)
# 3. Explicit flush requested

# Manual flush if needed
env.spatial_index.flush_updates()

# Check statistics
stats = env.get_spatial_performance_stats()
print(f"Batch updates: {stats['batch_updates']['total_batch_updates']}")
print(f"Avg batch size: {stats['batch_updates']['average_batch_size']:.1f}")
print(f"Efficiency gain: {stats['batch_updates']['efficiency_percentage']:.1%}")
```

#### Partial Flushing

For fine-grained control over update timing:

```python
# Process only a subset of pending updates
processed = env.spatial_index.flush_partial_updates(max_updates=25)
print(f"Processed {processed} updates")

# Check remaining
pending = len(env.spatial_index._pending_position_updates)
print(f"Remaining: {pending}")

# Use in responsive applications
total_processed = 0
while env.spatial_index.has_pending_updates():
    # Process small batch
    processed = env.spatial_index.flush_partial_updates(max_updates=10)
    total_processed += processed
    
    # Maintain responsiveness
    handle_user_input()
    render_frame()
    
print(f"Total processed: {total_processed}")
```

**Benefits:**
- **Modest Reduction** in update overhead for dynamic simulations (2-3% speedup)
- **Improved Scalability**: Performance scales better with population
- **Data Integrity**: Ensures consistent state across indices
- **Fine-Grained Control**: Partial flushing for responsive apps

**When to Use:**
- Simulations with frequent agent movement
- Large-scale environments (>1000 entities)
- Performance-critical applications
- Dynamic resource spawning/despawning

---

### 3. Multi-Index Support

Use multiple spatial indices simultaneously for optimal performance.

#### Choosing the Right Index

```python
from farm.core.environment import Environment

# Create environment with multiple indices
env = Environment(width=200, height=200)

# Enable all index types
env.enable_quadtree_indices()        # For range queries
env.enable_spatial_hash_indices()    # For fast neighbors

# Use appropriate index for each query type

# Radial query → KD-tree (default, best for radius)
allies = env.spatial_index.get_nearby(
    position=agent.position,
    radius=5.0,
    entity_types=["agents"]  # Uses KD-tree
)

# Range query → Quadtree (best for rectangles)
enemies_in_area = env.spatial_index.get_nearby_range(
    bounds=(x, y, width, height),
    entity_types=["agents_quadtree"]
)

# Frequent neighbor queries → Spatial Hash (fastest average)
nearby_resources = env.spatial_index.get_nearby(
    position=agent.position,
    radius=3.0,
    entity_types=["resources_hash"]
)
```

#### Index Selection Guide

| Query Type | Best Index | Time Complexity | Use When |
|------------|-----------|-----------------|----------|
| **Radial proximity** | KD-Tree | O(log n) | Perception, detection |
| **Nearest neighbor** | KD-Tree | O(log n) | Target selection |
| **Rectangular range** | Quadtree | O(log n) | Area effects, vision cones |
| **Frequent neighbors** | Spatial Hash | ~O(1) | Dynamic crowds |
| **Dynamic updates** | Spatial Hash | O(1) | Moving entities |

#### Custom Named Indices

Create specialized indices for specific entity types:

```python
# Define custom index for projectiles
projectile_config = {
    "index_type": "spatial_hash",
    "cell_size": 2.0,
    "enable_batch_updates": True
}

env.spatial_index.add_named_index(
    "projectiles",
    projectile_config
)

# Use custom index
nearby_projectiles = env.spatial_index.get_nearby(
    position=agent.position,
    radius=10.0,
    entity_types=["projectiles"]
)
```

---

### 4. Performance Monitoring

Comprehensive metrics and statistics for optimization.

#### Real-Time Performance Metrics

```python
# Get detailed performance statistics
stats = env.get_spatial_performance_stats()

# Query performance
print("Query Performance:")
print(f"  Total queries: {stats['queries']['total_count']}")
print(f"  Avg query time: {stats['queries']['avg_time_ms']:.3f}ms")
print(f"  Max query time: {stats['queries']['max_time_ms']:.3f}ms")
print(f"  Queries per second: {stats['queries']['queries_per_second']:.1f}")

# Update performance
print("\nUpdate Performance:")
print(f"  Total updates: {stats['updates']['total_count']}")
print(f"  Avg update time: {stats['updates']['avg_time_ms']:.3f}ms")
print(f"  Updates per second: {stats['updates']['updates_per_second']:.1f}")

# Batch update efficiency
print("\nBatch Updates:")
print(f"  Total batches: {stats['batch_updates']['total_batch_updates']}")
print(f"  Avg batch size: {stats['batch_updates']['average_batch_size']:.1f}")
print(f"  Efficiency gain: {stats['batch_updates']['efficiency_percentage']:.1%}")

# Memory usage
print("\nMemory:")
print(f"  Total memory: {stats['memory']['total_mb']:.2f}MB")
print(f"  Per entity: {stats['memory']['per_entity_kb']:.2f}KB")

# Index statistics
print("\nIndex Statistics:")
for index_name, index_stats in stats['indices'].items():
    print(f"  {index_name}:")
    print(f"    Entity count: {index_stats['entity_count']}")
    print(f"    Memory: {index_stats['memory_mb']:.2f}MB")
    print(f"    Last rebuild: {index_stats['last_rebuild_ms']:.3f}ms")
```

#### Profiling and Optimization

```python
from farm.utils import log_performance

# Profile spatial operations
@log_performance(operation_name="spatial_query", slow_threshold_ms=1.0)
def find_nearby_threats(agent):
    """Find threats near agent."""
    return env.spatial_index.get_nearby(
        position=agent.position,
        radius=agent.threat_radius,
        entity_types=["enemies"]
    )

# Automatically logs:
# - Execution time
# - Slow operations (>threshold)
# - Operation statistics

# Batch profiling
with log_simulation(simulation_id="perf_test"):
    for step in range(1000):
        with log_step(step_number=step):
            # Spatial operations logged with context
            for agent in agents:
                nearby = find_nearby_threats(agent)
                
# Analyze logs for bottlenecks
```

#### Benchmarking

Compare performance across configurations:

> **Note**: The `run_spatial_benchmark` function is planned for a future release. Currently, spatial performance benchmarking can be implemented using the existing spatial index APIs and timing utilities.

```python
# Custom spatial benchmarking implementation
import time
from farm.core.spatial.index import SpatialIndex

def run_custom_spatial_benchmark():
    """Custom spatial performance benchmark."""
    entity_counts = [100, 500, 1000, 2000]
    implementations = ["kdtree", "quadtree", "spatial_hash"]
    results = []

    for impl_name in implementations:
        for count in entity_counts:
            # Generate test entities
            entities = generate_test_entities(count)

            # Create appropriate index
            if impl_name == "kdtree":
                index = SpatialIndex.kd_tree_index()
            elif impl_name == "quadtree":
                index = SpatialIndex.quadtree_index()
            else:  # spatial_hash
                index = SpatialIndex.spatial_hash_index()

            # Benchmark build time
            start = time.perf_counter()
            for entity in entities:
                index.add_entity(entity)
            build_time = (time.perf_counter() - start) * 1000  # ms

            # Benchmark query time
            query_times = []
            for _ in range(100):
                start = time.perf_counter()
                results = index.query_radius((50, 50), 10.0)
                query_times.append((time.perf_counter() - start) * 1e6)  # μs

            avg_query_time = sum(query_times) / len(query_times)

            results.append({
                'name': f"{impl_name}_{count}",
                'build_time_ms': build_time,
                'query_time_us': avg_query_time,
                'memory_mb': estimate_memory_usage(index),
                'efficiency_score': calculate_efficiency(build_time, avg_query_time)
            })

    return results

# Run benchmark
results = run_custom_spatial_benchmark()

# Analyze results
print("Benchmark Results:")
for impl in results:
    print(f"\n{impl['name']}:")
    print(f"  Build time: {impl['build_time_ms']:.2f}ms")
    print(f"  Query time: {impl['query_time_us']:.2f}μs")
    print(f"  Memory: {impl['memory_mb']:.2f}MB")
    print(f"  Efficiency score: {impl['efficiency_score']:.3f}")

# Generate report
generate_benchmark_report(results, output="spatial_benchmark.md")
```

---

### 5. Scalable Architecture

Efficiently handle thousands of agents with minimal performance degradation.

#### Scaling Characteristics

**AgentFarm Spatial Indices - Scaling Analysis:**

| Implementation | 100 Entities | 1000 Entities | 10000 Entities | Scaling Factor |
|----------------|--------------|---------------|----------------|----------------|
| **KD-Tree Build** | 0.35ms | 1.41ms | 19.2ms | 0.94 log(n) |
| **KD-Tree Query** | 4.46μs | 5.02μs | 18.6μs | 0.87 log(n) |
| **Quadtree Build** | 0.91ms | 9.15ms | 197ms | 1.02 log(n) |
| **Quadtree Query** | 3.10μs | 7.62μs | 141μs | 0.95 log(n) |
| **Hash Build** | 0.45ms | 3.48ms | 45ms | 0.89 linear |
| **Hash Query** | 2.69μs | 3.39μs | 45μs | 0.71 const |

#### Performance vs Industry Standards

**Compared to SciPy and Scikit-learn:**

| Metric | AgentFarm KD-Tree | SciPy KD-Tree | Scikit-learn |
|--------|-------------------|---------------|--------------|
| **Build Time** | 1.26ms (avg) | 0.23ms ⭐ | 0.40ms |
| **Query Time** | 4.85μs (avg) | 3.95μs ⭐ | 23.48μs |
| **Memory** | 0.1MB/1000 | 0.0MB ⭐ | 0.0MB |
| **Batch Updates** | 2-3% speedup | N/A | N/A |
| **Multi-Index** | Yes ⭐ | No | No |

**Key Advantages:**
- **Query Performance**: Beats Scikit-learn (4.85μs vs 23.48μs)
- **Batch Updates**: Provides incremental improvement for dynamic simulations
- **Flexibility**: Multiple index types for different query patterns
- **Specialization**: Quadtree optimized for range queries

#### Optimization Strategies

```python
# 1. Use appropriate index for query type
# Radial → KD-tree
nearby = env.spatial_index.get_nearby(pos, radius, ["agents"])

# Range → Quadtree
in_area = env.spatial_index.get_nearby_range(bounds, ["agents_quadtree"])

# Frequent → Spatial Hash
neighbors = env.spatial_index.get_nearby(pos, radius, ["agents_hash"])

# 2. Enable batch updates
config = SpatialIndexConfig(
    enable_batch_updates=True,
    max_batch_size=100
)

# 3. Tune flush policies
config.flush_interval_seconds = 0.5  # More frequent flushes
config.max_pending_updates_before_flush = 25  # Smaller batches

# 4. Use partial flushing for responsiveness
while has_pending:
    env.spatial_index.flush_partial_updates(max_updates=10)
    yield_control()

# 5. Cache frequent queries
class Agent:
    def __init__(self):
        self._nearby_cache = {}
        self._cache_timestamp = 0
        
    def get_nearby(self, radius):
        # Cache valid for 10 steps
        if self.env.current_step - self._cache_timestamp > 10:
            self._nearby_cache = self.env.spatial_index.get_nearby(
                self.position, radius, ["agents"]
            )
            self._cache_timestamp = self.env.current_step
        return self._nearby_cache

# 6. Use allow_stale_reads for non-critical queries
# Faster, but may return slightly outdated results
nearby = env.spatial_index.get_nearby(
    position=agent.position,
    radius=10.0,
    entity_types=["agents"],
    allow_stale_reads=True  # Skip batch flush
)
```

---

## Advanced Usage

### Priority-Based Updates

Control update processing order:

```python
from farm.core.spatial.index import (
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL
)

# Set entity priority
env.spatial_index.update_position(
    entity=background_agent,
    old_position=old_pos,
    new_position=new_pos,
    priority=PRIORITY_LOW  # Process last
)

env.spatial_index.update_position(
    entity=player,
    old_position=old_pos,
    new_position=new_pos,
    priority=PRIORITY_CRITICAL  # Process first
)

# Higher priority updates processed first in batch
```

### Custom Query Filters

Create specialized spatial queries:

```python
def find_vulnerable_targets(self, agent, max_range):
    """Find low-health enemies within range."""
    
    # Get all nearby entities
    nearby = env.spatial_index.get_nearby(
        position=agent.position,
        radius=max_range,
        entity_types=["agents"]
    )
    
    # Filter for vulnerable enemies
    vulnerable = []
    for aid in nearby:
        target = env.get_agent(aid)
        if (target.team != agent.team and 
            target.health < target.max_health * 0.3):
            vulnerable.append(aid)
    
    return vulnerable

def find_resource_clusters(self, min_cluster_size=3):
    """Find areas with high resource density."""
    
    clusters = []
    checked = set()
    
    for resource in env.resources.values():
        if resource.id in checked:
            continue
            
        # Find nearby resources
        nearby = env.spatial_index.get_nearby(
            position=resource.position,
            radius=5.0,
            entity_types=["resources"]
        )
        
        if len(nearby) >= min_cluster_size:
            # Found a cluster
            cluster = {
                'center': resource.position,
                'resources': nearby,
                'density': len(nearby) / (math.pi * 5.0 ** 2)
            }
            clusters.append(cluster)
            checked.update(nearby)
    
    return clusters
```

### Spatial Analysis

Analyze spatial patterns:

```python
def analyze_spatial_distribution(env):
    """Analyze entity distribution patterns."""
    
    # Divide space into grid
    grid_size = 10
    grid = defaultdict(int)
    
    for agent in env.agents.values():
        grid_x = int(agent.position[0] // grid_size)
        grid_y = int(agent.position[1] // grid_size)
        grid[(grid_x, grid_y)] += 1
    
    # Calculate metrics
    densities = list(grid.values())
    
    analysis = {
        'mean_density': np.mean(densities),
        'max_density': np.max(densities),
        'std_density': np.std(densities),
        'num_hotspots': sum(1 for d in densities if d > np.mean(densities) + np.std(densities)),
        'clustering_coefficient': calculate_clustering(env)
    }
    
    return analysis

def calculate_clustering(env, radius=10.0):
    """Calculate how clustered entities are."""
    
    clustering_scores = []
    
    for agent in env.agents.values():
        nearby = env.spatial_index.get_nearby(
            position=agent.position,
            radius=radius,
            entity_types=["agents"]
        )
        
        # Score based on local density
        density = len(nearby) / (math.pi * radius ** 2)
        clustering_scores.append(density)
    
    return np.mean(clustering_scores)
```

---

## Performance Benchmarks

### Real-World Performance

**Test Configuration:**
- Environment: 1000×1000 world
- Entities: 100-10,000
- Queries: 100 radius + 50 nearest
- Hardware: Standard development machine

**Results:**

| Entity Count | KD-Tree Query | Quadtree Query | Hash Query | Build Time |
|--------------|---------------|----------------|------------|------------|
| 100 | 4.46μs | 3.10μs | 2.69μs | 0.35ms |
| 500 | 4.78μs | 5.40μs | 3.10μs | 0.78ms |
| 1,000 | 5.02μs | 7.62μs | 3.39μs | 1.41ms |
| 2,000 | 5.12μs | 10.94μs | 5.28μs | 2.50ms |
| 10,000 | 18.6μs | 141μs | 45μs | 19.2ms |

**Batch Update Performance:**

Based on actual benchmark results, batch updates provide modest improvements:

| Scenario | Batch Time | Individual Time | Measured Speedup |
|----------|------------|-----------------|------------------|
| 100 entities | ~0.46ms | ~1.10ms | **2.4%** |
| 500 entities | ~1.05ms | ~2.66ms | **2.5%** |
| 1000 entities | ~1.56ms | ~5.42ms | **3.5%** |

---

## Example: Complete Performance Optimization

```python
#!/usr/bin/env python3
"""
Complete spatial indexing performance optimization example.
Demonstrates all optimization techniques.
"""

from farm.config import SimulationConfig, SpatialIndexConfig
from farm.core.simulation import run_simulation
from farm.core.environment import Environment
import time

def main():
    print("=== Spatial Indexing Performance Demo ===\n")
    
    # Step 1: Configure optimized spatial indexing
    print("Step 1: Configuring spatial indices...")
    
    spatial_config = SpatialIndexConfig(
        # Batch updates for dynamic environments
        enable_batch_updates=True,
        region_size=50.0,
        max_batch_size=100,
        
        # Performance monitoring
        performance_monitoring=True,
        
        # Responsive flush policy
        flush_interval_seconds=0.5,
        max_pending_updates_before_flush=25,
        dirty_region_batch_size=10
    )
    
    config = SimulationConfig(
        width=200,
        height=200,
        system_agents=500,
        independent_agents=500,
        max_steps=1000,
        spatial_index_config=spatial_config,
        seed=42
    )
    
    # Step 2: Create environment with multiple indices
    print("\nStep 2: Initializing environment...")
    
    env = Environment(
        width=config.width,
        height=config.height,
        config=config
    )
    
    # Enable all index types
    env.enable_quadtree_indices()
    env.enable_spatial_hash_indices(cell_size=5.0)
    
    print("  Enabled indices:")
    print("    - KD-Tree (default, radial queries)")
    print("    - Quadtree (range queries)")
    print("    - Spatial Hash (neighbor queries)")
    
    # Step 3: Run simulation with performance tracking
    print("\nStep 3: Running simulation...")
    
    start_time = time.time()
    results = run_simulation(config)
    elapsed = time.time() - start_time
    
    print(f"  Simulation complete in {elapsed:.2f}s")
    print(f"  Final population: {results['surviving_agents']}")
    
    # Step 4: Analyze spatial performance
    print("\nStep 4: Analyzing spatial performance...")
    
    stats = env.get_spatial_performance_stats()
    
    print("\n  Query Performance:")
    print(f"    Total queries: {stats['queries']['total_count']:,}")
    print(f"    Avg time: {stats['queries']['avg_time_ms']:.3f}ms")
    print(f"    Throughput: {stats['queries']['queries_per_second']:.1f} queries/sec")
    
    print("\n  Batch Update Efficiency:")
    print(f"    Total batches: {stats['batch_updates']['total_batch_updates']}")
    print(f"    Avg batch size: {stats['batch_updates']['average_batch_size']:.1f}")
    print(f"    Efficiency gain: {stats['batch_updates']['efficiency_percentage']:.1%}")
    
    print("\n  Memory Usage:")
    print(f"    Total: {stats['memory']['total_mb']:.2f}MB")
    print(f"    Per entity: {stats['memory']['per_entity_kb']:.2f}KB")
    
    # Step 5: Demonstrate different query types
    print("\nStep 5: Query performance comparison...")
    
    test_position = (100, 100)
    test_radius = 20.0
    num_tests = 1000
    
    # KD-Tree radial query
    start = time.perf_counter()
    for _ in range(num_tests):
        nearby_kd = env.spatial_index.get_nearby(
            test_position, test_radius, ["agents"]
        )
    kdtree_time = (time.perf_counter() - start) / num_tests * 1000000
    
    # Quadtree range query
    bounds = (90, 90, 20, 20)
    start = time.perf_counter()
    for _ in range(num_tests):
        nearby_quad = env.spatial_index.get_nearby_range(
            bounds, ["agents_quadtree"]
        )
    quadtree_time = (time.perf_counter() - start) / num_tests * 1000000
    
    # Spatial Hash query
    start = time.perf_counter()
    for _ in range(num_tests):
        nearby_hash = env.spatial_index.get_nearby(
            test_position, test_radius, ["agents_hash"]
        )
    hash_time = (time.perf_counter() - start) / num_tests * 1000000
    
    print(f"  KD-Tree (radial): {kdtree_time:.2f}μs")
    print(f"  Quadtree (range): {quadtree_time:.2f}μs")
    print(f"  Spatial Hash: {hash_time:.2f}μs")
    
    # Step 6: Recommendations
    print("\n=== Performance Recommendations ===")
    
    if stats['batch_updates']['efficiency_percentage'] > 0.5:
        print("  ✓ Batch updates working efficiently")
    else:
        print("  ⚠ Consider enabling batch updates")
    
    if stats['queries']['avg_time_ms'] < 0.1:
        print("  ✓ Query performance excellent")
    elif stats['queries']['avg_time_ms'] < 1.0:
        print("  ✓ Query performance good")
    else:
        print("  ⚠ Query performance could be improved")
    
    if stats['memory']['per_entity_kb'] < 1.0:
        print("  ✓ Memory usage efficient")
    else:
        print("  ⚠ High memory per entity")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
```

---

## Additional Resources

### Documentation
- [Spatial Indexing Details](spatial/spatial_indexing.md) - Technical implementation
- [Performance Summary](spatial/spatial_module_performance_summary.md) - Benchmark results
- [Configuration Guide](config/README.md) - Spatial config options

### Benchmarks
- [Spatial Benchmark Report](../benchmarks/reports/0.1.0/spatial_benchmark_report.md)
- [Benchmark Guide](../BENCHMARK_GUIDE.md)

### Implementation
- `farm/core/spatial/index.py` - Main spatial index
- `farm/core/spatial/quadtree.py` - Quadtree implementation
- `farm/core/spatial/hash_grid.py` - Spatial hash grid
- `farm/core/spatial/dirty_regions.py` - Batch update system

---

## Support

For spatial indexing questions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [Full documentation index](README.md)
- **Benchmarks**: Check `benchmarks/` directory for performance data

---

**Ready to optimize your simulations?** Start with [KD-Tree Indexing](#kd-tree-indexing) or explore [Batch Updates](#batch-spatial-updates) for maximum performance!
