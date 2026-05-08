# Spatial Indexing & Performance

![Feature](https://img.shields.io/badge/feature-spatial%20%26%20performance-brightgreen)

## Table of Contents

1. [Overview](#overview)
   - [Why Spatial Indexing Matters](#why-spatial-indexing-matters)
2. [Core Capabilities](#core-capabilities)
   - [1. Advanced Spatial Indexing](#1-advanced-spatial-indexing)
     - [KD-Tree Indexing](#kd-tree-indexing)
     - [Quadtree Indexing](#quadtree-indexing)
     - [Spatial Hash Grid Indexing](#spatial-hash-grid-indexing)
   - [2. Batch Spatial Updates](#2-batch-spatial-updates)
     - [How Batch Updates Work](#how-batch-updates-work)
     - [Partial Flushing](#partial-flushing)
   - [3. Multi-Index Support](#3-multi-index-support)
     - [Choosing the Right Index](#choosing-the-right-index)
     - [Index Selection Guide](#index-selection-guide)
     - [Custom Named Indices](#custom-named-indices)
   - [4. Performance Monitoring](#4-performance-monitoring)
     - [Real-Time Performance Metrics](#real-time-performance-metrics)
     - [Profiling and Optimization](#profiling-and-optimization)
     - [Benchmarking](#benchmarking)
   - [5. Scalable architecture](#5-scalable-architecture)
     - [Optimization strategies](#optimization-strategies)
3. [Advanced Usage](#advanced-usage)
   - [Priority-Based Updates](#priority-based-updates)
   - [Custom Query Filters](#custom-query-filters)
   - [Spatial Analysis](#spatial-analysis)
4. [Performance benchmarks](#performance-benchmarks)
5. [Example: complete performance optimization](#example-complete-performance-optimization)
6. [Additional Resources](#additional-resources)
   - [Documentation](#documentation)
   - [Benchmarks](#benchmarks)
   - [Implementation](#implementation)
7. [Support](#support)

---

## Verified performance (repository artifacts)

Quantitative timings belong in **committed benchmark output**, not hand-edited prose.

| Artifact | Purpose |
|----------|---------|
| [`benchmarks/results/spatial_benchmark_verified.json`](../../benchmarks/results/spatial_benchmark_verified.json) | Raw timings, batch microbenchmark rows, and host metadata |
| [`benchmarks/results/SPATIAL_BENCHMARK_VERIFIED.md`](../../benchmarks/results/SPATIAL_BENCHMARK_VERIFIED.md) | Markdown tables generated from that JSON |

Regenerate from the repository root (after `pip install -e .`):

```bash
PYTHONHASHSEED=0 python benchmarks/implementations/spatial/comprehensive_spatial_benchmark.py --verified
```

`tests/benchmarks/test_spatial_benchmark_verified_artifact.py` asserts that both files exist and that the JSON contains the expected schema (including fixed RNG seed `42`). When you intentionally change spatial performance characteristics, re-run the command above and commit the refreshed artifacts alongside your code.

---

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
- **Sub-linear Query Time**: Typical tree-backed radius and nearest-neighbor queries scale roughly as O(log n) in many regimes (depends on radius, density, and implementation).
- **Continuous Positions**: Supports floating-point coordinates
- **Automatic Cache Invalidation**: Rebuilds only when positions change
- **Multi-Entity Support**: Separate trees for agents, resources, obstacles

**Measured performance:** Do not rely on hand-copied microsecond figures in feature docs. Run the deterministic harness and read the committed artifacts ([`benchmarks/results/spatial_benchmark_verified.json`](../../benchmarks/results/spatial_benchmark_verified.json) and [`benchmarks/results/SPATIAL_BENCHMARK_VERIFIED.md`](../../benchmarks/results/SPATIAL_BENCHMARK_VERIFIED.md)); see [Verified performance](#verified-performance-repository-artifacts).

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

**Measured performance:** Use the verified benchmark artifacts linked above (same harness for all index types).

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

**Measured performance:** Use the verified benchmark artifacts linked above.

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

#### Custom named indices

Register a dedicated index over your own entity list (here, hypothetical `projectiles`):

```python
projectiles = []  # populate with objects exposing .position

env.spatial_index.register_index(
    "projectiles",
    data_reference=projectiles,
    position_getter=lambda p: p.position,
    filter_func=lambda p: getattr(p, "alive", True),
    index_type="spatial_hash",
    cell_size=2.0,
)
env.spatial_index.update()

nearby_projectiles = env.spatial_index.get_nearby(
    position=agent.position,
    radius=10.0,
    entity_types=["projectiles"],
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

Use the checked-in harness described in [Verified performance](#verified-performance-repository-artifacts). It compares AgentFarm’s `SpatialIndex` orchestration (KD-tree, quadtree, and spatial-hash modes) against SciPy `cKDTree` and scikit-learn tree queries under identical synthetic workloads, records memory from `tracemalloc`, and appends a small batch-update microbenchmark.

For ad hoc experiments in notebooks, wrap `get_nearby` / `get_nearest` with `time.perf_counter()` and always record entity count, world size, query radius distribution, and hardware.

---

### 5. Scalable architecture

Large populations benefit from choosing the right index and tuning batch flush policy; see the verified tables for how build and query costs grew on the reference machine when scaling from 100 to 2 000 entities.

#### Optimization strategies

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

# Set entity priority (batch queue; use entity_type matching your index)
env.spatial_index.add_position_update(
    background_agent,
    old_pos,
    new_pos,
    entity_type="agent",
    priority=PRIORITY_LOW,
)

env.spatial_index.add_position_update(
    player,
    old_pos,
    new_pos,
    entity_type="agent",
    priority=PRIORITY_CRITICAL,
)

# Higher priority updates are processed first when the batch flushes.
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

## Performance benchmarks

Timings for the bundled harness (100–2 000 entities, uniform and clustered layouts, SciPy and scikit-learn baselines) live in the verified artifacts linked at the top of this page. Open [`SPATIAL_BENCHMARK_VERIFIED.md`](../../benchmarks/results/SPATIAL_BENCHMARK_VERIFIED.md) for tables; use the JSON if you are plotting regressions.

---

## Example: complete performance optimization

```python
#!/usr/bin/env python3
"""Short simulation with multi-index setup; pair with the verified harness for hard numbers."""

import time

from farm.config import SimulationConfig, SpatialIndexConfig
from farm.core.environment import Environment
from farm.core.simulation import run_simulation


def main() -> None:
    spatial_config = SpatialIndexConfig(
        enable_batch_updates=True,
        region_size=50.0,
        max_batch_size=100,
        performance_monitoring=True,
        dirty_region_batch_size=10,
    )
    config = SimulationConfig(seed=42, max_steps=120)
    config.environment.width = 200
    config.environment.height = 200
    config.environment.spatial_index = spatial_config
    config.population.system_agents = 40
    config.population.independent_agents = 40

    env = Environment(
        width=config.environment.width,
        height=config.environment.height,
        config=config,
    )
    env.enable_quadtree_indices()
    env.enable_spatial_hash_indices(5.0)

    t0 = time.perf_counter()
    env_after = run_simulation(config.max_steps, config)
    elapsed = time.perf_counter() - t0
    print(f"Simulation finished in {elapsed:.2f}s")

    stats = env_after.get_spatial_performance_stats()
    print("Tracked spatial queries:", stats["queries"]["total_count"])

    # Local micro-sample (not the committed benchmark artifact)
    pos, radius, n = (100.0, 100.0), 20.0, 200
    t1 = time.perf_counter()
    for _ in range(n):
        env_after.spatial_index.get_nearby(pos, radius, ["agents"])
    per_query_us = (time.perf_counter() - t1) / n * 1e6
    print(f"Sample get_nearby mean: {per_query_us:.2f} μs ({n} iterations)")


if __name__ == "__main__":
    main()
```

---

## Additional Resources

### Documentation
- [Spatial Indexing Details](../spatial/spatial_indexing.md) - Technical implementation
- [Verified tables (generated)](../../benchmarks/results/SPATIAL_BENCHMARK_VERIFIED.md) - Committed benchmark output
- [Configuration Guide](../config/configuration_guide.md) - Spatial config options

### Benchmarks
- [`spatial_benchmark_verified.json`](../../benchmarks/results/spatial_benchmark_verified.json) - Raw verified timings
- [Historical report (0.1.0 snapshot)](../../benchmarks/reports/0.1.0/spatial_benchmark_report.md) - Legacy narrative; prefer verified artifacts above
- [Benchmarks](../../benchmarks/README.md)

### Implementation
- `farm/core/spatial/index.py` - Main spatial index
- `farm/core/spatial/quadtree.py` - Quadtree implementation
- `farm/core/spatial/hash_grid.py` - Spatial hash grid
- `farm/core/spatial/dirty_regions.py` - Batch update system

---

## Support

For spatial indexing questions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [Full documentation index](../README.md)
- **Benchmarks**: Check `benchmarks/` directory for performance data

---

**Ready to optimize your simulations?** Start with [KD-Tree Indexing](#kd-tree-indexing) or explore [Batch Updates](#batch-spatial-updates) for maximum performance!
