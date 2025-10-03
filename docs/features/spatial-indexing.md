# Spatial Indexing & Performance

## Overview

AgentFarm implements advanced spatial indexing techniques to efficiently handle proximity queries and spatial operations in large-scale simulations. The system provides multiple index types optimized for different query patterns and includes sophisticated optimization strategies to minimize computational overhead.

Spatial queries - finding nearby agents, identifying neighbors within a radius, locating agents in a region - are fundamental operations in spatial agent-based models. Without optimization, these queries require checking every agent against every other, resulting in O(n²) complexity that becomes prohibitive as populations grow. Spatial indexing reduces this to O(log n) or even O(1) in ideal cases, making simulations with tens of thousands of agents practical.

## Spatial Index Types

AgentFarm provides multiple spatial index types, each optimized for different scenarios. The choice of index depends on your query patterns, spatial distribution, and update frequency.

### KD-Tree

KD-trees use binary space partitioning for efficient nearest-neighbor queries. They work by recursively dividing space along different dimensions, creating a balanced tree structure that enables fast spatial searches.

```python
from farm.spatial import KDTree

# Create KD-tree index
kdtree = KDTree(dimensions=2)

# Insert agents
for agent in agents:
    kdtree.insert(agent.position, agent.id)

# Nearest neighbor query
nearest = kdtree.nearest_neighbor(query_point=(50, 50))

# K-nearest neighbors
k_nearest = kdtree.k_nearest_neighbors(
    query_point=(50, 50),
    k=10
)

# Radial query
in_radius = kdtree.query_radius(
    center=(50, 50),
    radius=15
)
```

KD-trees excel at nearest-neighbor queries and work well for static or slowly changing agent populations. They're particularly effective when the spatial distribution is relatively uniform and when you primarily need to find closest neighbors rather than all neighbors within a region.

### Quadtree

Quadtrees use hierarchical grid structure for dynamic spatial partitioning. They recursively subdivide space into four quadrants, creating finer divisions in dense regions while keeping sparse regions coarse.

```python
from farm.spatial import Quadtree

# Create quadtree with bounds
quadtree = Quadtree(
    bounds=(0, 0, 100, 100),
    max_depth=8,
    max_objects=4
)

# Insert agents
for agent in agents:
    quadtree.insert(agent.position, agent.id, agent)

# Range query
in_range = quadtree.query_range(
    x_min=40, y_min=40,
    x_max=60, y_max=60
)

# Radial query
in_radius = quadtree.query_radius(
    center=(50, 50),
    radius=15
)
```

Quadtrees are excellent for dynamic simulations where agents frequently move and where spatial density varies significantly across the environment. They automatically adapt to density patterns, providing efficient queries in both dense and sparse regions.

### Spatial Hash Grid

Spatial hash grids provide fast uniform grid-based lookups. They divide space into a regular grid of cells and use hashing to quickly locate which cell any point falls into.

```python
from farm.spatial import SpatialHashGrid

# Create spatial hash grid
grid = SpatialHashGrid(
    cell_size=10,
    bounds=(0, 0, 100, 100)
)

# Insert agents
for agent in agents:
    grid.insert(agent.position, agent.id)

# Query cell
agents_in_cell = grid.query_cell(50, 50)

# Query radius (checks nearby cells)
agents_nearby = grid.query_radius(
    center=(50, 50),
    radius=15
)

# Get neighboring cells
neighbors = grid.get_neighbors(cell=(5, 5))
```

Spatial hash grids offer O(1) query time in ideal cases and are particularly effective when the spatial distribution is relatively uniform and when you primarily query local neighborhoods. They're also simple to implement and understand, making debugging easier.

## Batch Spatial Updates

One of the most significant performance optimizations in AgentFarm is batch spatial updates using dirty region tracking. Instead of rebuilding the entire spatial index every time an agent moves, the system tracks which regions have changed and only updates those regions.

```python
from farm.spatial import SpatialIndexManager

# Create manager with dirty tracking
manager = SpatialIndexManager(
    index_type='quadtree',
    enable_dirty_tracking=True
)

# Update agent positions
for agent in agents:
    old_position = agent.previous_position
    new_position = agent.position
    
    # Manager tracks dirty regions
    manager.update_agent(agent.id, old_position, new_position)

# Rebuild only dirty regions
manager.rebuild_dirty_regions()

# Query (uses optimized index)
nearby = manager.query_radius(center=(50, 50), radius=15)
```

This dirty region tracking can reduce computational overhead by up to 70% compared to full index rebuilds, especially in scenarios where only a fraction of agents move each timestep or where agents move short distances that keep them in the same spatial regions.

Lazy updates take this further by deferring updates until queries are performed. If multiple agents move but no queries occur, no index updates happen. When a query finally occurs, the system updates only the portions of the index needed to answer that query.

```python
from farm.spatial import LazySpatialIndex

# Create lazy index
index = LazySpatialIndex(
    base_index_type='kdtree',
    update_threshold=100  # Update after 100 changes
)

# Make changes (not immediately applied)
for agent in agents:
    index.mark_moved(agent.id, agent.position)

# Query triggers update if needed
nearby = index.query_radius((50, 50), 15)
```

Incremental updates modify the index directly rather than rebuilding from scratch. When an agent moves, the system removes it from its old position and inserts it at its new position, avoiding the cost of processing all agents.

## Index Selection

Choosing the optimal index for your use case significantly affects performance. AgentFarm can help guide this choice through analysis of your usage patterns.

```python
from farm.spatial import SpatialIndexSelector

selector = SpatialIndexSelector()

# Recommend index based on usage pattern
recommendation = selector.recommend_index(
    num_agents=1000,
    query_types=['radius', 'nearest_neighbor'],
    update_frequency='high',
    spatial_distribution='clustered'
)

print(f"Recommended index: {recommendation.index_type}")
print(f"Expected performance: {recommendation.performance_estimate}")
```

General guidelines for index selection:

KD-trees work best for nearest-neighbor queries with static or slowly changing agent populations. They provide excellent query performance but can be expensive to rebuild.

Quadtrees excel in dynamic simulations with varying density. They automatically adapt to spatial structure and handle frequent updates efficiently.

Spatial hash grids offer the best performance when distribution is relatively uniform and when you primarily query local neighborhoods. They're simple and provide O(1) lookups in ideal cases.

## Performance Monitoring

AgentFarm provides tools for monitoring and analyzing spatial index performance, helping you identify bottlenecks and optimize your simulations.

```python
from farm.spatial import SpatialIndexProfiler

# Enable profiling
profiler = SpatialIndexProfiler(index)

# Perform queries
for _ in range(1000):
    index.query_radius((50, 50), 15)

# Get statistics
stats = profiler.get_statistics()
print(f"Average query time: {stats.avg_query_time_ms} ms")
print(f"Total queries: {stats.total_queries}")
print(f"Cache hit rate: {stats.cache_hit_rate}")
```

Index statistics provide insights into structure and memory usage:

```python
# Get index statistics
stats = index.get_statistics()

print(f"Total agents: {stats.total_agents}")
print(f"Index depth: {stats.depth}")
print(f"Memory usage: {stats.memory_bytes / 1024 / 1024} MB")
print(f"Load factor: {stats.load_factor}")
```

Benchmarking tools compare performance across different index types and configurations, helping you make informed choices about which approach works best for your specific scenario.

## Scalability

AgentFarm's spatial indexing system is designed to scale efficiently to large populations. With appropriate index selection and optimization, simulations with tens of thousands of agents maintain reasonable performance.

```python
from farm.spatial import ScalableSpatialIndex

# Create scalable index
index = ScalableSpatialIndex(
    index_type='adaptive',  # Automatically switches based on load
    max_agents=50000,
    optimization_level='high'
)

# Add agents
for agent in agents:
    index.insert(agent.id, agent.position)

# Automatic optimization
index.optimize()  # Rebalances and optimizes structure

# Efficient queries even with 50k agents
nearby = index.query_radius((50, 50), 15)
```

Parallel queries can execute concurrent spatial queries for even better performance in multi-threaded scenarios. Dynamic adaptation automatically adjusts index structure based on population density and query patterns, ensuring optimal performance across simulation phases.

## Best Practices

Choose index types appropriate for your scenario. KD-trees for static populations with nearest-neighbor queries. Quadtrees for dynamic simulations with varying density. Spatial hash grids for uniform distributions with local queries.

Use batch updates when possible to minimize overhead. Group agent updates and rebuild indexes once rather than after each individual movement. Enable dirty tracking to update only changed regions.

Profile performance to identify bottlenecks. Use the profiling tools to measure actual query times and identify optimization opportunities. Compare different index types for your specific scenario rather than assuming one is universally better.

## Related Documentation

For detailed information, see [Spatial Indexing Guide](../spatial/spatial_indexing.md), [Batch Spatial Updates Guide](../spatial/batch_spatial_updates_guide.md), [Spatial Benchmark Report](../spatial/direct_spatial_benchmark_report.md), [Spatial Module Performance Summary](../spatial/spatial_module_performance_summary.md), and [Core Architecture](../core_architecture.md).

## Examples

Practical examples can be found in [Spatial Benchmark README](../spatial/spatial_benchmark_README.md), [Usage Examples](../usage_examples.md), and [Performance Monitoring](../monitoring.md).
