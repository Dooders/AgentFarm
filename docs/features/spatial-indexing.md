# Spatial Indexing & Performance

## Overview

AgentFarm implements advanced spatial indexing techniques to efficiently handle proximity queries and spatial operations in large-scale simulations. The system provides multiple index types optimized for different query patterns and includes sophisticated optimization strategies to minimize computational overhead.

## Key Capabilities

### Advanced Spatial Indexing
- **KD-Tree**: Binary space partitioning for efficient nearest-neighbor queries
- **Quadtree**: Hierarchical grid structure for dynamic spatial queries
- **Spatial Hash Grid**: Fast uniform grid-based lookups
- **R-Tree**: Bounding box-based indexing for range queries

### Batch Spatial Updates
- **Dirty Region Tracking**: Only update changed regions
- **Lazy Evaluation**: Defer updates until queries are performed
- **Incremental Updates**: Update index incrementally rather than rebuilding
- **Performance Gains**: Reduce computational overhead by up to 70%

### Multi-Index Support
- **Index Selection**: Choose optimal index for query pattern
- **Radial Queries**: Find agents within radius
- **Range Queries**: Find agents in rectangular region
- **Nearest Neighbor**: Find k-nearest neighbors

### Performance Monitoring
- **Query Metrics**: Track query performance statistics
- **Index Statistics**: Monitor index size and structure
- **Profiling Tools**: Identify bottlenecks in spatial operations
- **Optimization Recommendations**: Automated suggestions for improvement

### Scalable Architecture
- **Efficient Scaling**: Handle thousands of agents with minimal degradation
- **Memory Efficiency**: Optimize memory usage for large populations
- **Parallel Queries**: Support concurrent spatial queries
- **Dynamic Adaptation**: Automatically adjust to population density

## Spatial Index Types

### KD-Tree

Binary space partitioning for balanced nearest-neighbor queries:

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

### Quadtree

Hierarchical grid structure for dynamic spatial partitioning:

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

# Get all objects in region
objects = quadtree.get_objects_in_region(
    x=50, y=50,
    width=20, height=20
)
```

### Spatial Hash Grid

Fast uniform grid-based lookups:

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

### R-Tree

Bounding box-based indexing for range queries:

```python
from farm.spatial import RTree

# Create R-tree
rtree = RTree()

# Insert agents with bounding boxes
for agent in agents:
    rtree.insert(
        agent.id,
        bbox=(
            agent.x - agent.radius,
            agent.y - agent.radius,
            agent.x + agent.radius,
            agent.y + agent.radius
        )
    )

# Range query
in_range = rtree.query_range(
    bbox=(40, 40, 60, 60)
)

# Intersection query
intersecting = rtree.query_intersection(
    bbox=(45, 45, 55, 55)
)
```

## Batch Spatial Updates

### Dirty Region Tracking

Only update changed regions:

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

### Lazy Updates

Defer updates until queries are performed:

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

### Incremental Updates

Update index incrementally:

```python
from farm.spatial import IncrementalSpatialIndex

# Create incremental index
index = IncrementalSpatialIndex(index_type='quadtree')

# Incremental updates
for agent in moved_agents:
    # Remove from old position
    index.remove(agent.id, agent.old_position)
    
    # Insert at new position
    index.insert(agent.id, agent.position)

# Index stays balanced and efficient
```

## Performance Optimization

### Index Selection

Choose optimal index for your use case:

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

### Query Optimization

Optimize spatial queries:

```python
# Batch queries together
queries = [
    ('radius', (50, 50), 15),
    ('radius', (60, 60), 15),
    ('radius', (70, 70), 15)
]

# Execute batch query
results = index.batch_query(queries)

# Use query caching for repeated queries
index.enable_query_cache(max_size=1000)
nearby_1 = index.query_radius((50, 50), 15)  # Computed
nearby_2 = index.query_radius((50, 50), 15)  # Cached
```

### Memory Optimization

Reduce memory usage:

```python
# Use compact representation
index = SpatialHashGrid(
    cell_size=10,
    bounds=(0, 0, 100, 100),
    compact_mode=True  # Store only IDs, not full objects
)

# Prune empty regions
quadtree.prune_empty()

# Use memory-mapped storage for very large datasets
from farm.spatial import MemoryMappedSpatialIndex

mm_index = MemoryMappedSpatialIndex(
    index_file='spatial_index.mmap',
    max_agents=100000
)
```

## Performance Monitoring

### Query Metrics

Track query performance:

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

### Index Statistics

Monitor index structure:

```python
# Get index statistics
stats = index.get_statistics()

print(f"Total agents: {stats.total_agents}")
print(f"Index depth: {stats.depth}")
print(f"Memory usage: {stats.memory_bytes / 1024 / 1024} MB")
print(f"Load factor: {stats.load_factor}")
```

### Benchmarking

Compare index performance:

```python
from farm.spatial.benchmarks import SpatialIndexBenchmark

# Create benchmark
benchmark = SpatialIndexBenchmark()

# Test different indices
results = benchmark.compare_indices(
    index_types=['kdtree', 'quadtree', 'spatial_hash'],
    num_agents=[100, 1000, 10000],
    query_types=['radius', 'range', 'nearest'],
    num_queries=1000
)

# Generate report
benchmark.generate_report(results, output='benchmark_report.html')
```

## Scalability

### Large-Scale Simulations

Handle thousands of agents efficiently:

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

### Parallel Queries

Execute concurrent spatial queries:

```python
from farm.spatial import ParallelSpatialIndex
import concurrent.futures

# Create thread-safe index
index = ParallelSpatialIndex(base_index='quadtree')

# Parallel query execution
query_points = [(x, y) for x in range(0, 100, 10) 
                        for y in range(0, 100, 10)]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(index.query_radius, point, 15)
        for point in query_points
    ]
    
    results = [f.result() for f in futures]
```

### Dynamic Adaptation

Automatically adapt to population density:

```python
from farm.spatial import AdaptiveSpatialIndex

# Create adaptive index
index = AdaptiveSpatialIndex(
    initial_index='spatial_hash',
    adaptation_threshold=1000,
    density_monitoring=True
)

# Index automatically adapts
for agent in agents:
    index.insert(agent.id, agent.position)

# Switches to quadtree if density becomes non-uniform
if index.current_index_type == 'quadtree':
    print("Switched to quadtree for better performance")
```

## Best Practices

### Index Selection Guidelines

- **KD-Tree**: Best for nearest-neighbor queries, static or slowly changing agents
- **Quadtree**: Best for dynamic simulations with varying density
- **Spatial Hash Grid**: Best for uniform distributions and fast lookups
- **R-Tree**: Best for objects with extent (not just points)

### Update Strategies

- **High Frequency Updates**: Use dirty tracking and incremental updates
- **Batch Updates**: Group updates together and rebuild once
- **Mixed Pattern**: Use adaptive indexing to automatically optimize

### Query Patterns

- **Few Queries**: Build index once, query multiple times
- **Many Queries**: Use query caching and batch queries
- **Mixed Pattern**: Profile and optimize based on actual usage

## Related Documentation

- [Spatial Indexing Guide](../spatial/spatial_indexing.md)
- [Batch Spatial Updates Guide](../spatial/batch_spatial_updates_guide.md)
- [Spatial Benchmark Report](../spatial/direct_spatial_benchmark_report.md)
- [Spatial Module Performance Summary](../spatial/spatial_module_performance_summary.md)
- [Core Architecture](../core_architecture.md)

## Examples

For practical examples:
- [Spatial Benchmark README](../spatial/spatial_benchmark_README.md)
- [Usage Examples](../usage_examples.md)
- [Performance Monitoring](../monitoring.md)
