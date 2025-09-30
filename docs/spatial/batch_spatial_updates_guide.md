# Batch Spatial Updates with Dirty Region Tracking

This guide explains how to use the new batch spatial updates feature in AgentFarm, which significantly improves performance by only updating regions that have actually changed.

## Overview

The batch spatial updates feature implements dirty region tracking to optimize spatial index updates. Instead of rebuilding entire spatial indices every time an agent moves, the system:

1. **Tracks dirty regions** - Only regions that have changed are marked for updates
2. **Batches position updates** - Multiple position changes are collected and processed together
3. **Processes updates efficiently** - Updates are applied in batches with priority-based scheduling
4. **Provides performance metrics** - Detailed statistics about update efficiency

## Key Benefits

- **Enhanced Performance**: Only update regions that have changed, reducing computational overhead
- **Improved Scalability**: Better performance as simulation size increases
- **Data Integrity**: Systematic approach ensures all changes are captured and applied
- **Memory Efficiency**: Reduced memory allocations and better cache locality
- **Configurable**: Flexible configuration options for different use cases

## Configuration

### Basic Configuration

```python
from farm.config.config import SpatialIndexConfig, EnvironmentConfig, SimulationConfig

# Create spatial index configuration
spatial_config = SpatialIndexConfig(
    enable_batch_updates=True,      # Enable batch updates
    region_size=50.0,               # Size of each region for tracking
    max_batch_size=100,             # Maximum updates per batch
    max_regions=1000,               # Maximum regions to track
    enable_quadtree_indices=False,  # Enable quadtree indices
    enable_spatial_hash_indices=False,  # Enable spatial hash indices
    performance_monitoring=True,    # Enable performance monitoring
    debug_queries=False             # Enable debug logging
)

# Create environment configuration
env_config = EnvironmentConfig(
    width=200,
    height=200,
    spatial_index=spatial_config
)

# Create simulation configuration
config = SimulationConfig()
config.environment = env_config
```

### Advanced Configuration

```python
# Advanced configuration with multiple index types
spatial_config = SpatialIndexConfig(
    enable_batch_updates=True,
    region_size=25.0,                    # Smaller regions for finer granularity
    max_batch_size=50,                   # Smaller batches for more frequent updates
    max_regions=2000,                    # More regions for larger environments
    enable_quadtree_indices=True,        # Enable quadtree for range queries
    enable_spatial_hash_indices=True,    # Enable spatial hash for neighbor queries
    spatial_hash_cell_size=10.0,         # Custom cell size for spatial hash
    performance_monitoring=True,
    debug_queries=True                   # Enable debug logging
)
```

## Usage Examples

### Basic Usage

```python
from farm.core.environment import Environment

# Create environment with batch updates enabled
env = Environment(
    width=200,
    height=200,
    resource_distribution="uniform",
    config=config  # Use the configured spatial settings
)

# The environment automatically uses batch updates
# No additional code changes needed for basic usage
```

### Manual Batch Update Control

```python
# Process pending batch updates manually
env.process_batch_spatial_updates(force=True)

# Enable/disable batch updates at runtime
env.enable_batch_spatial_updates(region_size=30.0, max_batch_size=75)
env.disable_batch_spatial_updates()
```

### Performance Monitoring

```python
# Get spatial performance statistics
stats = env.get_spatial_performance_stats()

print(f"Total batch updates: {stats['batch_updates']['total_batch_updates']}")
print(f"Average batch size: {stats['batch_updates']['average_batch_size']}")
print(f"Total regions processed: {stats['batch_updates']['total_regions_processed']}")
print(f"Last batch processing time: {stats['batch_updates']['last_batch_time']:.3f}s")

# Get perception performance metrics
perception_stats = stats['perception']
print(f"Spatial query time: {perception_stats['spatial_query_time_s']:.3f}s")
print(f"Bilinear interpolation time: {perception_stats['bilinear_time_s']:.3f}s")
```

### Direct Spatial Index Usage

```python
# Access spatial index directly
spatial_index = env.spatial_index

# Add position updates manually
entity = some_agent
spatial_index.add_position_update(
    entity, 
    old_position=(10.0, 10.0), 
    new_position=(20.0, 20.0),
    entity_type="agent",
    priority=1
)

# Process batch updates
spatial_index.process_batch_updates(force=True)

# Get batch update statistics
batch_stats = spatial_index.get_batch_update_stats()
print(f"Pending updates: {len(spatial_index._pending_position_updates)}")
```

## Performance Tuning

### Region Size Optimization

The `region_size` parameter controls the granularity of dirty region tracking:

- **Smaller regions** (10-25): More precise tracking, better for dense populations
- **Larger regions** (50-100): Less overhead, better for sparse populations
- **Default** (50): Good balance for most use cases

```python
# For dense populations with many agents
spatial_config.region_size = 25.0

# For sparse populations with few agents
spatial_config.region_size = 100.0
```

### Batch Size Optimization

The `max_batch_size` parameter controls how many updates are processed together:

- **Smaller batches** (25-50): More frequent updates, lower latency
- **Larger batches** (100-200): Better throughput, higher latency
- **Default** (100): Good balance for most use cases

```python
# For real-time applications requiring low latency
spatial_config.max_batch_size = 25

# For batch processing requiring high throughput
spatial_config.max_batch_size = 200
```

### Index Type Selection

Choose the appropriate spatial index types for your use case:

```python
# For radial proximity queries (default)
spatial_config.enable_quadtree_indices = False
spatial_config.enable_spatial_hash_indices = False

# For rectangular range queries
spatial_config.enable_quadtree_indices = True

# For frequent neighbor queries with large populations
spatial_config.enable_spatial_hash_indices = True
spatial_config.spatial_hash_cell_size = 20.0
```

## Performance Monitoring

### Key Metrics to Monitor

1. **Batch Update Efficiency**
   - `total_batch_updates`: Number of batch operations performed
   - `average_batch_size`: Average number of updates per batch
   - `last_batch_time`: Time taken for last batch processing

2. **Region Tracking Efficiency**
   - `total_dirty_regions`: Current number of dirty regions
   - `total_regions_processed`: Total regions processed
   - `regions_by_type`: Regions by entity type

3. **Spatial Query Performance**
   - `spatial_query_time_s`: Time spent on spatial queries
   - `bilinear_time_s`: Time spent on bilinear interpolation
   - `nearest_time_s`: Time spent on nearest neighbor queries

### Performance Analysis

```python
def analyze_spatial_performance(env):
    """Analyze spatial performance and provide recommendations."""
    stats = env.get_spatial_performance_stats()
    batch_stats = stats['batch_updates']
    perception_stats = stats['perception']
    
    # Analyze batch efficiency
    avg_batch_size = batch_stats['average_batch_size']
    if avg_batch_size < 10:
        print("Warning: Low average batch size. Consider increasing max_batch_size.")
    elif avg_batch_size > 150:
        print("Info: High average batch size. Consider decreasing max_batch_size for lower latency.")
    
    # Analyze region efficiency
    dirty_regions = batch_stats['total_dirty_regions']
    if dirty_regions > 500:
        print("Warning: High number of dirty regions. Consider increasing region_size.")
    
    # Analyze query performance
    query_time = perception_stats['spatial_query_time_s']
    if query_time > 0.01:  # 10ms
        print("Warning: High spatial query time. Consider enabling quadtree or spatial hash indices.")
    
    return {
        'batch_efficiency': avg_batch_size,
        'region_efficiency': dirty_regions,
        'query_performance': query_time
    }
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce `max_regions` parameter
   - Increase `region_size` to reduce number of regions
   - Disable unused index types

2. **Low Performance**
   - Enable quadtree indices for range queries
   - Enable spatial hash indices for neighbor queries
   - Optimize batch size and region size

3. **High Latency**
   - Reduce `max_batch_size` for more frequent updates
   - Use `process_batch_spatial_updates(force=True)` for immediate processing

### Debug Mode

Enable debug mode to get detailed logging:

```python
spatial_config = SpatialIndexConfig(
    debug_queries=True,
    performance_monitoring=True
)
```

This will log detailed information about:
- Batch update processing
- Region marking and clearing
- Spatial query performance
- Index rebuild operations

## Migration Guide

### From Previous Versions

The batch spatial updates feature is backward compatible. Existing code will work without changes, but you can enable batch updates for better performance:

```python
# Old way (still works)
env = Environment(width=100, height=100, resource_distribution="uniform")

# New way with batch updates
spatial_config = SpatialIndexConfig(enable_batch_updates=True)
env_config = EnvironmentConfig(spatial_index=spatial_config)
config = SimulationConfig()
config.environment = env_config

env = Environment(width=100, height=100, resource_distribution="uniform", config=config)
```

### Gradual Migration

1. **Start with defaults**: Use default batch update settings
2. **Monitor performance**: Use performance monitoring to identify bottlenecks
3. **Optimize configuration**: Adjust parameters based on your specific use case
4. **Enable additional indices**: Add quadtree or spatial hash indices as needed

## Best Practices

1. **Start with default settings** and optimize based on your specific use case
2. **Monitor performance metrics** regularly to identify optimization opportunities
3. **Use appropriate index types** for your query patterns
4. **Balance batch size and latency** based on your application requirements
5. **Enable debug mode** during development to understand system behavior
6. **Test with realistic data** to ensure optimal configuration

## Conclusion

The batch spatial updates feature provides significant performance improvements for AgentFarm simulations, especially in scenarios with:

- High agent mobility
- Large simulation environments
- Frequent spatial queries
- Complex multi-agent interactions

By following this guide and monitoring performance metrics, you can optimize your simulation for maximum efficiency while maintaining data integrity and system reliability.