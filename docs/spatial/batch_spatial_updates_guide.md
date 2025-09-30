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

### Preset Configuration Profiles

For common use cases, use one of the preset configuration profiles:

```python
from farm.config.config import SpatialIndexConfig, EnvironmentConfig, SimulationConfig

# Profile 1: Dense Population (many agents in small area)
# Optimized for: High agent density, frequent neighbor queries
spatial_config = SpatialIndexConfig.preset_dense_population()
# Equivalent to:
# - region_size=25.0 (finer granularity)
# - max_batch_size=50 (more frequent updates)
# - enable_spatial_hash_indices=True (fast neighbor queries)
# - spatial_hash_cell_size=15.0 (smaller cells)

# Profile 2: Sparse Realtime (few agents, low latency required)
# Optimized for: Real-time responsiveness, low agent count
spatial_config = SpatialIndexConfig.preset_sparse_realtime()
# Equivalent to:
# - region_size=100.0 (less tracking overhead)
# - max_batch_size=25 (immediate updates)
# - enable_batch_updates=True (batching still helps)

# Profile 3: Large Environment (big simulation area)
# Optimized for: Large worlds, range queries, moderate agent count
spatial_config = SpatialIndexConfig.preset_large_environment()
# Equivalent to:
# - region_size=75.0 (balanced tracking)
# - max_batch_size=150 (higher throughput)
# - max_regions=2000 (more regions for larger area)
# - enable_quadtree_indices=True (efficient range queries)

# Profile 4: High Throughput (batch processing, latency tolerant)
# Optimized for: Maximum throughput, offline analysis
spatial_config = SpatialIndexConfig.preset_high_throughput()
# Equivalent to:
# - region_size=50.0 (balanced)
# - max_batch_size=200 (large batches)
# - enable_batch_updates=True

# Apply preset and customize if needed
env_config = EnvironmentConfig(
    width=200,
    height=200,
    spatial_index=spatial_config
)

config = SimulationConfig()
config.environment = env_config
```

### Basic Configuration

For custom configurations, create a `SpatialIndexConfig` manually:

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

### Accessing Performance Statistics

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

# Add position updates manually with priority
entity = some_agent
spatial_index.add_position_update(
    entity, 
    old_position=(10.0, 10.0), 
    new_position=(20.0, 20.0),
    entity_type="agent",
    priority=1  # See Priority System section below
)

# Process batch updates
spatial_index.process_batch_updates(force=True)

# Get batch update statistics (using public API)
batch_stats = spatial_index.get_batch_update_stats()
print(f"Pending updates: {batch_stats['pending_updates_count']}")
print(f"Total batch updates: {batch_stats['total_batch_updates']}")
```

### Priority System

The priority system controls the order in which position updates are processed within a batch:

```python
# Priority levels (higher = processed first)
PRIORITY_CRITICAL = 3   # Player entities, important NPCs
PRIORITY_HIGH = 2       # Active combat participants, quest-critical agents
PRIORITY_NORMAL = 1     # Regular agents (default)
PRIORITY_LOW = 0        # Background entities, decorative elements

# Example usage
spatial_index.add_position_update(
    player_entity,
    old_position=(0, 0),
    new_position=(10, 10),
    entity_type="player",
    priority=3  # PRIORITY_CRITICAL - processed first
)

spatial_index.add_position_update(
    background_entity,
    old_position=(5, 5),
    new_position=(6, 6),
    entity_type="decoration",
    priority=0  # PRIORITY_LOW - processed last
)
```

**When to use different priorities:**

- **Critical (3)**: Entities that must be updated immediately (player, boss enemies)
- **High (2)**: Entities actively involved in important interactions (combat, cutscenes)
- **Normal (1)**: Most agents in the simulation (default)
- **Low (0)**: Non-essential entities that can tolerate update delays

**Note**: Within the same priority level, updates are processed in FIFO order. Priority only affects ordering between different priority levels.

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

## Error Handling

### Handling Configuration Errors

```python
from farm.config.config import SpatialIndexConfig

try:
    spatial_config = SpatialIndexConfig(
        region_size=-10.0  # Invalid: negative region size
    )
except ValueError as e:
    print(f"Configuration error: {e}")
    # Use safe defaults
    spatial_config = SpatialIndexConfig.preset_dense_population()
```

### Handling Runtime Errors

```python
# Check if batch updates are enabled before manual control
if env.spatial_index.is_batch_updates_enabled():
    try:
        env.process_batch_spatial_updates(force=True)
    except MemoryError:
        # Handle out-of-memory during batch processing
        print("Memory limit reached. Reducing batch size...")
        env.disable_batch_spatial_updates()
        env.enable_batch_spatial_updates(
            region_size=100.0,  # Larger regions = less memory
            max_batch_size=50    # Smaller batches
        )
    except TimeoutError as e:
        # Handle batch processing timeout
        print(f"Batch processing timeout: {e}")
        # Process in smaller chunks
        stats = env.spatial_index.get_batch_update_stats()
        if stats['pending_updates_count'] > 1000:
            env.spatial_index.clear_pending_updates()
else:
    print("Batch updates not enabled")
```

### Monitoring for Issues

```python
def check_spatial_health(env):
    """Check spatial index health and detect potential issues."""
    stats = env.get_spatial_performance_stats()
    batch_stats = stats['batch_updates']
    
    issues = []
    
    # Check for excessive pending updates
    pending = batch_stats.get('pending_updates_count', 0)
    if pending > 1000:
        issues.append({
            'severity': 'high',
            'message': f'Excessive pending updates ({pending}). Consider processing batch or increasing max_batch_size.',
            'metric': 'pending_updates',
            'value': pending
        })
    
    # Check for region overflow
    dirty_regions = batch_stats.get('total_dirty_regions', 0)
    max_regions = env.spatial_index.max_regions
    if dirty_regions > max_regions * 0.9:
        issues.append({
            'severity': 'medium',
            'message': f'Approaching max regions limit ({dirty_regions}/{max_regions}). Consider increasing region_size or max_regions.',
            'metric': 'dirty_regions',
            'value': dirty_regions
        })
    
    # Check for slow batch processing
    last_batch_time = batch_stats.get('last_batch_time', 0)
    if last_batch_time > 0.1:  # 100ms threshold
        issues.append({
            'severity': 'medium',
            'message': f'Slow batch processing ({last_batch_time:.3f}s). Consider reducing batch size or enabling spatial indices.',
            'metric': 'batch_processing_time',
            'value': last_batch_time
        })
    
    return issues

# Usage
issues = check_spatial_health(env)
for issue in issues:
    print(f"[{issue['severity'].upper()}] {issue['message']}")
```

### Graceful Degradation

```python
def setup_spatial_with_fallback(env_config):
    """Setup spatial indices with graceful fallback on failure."""
    try:
        # Try optimal configuration
        spatial_config = SpatialIndexConfig.preset_dense_population()
        env_config.spatial_index = spatial_config
        return spatial_config
    except MemoryError:
        # Fall back to minimal configuration
        print("Warning: Insufficient memory for optimal config. Using minimal settings.")
        spatial_config = SpatialIndexConfig(
            enable_batch_updates=True,
            region_size=100.0,
            max_batch_size=25,
            max_regions=500,
            enable_quadtree_indices=False,
            enable_spatial_hash_indices=False
        )
        env_config.spatial_index = spatial_config
        return spatial_config
```

## Troubleshooting

### Diagnosing Issues

1. **High Memory Usage**
   - **Symptom**: MemoryError or high RAM consumption
   - **Solutions**:
     - Reduce `max_regions` parameter (e.g., from 1000 to 500)
     - Increase `region_size` to reduce number of tracked regions (e.g., from 25 to 100)
     - Disable unused index types (set `enable_quadtree_indices=False`)
     - Use `SpatialIndexConfig.preset_sparse_realtime()` profile

2. **Low Performance**
   - **Symptom**: High spatial query times (>10ms)
   - **Solutions**:
     - Enable quadtree indices for range queries (`enable_quadtree_indices=True`)
     - Enable spatial hash indices for neighbor queries (`enable_spatial_hash_indices=True`)
     - Optimize batch size and region size based on workload
     - Use `SpatialIndexConfig.preset_large_environment()` for large worlds

3. **High Latency**
   - **Symptom**: Delayed entity position updates
   - **Solutions**:
     - Reduce `max_batch_size` for more frequent updates (e.g., from 100 to 25)
     - Use `process_batch_spatial_updates(force=True)` for immediate processing
     - Use `SpatialIndexConfig.preset_sparse_realtime()` profile
     - Increase priority for critical entities

4. **Pending Updates Accumulation**
   - **Symptom**: `pending_updates_count` keeps growing
   - **Solutions**:
     - Increase `max_batch_size` to process more updates per batch
     - Call `process_batch_updates(force=True)` more frequently
     - Check for bottlenecks using `get_spatial_performance_stats()`
     - Consider if batch updates are appropriate for your use case

5. **Region Overflow**
   - **Symptom**: Warning about exceeding `max_regions`
   - **Solutions**:
     - Increase `max_regions` parameter
     - Increase `region_size` to reduce total region count
     - Process batch updates more frequently to clear dirty regions

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