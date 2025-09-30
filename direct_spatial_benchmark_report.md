# Direct Spatial Indexing Performance Report
============================================================

**Generated**: 2025-09-30 00:13:53

## Performance Comparison

| Implementation | Avg Build Time (ms) | Avg Query Time (μs) | Avg Range Time (μs) | Avg Memory (MB) |
|----------------|-------------------|-------------------|-------------------|----------------|
| AgentFarm Quadtree | 6.43 | 12.74 | 35.89 | 72.0 |
| AgentFarm Spatial Hash | 0.54 | 5.78 | 19.24 | 54.0 |

## Scaling Analysis

### AgentFarm Quadtree

| Entity Count | Build Time (ms) | Query Time (μs) | Range Time (μs) | Memory (MB) |
|--------------|----------------|----------------|----------------|-------------|
| 100 | 0.83 | 7.64 | 9.09 | 8.0 |
| 100 | 0.48 | 7.18 | 9.84 | 8.0 |
| 100 | 0.50 | 7.14 | 8.42 | 8.0 |
| 500 | 2.82 | 12.18 | 23.19 | 40.0 |
| 500 | 2.91 | 10.37 | 25.43 | 40.0 |
| 500 | 3.19 | 9.56 | 18.00 | 40.0 |
| 1000 | 5.77 | 17.73 | 36.27 | 80.0 |
| 1000 | 6.57 | 9.97 | 36.16 | 80.0 |
| 1000 | 7.41 | 14.19 | 30.74 | 80.0 |
| 2000 | 15.77 | 25.54 | 58.35 | 160.0 |
| 2000 | 14.05 | 12.89 | 124.85 | 160.0 |
| 2000 | 16.84 | 18.51 | 50.41 | 160.0 |

### AgentFarm Spatial Hash

| Entity Count | Build Time (ms) | Query Time (μs) | Range Time (μs) | Memory (MB) |
|--------------|----------------|----------------|----------------|-------------|
| 100 | 0.09 | 4.45 | 15.10 | 6.0 |
| 100 | 0.07 | 4.78 | 15.22 | 6.0 |
| 100 | 0.05 | 5.80 | 12.95 | 6.0 |
| 500 | 0.31 | 5.79 | 18.70 | 30.0 |
| 500 | 0.26 | 4.30 | 20.99 | 30.0 |
| 500 | 0.24 | 4.71 | 14.86 | 30.0 |
| 1000 | 0.64 | 5.40 | 17.71 | 60.0 |
| 1000 | 0.53 | 5.62 | 20.31 | 60.0 |
| 1000 | 0.54 | 6.38 | 20.22 | 60.0 |
| 2000 | 1.54 | 7.74 | 20.60 | 120.0 |
| 2000 | 1.10 | 8.82 | 27.45 | 120.0 |
| 2000 | 1.10 | 5.53 | 26.85 | 120.0 |

## Distribution Pattern Analysis

### Clustered Distribution

| Implementation | Avg Query Time (μs) | Performance vs Uniform |
|----------------|-------------------|----------------------|
| AgentFarm Quadtree | 10.10 | 0.93x |
| AgentFarm Spatial Hash | 5.88 | 0.54x |

### Uniform Distribution

| Implementation | Avg Query Time (μs) | Performance vs Uniform |
|----------------|-------------------|----------------------|
| AgentFarm Quadtree | 15.77 | 1.46x |
| AgentFarm Spatial Hash | 5.85 | 0.54x |

### Linear Distribution

| Implementation | Avg Query Time (μs) | Performance vs Uniform |
|----------------|-------------------|----------------------|
| AgentFarm Quadtree | 12.35 | 1.14x |
| AgentFarm Spatial Hash | 5.60 | 0.52x |

## Dynamic Update Performance

| Entity Count | Quadtree Update (ms) | Spatial Hash Update (ms) | Speedup |
|--------------|---------------------|-------------------------|---------|
| 100 | 0.12 | 0.03 | 3.73x |
| 500 | 0.53 | 0.12 | 4.42x |
| 1000 | 1.23 | 0.27 | 4.59x |

## Performance Recommendations

### Best Implementation by Use Case:

- **Best for Radius Queries**: AgentFarm Spatial Hash
- **Best for Range Queries**: AgentFarm Spatial Hash
- **Best for Dynamic Updates**: AgentFarm Spatial Hash (faster move operations)

### Key Performance Insights:

1. **Quadtree** excels at hierarchical spatial queries and range operations
2. **Spatial Hash** provides faster dynamic updates and uniform query performance
3. **Memory usage** scales linearly with entity count for both implementations
4. **Distribution patterns** have minimal impact on spatial hash performance
5. **Dynamic updates** are significantly faster with spatial hash grid

### Best Practices:

1. **Use Quadtree** for applications with many range queries and hierarchical operations
2. **Use Spatial Hash** for applications with frequent dynamic updates
3. **Choose appropriate cell size** for spatial hash based on typical query radius
4. **Consider hybrid approaches** using both data structures for different operations
5. **Profile with realistic data** to choose the best implementation for your use case
