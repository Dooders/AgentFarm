# Redis Memory Benchmark

This directory contains benchmarks for measuring the performance of the Redis-backed agent memory system.

## Overview

The Redis memory benchmark suite measures:

- Memory write operations (iterations per second)
- Memory read operations (iterations per second)
- Memory search operations (iterations per second)
- Batch operations throughput
- Memory usage under different loads
- Impact of memory limits and cleanup

## Requirements

- Python 3.7+
- Redis server (running locally or accessible)
- `matplotlib` and `pandas` for visualization

## Basic Usage

### Single Benchmark Run

To run the Redis memory benchmark:

```bash
python -m benchmarks.run_benchmarks --benchmark redis_memory
```

### Comparing Redis Configurations

To compare different Redis configuration options:

```bash
python -m benchmarks.compare_redis_configs --plot
```

## Command Line Options

### Redis Memory Benchmark Options

- `--agents`: Number of agents to simulate (default: 30)
- `--memory-entries`: Number of memory entries per agent (default: 1000)
- `--batch-size`: Batch size for batch operations (default: 100)
- `--search-radius`: Radius for spatial searches (default: 10.0)
- `--memory-limit`: Memory limit per agent (default: 5000)
- `--ttl`: Time-to-live for memory entries in seconds (default: 3600)
- `--cleanup-interval`: Cleanup interval for memory entries (default: 100)
- `--iterations`: Number of iterations to run (default: 3)
- `--output`: Directory to save results (default: benchmarks/results)

### Redis Configuration Comparison Options

- `--iterations`: Number of iterations for each configuration (default: 3)
- `--memory-entries`: Base number of memory entries per agent (default: 1000)
- `--plot`: Generate plots of results (flag, no value needed)
- `--agents`: List of agent counts to test (default: 1 5 10 50 100)
- `--batch-sizes`: List of batch sizes to test (default: 1 10 50 100 500)
- `--memory-limits`: List of memory limits to test (default: 100 1000 5000 10000)
- `--output`: Base directory for saving results (default: benchmarks/results/redis_comparison)

## Example Commands

### Basic Benchmark

```bash
python -m benchmarks.run_benchmarks --benchmark redis_memory --agents 50 --memory-entries 2000
```

### Configuration Comparison

```bash
python -m benchmarks.compare_redis_configs --plot --agents 10 20 50 100 --batch-sizes 50 100 200
```

### Quick Single-Config Test

```bash
python -m benchmarks.run_benchmarks --benchmark redis_memory --iterations 1 --memory-entries 100
```

## Understanding Results

The benchmark generates detailed metrics for each operation type:

1. **Write Operations**: Measures how quickly memories can be stored
2. **Read Operations**: Measures how quickly memories can be retrieved
3. **Search Operations**: Measures spatial search performance
4. **Batch Operations**: Measures throughput of batched operations
5. **Memory Usage**: Measures memory consumption per entry
6. **Cleanup**: Measures time to prune old memories

### Key Performance Indicators

- **Operations Per Second**: Higher is better
- **Memory Per Entry**: Lower is better
- **Batch Throughput**: Higher is better
- **Cleanup Time**: Lower is better

## Interpreting Comparison Results

The comparison tool will generate both numeric results and (optionally) visual plots. Look for:

1. **Scaling with Agent Count**: How performance changes as you add more agents
2. **Batch Size Impact**: Optimal batch size for your workload
3. **Memory Limit Effects**: How different memory limits affect performance and cleanup time

## Tips for Accurate Results

1. Run the Redis server on the same machine for consistent results
2. Close other applications that might compete for resources
3. Run multiple iterations (3-5) for statistical significance
4. Test with realistic workloads that match your production use case
5. Try different Redis configurations (maxmemory, eviction policies)

## Customizing the Benchmark

To modify the benchmark for your specific needs:

1. Edit `benchmarks/implementations/redis_memory_benchmark.py` to adjust test logic
2. Add new metrics in the `run()` method return value
3. Add custom visualization in `compare_redis_configs.py`

## Known Limitations

- The benchmark assumes Redis is running locally by default
- Some operations might be affected by Redis configuration beyond the benchmark's control
- Memory usage metrics might vary depending on Redis version and configuration

## Troubleshooting

If you encounter issues:

1. Ensure Redis server is running: `redis-cli ping` should return "PONG"
2. Check Redis connection parameters (host, port, password if needed)
3. If getting memory errors, reduce the number of agents or entries
4. For slow benchmarks, try smaller iteration counts 