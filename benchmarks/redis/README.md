# Redis Benchmark Runner

A simple script to run Redis memory benchmarks and track performance metrics.

## Quick Start

Make sure Redis is running, then run:

```bash
# Navigate to the benchmarks/redis directory
cd benchmarks/redis

# Quick test to verify setup
python run_redis_benchmark.py --mode quick --check-redis

# Standard single configuration benchmark
python run_redis_benchmark.py --mode single --agents 20 --memory-entries 1000

# Compare different Redis configurations
python run_redis_benchmark.py --mode compare --plot
```

### Using the Batch/Shell Scripts

For Windows, use the batch script:
```batch
# Navigate to the benchmarks/redis directory
cd benchmarks\redis

# Run the batch script
run_redis_benchmark.bat
```

For Linux/macOS, use the shell script:
```bash
# Navigate to the benchmarks/redis directory
cd benchmarks/redis

# Make the script executable if needed
chmod +x run_redis_benchmark.sh

# Run the shell script
./run_redis_benchmark.sh
```

### Using Docker for Redis

We provide Docker support to easily run Redis in a container:

```bash
# Navigate to the benchmarks/redis directory
cd benchmarks/redis

# Start Redis using Docker
docker-compose up -d

# Run benchmarks with Docker Redis
./run_redis_benchmark.sh --docker

# Or on Windows
run_redis_benchmark.bat --docker

# To start Redis and run benchmarks in one command
./run_redis_benchmark.sh --docker --start-redis
```

For more details on the Docker setup, see [DOCKER.md](DOCKER.md).

## Usage

```
python run_redis_benchmark.py [options]
```

### Benchmark Modes

- `--mode single`: Run a single benchmark with one configuration
- `--mode compare`: Run benchmarks with multiple configurations and compare results
- `--mode quick`: Run a fast benchmark for quick feedback (reduced entries and iterations)

### Common Options

- `--check-redis`: Check if Redis server is running before starting
- `--output DIR`: Directory to save results (default: "benchmark_results")
- `--iterations N`: Number of iterations to run (default: 3)
- `--memory-entries N`: Number of memory entries per agent (default: 1000)

### Options for Single Mode

- `--agents N`: Number of agents to simulate (default: 10)
- `--batch-size N`: Batch size for batch operations (default: 100)

### Options for Compare Mode

- `--plot`: Generate plots of the results
- `--agent-counts N1 N2...`: List of agent counts to test (default: 1 5 10 50)
- `--batch-sizes N1 N2...`: List of batch sizes to test (default: 1 10 50 100 500)

## Examples

### Basic Benchmark

```bash
python run_redis_benchmark.py --mode single
```

### Compare Agent Counts

```bash
python run_redis_benchmark.py --mode compare --agent-counts 5 10 20 50 100 --plot
```

### Compare Batch Sizes

```bash
python run_redis_benchmark.py --mode compare --batch-sizes 1 10 50 100 500 1000 --plot
```

### Custom Output Directory

```bash
python run_redis_benchmark.py --mode single --output redis_perf_results
```

### Quick Test with Redis Check

```bash
python run_redis_benchmark.py --mode quick --check-redis
```

## Interpreting Results

The benchmark measures:

1. **Write Operations**: How quickly memories can be stored (ops/sec)
2. **Read Operations**: How quickly memories can be retrieved (ops/sec)
3. **Search Operations**: Performance of spatial search (ops/sec)
4. **Batch Operations**: Throughput of batched operations (ops/sec)
5. **Memory Usage**: Memory consumption per entry (bytes)
6. **Cleanup**: Time to prune old memories (seconds)

Results are saved to the specified output directory. For comparison mode, plots are generated to visualize the differences between configurations.

## Result Files

Benchmark results are stored in the `results/` directory, containing:
- JSON files with raw benchmark data
- PNG charts showing performance comparisons
- Performance history tracking changes over time
- README.md with latest benchmark summaries

## CI/CD Integration

This benchmark is integrated with GitHub Actions. See `WORKFLOW.md` for details on how the automated benchmarking works.

## Troubleshooting

- If Redis connection fails, make sure Redis server is running on localhost:6379
- For Windows: Start Redis server from the Redis installation
- For Linux/macOS: Run `redis-server` in terminal
- Using Docker: Run `docker-compose up -d` in the benchmarks/redis directory 