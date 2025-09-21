# AgentFarm Benchmarking Guide

This guide explains how to run and interpret benchmarks for the AgentFarm perception system and other components.

## Quick Start

### Running Perception Metrics Benchmark

```bash
# Basic run with default settings
python benchmarks/run_benchmarks.py --benchmark perception_metrics

# Custom configuration
python benchmarks/run_benchmarks.py --benchmark perception_metrics \
  --pm-agents "100,1000,10000" \
  --pm-radii "5,8,10" \
  --pm-modes "hybrid,dense" \
  --pm-bilinear "true,false" \
  --pm-steps 20 \
  --pm-device cpu
```

### Running All Benchmarks

```bash
# Run all available benchmarks
python benchmarks/run_benchmarks.py --benchmark all

# Run specific benchmark
python benchmarks/run_benchmarks.py --benchmark observation_flow
```

## Available Benchmarks

### 1. Perception Metrics Benchmark

**Purpose**: Measures perception system performance across different scales and configurations.

**Parameters**:
- `--pm-agents`: Comma-separated agent counts (e.g., "100,1000,10000")
- `--pm-radii`: Comma-separated observation radii (e.g., "5,8,10")
- `--pm-modes`: Storage modes ("hybrid,dense")
- `--pm-bilinear`: Interpolation methods ("true,false")
- `--pm-steps`: Steps per run (default: 10)
- `--pm-device`: Device ("cpu" or "cuda")

**Output Metrics**:
- Throughput (observations/second)
- Memory usage (dense vs sparse)
- Memory reduction percentage
- Step timing (mean, P95)
- GFLOPS estimation
- Cache hit rates

### 2. Observation Flow Benchmark

**Purpose**: Tests observation generation throughput under various loads.

**Parameters**:
- `--obs-agents`: Number of agents (default: 200)
- `--obs-steps`: Number of steps (default: 100)
- `--obs-width/height`: Environment dimensions
- `--obs-radius`: Observation radius
- `--obs-device`: Device for tensors

### 3. Memory Database Benchmark

**Purpose**: Tests database performance for agent memory storage.

**Parameters**:
- `--agents`: Number of agents
- `--steps`: Number of simulation steps

### 4. Redis Memory Benchmark

**Purpose**: Tests Redis-based agent memory system performance.

**Parameters**:
- `--memory-entries`: Entries per agent
- `--batch-size`: Batch operation size
- `--search-radius`: Spatial search radius
- `--memory-limit`: Memory limit per agent

### 5. Pragma Profile Benchmark

**Purpose**: Profiles database performance with various configurations.

**Parameters**:
- `--num-records`: Number of database records
- `--db-size-mb`: Database size in MB

## Understanding Results

### Perception Metrics Interpretation

**Memory Metrics**:
- **Dense bytes per agent**: Memory if using dense storage only
- **Sparse bytes per agent**: Actual memory with sparse storage
- **Memory reduction**: Percentage saved by sparse storage

**Performance Metrics**:
- **Observations/sec**: Throughput of the perception system
- **Mean step time**: Average time per simulation step
- **P95 step time**: 95th percentile step time (indicates worst-case performance)

**Computational Metrics**:
- **GFLOPS**: Estimated floating-point operations per second
- **Cache hit rate**: Efficiency of lazy dense construction

### Performance Scaling

**Linear Scaling**: Performance should scale linearly with agent count
**Quadratic Scaling**: Memory and computation scale quadratically with observation radius

**Expected Performance**:
- 100 agents: <1ms per step
- 1,000 agents: <1ms per step
- 10,000 agents: <10ms per step

### Memory Efficiency

**Hybrid Storage Benefits**:
- 85% memory reduction vs dense storage
- Minimal computational overhead
- Better cache locality for sparse environments

**Storage Mode Comparison**:
- **Hybrid**: Best memory efficiency, minimal performance impact
- **Dense**: Higher memory usage, slightly faster for very dense environments

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Set Python path
export PYTHONPATH=/path/to/AgentFarm:$PYTHONPATH
python benchmarks/run_benchmarks.py --benchmark perception_metrics
```

**Dependency Issues**:
```bash
# Install required packages
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Memory Issues**:
- Reduce agent counts for initial testing
- Use smaller observation radii
- Monitor system memory usage

### Performance Optimization

**CPU Optimization**:
- Use fewer steps for initial testing
- Start with smaller agent counts
- Use nearest-neighbor interpolation for speed

**GPU Acceleration**:
```bash
# Enable GPU acceleration
python benchmarks/run_benchmarks.py --benchmark perception_metrics --pm-device cuda
```

**Memory Optimization**:
- Use hybrid storage mode
- Reduce observation radius when possible
- Monitor memory usage with system tools

## Advanced Usage

### Custom Benchmark Scripts

Create custom benchmark scripts by extending the base `Benchmark` class:

```python
from benchmarks.base.benchmark import Benchmark

class CustomBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            name="custom",
            description="Custom benchmark",
            parameters={}
        )
    
    def setup(self):
        # Setup code
        pass
    
    def run(self):
        # Benchmark code
        return {"results": "data"}
    
    def cleanup(self):
        # Cleanup code
        pass
```

### Batch Benchmarking

Run multiple benchmark configurations:

```bash
# Create benchmark script
cat > run_batch_benchmarks.sh << 'EOF'
#!/bin/bash

# Test different agent counts
for agents in 100 500 1000; do
    python benchmarks/run_benchmarks.py \
        --benchmark perception_metrics \
        --pm-agents $agents \
        --pm-radii "5,8" \
        --pm-modes "hybrid" \
        --pm-bilinear "true,false"
done
EOF

chmod +x run_batch_benchmarks.sh
./run_batch_benchmarks.sh
```

### Results Analysis

**JSON Output**: All benchmarks save results in JSON format for analysis:

```python
import json
import pandas as pd

# Load results
with open('perception_benchmark_results_20250920_180135.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results['runs'])
print(df.groupby(['agents', 'R', 'storage_mode']).agg({
    'observes_per_sec': 'mean',
    'memory_reduction_percent': 'mean',
    'gflops_est': 'mean'
}))
```

## Best Practices

1. **Start Small**: Begin with small agent counts and gradually increase
2. **Multiple Runs**: Run benchmarks multiple times to get stable results
3. **System Monitoring**: Monitor CPU, memory, and GPU usage during benchmarks
4. **Documentation**: Document your benchmark configurations and results
5. **Comparison**: Compare results across different hardware configurations

## Hardware Recommendations

**CPU**: Modern multi-core processor (Intel i7/AMD Ryzen 7 or better)
**Memory**: 16GB+ RAM for large-scale benchmarks
**GPU**: RTX 3070 or better for GPU acceleration
**Storage**: SSD for database benchmarks

## Contributing

To add new benchmarks:

1. Create a new benchmark class in `benchmarks/implementations/`
2. Extend the base `Benchmark` class
3. Add command-line arguments in `benchmarks/run_benchmarks.py`
4. Update this guide with new benchmark documentation
5. Add tests for your benchmark

## References

- [Perception System Documentation](docs/perception_system.md)
- [Benchmark Results](BENCHMARK_RESULTS.md)
- [Core Architecture](docs/core_architecture.md)
