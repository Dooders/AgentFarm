# Benchmark Specs Reference

This directory contains YAML specification files for running various benchmarks in the AgentFarm system. Each spec file defines the parameters, instrumentation, and configuration for a specific benchmark experiment.

## Available Specs

### Database Benchmarks
- **`memory_db_baseline.yaml`** - Memory database performance baseline
- **`pragma_profile_baseline.yaml`** - SQLite pragma profile performance comparison
- **`database_profiler_baseline.yaml`** - Database operation profiling

### Memory System Benchmarks
- **`redis_memory_baseline.yaml`** - Redis memory system performance

### Perception & Observation Benchmarks
- **`observation_baseline.yaml`** - Observation flow baseline (existing)
- **`observation_sweep.yaml`** - Observation flow parameter sweep (existing)
- **`perception_metrics_baseline.yaml`** - Perception metrics performance
- **`observation_profiler_baseline.yaml`** - Observation system profiling

### Spatial Indexing Benchmarks
- **`spatial_comprehensive_baseline.yaml`** - Comprehensive spatial indexing performance
- **`spatial_index_profiler_baseline.yaml`** - Spatial indexing profiling
- **`spatial_memory_profiler_baseline.yaml`** - Spatial indexing memory usage
- **`spatial_performance_analyzer_baseline.yaml`** - Spatial performance analysis

### System Profiling
- **`system_profiler_baseline.yaml`** - System resource profiling

### Comprehensive Sweeps
- **`comprehensive_profiling_sweep.yaml`** - Multi-experiment profiling sweep

## Usage

### Running Individual Specs
```bash
# Run a specific benchmark
python -m benchmarks.run_benchmarks --spec benchmarks/specs/memory_db_baseline.yaml

# Run with custom output directory
python -m benchmarks.run_benchmarks --spec benchmarks/specs/pragma_profile_baseline.yaml --output-dir custom/results
```

### Running Sweeps
```bash
# Run parameter sweeps
python -m benchmarks.run_benchmarks --spec benchmarks/specs/observation_sweep.yaml

# Run comprehensive profiling sweep
python -m benchmarks.run_benchmarks --spec benchmarks/specs/comprehensive_profiling_sweep.yaml
```

## Spec File Structure

Each spec file follows this general structure:

```yaml
experiment: <experiment_name>
params:
  # Experiment-specific parameters
  param1: value1
  param2: value2
iterations:
  warmup: 1      # Number of warmup iterations
  measured: 3    # Number of measured iterations
instrumentation:
  - timing       # Basic timing instrumentation
  - psutil       # System resource monitoring
  - cprofile     # Python profiling (optional)
output_dir: benchmarks/results/<experiment_name>
tags: [tag1, tag2, tag3]
notes: "Description of the benchmark"
seed: 42
```

## Instrumentation Options

- **`timing`** - Basic execution time measurement
- **`psutil`** - System resource monitoring (CPU, memory, disk, network)
- **`cprofile`** - Python function-level profiling

## Tags

Tags help categorize and filter benchmarks:
- `baseline` - Baseline performance measurements
- `database` - Database-related benchmarks
- `spatial` - Spatial indexing benchmarks
- `perception` - Perception and observation benchmarks
- `profiling` - Detailed profiling benchmarks
- `sweep` - Parameter sweep benchmarks

## Best Practices

1. **Start with baselines** - Run baseline specs first to establish performance baselines
2. **Use appropriate iterations** - More iterations for stable results, fewer for quick tests
3. **Choose relevant instrumentation** - Add `cprofile` only when needed for detailed analysis
4. **Tag appropriately** - Use consistent tags for easy filtering and organization
5. **Document changes** - Update notes when modifying benchmark parameters

## Customization

You can create custom spec files by:
1. Copying an existing spec file
2. Modifying the parameters for your specific needs
3. Adjusting iterations and instrumentation as needed
4. Using descriptive names and tags

## Integration with CI/CD

These specs are designed to work with the GitHub Actions workflow in `.github/workflows/database-performance-baseline.yml`. The workflow can be extended to run additional specs as needed.
