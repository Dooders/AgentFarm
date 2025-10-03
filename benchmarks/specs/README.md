# Benchmark Specs Reference

This directory contains YAML specification files for running various benchmarks in the AgentFarm system. Each spec file defines the parameters, instrumentation, and configuration for a specific benchmark experiment.

## Available Specs

### Database Benchmarks
- **`memory_db_baseline.yaml`** - Memory database performance baseline
- **`pragma_profile_baseline.yaml`** - SQLite pragma profile performance comparison
- **`pragma_profile_sweep.yaml`** - SQLite pragma profile parameter sweep
- **`database_profiler_baseline.yaml`** - Database operation profiling *(standalone script)*

### Memory System Benchmarks
- **`redis_memory_baseline.yaml`** - Redis memory system performance

### Perception & Observation Benchmarks
- **`observation_baseline.yaml`** - Observation flow baseline (existing)
- **`observation_sweep.yaml`** - Observation flow parameter sweep (existing)
- **`perception_metrics_baseline.yaml`** - Perception metrics performance
- **`observation_profiler_baseline.yaml`** - Observation system profiling *(standalone script)*

### Spatial Indexing Benchmarks
- **`spatial_comprehensive_baseline.yaml`** - Comprehensive spatial indexing performance *(standalone script)*
- **`spatial_index_profiler_baseline.yaml`** - Spatial indexing profiling *(standalone script)*
- **`spatial_memory_profiler_baseline.yaml`** - Spatial indexing memory usage *(standalone script)*
- **`spatial_performance_analyzer_baseline.yaml`** - Spatial performance analysis *(standalone script)*

### System Profiling
- **`system_profiler_baseline.yaml`** - System resource profiling *(standalone script)*

### Comprehensive Sweeps
- **`pragma_profile_sweep.yaml`** - SQLite pragma profile parameter sweep
- **`perception_metrics_sweep.yaml`** - Perception metrics parameter sweep
- **`spatial_comprehensive_sweep.yaml`** - Spatial indexing parameter sweep

## Usage

### Running Individual Specs
```bash
# Run a specific benchmark
python -m benchmarks.run_benchmarks --spec benchmarks/specs/memory_db_baseline.yaml

# Output directory is configured in the spec file itself
```

### Running Sweeps
```bash
# Run parameter sweeps
python -m benchmarks.run_benchmarks --spec benchmarks/specs/observation_sweep.yaml

# Run individual comprehensive sweeps
python -m benchmarks.run_benchmarks --spec benchmarks/specs/pragma_profile_sweep.yaml
python -m benchmarks.run_benchmarks --spec benchmarks/specs/perception_metrics_sweep.yaml
python -m benchmarks.run_benchmarks --spec benchmarks/specs/spatial_comprehensive_sweep.yaml
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
- `comprehensive` - Comprehensive benchmarking suites
- `database` - Database-related benchmarks
- `memory` - Memory usage and profiling benchmarks
- `performance` - Performance-focused benchmarks
- `perception` - Perception and observation benchmarks
- `profiling` - Detailed profiling benchmarks
- `pragma` - SQLite pragma configuration benchmarks
- `redis` - Redis memory system benchmarks
- `spatial` - Spatial indexing benchmarks
- `sqlite` - SQLite-specific benchmarks
- `sweep` - Parameter sweep benchmarks
- `system` - System resource profiling benchmarks

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

## Standalone Scripts

Some benchmark specs are marked with *(standalone script)* because they reference Python files that are standalone scripts rather than registered benchmark experiment classes. These scripts can be run directly:

```bash
# Run standalone spatial benchmarks
python -m benchmarks.implementations.spatial.comprehensive_spatial_benchmark

# Run standalone profiling scripts
python -m benchmarks.implementations.profiling.database_profiler
python -m benchmarks.implementations.profiling.spatial_index_profiler
python -m benchmarks.implementations.spatial.spatial_memory_profiler
python -m benchmarks.implementations.spatial.spatial_performance_analyzer
python -m benchmarks.implementations.profiling.system_profiler
python -m benchmarks.implementations.profiling.observation_profiler
```

These standalone scripts typically generate their own results and reports.
