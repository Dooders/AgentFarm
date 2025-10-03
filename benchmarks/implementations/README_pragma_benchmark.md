# SQLite Pragma Profile Benchmark

This benchmark tests the performance of different SQLite pragma profiles in the AgentFarm database module under various workloads.

## Overview

SQLite performance can be significantly affected by pragma settings. This benchmark evaluates four predefined profiles:

1. **balanced**: Good balance of performance and data safety
2. **performance**: Maximum performance, reduced data safety
3. **safety**: Maximum data safety, reduced performance
4. **memory**: Optimized for low memory usage

Each profile is tested against three workload types:

1. **write_heavy**: Primarily insert operations
2. **read_heavy**: Primarily query operations
3. **mixed**: Balanced mix of reads and writes

## Running the Benchmark

### Basic Usage

```bash
# Run with default settings (using the dedicated example script)
python -m benchmarks.examples.run_pragma_benchmark

# Run with custom parameters
python -m benchmarks.examples.run_pragma_benchmark --num-records 50000 --db-size-mb 50 --iterations 5
```

### With Visualization

For detailed visualization of results, use the dedicated script:

```bash
# Run with default settings
python -m benchmarks.examples.run_pragma_benchmark

# Run with custom parameters
python -m benchmarks.examples.run_pragma_benchmark --num-records 50000 --db-size-mb 50 --iterations 5
```

## Parameters

- `--num-records`: Number of records to insert for write tests (default: 100,000)
- `--db-size-mb`: Target database size in MB for read tests (default: 100)
- `--iterations`: Number of benchmark iterations to run (default: 3)
- `--output`: Directory to save results (default: benchmarks/results)

## Interpreting Results

The benchmark produces several metrics:

1. **Raw durations**: Time taken (in seconds) for each profile and workload
2. **Relative performance**: Speedup factor relative to the balanced profile
3. **Visualization**: When using the visualization script, you'll get charts showing:
   - Absolute performance by workload
   - Relative performance by workload
   - Performance by profile
   - Profile characteristics radar chart

## Example Results

Here's an example of what the results might look like:

```
Testing balanced profile with write_heavy workload...
  balanced profile, write_heavy workload: 12.45 seconds
Testing balanced profile with read_heavy workload...
  balanced profile, read_heavy workload: 8.32 seconds
Testing balanced profile with mixed workload...
  balanced profile, mixed workload: 10.18 seconds

Testing performance profile with write_heavy workload...
  performance profile, write_heavy workload: 5.67 seconds
Testing performance profile with read_heavy workload...
  performance profile, read_heavy workload: 7.89 seconds
Testing performance profile with mixed workload...
  performance profile, mixed workload: 6.54 seconds

Testing safety profile with write_heavy workload...
  safety profile, write_heavy workload: 18.76 seconds
Testing safety profile with read_heavy workload...
  safety profile, read_heavy workload: 9.12 seconds
Testing safety profile with mixed workload...
  safety profile, mixed workload: 14.32 seconds

Testing memory profile with write_heavy workload...
  memory profile, write_heavy workload: 13.21 seconds
Testing memory profile with read_heavy workload...
  memory profile, read_heavy workload: 8.76 seconds
Testing memory profile with mixed workload...
  memory profile, mixed workload: 11.45 seconds

Relative Performance:
  performance profile is 2.20x faster than balanced for write_heavy workload
  performance profile is 1.05x faster than balanced for read_heavy workload
  performance profile is 1.56x faster than balanced for mixed workload

  safety profile is 0.66x slower than balanced for write_heavy workload
  safety profile is 0.91x slower than balanced for read_heavy workload
  safety profile is 0.71x slower than balanced for mixed workload

  memory profile is 0.94x slower than balanced for write_heavy workload
  memory profile is 0.95x slower than balanced for read_heavy workload
  memory profile is 0.89x slower than balanced for mixed workload
```

## Recommendations

Based on benchmark results, you can choose the appropriate profile for your use case:

1. **For write-heavy workloads** (e.g., data collection, logging):
   - Use the **performance** profile for maximum throughput
   - Consider **balanced** if data integrity is important

2. **For read-heavy workloads** (e.g., data analysis, reporting):
   - Use the **balanced** profile for good read performance with data safety
   - Consider **performance** if the data is non-critical and can be regenerated

3. **For mixed workloads** (e.g., interactive applications):
   - Use the **balanced** profile for general use
   - Consider **performance** for non-critical applications

4. **For memory-constrained environments**:
   - Use the **memory** profile to minimize memory usage
   - Accept some performance trade-offs

## Implementation Details

The benchmark tests each profile with three different workloads:

1. **Write Test**: Inserts a large number of records in batches
2. **Read Test**: Performs various query operations on a pre-populated database
3. **Mixed Test**: Interleaves read and write operations

Each test is timed, and the results are normalized relative to the balanced profile to show the relative performance advantage or disadvantage of each profile.

### Technical Notes

- The benchmark uses unique table names and database files for each test to avoid conflicts when running multiple iterations
- All database connections are properly tracked and closed to prevent resource leaks
- The benchmark includes robust error handling for cleanup operations, especially on Windows where file handles may be held longer

## Customizing Profiles

If you want to test custom pragma settings, you can modify the `SimulationConfig` object in your code:

```python
from farm.core.config import SimulationConfig

config = SimulationConfig()
config.db_pragma_profile = "balanced"  # Base profile

# Override specific pragmas
config.db_custom_pragmas = {
    "synchronous": "OFF",
    "cache_size": -524288,  # 512MB
    "mmap_size": 536870912,  # 512MB
}
```

## Further Reading

For more information about SQLite pragma settings and their performance implications, see:

- [SQLite Pragma Documentation](https://www.sqlite.org/pragma.html)
- [SQLite Performance Optimization](https://www.sqlite.org/optimization.html)
- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
- [AgentFarm Database Module Documentation](https://agentfarm.readthedocs.io/en/latest/api/database.html) 