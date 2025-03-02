# AgentFarm Benchmarks

This directory contains a framework for creating, running, and analyzing benchmarks for the AgentFarm system.

## Directory Structure

- `base/`: Base classes for benchmarks
  - `benchmark.py`: Abstract base class for all benchmarks
  - `runner.py`: Class for running benchmarks
  - `results.py`: Class for storing and analyzing benchmark results
- `implementations/`: Concrete benchmark implementations
  - `memory_db_benchmark.py`: Benchmark for comparing disk-based and in-memory databases
- `results/`: Directory for storing benchmark results
- `utils/`: Utility functions for benchmarks
  - `visualization.py`: Functions for visualizing benchmark results
  - `statistics.py`: Functions for statistical analysis of benchmark results
  - `config_helper.py`: Helper functions for configuring simulations
- `examples/`: Example scripts for using the benchmark framework
  - `run_with_recommended_config.py`: Example of running a simulation with the recommended configuration
- `run_benchmarks.py`: Main script for running benchmarks

## Recommended Configuration for Post-Simulation Analysis

Based on benchmark results, the recommended configuration for simulations that require post-simulation analysis is:

```python
config.use_in_memory_db = True
config.persist_in_memory_db = True
```

This configuration provides:
- **33.6% faster execution** than disk-based database
- **Full data persistence** for post-simulation analysis
- **Good balance** between performance and data durability

You can use the helper functions in `benchmarks.utils.config_helper` to apply these settings:

```python
from benchmarks.utils.config_helper import configure_for_performance_with_persistence

# Apply recommended settings to an existing config
config = configure_for_performance_with_persistence(config)

# Or get a fully configured SimulationConfig
from benchmarks.utils.config_helper import get_recommended_config
config = get_recommended_config(num_agents=30, num_steps=100)
```

## Usage

### Running Benchmarks

To run all available benchmarks with default parameters:

```bash
python -m benchmarks.run_benchmarks
```

To run a specific benchmark:

```bash
python -m benchmarks.run_benchmarks --benchmark memory_db
```

To customize benchmark parameters:

```bash
python -m benchmarks.run_benchmarks --benchmark memory_db --steps 200 --agents 50 --iterations 5
```

### Running a Simulation with Recommended Configuration

To run a simulation with the recommended configuration for post-simulation analysis:

```bash
python -m benchmarks.examples.run_with_recommended_config
```

You can customize the simulation parameters:

```bash
python -m benchmarks.examples.run_with_recommended_config --steps 200 --agents 50 --output my_simulation_results
```

### Creating a New Benchmark

To create a new benchmark, follow these steps:

1. Create a new Python file in the `implementations/` directory
2. Define a class that inherits from `Benchmark`
3. Implement the required methods: `setup()`, `run()`, and `cleanup()`
4. Register the benchmark in `run_benchmarks.py`

Example:

```python
from benchmarks.base.benchmark import Benchmark

class MyBenchmark(Benchmark):
    def __init__(self, param1=10, param2=20, parameters=None):
        super().__init__(
            name="my_benchmark",
            description="My custom benchmark",
            parameters=parameters or {},
        )
        
        self.parameters.update({
            "param1": param1,
            "param2": param2,
        })
    
    def setup(self):
        # Set up the benchmark environment
        pass
    
    def run(self):
        # Run the benchmark
        # Return a dictionary of results
        return {"result1": 1.0, "result2": 2.0}
    
    def cleanup(self):
        # Clean up after the benchmark
        pass
```

### Analyzing Benchmark Results

The `BenchmarkResults` class provides methods for analyzing benchmark results:

```python
from benchmarks.base.runner import BenchmarkRunner

# Load results from a file
runner = BenchmarkRunner()
results = runner.load_results("benchmarks/results/my_benchmark_20250101_120000.json")

# Get summary statistics
summary = results.get_summary()
print(f"Mean duration: {summary['mean_duration']:.2f} seconds")

# Plot durations
results.plot_durations()

# Compare with another benchmark
other_results = runner.load_results("benchmarks/results/other_benchmark_20250101_120000.json")
results.compare_with(other_results)
```

## Extending the Framework

The benchmark framework is designed to be extensible. You can add new functionality by:

1. Adding new methods to the base classes
2. Creating new utility functions
3. Implementing new benchmark types

## License

This benchmark framework is part of the AgentFarm project and is subject to the same license terms. 