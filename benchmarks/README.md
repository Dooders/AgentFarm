# AgentFarm Benchmarks (Spec-driven)

This directory contains a spec-driven framework for creating, running, and analyzing benchmarks and profiles for the AgentFarm system.

## Directory Structure

- `core/`: Core framework
  - `experiments.py`: Experiment API and context
  - `registry.py`: Experiment registry and discovery
  - `runner.py`: Runner orchestration (iterations, instruments, reports)
  - `results.py`: Result schema with env/VCS capture
  - `spec.py`: YAML/JSON spec loader
  - `sweep.py`: SweepRunner for parameter grids
  - `compare.py`: Compare two run result JSON files
  - `reporting/markdown.py`: Markdown report generators
  - `instrumentation/`: timing, cProfile, psutil
- `implementations/`: Concrete experiments
  - `memory_db_benchmark.py`: Memory database performance benchmark
  - `observation_flow_benchmark.py`: Observation system benchmark
  - `perception_metrics_benchmark.py`: Perception metrics benchmark
  - `pragma_profile_benchmark.py`: SQLite pragma profile benchmark
  - `redis_memory_benchmark.py`: Redis memory system benchmark
  - `profiling/`: Profiling experiment implementations
  - `spatial/`: Spatial indexing benchmarks
- `examples/`: Example scripts and usage demonstrations
- `specs/`: YAML specification files for running benchmarks
- `utils/`: Utility functions for configuration, statistics, and visualization
- `run_benchmarks.py`: Spec-driven CLI

## Recommended Configuration for Post-Simulation Analysis

Based on benchmark results, the recommended configuration for simulations that require post-simulation analysis is:

```python
config.use_in_memory_db = True
config.persist_db_on_completion = True
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

### List available experiments

```bash
python -m benchmarks.run_benchmarks --list
```

### Run a single experiment (from spec)

```bash
python -m benchmarks.run_benchmarks --spec benchmarks/specs/observation_baseline.yaml
```

### Run a sweep (cartesian)

```bash
python -m benchmarks.run_benchmarks --spec benchmarks/specs/observation_sweep.yaml
```

### Run comprehensive profiling

```bash
# Run individual profiling benchmarks
python -m benchmarks.run_benchmarks --spec benchmarks/specs/pragma_profile_baseline.yaml
python -m benchmarks.run_benchmarks --spec benchmarks/specs/spatial_comprehensive_baseline.yaml
python -m benchmarks.run_benchmarks --spec benchmarks/specs/perception_metrics_baseline.yaml

# Run comprehensive profiling sweep
python -m benchmarks.run_benchmarks --spec benchmarks/specs/comprehensive_profiling_sweep.yaml
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

### Creating a New Experiment

1. Create a new file in `benchmarks/implementations/`, subclass `Experiment`, implement `setup/execute_once/teardown`.
2. Register with a slug using `@register_experiment("your_slug")`.
3. Provide a simple `param_schema` for defaults/validation.
4. Add a spec in `benchmarks/specs/` to run it.

### Comparing two runs (A/B)

```bash
python -m benchmarks.run_benchmarks --compare path/to/A.json path/to/B.json
```

## Available Spec Files

The `benchmarks/specs/` directory contains comprehensive specification files for various benchmarks:

### Database Benchmarks
- `memory_db_baseline.yaml` - Memory database performance baseline
- `pragma_profile_baseline.yaml` - SQLite pragma profile performance comparison
- `database_profiler_baseline.yaml` - Database operation profiling

### Memory System Benchmarks
- `redis_memory_baseline.yaml` - Redis memory system performance

### Perception & Observation Benchmarks
- `observation_baseline.yaml` - Observation flow baseline
- `observation_sweep.yaml` - Observation flow parameter sweep
- `perception_metrics_baseline.yaml` - Perception metrics performance
- `observation_profiler_baseline.yaml` - Observation system profiling

### Spatial Indexing Benchmarks
- `spatial_comprehensive_baseline.yaml` - Comprehensive spatial indexing performance
- `spatial_index_profiler_baseline.yaml` - Spatial indexing profiling
- `spatial_memory_profiler_baseline.yaml` - Spatial indexing memory usage
- `spatial_performance_analyzer_baseline.yaml` - Spatial performance analysis

### System Profiling
- `system_profiler_baseline.yaml` - System resource profiling

### Comprehensive Sweeps
- `comprehensive_profiling_sweep.yaml` - Multi-experiment profiling sweep

See `benchmarks/specs/README.md` for detailed information about each spec file.

## How to Extend and Add New Benchmarks

This section provides a comprehensive guide for extending the benchmark framework and adding new benchmarks.

### Adding a New Benchmark Experiment

#### 1. Create the Experiment Implementation

Create a new file in `benchmarks/implementations/` (or appropriate subdirectory):

```python
# benchmarks/implementations/my_new_benchmark.py
from typing import Any, Dict, Optional
from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.registry import register_experiment

@register_experiment("my_new_benchmark")
class MyNewBenchmark(Experiment):
    """
    Description of what this benchmark measures.
    
    This benchmark tests [specific functionality] and measures [specific metrics].
    """
    
    def __init__(
        self,
        param1: int = 100,
        param2: str = "default",
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the benchmark.
        
        Parameters
        ----------
        param1 : int
            Description of parameter 1
        param2 : str
            Description of parameter 2
        parameters : Dict[str, Any], optional
            Additional parameters for the benchmark
        """
        super().__init__(parameters or {})
        
        # Set benchmark-specific parameters
        self.params.update({
            "param1": param1,
            "param2": param2,
        })
        
        # Initialize any benchmark-specific attributes
        self.setup_data = None
    
    def setup(self, context: ExperimentContext) -> None:
        """Set up the benchmark environment."""
        # Initialize any resources needed for the benchmark
        # This runs once before all iterations
        self.setup_data = self._prepare_test_data()
    
    def execute_once(self, context: ExperimentContext) -> Dict[str, Any]:
        """
        Execute a single benchmark iteration.
        
        Returns
        -------
        Dict[str, Any]
            Raw results from this iteration
        """
        # Perform the actual benchmark work
        start_time = time.time()
        
        # Your benchmark logic here
        result = self._run_benchmark_logic()
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "duration": duration,
            "result_value": result,
            "param1": self.params["param1"],
            "param2": self.params["param2"],
        }
    
    def teardown(self, context: ExperimentContext) -> None:
        """Clean up resources after the benchmark."""
        # Clean up any resources allocated in setup()
        self.setup_data = None
    
    def _prepare_test_data(self):
        """Helper method to prepare test data."""
        # Implementation here
        pass
    
    def _run_benchmark_logic(self):
        """Helper method containing the actual benchmark logic."""
        # Implementation here
        pass
```

#### 2. Create a Spec File

Create a YAML specification file in `benchmarks/specs/`:

```yaml
# benchmarks/specs/my_new_benchmark_baseline.yaml
experiment: my_new_benchmark
params:
  # Test different parameter values
  param1: 100
  param2: "test_value"
  # Additional parameters
  parameters:
    custom_setting: true
    test_mode: "baseline"
iterations:
  warmup: 1      # Number of warmup iterations
  measured: 3    # Number of measured iterations
instrumentation:
  - timing       # Basic timing instrumentation
  - psutil       # System resource monitoring
output_dir: benchmarks/results/my_new_benchmark
tags: [custom, baseline]
notes: "Baseline test for my new benchmark"
seed: 42
```

#### 3. Test Your Benchmark

```bash
# Test the benchmark
python -m benchmarks.run_benchmarks --spec benchmarks/specs/my_new_benchmark_baseline.yaml

# List available experiments to verify registration
python -m benchmarks.run_benchmarks --list
```

### Creating Parameter Sweeps

For benchmarks that need to test multiple parameter combinations:

```yaml
# benchmarks/specs/my_new_benchmark_sweep.yaml
experiment: my_new_benchmark
params:
  param1: [50, 100, 200, 500]  # Multiple values for sweeping
  param2: ["option1", "option2", "option3"]
  parameters:
    custom_setting: true
iterations:
  warmup: 1
  measured: 2
instrumentation:
  - timing
  - psutil
output_dir: benchmarks/results/my_new_benchmark_sweep
tags: [custom, sweep]
notes: "Parameter sweep for my new benchmark"
seed: 42
```

### Adding Custom Instrumentation

To add new instrumentation beyond the built-in options:

#### 1. Create the Instrument

```python
# benchmarks/core/instrumentation/my_custom_instrument.py
import time
from contextlib import contextmanager
from typing import Dict, Any

@contextmanager
def my_custom_instrument(context: Dict[str, Any]):
    """Custom instrumentation for measuring specific metrics."""
    start_time = time.time()
    start_memory = get_memory_usage()  # Your custom metric
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # Record metrics
        context["my_custom_metric"] = {
            "duration": end_time - start_time,
            "memory_delta": end_memory - start_memory,
        }
```

#### 2. Register the Instrument

Update `benchmarks/core/instrumentation/__init__.py`:

```python
from .my_custom_instrument import my_custom_instrument

__all__ = [
    "timing",
    "psutil_monitor", 
    "cprofile",
    "my_custom_instrument",  # Add your instrument
]
```

#### 3. Use in Spec Files

```yaml
instrumentation:
  - timing
  - psutil
  - my_custom_instrument  # Use your custom instrument
```

### Adding Custom Reporters

To add new report generation capabilities:

#### 1. Create the Reporter

```python
# benchmarks/core/reporting/my_custom_reporter.py
from typing import List
from benchmarks.core.results import RunResult

def generate_custom_report(results: List[RunResult], output_dir: str) -> None:
    """Generate a custom report format."""
    # Your custom reporting logic here
    with open(f"{output_dir}/custom_report.txt", "w") as f:
        f.write("Custom Report\n")
        f.write("=" * 50 + "\n")
        
        for result in results:
            f.write(f"Experiment: {result.name}\n")
            f.write(f"Duration: {result.metrics.get('duration_s', {}).get('mean', 0):.3f}s\n")
            f.write("-" * 30 + "\n")
```

#### 2. Integrate with Runner

Update `benchmarks/core/runner.py` to include your reporter in the reporting pipeline.

### Best Practices for New Benchmarks

#### 1. **Parameter Design**
- Use meaningful parameter names
- Provide sensible defaults
- Document parameter purposes clearly
- Consider parameter ranges for sweeps

#### 2. **Error Handling**
- Handle expected errors gracefully
- Provide meaningful error messages
- Clean up resources in teardown even if errors occur

#### 3. **Resource Management**
- Initialize resources in `setup()`
- Clean up resources in `teardown()`
- Use context managers where appropriate

#### 4. **Metrics and Results**
- Return consistent result structures
- Include both raw data and derived metrics
- Use appropriate units and precision
- Document what each metric represents

#### 5. **Testing and Validation**
- Test with small parameter values first
- Validate results make sense
- Test error conditions
- Verify cleanup works properly

### Example: Complete Benchmark Implementation

Here's a complete example of a simple benchmark:

```python
# benchmarks/implementations/example_benchmark.py
import time
import random
from typing import Any, Dict, Optional
from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.registry import register_experiment

@register_experiment("example_benchmark")
class ExampleBenchmark(Experiment):
    """Example benchmark that demonstrates the framework."""
    
    def __init__(
        self,
        data_size: int = 1000,
        iterations: int = 100,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parameters or {})
        self.params.update({
            "data_size": data_size,
            "iterations": iterations,
        })
        self.test_data = None
    
    def setup(self, context: ExperimentContext) -> None:
        """Prepare test data."""
        self.test_data = [random.random() for _ in range(self.params["data_size"])]
    
    def execute_once(self, context: ExperimentContext) -> Dict[str, Any]:
        """Run the benchmark."""
        start_time = time.time()
        
        # Simulate some work
        result = 0
        for _ in range(self.params["iterations"]):
            result += sum(self.test_data)
        
        end_time = time.time()
        
        return {
            "duration": end_time - start_time,
            "result_sum": result,
            "data_size": self.params["data_size"],
            "iterations": self.params["iterations"],
        }
    
    def teardown(self, context: ExperimentContext) -> None:
        """Clean up."""
        self.test_data = None
```

### Integration with CI/CD

To integrate your new benchmark with the GitHub Actions workflow:

1. **Add to existing workflow** - Update `.github/workflows/database-performance-baseline.yml`
2. **Create dedicated workflow** - Create a new workflow file for your benchmark category
3. **Use spec files** - Reference your spec files in the workflow steps

Example workflow step:
```yaml
- name: Run custom benchmark
  run: |
    python -m benchmarks.run_benchmarks --spec benchmarks/specs/my_new_benchmark_baseline.yaml
```

### Troubleshooting Common Issues

#### **Benchmark not appearing in `--list`**
- Ensure `@register_experiment()` decorator is used
- Check that the module is imported somewhere
- Verify the experiment name is unique

#### **Import errors**
- Check that all dependencies are available
- Ensure proper Python path setup
- Verify module structure matches imports

#### **Parameter validation errors**
- Check parameter types match the spec file
- Ensure all required parameters are provided
- Verify parameter names match exactly

#### **Resource cleanup issues**
- Always implement `teardown()` method
- Use try/finally blocks for critical cleanup
- Check for resource leaks in long-running benchmarks

## Extending the Framework

- Add new instruments by implementing a context manager and wiring it in `Runner`.
- Add new reporters by consuming `RunResult` and generating artifacts.

## License

This benchmark framework is part of the AgentFarm project and is subject to the same license terms. 