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
  - `reporting/markdown.py`: Markdown report generators
  - `instrumentation/`: timing, cProfile, psutil
- `implementations/`: Concrete experiments (e.g., `observation_flow_experiment.py`)
- `run_benchmarks.py`: Spec-driven CLI
- `specs/`: Example specs (single-run, sweep)

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

## Extending the Framework

- Add new instruments by implementing a context manager and wiring it in `Runner`.
- Add new reporters by consuming `RunResult` and generating artifacts.

## License

This benchmark framework is part of the AgentFarm project and is subject to the same license terms. 