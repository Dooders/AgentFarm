# ExperimentRunner — Running Multi-Iteration Simulations

The generic `ExperimentRunner` provides functionality to run multiple
simulation iterations with different parameters and analyze the results.
This page describes how to use the runner directly; for the catalog of
**defined experiments** (intrinsic evolution, hyperparameter convergence,
multi-seed cohorts, etc.), see [Experiments](experiments.md).

## Overview

The runner allows you to:

- Run multiple iterations of simulations
- Test different parameter configurations
- Collect and analyze results
- Generate summary reports
- Compare outcomes across iterations

The implementation lives in
[`farm/runners/experiment_runner.py`](../farm/runners/experiment_runner.py).

## Usage

### Basic example

```python
from farm.runners import ExperimentRunner
from farm.config import SimulationConfig

base_config = SimulationConfig(
    num_steps=1000,
    num_agents=100,
)

runner = ExperimentRunner(
    base_config=base_config,
    experiment_name="my_experiment",
)

runner.run_iterations(num_iterations=5)
runner.generate_report()
```

### Parameter variations

You can test different parameter configurations across iterations:

```python
variations = [
    {"num_agents": 50},
    {"num_agents": 150},
    {"resource_rate": 2.0},
    {"system_threshold": 0.8},
]

runner.run_iterations(
    num_iterations=len(variations),
    config_variations=variations,
)
```

## Results analysis

The experiment runner generates two main result files:

1. `{experiment_name}_results.csv` — Detailed results for each iteration,
   including:
   - Final number of system and independent agents
   - Average resources per agent type
   - Timestamp
   - Configuration variation used
   - Iteration number
2. `{experiment_name}_summary.csv` — Statistical summary of results across
   all iterations, including:
   - Mean, standard deviation, min, max values
   - Quartile distributions
   - Count of successful iterations

## Logging

Each experiment maintains its own log file (`{experiment_name}.log`) that
captures:

- Experiment progress
- Iteration status
- Error messages
- Configuration details

The log file lives in the experiment's root directory.

## Best practices

1. Use meaningful experiment names that reflect the purpose of the test.
2. Start with a small number of iterations to validate configuration.
3. Save base configurations for reproducibility.
4. Document parameter variations used in experiments.
5. Review logs for any warnings or errors before analyzing results.

## Error handling

The experiment runner includes built-in error handling to:

- Continue running if individual iterations fail
- Log error details for debugging
- Mark failed iterations in results

Failed iterations are logged but do not prevent the completion of the
overall experiment.
