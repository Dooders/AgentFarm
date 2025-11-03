# Hydra Multi-Run and Sweep Guide

This guide explains how to run multiple simulations with different configurations using Hydra's multi-run and sweep capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Native Hydra Multi-Run](#native-hydra-multi-run)
3. [Sweep Configurations](#sweep-configurations)
4. [Examples](#examples)
5. [Output Management](#output-management)
6. [Best Practices](#best-practices)

## Overview

Hydra provides two main approaches for running multiple experiments:

1. **Multi-Run**: Command-line grid search using `-m` flag
2. **Sweep Configs**: Predefined sweep configurations in YAML files

Both approaches automatically:
- Create separate output directories for each run
- Save configuration files with results
- Organize outputs by date/time
- Enable parallel execution (with joblib launcher)

## Native Hydra Multi-Run

### Basic Multi-Run

Use the `run_simulation_hydra.py` entry point with the `-m` flag:

```bash
python run_simulation_hydra.py -m learning.learning_rate=0.0001,0.0005,0.001
```

This runs 3 simulations with different learning rates.

### Grid Search

Multiple parameters create a cartesian product:

```bash
python run_simulation_hydra.py -m \
    learning.learning_rate=0.0001,0.0005,0.001 \
    learning.gamma=0.9,0.99
```

This runs 6 simulations (3 ? 2 = 6 combinations).

### With Environment and Profile

```bash
python run_simulation_hydra.py -m \
    environment=production \
    profile=benchmark \
    learning.learning_rate=0.0001,0.0005,0.001
```

### Override During Multi-Run

```bash
python run_simulation_hydra.py -m \
    learning.learning_rate=0.0001,0.0005,0.001 \
    simulation_steps=200 \
    population.system_agents=50
```

All runs use `simulation_steps=200` and `population.system_agents=50`.

## Sweep Configurations

### Using Predefined Sweeps

Load a sweep configuration from `conf/sweeps/`:

```bash
python run_simulation_hydra.py \
    --config-path=conf/sweeps \
    --config-name=learning_rate_sweep \
    -m
```

### Available Sweeps

See `conf/sweeps/README.md` for all available sweep configurations:

- `learning_rate_sweep.yaml` - Learning rate optimization
- `population_sweep.yaml` - Population size studies
- `environment_size_sweep.yaml` - Environment dimension tests
- `hyperparameter_grid.yaml` - Comprehensive grid search
- `agent_behavior_sweep.yaml` - Agent behavior parameters

### Override Sweep Parameters

You can override parameters when using a sweep:

```bash
python run_simulation_hydra.py \
    --config-path=conf/sweeps \
    --config-name=learning_rate_sweep \
    -m \
    simulation_steps=200
```

This uses the sweep's learning rate range but overrides `simulation_steps`.

## Examples

### Example 1: Quick Learning Rate Test

```bash
python run_simulation_hydra.py -m \
    learning.learning_rate=0.0001,0.0005,0.001 \
    simulation_steps=100 \
    environment=development
```

**Runs:** 3 simulations  
**Time:** ~5-10 minutes per run  
**Output:** `outputs/YYYY-MM-DD/HH-MM-SS/0/`, `1/`, `2/`

### Example 2: Population Study

```bash
python run_simulation_hydra.py -m \
    population.system_agents=10,20,30,40,50 \
    population.independent_agents=5,10,15 \
    simulation_steps=200 \
    environment=production
```

**Runs:** 15 simulations (5 ? 3)  
**Time:** ~10-15 minutes per run  
**Output:** Organized in `outputs/` directory

### Example 3: Comprehensive Hyperparameter Search

```bash
python run_simulation_hydra.py \
    --config-path=conf/sweeps \
    --config-name=hyperparameter_grid \
    -m
```

**Runs:** 144 simulations (4 ? 3 ? 3 ? 4)  
**Time:** ~10-15 minutes per run  
**Note:** This is a large sweep - consider reducing parameters or steps

### Example 4: Custom Grid Search

```bash
python run_simulation_hydra.py -m \
    learning.learning_rate=0.0001,0.0005,0.001 \
    learning.gamma=0.9,0.95,0.99 \
    learning.batch_size=32,64 \
    simulation_steps=200 \
    environment=production \
    profile=benchmark
```

**Runs:** 18 simulations (3 ? 3 ? 2)

### Example 5: Environment Size Sweep

```bash
python run_simulation_hydra.py -m \
    environment.width=50,100,150,200 \
    environment.height=50,100,150,200 \
    simulation_steps=100
```

**Runs:** 16 simulations (4 ? 4)

## Output Management

### Output Structure

Hydra creates organized output directories:

```
outputs/
??? 2024-01-15/
    ??? 14-30-25_sweep_name/
        ??? 0/
        ?   ??? .hydra/
        ?   ?   ??? config.yaml          # Config used for this run
        ?   ?   ??? hydra.yaml           # Hydra settings
        ?   ??? simulation_*.db          # Simulation database
        ??? 1/
        ?   ??? ...
        ??? multirun.yaml                # Multi-run summary
        ??? .hydra/
            ??? ...
```

### Single Run Output

```
outputs/
??? 2024-01-15/
    ??? 14-30-25/
        ??? .hydra/
        ?   ??? config.yaml
        ?   ??? hydra.yaml
        ??? simulation_*.db
```

### Accessing Results

Each run's config is saved in `.hydra/config.yaml`:

```bash
# View config for run 0
cat outputs/2024-01-15/14-30-25_sweep_name/0/.hydra/config.yaml

# View multi-run summary
cat outputs/2024-01-15/14-30-25_sweep_name/multirun.yaml
```

### Custom Output Directory

```bash
python run_simulation_hydra.py -m \
    learning.learning_rate=0.0001,0.0005,0.001 \
    hydra.run.dir=./my_sweep_results
```

## Best Practices

### 1. Start Small

Test with 2-3 parameter values first:

```bash
# Test with 2 values
python run_simulation_hydra.py -m learning.learning_rate=0.0001,0.001

# Then expand
python run_simulation_hydra.py -m learning.learning_rate=0.0001,0.0005,0.001,0.005
```

### 2. Use Appropriate Steps

Reduce `simulation_steps` for faster sweeps:

```bash
# Quick sweep
python run_simulation_hydra.py -m \
    learning.learning_rate=0.0001,0.0005,0.001 \
    simulation_steps=50

# Full sweep
python run_simulation_hydra.py -m \
    learning.learning_rate=0.0001,0.0005,0.001 \
    simulation_steps=1000
```

### 3. Use Benchmark Profile

For faster runs during parameter sweeps:

```bash
python run_simulation_hydra.py -m \
    profile=benchmark \
    learning.learning_rate=0.0001,0.0005,0.001
```

### 4. Monitor Resource Usage

Large sweeps can be computationally expensive:

- **Small (< 10 runs)**: Use full simulation steps
- **Medium (10-50 runs)**: Reduce steps to 100-200
- **Large (> 50 runs)**: Reduce steps to 50-100

### 5. Organize Sweeps

Use descriptive sweep names and organize outputs:

```bash
# Learning rate study
python run_simulation_hydra.py -m \
    hydra.run.dir=./sweeps/learning_rate_$(date +%Y%m%d) \
    learning.learning_rate=0.0001,0.0005,0.001

# Population study
python run_simulation_hydra.py -m \
    hydra.run.dir=./sweeps/population_$(date +%Y%m%d) \
    population.system_agents=10,20,30,40,50
```

### 6. Save Sweep Definitions

Create custom sweep configs in `conf/sweeps/` for reproducibility:

```yaml
# conf/sweeps/my_custom_sweep.yaml
defaults:
  - /config
  - override /hydra/sweeper: basic

hydra:
  sweeper:
    params:
      learning.learning_rate: range(0.0001, 0.001, 0.0001)
      simulation_steps: 100
```

### 7. Parallel Execution

For faster sweeps, use Hydra's joblib launcher:

```bash
python run_simulation_hydra.py -m \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=4 \
    learning.learning_rate=0.0001,0.0005,0.001
```

**Note:** Requires `joblib` plugin: `pip install hydra-joblib-launcher`

## Troubleshooting

### Sweep Not Running

**Problem:** Sweep runs only once instead of multiple times

**Solution:** Ensure you're using `-m` flag:
```bash
python run_simulation_hydra.py -m learning.learning_rate=0.0001,0.0005
```

### Too Many Runs

**Problem:** Grid search creates too many combinations

**Solution:** Reduce parameter values or use a sweep config:
```bash
# Instead of 100 combinations
python run_simulation_hydra.py -m \
    param1=1,2,3,4,5,6,7,8,9,10 \
    param2=1,2,3,4,5,6,7,8,9,10

# Use smaller set
python run_simulation_hydra.py -m \
    param1=1,3,5,7,9 \
    param2=1,3,5,7,9
```

### Output Directory Full

**Problem:** Sweeps create many output directories

**Solution:** Use custom output directory and clean up:
```bash
python run_simulation_hydra.py -m \
    hydra.run.dir=./sweep_results \
    learning.learning_rate=0.0001,0.0005,0.001

# After analysis, clean up
rm -rf ./sweep_results
```

## Advanced Usage

### Custom Sweeper

Hydra supports custom sweepers (Optuna, Ax, etc.):

```yaml
# conf/sweeps/optuna_sweep.yaml
defaults:
  - /config
  - override /hydra/sweeper: optuna

hydra:
  sweeper:
    study_name: learning_rate_study
    direction: maximize
    n_trials: 50
    params:
      learning.learning_rate: interval(0.0001, 0.01)
```

**Note:** Requires installing sweeper plugin (e.g., `hydra-optuna-sweeper`)

### Combining Multi-Run with Overrides

```bash
python run_simulation_hydra.py -m \
    learning.learning_rate=0.0001,0.0005,0.001 \
    +custom_param=new_value \
    ~override_to_remove
```

## Next Steps

- See [HYDRA_CLI_EXAMPLES.md](./HYDRA_CLI_EXAMPLES.md) for more CLI examples
- See [conf/sweeps/README.md](../../conf/sweeps/README.md) for sweep configurations
- See [HYDRA_USAGE.md](./HYDRA_USAGE.md) for general Hydra usage
