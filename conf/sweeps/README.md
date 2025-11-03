# Hydra Sweep Configurations

This directory contains predefined sweep configurations for hyperparameter tuning and multi-run experiments.

## Available Sweeps

### 1. `learning_rate_sweep.yaml`
Tests different learning rates for the DQN agent.

**Parameters swept:**
- `learning.learning_rate`: 0.0001 to 0.001 (step 0.0001)

**Total runs:** ~10 experiments

### 2. `population_sweep.yaml`
Tests different agent population sizes.

**Parameters swept:**
- `population.system_agents`: 10, 20, 30, 40, 50
- `population.independent_agents`: 5, 10, 15, 20, 25

**Total runs:** 25 experiments (5 ? 5 grid)

### 3. `environment_size_sweep.yaml`
Tests different environment dimensions.

**Parameters swept:**
- `environment.width`: 50, 100, 150, 200, 250
- `environment.height`: 50, 100, 150, 200, 250

**Total runs:** 25 experiments (5 ? 5 grid)

### 4. `hyperparameter_grid.yaml`
Comprehensive grid search over learning parameters.

**Parameters swept:**
- `learning.learning_rate`: 0.0001, 0.0005, 0.001, 0.005
- `learning.gamma`: 0.9, 0.95, 0.99
- `learning.batch_size`: 32, 64, 128
- `learning.epsilon_start`: 0.8, 0.9, 1.0

**Total runs:** 144 experiments (4 ? 3 ? 3 ? 4 grid)

### 5. `agent_behavior_sweep.yaml`
Tests different agent behavior parameters.

**Parameters swept:**
- `agent_behavior.max_movement`: 5, 10, 15, 20
- `agent_behavior.gathering_range`: 20, 30, 40, 50
- `agent_behavior.base_consumption_rate`: 0.1, 0.2, 0.3, 0.4

**Total runs:** 64 experiments (4 ? 4 ? 4 grid)

## Usage

### Method 1: Using run_simulation.py with --sweep flag

```bash
python run_simulation.py --use-hydra --sweep learning_rate_sweep
```

### Method 2: Using Hydra's native multi-run

Create a script using `@hydra.main()` decorator (see `run_simulation_hydra.py`):

```bash
python run_simulation_hydra.py -m learning.learning_rate=0.0001,0.0005,0.001
```

### Method 3: Manual multi-run with command-line

```bash
python run_simulation.py --use-hydra \
    --hydra-overrides learning.learning_rate=0.0001
python run_simulation.py --use-hydra \
    --hydra-overrides learning.learning_rate=0.0005
python run_simulation.py --use-hydra \
    --hydra-overrides learning.learning_rate=0.001
```

## Creating Custom Sweeps

Create a new YAML file in this directory:

```yaml
defaults:
  - /config
  - override /hydra/sweeper: basic

hydra:
  sweeper:
    params:
      parameter.name: range(start, end, step)  # For numeric ranges
      parameter.name: choice(val1, val2, val3)  # For discrete choices

simulation_steps: 100
environment: development
profile: null
```

### Sweep Parameter Syntax

**Range (numeric):**
```yaml
learning.learning_rate: range(0.0001, 0.001, 0.0001)
# Generates: 0.0001, 0.0002, 0.0003, ..., 0.001
```

**Choice (discrete):**
```yaml
learning.gamma: choice(0.9, 0.95, 0.99)
# Generates: 0.9, 0.95, 0.99
```

**Grid (multiple parameters):**
```yaml
params:
  learning.learning_rate: choice(0.0001, 0.001)
  learning.gamma: choice(0.9, 0.99)
# Generates: (0.0001, 0.9), (0.0001, 0.99), (0.001, 0.9), (0.001, 0.99)
```

## Output Structure

When running sweeps, Hydra creates output directories:

```
outputs/
??? 2024-01-15/
?   ??? 14-30-25_sweep_name/
?   ?   ??? 0/
?   ?   ?   ??? .hydra/
?   ?   ?   ?   ??? config.yaml
?   ?   ?   ?   ??? hydra.yaml
?   ?   ?   ??? simulation.db
?   ?   ??? 1/
?   ?   ?   ??? ...
?   ?   ??? multirun.yaml
```

## Best Practices

1. **Start small**: Test with 2-3 parameter values first
2. **Use benchmark profile**: For faster runs during sweeps
3. **Monitor resources**: Large sweeps can be computationally expensive
4. **Save results**: Ensure simulation databases are saved for analysis
5. **Use appropriate steps**: Reduce `simulation_steps` for faster sweeps

## Performance Considerations

- **Grid search**: Exponential growth (cartesian product)
- **Small sweeps**: < 10 runs - use full simulation steps
- **Medium sweeps**: 10-50 runs - reduce steps to 100-200
- **Large sweeps**: > 50 runs - reduce steps to 50-100

## Examples

### Quick Learning Rate Test
```bash
python run_simulation.py --use-hydra \
    --sweep learning_rate_sweep \
    --hydra-overrides simulation_steps=50
```

### Population Study
```bash
python run_simulation.py --use-hydra \
    --sweep population_sweep \
    --environment production
```

### Custom Quick Sweep
```bash
python run_simulation.py --use-hydra \
    --hydra-overrides \
        learning.learning_rate=0.0001,0.0005,0.001 \
        simulation_steps=100
```
