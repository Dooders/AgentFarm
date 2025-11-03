# Hydra CLI Usage Examples

This document provides practical examples of using Hydra with the Agent Farm simulation framework.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Entry Point Examples](#entry-point-examples)
3. [CLI Override Examples](#cli-override-examples)
4. [Environment and Profile Examples](#environment-and-profile-examples)
5. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using run_simulation.py with Hydra

**Enable Hydra:**
```bash
python run_simulation.py --use-hydra
```

**With environment and profile:**
```bash
python run_simulation.py --use-hydra --environment production --profile benchmark
```

**With overrides:**
```bash
python run_simulation.py --use-hydra \
    --environment production \
    --profile benchmark \
    --hydra-overrides simulation_steps=2000 population.system_agents=50
```

## Entry Point Examples

### Example 1: Basic Simulation Run

**Legacy (default):**
```bash
python run_simulation.py --steps 100 --environment development
```

**With Hydra:**
```bash
python run_simulation.py --use-hydra --environment development
# Uses simulation_steps from config (default: 100)
```

**With Hydra and override:**
```bash
python run_simulation.py --use-hydra \
    --environment development \
    --hydra-overrides simulation_steps=200
```

### Example 2: Production Benchmark

**Legacy:**
```bash
python run_simulation.py \
    --environment production \
    --profile benchmark \
    --steps 2000 \
    --seed 42
```

**With Hydra:**
```bash
python run_simulation.py --use-hydra \
    --environment production \
    --profile benchmark \
    --hydra-overrides simulation_steps=2000 seed=42
```

### Example 3: Custom Agent Population

**With Hydra:**
```bash
python run_simulation.py --use-hydra \
    --environment production \
    --hydra-overrides \
        simulation_steps=1000 \
        population.system_agents=50 \
        population.independent_agents=30 \
        population.control_agents=20
```

### Example 4: Learning Parameter Tuning

**With Hydra:**
```bash
python run_simulation.py --use-hydra \
    --environment development \
    --hydra-overrides \
        learning.learning_rate=0.0005 \
        learning.gamma=0.99 \
        learning.epsilon_start=0.9 \
        learning.epsilon_decay=0.999
```

### Example 5: Environment Size Customization

**With Hydra:**
```bash
python run_simulation.py --use-hydra \
    --environment production \
    --hydra-overrides \
        environment.width=300 \
        environment.height=300 \
        simulation_steps=2000
```

## CLI Override Examples

### Overriding Simulation Parameters

```bash
# Number of steps
python run_simulation.py --use-hydra \
    --hydra-overrides simulation_steps=500

# Seed
python run_simulation.py --use-hydra \
    --hydra-overrides seed=12345

# Both
python run_simulation.py --use-hydra \
    --hydra-overrides simulation_steps=500 seed=12345
```

### Overriding Population Settings

```bash
# Agent counts
python run_simulation.py --use-hydra \
    --hydra-overrides \
        population.system_agents=50 \
        population.independent_agents=30 \
        population.control_agents=20

# Max population
python run_simulation.py --use-hydra \
    --hydra-overrides population.max_population=1000
```

### Overriding Environment Settings

```bash
# Environment size
python run_simulation.py --use-hydra \
    --hydra-overrides \
        environment.width=200 \
        environment.height=200

# Discretization method
python run_simulation.py --use-hydra \
    --hydra-overrides environment.position_discretization_method=round
```

### Overriding Learning Parameters

```bash
# DQN parameters
python run_simulation.py --use-hydra \
    --hydra-overrides \
        learning.learning_rate=0.0005 \
        learning.gamma=0.99 \
        learning.batch_size=64

# Epsilon schedule
python run_simulation.py --use-hydra \
    --hydra-overrides \
        learning.epsilon_start=0.9 \
        learning.epsilon_min=0.05 \
        learning.epsilon_decay=0.995
```

### Overriding Database Settings

```bash
# In-memory database
python run_simulation.py --use-hydra \
    --hydra-overrides \
        database.use_in_memory_db=true \
        database.persist_db_on_completion=true

# Cache size
python run_simulation.py --use-hydra \
    --hydra-overrides database.db_cache_size_mb=500
```

### Overriding Agent Behavior

```bash
# Movement and gathering
python run_simulation.py --use-hydra \
    --hydra-overrides \
        agent_behavior.max_movement=10 \
        agent_behavior.gathering_range=40 \
        agent_behavior.max_gather_amount=5

# Consumption and reproduction
python run_simulation.py --use-hydra \
    --hydra-overrides \
        agent_behavior.base_consumption_rate=0.2 \
        agent_behavior.offspring_cost=5 \
        agent_behavior.min_reproduction_resources=10
```

## Environment and Profile Examples

### Development Environment

```bash
# Default development
python run_simulation.py --use-hydra --environment development

# Development with custom steps
python run_simulation.py --use-hydra \
    --environment development \
    --hydra-overrides simulation_steps=50
```

### Production Environment

```bash
# Default production
python run_simulation.py --use-hydra --environment production

# Production with benchmark profile
python run_simulation.py --use-hydra \
    --environment production \
    --profile benchmark
```

### Testing Environment

```bash
# Quick test run
python run_simulation.py --use-hydra \
    --environment testing \
    --hydra-overrides simulation_steps=10
```

### Profile Combinations

```bash
# Benchmark profile
python run_simulation.py --use-hydra \
    --environment production \
    --profile benchmark

# Research profile
python run_simulation.py --use-hydra \
    --environment production \
    --profile research

# Simulation profile
python run_simulation.py --use-hydra \
    --environment production \
    --profile simulation
```

## Advanced Examples

### Complex Override Scenarios

**Multiple nested overrides:**
```bash
python run_simulation.py --use-hydra \
    --environment production \
    --profile benchmark \
    --hydra-overrides \
        simulation_steps=2000 \
        seed=42 \
        population.system_agents=50 \
        population.independent_agents=30 \
        environment.width=200 \
        environment.height=200 \
        learning.learning_rate=0.0005 \
        learning.gamma=0.99 \
        database.use_in_memory_db=true \
        database.db_cache_size_mb=500
```

**Per-module learning parameters:**
```bash
python run_simulation.py --use-hydra \
    --environment production \
    --hydra-overrides \
        modules.gather_learning_rate=0.001 \
        modules.share_learning_rate=0.0005 \
        modules.move_learning_rate=0.001 \
        modules.attack_learning_rate=0.001
```

### Using Environment Variable

**Set once, use everywhere:**
```bash
export USE_HYDRA_CONFIG=true
python run_simulation.py --environment production
python run_simulation.py --environment development --profile benchmark
```

**Disable for one run:**
```bash
export USE_HYDRA_CONFIG=true
python run_simulation.py --environment production  # Uses Hydra
USE_HYDRA_CONFIG=false python run_simulation.py --environment production  # Uses legacy
```

### Combining argparse and Hydra Overrides

**argparse for basic args, Hydra for config:**
```bash
python run_simulation.py --use-hydra \
    --environment production \
    --profile benchmark \
    --steps 2000 \
    --seed 42 \
    --perf-profile \
    --hydra-overrides \
        population.system_agents=50 \
        learning.learning_rate=0.0005
```

## Programmatic Usage Examples

### Using HydraConfigLoader Directly

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()
config = loader.load_config(
    environment="production",
    profile="benchmark",
    overrides=["simulation_steps=2000", "population.system_agents=50"]
)
```

### Using Compatibility Layer

```python
from farm.config import load_config

# With Hydra
config = load_config(
    environment="production",
    profile="benchmark",
    use_hydra=True,
    overrides=["simulation_steps=2000"]
)

# Without Hydra (legacy)
config = load_config(
    environment="production",
    profile="benchmark",
    use_hydra=False
)
```

### Using @hydra.main() Decorator

```python
import hydra
from omegaconf import DictConfig
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config = SimulationConfig.from_hydra(cfg)
    
    environment = run_simulation(
        num_steps=config.simulation_steps,
        config=config,
        path="simulations",
        save_config=True,
    )
    
    print(f"Simulation completed with {len(environment.agents)} agents")

if __name__ == "__main__":
    main()
```

Then run with:
```bash
python script.py environment=production profile=benchmark simulation_steps=2000
```

## Tips and Best Practices

### 1. Use Profiles for Common Scenarios

Instead of overriding many parameters each time, create profiles:
```bash
# Use benchmark profile instead of many overrides
python run_simulation.py --use-hydra --profile benchmark
```

### 2. Combine Environment and Profile

```bash
# Production + Benchmark = optimized for performance testing
python run_simulation.py --use-hydra \
    --environment production \
    --profile benchmark
```

### 3. Override Only What's Different

```bash
# Only override what you need to change
python run_simulation.py --use-hydra \
    --environment production \
    --hydra-overrides simulation_steps=2000  # Only change steps
```

### 4. Use Environment Variables for Consistency

```bash
# Set once per session
export USE_HYDRA_CONFIG=true
export HYDRA_ENV=production

# Use everywhere
python run_simulation.py --environment $HYDRA_ENV
```

### 5. Document Your Overrides

```bash
# Use comments in scripts
python run_simulation.py --use-hydra \
    --environment production \
    --hydra-overrides \
        simulation_steps=2000 \      # Longer run for stability
        population.system_agents=50 \ # More agents
        learning.learning_rate=0.0005  # Slower learning
```

## Troubleshooting

### Override Not Working

**Check syntax:**
```bash
# ? Correct
--hydra-overrides simulation_steps=200

# ? Wrong (spaces)
--hydra-overrides simulation_steps = 200

# ? Wrong (colon)
--hydra-overrides simulation_steps: 200
```

### Config Not Loading

**Check environment and profile:**
```bash
# List available options
python run_simulation.py --help

# Use valid options
python run_simulation.py --use-hydra \
    --environment development  # ? Valid
    # --environment invalid     # ? Invalid
```

### Hydra Not Found

**Install Hydra:**
```bash
pip install hydra-core>=1.3.0
```

## Next Steps

- See [HYDRA_USAGE.md](./HYDRA_USAGE.md) for more detailed usage
- See [HYDRA_MIGRATION_PLAN.md](./HYDRA_MIGRATION_PLAN.md) for migration details
- See Phase 4 documentation for multi-run and sweep examples
