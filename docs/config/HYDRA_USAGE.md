# Hydra Configuration Usage Guide

This guide explains how to use the Hydra-based configuration system in Agent Farm.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Environment and Profiles](#environment-and-profiles)
4. [Command-Line Overrides](#command-line-overrides)
5. [Programmatic Usage](#programmatic-usage)
6. [Compatibility Layer](#compatibility-layer)
7. [Migration from Legacy System](#migration-from-legacy-system)

## Quick Start

### Installation

First, install Hydra:
```bash
pip install hydra-core>=1.3.0
```

### Basic Example

```python
from farm.config import load_config

# Load config (always uses Hydra)
config = load_config(environment="production")

# Use the config
print(f"Simulation steps: {config.simulation_steps}")
print(f"Environment size: {config.environment.width}x{config.environment.height}")
```

## Basic Usage

### Using the Compatibility Layer (Recommended)

The easiest way to use Hydra is through the compatibility layer:

```python
from farm.config import load_config

# Load config (always uses Hydra)
config = load_config(
    environment="development",
    profile="benchmark"
)
```

### Using HydraConfigLoader Directly

For more control, use the loader directly:

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()
config = loader.load_config(
    environment="production",
    profile="simulation"
)
```

### Using Hydra Decorator (For Entry Points)

For command-line applications, use the `@hydra.main()` decorator:

```python
import hydra
from omegaconf import DictConfig
from farm.config import SimulationConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert Hydra config to SimulationConfig
    config = SimulationConfig.from_hydra(cfg)
    
    # Use config
    print(f"Steps: {config.simulation_steps}")
    # ... rest of your code

if __name__ == "__main__":
    main()
```

Then run with:
```bash
python script.py environment=production profile=benchmark simulation_steps=2000
```

## Environment and Profiles

### Available Environments

- **development**: Optimized for development and debugging
  - Smaller simulation size (50x50)
  - Debug enabled
  - In-memory database
  
- **production**: Optimized for performance and reliability
  - Larger simulation size (200x200)
  - Debug disabled
  - Persistent database
  
- **testing**: Optimized for automated testing
  - Minimal simulation size (25x25)
  - Fast execution
  - In-memory database

### Available Profiles

- **benchmark**: Performance benchmarking
- **simulation**: Standard simulation runs
- **research**: Experimental research settings
- **null**: No profile (default)

### Example: Combining Environment and Profile

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()

# Production environment with benchmark profile
config = loader.load_config(
    environment="production",
    profile="benchmark"
)
```

## Command-Line Overrides

### Using Hydra Decorator

When using `@hydra.main()`, you can override any config value:

```bash
# Override top-level parameters
python run_simulation.py simulation_steps=2000 seed=42

# Override nested parameters
python run_simulation.py environment.width=300 environment.height=300

# Override multiple values
python run_simulation.py \
    simulation_steps=2000 \
    population.system_agents=50 \
    learning.learning_rate=0.0005
```

### Using Loader with Overrides

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()
config = loader.load_config(
    environment="development",
    overrides=[
        "simulation_steps=200",
        "population.system_agents=50",
        "learning.learning_rate=0.0005"
    ]
)
```

### Convenience Method

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()
config = loader.load_config_with_overrides(
    "simulation_steps=200",
    "population.system_agents=50",
    environment="production"
)
```

## Programmatic Usage

### Example 1: Basic Loading

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()
config = loader.load_config(environment="development")

# Access config values
print(f"Steps: {config.simulation_steps}")
print(f"Width: {config.environment.width}")
print(f"Agents: {config.population.system_agents}")
```

### Example 2: With Profile

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()
config = loader.load_config(
    environment="production",
    profile="research"
)

# Config values are merged from:
# 1. Base config (config.yaml)
# 2. Production environment overrides
# 3. Research profile overrides
```

### Example 3: Using from_hydra

```python
from hydra import compose, initialize
from farm.config import SimulationConfig

with initialize(config_path="conf", version_base=None):
    cfg = compose(
        config_name="config",
        overrides=["environment=production", "profile=benchmark"]
    )
    config = SimulationConfig.from_hydra(cfg)
```

## Compatibility Layer

The compatibility layer allows you to switch between Hydra and legacy config systems:

### Environment Variable

Set `USE_HYDRA_CONFIG=true` to use Hydra by default:

```bash
export USE_HYDRA_CONFIG=true
python run_simulation.py  # Will use Hydra
```

### Explicit Flag

```python
from farm.config import load_config

# Use Hydra
config = load_config(environment="production", use_hydra=True)

# Use legacy system
config = load_config(environment="production", use_hydra=False)
```

### Automatic Fallback

If Hydra is not installed, the compatibility layer automatically falls back to the legacy system:

```python
from farm.config import load_config

# Will use legacy system if Hydra not installed
config = load_config(environment="production", use_hydra=True)
# Raises ImportError if Hydra not available and use_hydra=True
```

## Migration from Legacy System

### Step 1: Install Hydra

```bash
pip install hydra-core>=1.3.0
```

### Step 2: Test Hydra Loading

```python
from farm.config import load_config

# Test loading with Hydra
config = load_config(environment="development", use_hydra=True)
print(f"Loaded: {config.simulation_steps} steps")
```

### Step 3: Update Entry Points

Replace legacy config loading:

**Before:**
```python
from farm.config import SimulationConfig

config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="benchmark"
)
```

**After (Option 1 - Compatibility Layer):**
```python
from farm.config import load_config

config = load_config(
    environment="production",
    profile="benchmark",
    use_hydra=True
)
```

**After (Option 2 - Hydra Decorator):**
```python
import hydra
from omegaconf import DictConfig
from farm.config import SimulationConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config = SimulationConfig.from_hydra(cfg)
    # ... rest of code
```

### Step 4: Enable Gradually

Use environment variable for gradual migration:

```bash
# Test in development
export USE_HYDRA_CONFIG=true
python run_simulation.py environment=development

# Verify it works, then enable in production
```

## Advanced Usage

### Custom Config Path

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader(config_path="custom/conf/path")
config = loader.load_config(environment="development")
```

### Multiple Overrides

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()
config = loader.load_config(
    environment="production",
    overrides=[
        "simulation_steps=2000",
        "population.system_agents=50",
        "population.independent_agents=50",
        "learning.learning_rate=0.0005",
        "learning.gamma=0.99"
    ]
)
```

### Clearing Hydra State (for testing)

```python
from farm.config.hydra_loader import HydraConfigLoader

loader = HydraConfigLoader()
loader.load_config(environment="development")

# Clear state
loader.clear()

# Reinitialize
loader.load_config(environment="production")
```

## Troubleshooting

### Hydra Not Found

**Error**: `ImportError: Hydra is not available`

**Solution**: Install Hydra:
```bash
pip install hydra-core>=1.3.0
```

### Config Directory Not Found

**Error**: `FileNotFoundError: Config directory not found`

**Solution**: Ensure Phase 1 is complete and `conf/` directory exists.

### Invalid Environment/Profile

**Error**: `ValueError: Failed to load Hydra config`

**Solution**: Check that environment/profile names are correct:
- Environments: `development`, `production`, `testing`
- Profiles: `benchmark`, `simulation`, `research`, `null`

### Config Values Not Overridden

**Issue**: Overrides don't seem to work

**Solution**: Check override syntax:
- ? Correct: `"simulation_steps=200"`
- ? Correct: `"population.system_agents=50"`
- ? Wrong: `"simulation_steps = 200"` (spaces)
- ? Wrong: `"simulation_steps: 200"` (colon)

## Testing

Run the test script to verify Hydra config loading:

```bash
python scripts/test_hydra_config.py
```

This will test:
- Direct loader usage
- Profile loading
- Overrides
- Compatibility layer
- All environments and profiles

## Next Steps

- See [HYDRA_MIGRATION_PLAN.md](./HYDRA_MIGRATION_PLAN.md) for full migration details
- See [conf/README.md](../../conf/README.md) for config structure details
- See Phase 3 documentation for entry point updates
