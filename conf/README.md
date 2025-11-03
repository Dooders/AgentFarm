# Hydra Configuration Directory

This directory contains the Hydra-based configuration system for the Agent Farm simulation framework.

## Directory Structure

```
conf/
??? config.yaml                    # Main config entry point
??? defaults/
?   ??? environment/               # Environment config group
?   ?   ??? development.yaml       # Development environment
?   ?   ??? production.yaml        # Production environment
?   ?   ??? testing.yaml          # Testing environment
?   ??? profile/                   # Profile config group
?       ??? benchmark.yaml         # Benchmark profile
?       ??? research.yaml         # Research profile
?       ??? simulation.yaml       # Simulation profile
?       ??? null.yaml             # Empty profile (for optional profiles)
??? sweeps/                        # Sweep configurations (for multi-run)
```

## Usage

### Basic Usage

Load configuration with default environment (development):
```python
from hydra import compose, initialize
from omegaconf import OmegaConf

with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config")
```

### With Environment Override

```python
with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config", overrides=["environment=production"])
```

### With Profile

```python
with initialize(config_path="conf", version_base=None):
    cfg = compose(
        config_name="config",
        overrides=["environment=production", "profile=benchmark"]
    )
```

### Command Line Overrides

```bash
python run_simulation.py \
    environment=production \
    profile=benchmark \
    simulation_steps=2000 \
    population.system_agents=50
```

## Config Groups

### Environment (`environment/`)
- **development**: Optimized for development and debugging
- **production**: Optimized for performance and reliability
- **testing**: Optimized for automated testing and CI/CD

### Profile (`profile/`)
- **benchmark**: Optimized for performance benchmarking
- **research**: Optimized for experimental research
- **simulation**: Standard settings for regular simulation runs
- **null**: No profile (default)

## Migration Notes

This Hydra configuration system runs alongside the legacy config system in `farm/config/`. 
The legacy system remains functional until the migration is complete.

To use Hydra:
1. Set environment variable: `USE_HYDRA_CONFIG=true`
2. Or use the new Hydra-based entry points directly

## Next Steps

- Phase 2: Implement HydraConfigLoader
- Phase 3: Update entry points with @hydra.main()
- Phase 4: Add multi-run and sweep support
