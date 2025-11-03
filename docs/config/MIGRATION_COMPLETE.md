# Hydra Migration - Complete

## ? Migration Fully Complete

The migration to Hydra is **100% complete**. All legacy code and backward compatibility layers have been removed.

## What Changed

### Removed
- ? Compatibility layer (`use_hydra` parameter)
- ? `USE_HYDRA_CONFIG` environment variable
- ? `--use-hydra` flag from `run_simulation.py`
- ? Legacy config loading code paths
- ? Backward compatibility tests

### Updated
- ? `load_config()` now **always** uses Hydra
- ? `from_centralized_config()` uses Hydra internally
- ? `run_simulation.py` uses Hydra by default
- ? All tests updated to use Hydra only
- ? Documentation updated

## Current State

### Configuration Loading

**Always uses Hydra:**
```python
from farm.config import load_config

# This always uses Hydra now
config = load_config(environment="production", profile="benchmark")
```

**Legacy method name still works (uses Hydra internally):**
```python
from farm.config import SimulationConfig

# This also uses Hydra internally
config = SimulationConfig.from_centralized_config(environment="production")
```

### Entry Points

**Standard entry point (uses Hydra):**
```bash
python run_simulation.py --environment production
```

**With overrides:**
```bash
python run_simulation.py \
    --hydra-overrides simulation_steps=2000 population.system_agents=50
```

**Native Hydra entry point (for multi-run):**
```bash
python run_simulation_hydra.py -m learning.learning_rate=0.0001,0.0005,0.001
```

## Benefits

1. **Simplified Codebase** - No dual code paths
2. **Consistent Behavior** - Always uses Hydra
3. **Better Features** - Full Hydra capabilities available
4. **Cleaner API** - No feature flags or compatibility layers

## Migration Notes

- All existing code using `load_config()` will continue to work
- Code using `from_centralized_config()` will continue to work (now uses Hydra)
- No breaking changes to the public API
- Internal implementation simplified

## Documentation

- **HYDRA_USAGE.md** - Usage guide
- **HYDRA_CLI_EXAMPLES.md** - CLI examples
- **HYDRA_MULTIRUN_GUIDE.md** - Multi-run guide
- **HYDRA_MIGRATION_SUMMARY.md** - Complete migration summary

## Status

? **Migration Complete** - Hydra is the only configuration system.
