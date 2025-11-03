# Hydra Configuration Migration Plan

## Executive Summary

This document outlines a comprehensive plan to migrate the Agent Farm simulation framework from the current custom configuration system to **Hydra**, a powerful configuration management framework developed by Facebook Research. Hydra provides advanced features like command-line overrides, config composition, and automatic experiment management.

## Current Config System Analysis

### Architecture Overview

The current system uses:
- **Dataclass-based configs**: `SimulationConfig` with nested config objects (`EnvironmentConfig`, `PopulationConfig`, etc.)
- **File-based loading**: YAML files in a hierarchical structure:
  - `default.yaml` - base configuration
  - `environments/{env}.yaml` - environment-specific overrides
  - `profiles/{profile}.yaml` - profile-specific overrides
- **Caching system**: File modification tracking with LRU eviction
- **Validation**: Schema-based validation with auto-repair capabilities
- **Orchestrator pattern**: Coordinates loading, caching, and validation

### Key Components

1. **`farm/config/config.py`**: Core config dataclasses (~1,500 lines)
2. **`farm/config/cache.py`**: Config caching and optimization
3. **`farm/config/orchestrator.py`**: Configuration orchestrator
4. **`farm/config/validation.py`**: Validation system
5. **`farm/config/schema.py`**: Schema generation for UI

### Strengths of Current System

- ? Well-structured nested configs with type safety
- ? Efficient caching mechanism
- ? Comprehensive validation
- ? Backward compatibility with flat/flattened configs
- ? Versioning support

### Limitations

- ? No command-line overrides (requires editing YAML files)
- ? No automatic experiment tracking/versioning
- ? Manual config composition
- ? No built-in multi-run/sweep support
- ? Limited dynamic config resolution

## Hydra Overview

### What is Hydra?

Hydra is a framework for elegantly configuring complex applications. Key features:

1. **Command-line overrides**: `python script.py config.simulation_steps=200`
2. **Config composition**: Automatic merging of config groups
3. **Multi-run**: Run multiple experiments with different configs
4. **Config sweeps**: Hyperparameter tuning with grid/random search
5. **Automatic versioning**: Saves configs with outputs
6. **OmegaConf**: Structured configs with variable interpolation

### Benefits for Agent Farm

- **Experiment management**: Automatic experiment tracking and versioning
- **CLI flexibility**: Override any config without editing files
- **Multi-run support**: Run parameter sweeps for hyperparameter tuning
- **Config groups**: Organize configs by scenario/preset
- **Reproducibility**: Automatic config saving with outputs

## Migration Strategy

### Phase 1: Preparation & Setup (Week 1)

#### 1.1 Install Hydra
```bash
pip install hydra-core>=1.3.0
```

#### 1.2 Create Hydra Config Structure
```
conf/
??? config.yaml                    # Main config (references defaults)
??? defaults/
?   ??? environment/
?   ?   ??? development.yaml
?   ?   ??? production.yaml
?   ?   ??? testing.yaml
?   ??? profile/
?   ?   ??? benchmark.yaml
?   ?   ??? simulation.yaml
?   ?   ??? research.yaml
?   ??? model/
?       ??? dqn.yaml
?       ??? ppo.yaml
??? environment/
?   ??? development.yaml
?   ??? production.yaml
?   ??? testing.yaml
??? profile/
?   ??? benchmark.yaml
?   ??? simulation.yaml
?   ??? research.yaml
??? schema/
    ??? simulation_schema.yaml     # For validation
```

#### 1.3 Convert Existing Configs to Hydra Format

Transform `default.yaml` and environment/profile configs to Hydra's format, maintaining the same structure but using Hydra's config groups pattern.

### Phase 2: Core Migration (Week 2-3)

#### 2.1 Create Hydra-Compatible Config Dataclasses

Modify `SimulationConfig` to work with Hydra:
- Keep existing dataclass structure
- Add `@hydra.main()` decorator support
- Create config store registrations for config groups

#### 2.2 Implement Hydra Config Loader

Create `farm/config/hydra_loader.py`:
```python
from hydra import compose, initialize_config_dir, initialize_config_store
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from farm.config.config import SimulationConfig

class HydraConfigLoader:
    """Load configurations using Hydra."""
    
    def load_config(
        self,
        config_path: str = "conf",
        config_name: str = "config",
        overrides: Optional[List[str]] = None,
    ) -> SimulationConfig:
        """Load config using Hydra."""
        # Implementation
```

#### 2.3 Migration Wrapper

Create a compatibility layer that supports both old and new config systems:
```python
def load_config(
    environment: str = "development",
    profile: Optional[str] = None,
    use_hydra: bool = False,  # Feature flag
    **kwargs
) -> SimulationConfig:
    """Load config with support for both systems."""
    if use_hydra:
        return _load_hydra_config(environment, profile, **kwargs)
    else:
        return _load_legacy_config(environment, profile, **kwargs)
```

### Phase 3: Integration (Week 3-4)

#### 3.1 Update Entry Points

Modify `run_simulation.py` and other entry points:
```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert OmegaConf to SimulationConfig
    config = SimulationConfig.from_hydra(cfg)
    # Run simulation
    run_simulation(config=config, ...)
```

#### 3.2 CLI Integration

Enable command-line overrides:
```bash
python run_simulation.py \
    simulation_steps=200 \
    population.system_agents=50 \
    learning.learning_rate=0.0005 \
    +profile=benchmark
```

#### 3.3 Update All Config Access Points

Audit and update all code that accesses config:
- Direct attribute access: `config.simulation_steps` ? (no change needed)
- Nested access: `config.environment.width` ? (no change needed)
- Dictionary access: Convert to OmegaConf access patterns

### Phase 4: Advanced Features (Week 4-5)

#### 4.1 Multi-Run Support

Enable running multiple experiments:
```python
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Automatically runs multiple configs
    pass
```

Command-line:
```bash
python run_simulation.py -m \
    simulation_steps=100,200,300 \
    learning.learning_rate=0.001,0.0005
```

#### 4.2 Config Sweeps

Create sweep configs for hyperparameter tuning:
```yaml
# conf/sweeps/learning_rate_sweep.yaml
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
      learning.gamma: choice(0.9, 0.95, 0.99)
```

#### 4.3 Experiment Tracking

Hydra automatically creates output directories:
```
outputs/
??? 2024-01-15/
    ??? 14-30-25/
        ??? .hydra/
        ?   ??? config.yaml
        ?   ??? hydra.yaml
        ??? simulation.db
```

### Phase 5: Validation & Testing (Week 5-6)

#### 5.1 Compatibility Testing

- Test all existing configs load correctly
- Verify backward compatibility with legacy system
- Test config overrides work as expected

#### 5.2 Performance Testing

- Compare config loading performance
- Benchmark Hydra vs. current caching system
- Measure multi-run overhead

#### 5.3 Migration Testing

- Test gradual migration (feature flag)
- Verify no breaking changes
- Test rollback capability

## Detailed Implementation Plan

### Step 1: Config Structure Conversion

Convert existing YAML configs to Hydra format:

**Before (current `default.yaml`):**
```yaml
simulation_steps: 100
seed: 1234567890
environment:
  width: 100
  height: 100
```

**After (Hydra `conf/config.yaml`):**
```yaml
defaults:
  - environment: development
  - profile: null
  - _self_

simulation_steps: 100
seed: 1234567890
```

**After (Hydra `conf/defaults/environment/development.yaml`):**
```yaml
# @package _global_
environment:
  width: 100
  height: 100
```

### Step 2: Config Loader Implementation

```python
# farm/config/hydra_loader.py
from typing import Optional, List
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from .config import SimulationConfig


class HydraConfigLoader:
    """Load configurations using Hydra framework."""
    
    def __init__(self, config_path: str = "conf"):
        self.config_path = Path(config_path)
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure Hydra is initialized."""
        if not self._initialized:
            GlobalHydra.instance().clear()
            initialize_config_dir(
                config_dir=str(self.config_path.absolute()),
                config_name="config"
            )
            self._initialized = True
    
    def load_config(
        self,
        environment: str = "development",
        profile: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> SimulationConfig:
        """Load configuration using Hydra."""
        self._ensure_initialized()
        
        # Build override list
        override_list = [f"environment={environment}"]
        if profile:
            override_list.append(f"profile={profile}")
        if overrides:
            override_list.extend(overrides)
        
        # Compose config
        cfg = compose(
            config_name="config",
            overrides=override_list or [],
        )
        
        # Convert to SimulationConfig
        return self._omega_to_simulation_config(cfg)
    
    def _omega_to_simulation_config(self, cfg: DictConfig) -> SimulationConfig:
        """Convert OmegaConf DictConfig to SimulationConfig."""
        # Convert to dict and use existing from_dict method
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return SimulationConfig.from_dict(config_dict)
```

### Step 3: Entry Point Updates

Update `run_simulation.py`:

```python
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from farm.config.config import SimulationConfig

# Register config store
cs = ConfigStore.instance()
cs.store(name="simulation_config", node=SimulationConfig)

@hydra.main(
    config_path="conf",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra support."""
    # Convert OmegaConf to SimulationConfig
    config = SimulationConfig.from_hydra(cfg)
    
    # Rest of the simulation logic
    # ...
```

### Step 4: Backward Compatibility Layer

```python
# farm/config/__init__.py
from typing import Optional
import os

from .config import SimulationConfig

# Feature flag for gradual migration
USE_HYDRA = os.getenv("USE_HYDRA_CONFIG", "false").lower() == "true"

def load_config(
    environment: str = "development",
    profile: Optional[str] = None,
    **kwargs
) -> SimulationConfig:
    """
    Load configuration with backward compatibility.
    
    Uses Hydra if USE_HYDRA_CONFIG=true, otherwise uses legacy system.
    """
    if USE_HYDRA:
        from .hydra_loader import HydraConfigLoader
        loader = HydraConfigLoader()
        return loader.load_config(
            environment=environment,
            profile=profile,
            overrides=kwargs.get("overrides"),
        )
    else:
        # Legacy loading
        return SimulationConfig.from_centralized_config(
            environment=environment,
            profile=profile,
        )
```

## Config File Organization

### Recommended Hydra Structure

```
conf/
??? config.yaml                      # Main entry point
??? defaults/
?   ??? environment/
?   ?   ??? development.yaml
?   ?   ??? production.yaml
?   ?   ??? testing.yaml
?   ??? profile/
?   ?   ??? benchmark.yaml
?   ?   ??? simulation.yaml
?   ?   ??? research.yaml
?   ??? model/
?       ??? dqn.yaml
?       ??? ppo.yaml
??? environment/
?   ??? development.yaml
?   ??? production.yaml
?   ??? testing.yaml
??? profile/
?   ??? benchmark.yaml
?   ??? simulation.yaml
?   ??? research.yaml
??? sweeps/
    ??? learning_rate_sweep.yaml
    ??? population_sweep.yaml
```

### Example Config Files

**`conf/config.yaml`:**
```yaml
defaults:
  - environment: development
  - profile: null
  - _self_

# Core simulation parameters
simulation_steps: 100
max_steps: 1000
seed: 1234567890

# Environment configuration
environment:
  width: 100
  height: 100
  position_discretization_method: floor
  use_bilinear_interpolation: true

# Population configuration
population:
  system_agents: 10
  independent_agents: 10
  control_agents: 10
  max_population: 300

# ... rest of config
```

**`conf/defaults/environment/development.yaml`:**
```yaml
# @package _global_
environment:
  width: 100
  height: 100

database:
  use_in_memory_db: true
  persist_db_on_completion: true

logging:
  debug: true
  verbose_logging: true
```

**`conf/defaults/profile/benchmark.yaml`:**
```yaml
# @package _global_
simulation_steps: 2000
max_steps: 2000

performance:
  enable_parallel_processing: true
  max_worker_threads: 8

database:
  db_pragma_profile: performance
```

## Migration Checklist

### Pre-Migration
- [ ] Install Hydra: `pip install hydra-core>=1.3.0`
- [ ] Create `conf/` directory structure
- [ ] Document all config access patterns in codebase
- [ ] Create backup of existing configs

### Phase 1: Setup
- [ ] Convert `default.yaml` to Hydra format
- [ ] Convert environment configs to Hydra format
- [ ] Convert profile configs to Hydra format
- [ ] Create `HydraConfigLoader` class
- [ ] Test config loading with Hydra

### Phase 2: Integration
- [ ] Update `run_simulation.py` with `@hydra.main()`
- [ ] Update all entry points
- [ ] Add backward compatibility layer
- [ ] Test CLI overrides
- [ ] Update documentation

### Phase 3: Advanced Features
- [ ] Implement multi-run support
- [ ] Create sweep configs
- [ ] Test experiment tracking
- [ ] Update CI/CD for Hydra

### Phase 4: Migration
- [ ] Enable Hydra via feature flag
- [ ] Test in development environment
- [ ] Test in production environment
- [ ] Monitor for issues
- [ ] Deprecate legacy system

### Phase 5: Cleanup
- [ ] Remove legacy config loading code
- [ ] Remove feature flags
- [ ] Update all documentation
- [ ] Archive old config files

## Benefits After Migration

### Immediate Benefits
1. **CLI Overrides**: Override any config from command line
2. **Automatic Versioning**: Configs saved with each run
3. **Multi-run**: Run multiple experiments easily
4. **Config Groups**: Better organization with config groups

### Long-term Benefits
1. **Hyperparameter Tuning**: Built-in sweep support
2. **Experiment Management**: Automatic experiment tracking
3. **Reproducibility**: Configs automatically saved with outputs
4. **Community Standard**: Using industry-standard tool

## Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation**: Use feature flag for gradual migration, maintain backward compatibility layer

### Risk 2: Performance Impact
**Mitigation**: Benchmark Hydra loading vs. current caching, optimize if needed

### Risk 3: Learning Curve
**Mitigation**: Provide training, comprehensive documentation, migration guide

### Risk 4: Config Complexity
**Mitigation**: Keep config structure similar, use clear naming conventions

## Rollback Plan

If issues arise:

1. **Immediate**: Set `USE_HYDRA_CONFIG=false` environment variable
2. **Short-term**: Revert to legacy config loading
3. **Long-term**: Fix issues and re-enable Hydra

## Timeline

- **Week 1**: Setup and preparation
- **Week 2-3**: Core migration
- **Week 3-4**: Integration
- **Week 4-5**: Advanced features
- **Week 5-6**: Testing and validation
- **Week 6+**: Gradual rollout and cleanup

## Success Criteria

- ? All existing configs load correctly
- ? CLI overrides work as expected
- ? Multi-run functionality works
- ? No performance regression
- ? Backward compatibility maintained
- ? Documentation updated
- ? Team trained on Hydra

## Resources

- [Hydra Documentation](https://hydra.cc/)
- [Hydra Tutorials](https://hydra.cc/docs/tutorials/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Examples](https://github.com/facebookresearch/hydra/tree/main/examples)

## Next Steps

1. Review and approve this migration plan
2. Set up development environment with Hydra
3. Create proof-of-concept implementation
4. Test with one entry point (`run_simulation.py`)
5. Gradually expand to all entry points
6. Full migration and cleanup
