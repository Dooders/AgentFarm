# Configuration Quick Reference

## Hardcoded Values Not in Config System

### 🔴 Critical Priority

| Location | Value | Current | Should Be In |
|----------|-------|---------|-------------|
| `farm/core/action.py:1289` | `reward = 0.02` | Hardcoded | `ActionRewardConfig.defend_reward` |
| `farm/core/action.py:1364` | `reward = 0.01` | Hardcoded | `ActionRewardConfig.pass_reward` |
| `farm/core/simulation.py:390` | `batch_size = 32` | Hardcoded | `PerformanceConfig.agent_processing_batch_size` |
| `farm/core/spatial/dirty_regions.py:47` | `self._batch_size = 10` | Hardcoded | `SpatialIndexConfig.dirty_region_batch_size` |
| `farm/database/database.py:219` | `pool_size=10` | Hardcoded | `DatabaseConfig.connection_pool_size` |
| `farm/database/database.py:226` | `timeout=30` | Hardcoded | `DatabaseConfig.connection_timeout` |

### 🟡 High Priority

| Location | Value | Current | Should Be In |
|----------|-------|---------|-------------|
| `farm/core/decision/base_dqn.py:221` | `self._max_cache_size = 100` | Hardcoded | `LearningConfig.dqn_state_cache_size` |
| `farm/database/data_logging.py:45` | `buffer_size: int = 1000` | Has config class | `DatabaseConfig.log_buffer_size` |
| `farm/database/data_logging.py:46` | `commit_interval: int = 30` | Has config class | `DatabaseConfig.commit_interval_seconds` |
| `farm/database/database.py:1475` | `batch_size = 1000` | Hardcoded | `DatabaseConfig.export_batch_size` |

### 🟢 Medium Priority

| Location | Value | Current | Should Be In |
|----------|-------|---------|-------------|
| `farm/core/resource_manager.py:695` | `amount = 5` | Hardcoded | `ResourceConfig.default_spawn_amount` |
| `farm/core/observations.py:300` | `random_max=0.1` | Hardcoded | `ObservationConfig.random_noise_max` |
| `farm/core/experiment_tracker.py:195` | `chunk_size = 1000` | Hardcoded | `ExperimentConfig.chunk_size` |

---

## Recommended New Config Classes

### ActionRewardConfig (NEW - Critical)
```python
@dataclass
class ActionRewardConfig:
    defend_success_reward: float = 0.02
    pass_action_reward: float = 0.01
    successful_gather_bonus: float = 0.05
    successful_share_bonus: float = 0.03
    successful_attack_bonus: float = 0.1
    reproduction_success_bonus: float = 0.15
    failed_action_penalty: float = -0.05
    collision_penalty: float = -0.02
```

### PerformanceConfig (NEW - Critical)
```python
@dataclass
class PerformanceConfig:
    agent_processing_batch_size: int = 32
    resource_processing_batch_size: int = 100
    enable_parallel_processing: bool = False
    max_worker_threads: int = 4
    enable_memory_pooling: bool = True
    memory_pool_size_mb: int = 100
    enable_state_caching: bool = True
    cache_ttl_seconds: int = 60
```

---

## Enhanced Existing Configs

### DatabaseConfig (Enhancements - High Priority)
```python
@dataclass
class DatabaseConfig:
    # ... existing fields ...
    
    # NEW - Connection pooling
    connection_pool_size: int = 10
    connection_pool_recycle: int = 3600
    connection_timeout: int = 30
    
    # NEW - Buffering and commits
    log_buffer_size: int = 1000
    commit_interval_seconds: int = 30
    
    # NEW - Export settings
    export_batch_size: int = 1000
```

### LearningConfig (Enhancements - High Priority)
```python
@dataclass
class LearningConfig:
    # ... existing fields ...
    
    # NEW - Caching
    dqn_state_cache_size: int = 100
```

### SpatialIndexConfig (Enhancements - Critical)
```python
@dataclass
class SpatialIndexConfig:
    # ... existing fields ...
    
    # NEW
    dirty_region_batch_size: int = 10
```

### ResourceConfig (Enhancements - Medium Priority)
```python
@dataclass
class ResourceConfig:
    # ... existing fields ...
    
    # NEW
    default_spawn_amount: int = 5
```

---

## Implementation Checklist

### Phase 1 (Do First)
- [ ] Create `ActionRewardConfig` class in `farm/config/config.py`
- [ ] Create `PerformanceConfig` class in `farm/config/config.py`
- [ ] Add new fields to `DatabaseConfig`
- [ ] Add `dirty_region_batch_size` to `SpatialIndexConfig`
- [ ] Update `farm/core/action.py` to use `config.action_rewards.defend_reward`
- [ ] Update `farm/core/action.py` to use `config.action_rewards.pass_reward`
- [ ] Update `farm/core/simulation.py` to use `config.performance.agent_processing_batch_size`
- [ ] Update `farm/core/spatial/dirty_regions.py` to use `config.environment.spatial_index.dirty_region_batch_size`
- [ ] Update `farm/database/database.py` to use new database config fields
- [ ] Add new sections to `farm/config/default.yaml`
- [ ] Update schema generation in `farm/config/schema.py`

### Phase 2 (Do Next)
- [ ] Add `dqn_state_cache_size` to `LearningConfig`
- [ ] Update `farm/core/decision/base_dqn.py` to use config
- [ ] Add `default_spawn_amount` to `ResourceConfig`
- [ ] Update `farm/core/resource_manager.py` to use config
- [ ] Remove duplicate attack parameters from `ModuleConfig`
- [ ] Update all references to use `CombatConfig` instead

### Phase 3 (Nice to Have)
- [ ] Create `ExperimentConfig` for experiment tracking
- [ ] Add `random_noise_max` to `ObservationConfig`
- [ ] Consider `ChannelDecayConfig` if decay rates need tuning
- [ ] Standardize naming conventions across all configs

---

## Files That Need Updates

### Core Files
- `farm/config/config.py` - Add new config classes
- `farm/config/default.yaml` - Add new config sections
- `farm/config/schema.py` - Update schema generation

### Code Files Using Hardcoded Values
- `farm/core/action.py` - Use ActionRewardConfig
- `farm/core/simulation.py` - Use PerformanceConfig
- `farm/core/spatial/dirty_regions.py` - Use SpatialIndexConfig
- `farm/database/database.py` - Use enhanced DatabaseConfig
- `farm/core/decision/base_dqn.py` - Use enhanced LearningConfig
- `farm/core/resource_manager.py` - Use enhanced ResourceConfig

### Environment/Profile Files
- `farm/config/environments/development.yaml`
- `farm/config/environments/production.yaml`
- `farm/config/environments/testing.yaml`
- `farm/config/profiles/benchmark.yaml`
- `farm/config/profiles/simulation.yaml`
- `farm/config/profiles/research.yaml`

---

## Example Usage After Implementation

```python
from farm.config import SimulationConfig

# Load with new configs
config = SimulationConfig.from_centralized_config(
    environment="production",
    profile="benchmark"
)

# Access new config values
print(f"Defend reward: {config.action_rewards.defend_reward}")
print(f"Agent batch size: {config.performance.agent_processing_batch_size}")
print(f"DB pool size: {config.database.connection_pool_size}")
print(f"DQN cache size: {config.learning.dqn_state_cache_size}")
```

---

## Notes

- **Backward Compatibility**: The existing `_convert_flat_to_nested()` method will need updates to handle new config classes
- **Testing**: Update tests to use new config fields
- **Documentation**: Update README and config documentation
- **Migration**: Existing YAML configs will need migration or will use defaults