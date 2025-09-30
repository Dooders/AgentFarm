# Configuration System Analysis Report

## Executive Summary

This document provides a comprehensive analysis of hardcoded values, constants, and opportunities for making additional parameters configurable in the Agent Farm Simulation Framework.

---

## 1. Currently Missing from Config System

### 1.1 Visualization Hardcoded Constants (`farm/core/visualization.py`)

**Status**: Some visualization constants are configurable, but several are hardcoded

**Hardcoded Constants:**
```python
DEFAULT_CANVAS_SIZE = (400, 400)          # Already in VisualizationConfig ✓
PADDING = 20                              # Already in VisualizationConfig ✓
MAX_ANIMATION_FRAMES = 5                  # Already in VisualizationConfig ✓
ANIMATION_MIN_DELAY = 50                  # Already in VisualizationConfig ✓
MAX_RESOURCE_AMOUNT = 30                  # Already in VisualizationConfig ✓
RESOURCE_GLOW_RED = 50                    # Partially configurable
RESOURCE_GLOW_GREEN = 255                 # Partially configurable
RESOURCE_GLOW_BLUE = 50                   # Partially configurable
AGENT_RADIUS_SCALE = 2                    # Already in VisualizationConfig ✓
BIRTH_RADIUS_SCALE = 4                    # Already in VisualizationConfig ✓
DEATH_MARK_SCALE = 1.5                    # Already in VisualizationConfig ✓
MIN_FONT_SIZE = 10                        # Already in VisualizationConfig ✓
FONT_SCALE_FACTOR = 40                    # Already in VisualizationConfig ✓
```

**Recommendation**: These are already properly configurable via `VisualizationConfig`. No changes needed.

---

### 1.2 Action Reward Constants (`farm/core/action.py`)

**Status**: NOT CONFIGURABLE - High Priority

**Hardcoded Values:**
```python
# Line 1289
reward = 0.02  # Small reward for successful defense

# Line 1364
reward = 0.01  # Minimal reward for passing
```

**Recommendation**: Add to `AgentBehaviorConfig` or create an `ActionRewardConfig`:

```python
@dataclass
class ActionRewardConfig:
    """Configuration for action rewards."""
    defend_reward: float = 0.02
    pass_reward: float = 0.01
    # Add more as needed
```

---

### 1.3 Spatial Index Constants (`farm/core/spatial/`)

**Status**: PARTIALLY CONFIGURABLE - Medium Priority

#### dirty_regions.py
```python
# Line 47
self._batch_size = 10  # NOT CONFIGURABLE
```

**Found in**: `DirtyRegionTracker.__init__()`

**Recommendation**: Add to `SpatialIndexConfig`:
```python
@dataclass
class SpatialIndexConfig:
    # ... existing fields ...
    dirty_region_batch_size: int = 10
```

#### index.py (Priority Constants)
```python
# Lines 43-46
PRIORITY_LOW = 0  # Background entities, decorative elements
PRIORITY_NORMAL = 1  # Regular agents (default)
PRIORITY_HIGH = 2  # Active combat participants, quest-critical agents
PRIORITY_CRITICAL = 3  # Player entities, important NPCs
```

**Recommendation**: These are constants that define the priority system. Consider making them configurable only if you plan to support custom priority levels.

---

### 1.4 Channel System Constants (`farm/core/channels.py`)

**Status**: PARTIALLY CONFIGURABLE - Low Priority

**Channel Indices** (Lines 1357-1366):
```python
SELF_HP = 0
ALLIES_HP = 1
ENEMIES_HP = 2
RESOURCES = 3
OBSTACLES = 4
TERRAIN_COST = 5
VISIBILITY = 6
KNOWN_EMPTY = 7
DAMAGE_HEAT = 8
TRAILS = 9
```

**Recommendation**: These are enum-like constants defining the channel order. They should remain hardcoded as they define the system architecture. However, consider adding channel-specific configuration:

```python
@dataclass
class ChannelConfig:
    """Configuration for observation channels."""
    known_empty_decay: float = 0.9
    damage_heat_decay: float = 0.95
    trails_decay: float = 0.98
    ally_signal_decay: float = 0.92
```

---

### 1.5 Observation System Constants (`farm/core/observations.py`)

**Status**: PARTIALLY CONFIGURABLE

**Random Observation Noise** (Line 300):
```python
random_max=0.1
```

**Recommendation**: Add to `ObservationConfig`:
```python
class ObservationConfig(BaseModel):
    # ... existing fields ...
    random_noise_max: float = Field(default=0.1, description="Maximum random noise for observation augmentation")
```

---

### 1.6 Resource Manager Constants (`farm/core/resource_manager.py`)

**Status**: MOSTLY CONFIGURABLE

**Default Resource Amount** (Line 695):
```python
amount = 5  # Default amount
```

**Recommendation**: Add to `ResourceConfig`:
```python
@dataclass
class ResourceConfig:
    # ... existing fields ...
    default_spawn_amount: int = 5
```

---

### 1.7 Simulation Constants (`farm/core/simulation.py`)

**Status**: NOT CONFIGURABLE

**Batch Size for Processing** (Line 390):
```python
batch_size = 32  # Adjust based on your needs
```

**Recommendation**: Add to `SimulationConfig` or create a `PerformanceConfig`:
```python
@dataclass
class PerformanceConfig:
    """Configuration for performance tuning."""
    agent_processing_batch_size: int = 32
    enable_parallel_processing: bool = False
    max_worker_threads: int = 4
```

---

### 1.8 Agent Observation Rendering Constants (`farm/core/agent.py`)

**Status**: PARTIALLY CONFIGURABLE

**Default Radius Values** (Lines 404-405, 584-585, 603-604):
```python
radius = 5
size = 11
```

**Recommendation**: These appear to be fallback values. Add to `AgentBehaviorConfig`:
```python
@dataclass
class AgentBehaviorConfig:
    # ... existing fields ...
    default_observation_radius: int = 5
```

---

## 2. Configuration Improvements & Suggestions

### 2.1 Missing Config Sections

The following logical config groupings could be added:

#### A. Performance Configuration
```python
@dataclass
class PerformanceConfig:
    """Configuration for performance optimization settings."""
    
    # Batch processing
    agent_processing_batch_size: int = 32
    resource_processing_batch_size: int = 100
    
    # Parallel processing
    enable_parallel_processing: bool = False
    max_worker_threads: int = 4
    
    # Memory management
    enable_memory_pooling: bool = True
    memory_pool_size_mb: int = 100
    
    # Caching
    enable_state_caching: bool = True
    cache_ttl_seconds: int = 60
```

#### B. Channel Decay Configuration
```python
@dataclass
class ChannelDecayConfig:
    """Configuration for observation channel decay rates."""
    
    known_empty_decay: float = 0.9
    damage_heat_decay: float = 0.95
    trails_decay: float = 0.98
    ally_signal_decay: float = 0.92
    
    # Enable/disable specific channels
    enable_known_empty: bool = True
    enable_damage_heat: bool = True
    enable_trails: bool = True
    enable_ally_signal: bool = True
```

#### C. Action Reward Configuration
```python
@dataclass
class ActionRewardConfig:
    """Configuration for action-specific rewards."""
    
    # Core action rewards
    defend_success_reward: float = 0.02
    pass_action_reward: float = 0.01
    
    # Extended rewards
    successful_gather_bonus: float = 0.05
    successful_share_bonus: float = 0.03
    successful_attack_bonus: float = 0.1
    reproduction_success_bonus: float = 0.15
    
    # Penalties
    failed_action_penalty: float = -0.05
    collision_penalty: float = -0.02
```

#### D. Networking Configuration (for future multi-node support)
```python
@dataclass
class NetworkingConfig:
    """Configuration for distributed simulation networking."""
    
    enable_distributed_mode: bool = False
    master_node_host: str = "localhost"
    master_node_port: int = 5555
    worker_node_count: int = 1
    communication_protocol: str = "tcp"  # Options: "tcp", "udp", "ipc"
```

---

### 2.2 Config Organization Issues

#### Issue 1: Duplicate Attack Parameters

The following attack-related parameters appear in multiple places:

**In `CombatConfig`:**
- `attack_range: float = 20.0`
- `attack_base_damage: float = 10.0`
- `attack_kill_reward: float = 5.0`

**In `ModuleConfig`:**
- `attack_range: float = 20.0`
- `attack_base_damage: float = 10.0`
- `attack_kill_reward: float = 5.0`

**Recommendation**: Remove duplication. Keep combat-related parameters in `CombatConfig` only.

#### Issue 2: Inconsistent Naming Convention

Some parameters use underscores, some don't have consistent prefixes:

```python
# Inconsistent prefixes
gather_success_reward  vs  success_reward
move_base_cost        vs  base_cost
```

**Recommendation**: Adopt consistent naming: `{module}_{aspect}_{metric}`

Example:
```python
# Gathering module
gather_reward_success: float = 0.5
gather_reward_failure: float = -0.1
gather_cost_base: float = -0.05

# Movement module
move_reward_approach: float = 0.3
move_penalty_retreat: float = -0.2
move_cost_base: float = -0.1
```

---

### 2.3 Agent-Specific Parameters Not in Config

The `default.yaml` includes agent-specific parameters that aren't present in the main `SimulationConfig`:

**In `default.yaml` but missing from Python config:**
```yaml
agent_parameters:
  SystemAgent:
    cooperation_threshold: 0.5  # NOT IN CONFIG
    learning_rate: 0.01        # NOT IN CONFIG
    exploration_rate: 0.1      # NOT IN CONFIG
```

**Recommendation**: Either:
1. Add these to the `agent_parameters` dict validation
2. Create a proper `AgentTypeConfig` class:

```python
@dataclass
class AgentTypeParameters:
    """Configuration for agent type-specific parameters."""
    gather_efficiency_multiplier: float
    gather_cost_multiplier: float
    min_resource_threshold: float
    share_weight: float
    attack_weight: float
    cooperation_threshold: float = 0.5
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
```

---

### 2.4 Missing Environment-Specific Configurations

These are mentioned in config README but could be expanded:

```python
@dataclass
class EnvironmentConfig:
    # ... existing fields ...
    
    # Terrain features (not yet implemented)
    enable_terrain: bool = False
    terrain_types: List[str] = field(default_factory=list)
    terrain_cost_multipliers: Dict[str, float] = field(default_factory=dict)
    
    # Weather/environmental effects (not yet implemented)
    enable_weather: bool = False
    weather_types: List[str] = field(default_factory=list)
    
    # Obstacles (not yet implemented)
    enable_obstacles: bool = False
    obstacle_density: float = 0.1
```

---

## 3. Config System Strengths

### What's Working Well:

1. **Hierarchical Organization**: The nested config structure (environment, population, resources, etc.) is well-designed
2. **Backward Compatibility**: The `_convert_flat_to_nested()` method handles legacy configs well
3. **Validation**: Good use of `__post_init__` for validation in dataclasses
4. **Versioning**: Excellent versioning system with `generate_version_hash()`
5. **Centralized Loading**: `from_centralized_config()` provides good environment/profile support
6. **Schema Generation**: The `schema.py` module generates JSON schemas for UI integration

---

## 4. Priority Recommendations

### High Priority (Implement First):

1. **Add `ActionRewardConfig`** for defend_reward and pass_reward
2. **Add `dirty_region_batch_size`** to `SpatialIndexConfig`
3. **Add `agent_processing_batch_size`** to a new `PerformanceConfig`
4. **Remove duplicate attack parameters** from `ModuleConfig` (use `CombatConfig` only)
5. **Add missing agent_parameters fields** (cooperation_threshold, learning_rate, exploration_rate)

### Medium Priority:

6. **Add `ChannelDecayConfig`** for observation channel decay rates
7. **Add `default_spawn_amount`** to `ResourceConfig`
8. **Add `random_noise_max`** to `ObservationConfig`
9. **Standardize naming conventions** across all config classes

### Low Priority:

10. **Add `PerformanceConfig`** for comprehensive performance tuning
11. **Add terrain/weather configs** to `EnvironmentConfig` (for future features)
12. **Consider `NetworkingConfig`** for distributed simulations (future)

---

## 5. Configuration Coverage Analysis

### Well Covered (✓):
- Visualization settings
- Learning parameters
- Agent behavior parameters
- Database settings
- Device configuration
- Curriculum learning
- Spatial indexing (mostly)
- Combat parameters
- Resource management

### Partially Covered (⚠):
- Action rewards (only some)
- Observation system (missing noise config)
- Performance tuning (scattered across different places)
- Channel decay rates (hardcoded)

### Not Covered (✗):
- Batch processing sizes
- Parallel processing settings
- Memory pooling settings
- Cache configuration
- Networking/distributed settings

---

## 6. Suggested New Config Structure

```python
@dataclass
class SimulationConfig:
    """Main simulation configuration."""
    
    # Core settings (existing)
    simulation_steps: int = 100
    max_steps: int = 1000
    seed: Optional[int] = 1234567890
    agent_parameters: Dict[str, Dict[str, float]] = field(default_factory=...)
    
    # Nested configs (existing)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    agent_behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    modules: ModuleConfig = field(default_factory=ModuleConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    observation: Optional[ObservationConfig] = None
    redis: RedisMemoryConfig = field(default_factory=RedisMemoryConfig)
    
    # NEW CONFIGS (recommended additions)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    action_rewards: ActionRewardConfig = field(default_factory=ActionRewardConfig)
    channel_decay: ChannelDecayConfig = field(default_factory=ChannelDecayConfig)
```

---

## 7. Implementation Guide

### Step 1: Add Missing Simple Parameters

```python
# In farm/config/config.py

@dataclass
class ResourceConfig:
    # ... existing fields ...
    default_spawn_amount: int = 5

@dataclass
class SpatialIndexConfig:
    # ... existing fields ...
    dirty_region_batch_size: int = 10

# Update default.yaml accordingly
```

### Step 2: Create New Config Classes

```python
# Add to farm/config/config.py

@dataclass
class ActionRewardConfig:
    """Configuration for action-specific rewards."""
    defend_reward: float = 0.02
    pass_reward: float = 0.01

@dataclass
class PerformanceConfig:
    """Configuration for performance tuning."""
    agent_processing_batch_size: int = 32

@dataclass
class ChannelDecayConfig:
    """Configuration for observation channel decay rates."""
    known_empty_decay: float = 0.9
    damage_heat_decay: float = 0.95
    trails_decay: float = 0.98
```

### Step 3: Update Code to Use Config

```python
# In farm/core/action.py
def defend_action(agent: "BaseAgent") -> dict:
    # OLD:
    # reward = 0.02
    
    # NEW:
    reward = agent.config.action_rewards.defend_reward if agent.config else 0.02
```

### Step 4: Update default.yaml

```yaml
# Add new sections
action_rewards:
  defend_reward: 0.02
  pass_reward: 0.01

performance:
  agent_processing_batch_size: 32

channel_decay:
  known_empty_decay: 0.9
  damage_heat_decay: 0.95
  trails_decay: 0.98
```

### Step 5: Update Schema and Validation

Update `farm/config/schema.py` to include new configs in schema generation.

---

## 8. Additional Hardcoded Values Found

### 8.1 Database Configuration

**Status**: PARTIALLY CONFIGURABLE

**Database Connection Settings** (`farm/database/database.py` lines 219-227):
```python
pool_size=10,
pool_recycle=3600,
connect_args={
    "timeout": 30,
    "check_same_thread": False,
}
```

**Data Logging Buffer Settings** (`farm/database/data_logging.py` lines 45-46):
```python
buffer_size: int = 1000
commit_interval: int = 30  # seconds
```

**Recommendation**: Add to `DatabaseConfig`:
```python
@dataclass
class DatabaseConfig:
    # ... existing fields ...
    
    # Connection pooling
    connection_pool_size: int = 10
    connection_pool_recycle: int = 3600  # seconds
    connection_timeout: int = 30  # seconds
    
    # Buffering and commits
    log_buffer_size: int = 1000
    commit_interval_seconds: int = 30
```

---

### 8.2 Redis Memory Configuration

**Status**: PARTIALLY CONFIGURABLE

**Redis Memory Settings** (`farm/memory/redis_memory.py` lines 41-50):
```python
memory_limit: int = 1000
ttl: int = 3600  # seconds
enable_semantic_search: bool = True
embedding_dimension: int = 128
memory_priority_decay: float = 0.95
cleanup_interval: int = 100
```

**Recommendation**: These ARE configurable via `RedisMemoryConfig`, which is good! However, ensure they're accessible in the main `SimulationConfig.redis` section.

---

### 8.3 Decision Module Cache Settings

**Status**: NOT CONFIGURABLE

**DQN Cache Size** (`farm/core/decision/base_dqn.py` line 221):
```python
self._max_cache_size = 100
```

**Recommendation**: Add to `LearningConfig` or create a `CacheConfig`:
```python
@dataclass
class LearningConfig:
    # ... existing fields ...
    
    # DQN caching
    dqn_state_cache_size: int = 100
```

---

### 8.4 Experiment Runner Settings

**Status**: NOT CONFIGURABLE - Low Priority

**Default Number of Steps** (`farm/runners/experiment_runner.py` line 26):
```python
DEFAULT_NUM_STEPS = 1000
```

**Recommendation**: This is already configurable via `SimulationConfig.simulation_steps`, so this constant is just a fallback. Consider renaming it to `FALLBACK_NUM_STEPS` for clarity.

---

### 8.5 Database Export Batch Size

**Status**: NOT CONFIGURABLE

**Export Batch Size** (`farm/database/database.py` line 1475):
```python
batch_size = 1000  # For data export operations
```

**Recommendation**: Add to `DatabaseConfig`:
```python
@dataclass
class DatabaseConfig:
    # ... existing fields ...
    export_batch_size: int = 1000
```

---

### 8.6 Experiment Tracker Settings

**Status**: NOT CONFIGURABLE

**Chunk Size** (`farm/core/experiment_tracker.py` line 195):
```python
chunk_size = 1000
```

**Recommendation**: Add to a new `ExperimentConfig` or add to existing config:
```python
@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and management."""
    
    chunk_size: int = 1000
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
```

---

## 9. Summary of All Missing/Hardcoded Parameters

### Critical Priority (Fix Immediately):
1. ✅ **Action rewards** (`defend_reward=0.02`, `pass_reward=0.01`)
2. ✅ **Simulation batch size** (`batch_size=32`)
3. ✅ **Dirty region batch size** (`_batch_size=10`)
4. ✅ **Database connection settings** (`pool_size=10`, `timeout=30`)

### High Priority (Fix Soon):
5. ✅ **DQN cache size** (`_max_cache_size=100`)
6. ✅ **Database buffer settings** (`buffer_size=1000`, `commit_interval=30`)
7. ✅ **Database export batch size** (`batch_size=1000`)
8. ✅ **Channel decay rates** (if used - check actual usage)

### Medium Priority:
9. ✅ **Resource default spawn amount** (`amount=5`)
10. ✅ **Observation random noise** (`random_max=0.1`)
11. ✅ **Experiment tracker chunk size** (`chunk_size=1000`)

### Low Priority (Nice to Have):
12. Priority constants (`PRIORITY_LOW=0`, etc.) - These are architectural
13. Channel indices (`SELF_HP=0`, etc.) - These are architectural
14. Visualization constants - Already mostly configurable
15. Default fallback values - These are safety fallbacks

---

## 10. Complete Updated Config Structure

```python
@dataclass
class SimulationConfig:
    """Main simulation configuration."""
    
    # Core settings (existing)
    simulation_steps: int = 100
    max_steps: int = 1000
    seed: Optional[int] = 1234567890
    agent_parameters: Dict[str, Dict[str, float]] = field(default_factory=...)
    
    # Nested configs (existing)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    agent_behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)  # ENHANCED
    device: DeviceConfig = field(default_factory=DeviceConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    modules: ModuleConfig = field(default_factory=ModuleConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    observation: Optional[ObservationConfig] = None  # ENHANCED
    redis: RedisMemoryConfig = field(default_factory=RedisMemoryConfig)
    
    # NEW CONFIGS (recommended additions)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)  # NEW
    action_rewards: ActionRewardConfig = field(default_factory=ActionRewardConfig)  # NEW
    channel_decay: ChannelDecayConfig = field(default_factory=ChannelDecayConfig)  # NEW (optional)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)  # NEW (optional)


@dataclass
class ResourceConfig:
    """Configuration for resource system settings."""
    initial_resources: int = 20
    resource_regen_rate: float = 0.1
    resource_regen_amount: int = 2
    max_resource_amount: int = 30
    memmap_delete_on_close: bool = False
    default_spawn_amount: int = 5  # NEW


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    use_in_memory_db: bool = False
    persist_db_on_completion: bool = True
    in_memory_db_memory_limit_mb: Optional[int] = None
    in_memory_tables_to_persist: Optional[List[str]] = None
    db_pragma_profile: str = "balanced"
    db_cache_size_mb: int = 200
    db_synchronous_mode: str = "NORMAL"
    db_journal_mode: str = "WAL"
    db_custom_pragmas: Dict[str, str] = field(default_factory=dict)
    
    # NEW - Connection pooling
    connection_pool_size: int = 10
    connection_pool_recycle: int = 3600
    connection_timeout: int = 30
    
    # NEW - Buffering and commits
    log_buffer_size: int = 1000
    commit_interval_seconds: int = 30
    
    # NEW - Export settings
    export_batch_size: int = 1000


@dataclass
class LearningConfig:
    """Configuration for reinforcement learning parameters."""
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 2000
    batch_size: int = 32
    training_frequency: int = 4
    dqn_hidden_size: int = 24
    tau: float = 0.005
    
    # NEW - Caching
    dqn_state_cache_size: int = 100


@dataclass
class SpatialIndexConfig:
    """Configuration for spatial indexing and batch updates."""
    enable_batch_updates: bool = True
    region_size: float = 50.0
    max_batch_size: int = 100
    max_regions: int = 1000
    enable_quadtree_indices: bool = False
    enable_spatial_hash_indices: bool = False
    spatial_hash_cell_size: Optional[float] = None
    performance_monitoring: bool = True
    debug_queries: bool = False
    
    # NEW
    dirty_region_batch_size: int = 10


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization settings."""
    
    # Batch processing
    agent_processing_batch_size: int = 32
    resource_processing_batch_size: int = 100
    
    # Parallel processing
    enable_parallel_processing: bool = False
    max_worker_threads: int = 4
    
    # Memory management
    enable_memory_pooling: bool = True
    memory_pool_size_mb: int = 100
    
    # Caching
    enable_state_caching: bool = True
    cache_ttl_seconds: int = 60


@dataclass
class ActionRewardConfig:
    """Configuration for action-specific rewards."""
    
    # Core action rewards
    defend_success_reward: float = 0.02
    pass_action_reward: float = 0.01
    
    # Extended rewards (optional - for future use)
    successful_gather_bonus: float = 0.05
    successful_share_bonus: float = 0.03
    successful_attack_bonus: float = 0.1
    reproduction_success_bonus: float = 0.15
    
    # Penalties
    failed_action_penalty: float = -0.05
    collision_penalty: float = -0.02


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and management."""
    
    # Chunk sizes for processing
    chunk_size: int = 1000
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
    checkpoint_directory: str = "checkpoints"
    
    # Results
    results_directory: str = "results"
    auto_save_results: bool = True
```

---

## 11. Conclusion

Your configuration system is **exceptionally well-structured** with excellent hierarchical organization, versioning, and backward compatibility. The analysis reveals:

### Current Strengths:
- ✅ Comprehensive coverage of core simulation parameters
- ✅ Excellent versioning system with hash-based config tracking
- ✅ Strong backward compatibility with flat-to-nested conversion
- ✅ Good validation in dataclass `__post_init__` methods
- ✅ Centralized loading with environment/profile support
- ✅ JSON schema generation for UI integration

### Main Gaps Found:
1. **Action rewards** (defend=0.02, pass=0.01) - **Critical**
2. **Performance tuning** (batch sizes scattered) - **Critical**
3. **Database connection pooling** (pool_size, timeout) - **High**
4. **Caching parameters** (DQN cache, state cache) - **High**
5. **Channel decay rates** (if used) - **Medium**
6. **Minor utility defaults** (spawn amounts, noise) - **Low**

### Implementation Priority:

**Phase 1 (Immediate):**
- Add `ActionRewardConfig` with defend_reward and pass_reward
- Add `PerformanceConfig` with agent_processing_batch_size
- Enhance `DatabaseConfig` with connection pool and buffer settings
- Add dirty_region_batch_size to `SpatialIndexConfig`

**Phase 2 (Soon):**
- Add DQN cache size to `LearningConfig`
- Add export_batch_size to `DatabaseConfig`
- Add default_spawn_amount to `ResourceConfig`
- Clean up duplicate attack parameters

**Phase 3 (Nice to Have):**
- Add `ExperimentConfig` for experiment tracking
- Add `ChannelDecayConfig` if decay rates need tuning
- Standardize naming conventions across all configs

The highest priority fixes would give you control over the most frequently adjusted experimental parameters while maintaining the excellent structure you've already built.
