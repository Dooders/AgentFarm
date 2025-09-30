# Additional Configuration Analysis - Second Pass

## Overview

After the initial high-priority implementation, this document captures additional hardcoded values found in the codebase that could potentially be made configurable in future updates.

---

## New Findings

### 1. Gradient Clipping (Medium Priority)

**Location**: `farm/core/decision/base_dqn.py:424`
```python
torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
```

**Also in**: 
- `farm/core/decision/decision.py:185,302` - `max_grad_norm: 0.5`
- `farm/core/decision/algorithms/tianshou.py:107,141` - `max_grad_norm: 0.5`

**Status**: ⚠️ **INCONSISTENT VALUES** - Different modules use different values (1.0 vs 0.5)

**Recommendation**: Add to `LearningConfig`:
```python
@dataclass
class LearningConfig:
    # ... existing fields ...
    gradient_clip_norm: float = 1.0  # Gradient clipping max norm
```

**Impact**: Medium - affects training stability and convergence

---

### 2. Thread/Worker Timeouts (Low Priority)

#### A. Database Worker Thread Timeout
**Location**: `farm/database/database.py:1167`
```python
self.worker_thread.join(timeout=10.0)
```

#### B. Config Watcher Thread Timeout
**Location**: `farm/config/watcher.py:95`
```python
self.thread.join(timeout=5.0)
```

**Recommendation**: Add to `DatabaseConfig` and a potential `SystemConfig`:
```python
@dataclass
class DatabaseConfig:
    # ... existing fields ...
    worker_thread_timeout: float = 10.0  # Timeout for worker thread shutdown

@dataclass
class SystemConfig:
    """Configuration for system-level settings."""
    config_watcher_timeout: float = 5.0
    ui_update_interval: float = 0.1
```

**Impact**: Low - mostly affects graceful shutdown behavior

---

### 3. Resource Monitoring Settings (Low Priority)

**Location**: `farm/runners/parallel_experiment_runner.py:281,289`
```python
cpu_percent = psutil.cpu_percent(interval=0.1)
# ...
if memory_percent > 90:  # Warning threshold
```

**Recommendation**: Add to `PerformanceConfig`:
```python
@dataclass
class PerformanceConfig:
    # ... existing fields ...
    
    # Resource monitoring
    cpu_monitor_interval: float = 0.1  # Seconds between CPU checks
    memory_warning_threshold: float = 90.0  # Percent memory before warning
    memory_critical_threshold: float = 95.0  # Percent memory before critical
```

**Impact**: Low - affects monitoring and warnings only

---

### 4. Analysis/Clustering Parameters (Low Priority)

#### A. Social Behavior Clustering
**Location**: `farm/analysis/social_behavior/compute.py:340,343`
```python
epsilon = 50.0  # Adjust based on your simulation's spatial scale
clustering = DBSCAN(eps=epsilon, min_samples=2).fit(coordinates)
```

#### B. Data Processing Windows
**Location**: `farm/analysis/data/processors.py:450,455`
```python
.rolling(window=10, min_periods=1).std()
.rolling(window=10, min_periods=1).mean()
```

**Location**: `farm/analysis/health_resource_dynamics.py:231`
```python
.rolling(window=50)
```

**Recommendation**: Create `AnalysisConfig`:
```python
@dataclass
class AnalysisConfig:
    """Configuration for analysis and post-processing."""
    
    # Clustering parameters
    spatial_clustering_epsilon: float = 50.0  # DBSCAN epsilon for spatial clustering
    min_cluster_samples: int = 2  # Minimum samples for cluster formation
    
    # Rolling window analysis
    short_window_size: int = 10  # Short-term rolling window
    long_window_size: int = 50  # Long-term rolling window
    min_window_periods: int = 1  # Minimum periods for valid window
    
    # Feature computation
    min_data_points: int = 5  # Minimum data points for feature computation
    min_group_size: int = 5  # Minimum group size for analysis
```

**Impact**: Low - only affects post-simulation analysis, not core simulation

---

### 5. Config Monitor Settings (Low Priority)

**Location**: `farm/config/monitor.py:53`
```python
self.max_metrics_history = 1000
```

**Recommendation**: Add to a new `MonitoringConfig`:
```python
@dataclass
class MonitoringConfig:
    """Configuration for monitoring and telemetry."""
    
    max_metrics_history: int = 1000  # Maximum metrics to keep in memory
    enable_performance_tracking: bool = True
    log_config_operations: bool = False  # Log all config operations
```

**Impact**: Low - affects monitoring overhead only

---

### 6. UI/Controller Sleep Intervals (Very Low Priority)

**Location**: Various controller files
```python
# farm/controllers/experiment_controller.py:201
time.sleep(0.1)

# farm/controllers/simulation_controller.py:59,277
time.sleep(1)
time.sleep(0.1)

# farm/config/cli.py:227
time.sleep(1)
```

**Recommendation**: Add to `SystemConfig`:
```python
@dataclass
class SystemConfig:
    """Configuration for system-level settings."""
    
    ui_update_interval: float = 0.1  # Seconds between UI updates
    controller_poll_interval: float = 1.0  # Seconds between controller polls
    retry_delay: float = 1.0  # Seconds to wait before retrying operations
```

**Impact**: Very Low - affects UI responsiveness only

---

## Already Configurable (No Action Needed)

The following were checked and are already properly configurable:

### ✅ Hidden Layer Sizes
- `dqn_hidden_size` - Already in `LearningConfig` and module configs
- `mlp.hidden_layer_sizes` - Configurable via algorithm params

### ✅ Spatial Index Settings
- `region_size`, `max_batch_size` - Already in `SpatialIndexConfig` presets
- `max_regions` - Calculated dynamically from environment size

### ✅ Default Num Steps
- `DEFAULT_NUM_STEPS = 1000` in `experiment_runner.py` - Just a fallback
- Actual value comes from `config.simulation_steps`

### ✅ Visualization Constants
- Most are in `VisualizationConfig` already
- `MAX_ANIMATION_FRAMES`, `MAX_RESOURCE_AMOUNT`, etc. - all configurable

---

## Priority Summary

### Critical (Already Done ✅)
- Action rewards
- Batch processing sizes
- Database connection pooling
- Cache sizes

### High Priority (Not Yet Done)
None identified in second pass.

### Medium Priority (Consider for Phase 2)
1. **Gradient clipping norm** - Currently inconsistent (1.0 vs 0.5)
   - Impact: Training stability
   - Files affected: 4
   - Recommendation: Add to `LearningConfig`

### Low Priority (Phase 3 or Later)
2. **Thread timeouts** - Worker and watcher timeouts
3. **Resource monitoring thresholds** - Memory/CPU warning levels
4. **Analysis parameters** - Clustering and windowing settings
5. **Config monitor settings** - Metrics history size

### Very Low Priority (Optional)
6. **UI sleep intervals** - Controller polling rates

---

## Recommendation: Focus on Gradient Clipping

The most important finding from this second pass is the **inconsistent gradient clipping values**:

- `base_dqn.py` uses `max_norm=1.0`
- `decision.py` and `tianshou.py` use `max_grad_norm=0.5`

### Suggested Action:

Add to `LearningConfig`:
```python
@dataclass
class LearningConfig:
    # ... existing fields ...
    
    # Gradient clipping
    gradient_clip_norm: float = 1.0  # Max norm for gradient clipping
    enable_gradient_clipping: bool = True  # Whether to clip gradients
```

Then update the code:
```python
# In base_dqn.py:424
if config.enable_gradient_clipping:
    torch.nn.utils.clip_grad_norm_(
        self.q_network.parameters(), 
        max_norm=config.gradient_clip_norm
    )

# In decision.py and tianshou.py - use config value instead of hardcoded 0.5
```

This would:
1. Make the value consistent and configurable
2. Allow disabling gradient clipping if desired
3. Enable experimentation with different clipping values
4. Resolve the current inconsistency

---

## Files Requiring Updates (If Implementing Medium Priority)

### For Gradient Clipping:
- `farm/config/config.py` - Add fields to `LearningConfig`
- `farm/core/decision/base_dqn.py` - Use config value
- `farm/core/decision/decision.py` - Use config value
- `farm/core/decision/algorithms/tianshou.py` - Use config value
- `farm/config/default.yaml` - Add default values

---

## Validation of Initial Implementation

### ✅ Confirmed Complete:
- All hardcoded batch sizes now configurable
- All hardcoded rewards now configurable
- All hardcoded database settings now configurable
- All hardcoded cache sizes now configurable
- All hardcoded resource settings now configurable

### ✅ No Critical Items Missed

The second pass confirms that all **critical and high priority** items were addressed in the initial implementation.

---

## Summary Table: All Potential Configurable Values

| Item | Location | Current Value | Priority | Status |
|------|----------|---------------|----------|--------|
| **Phase 1 (Done)** |
| Defend reward | action.py | 0.02 | Critical | ✅ Done |
| Pass reward | action.py | 0.01 | Critical | ✅ Done |
| Agent batch size | simulation.py | 32 | Critical | ✅ Done |
| Dirty region batch | dirty_regions.py | 10 | Critical | ✅ Done |
| DB pool size | database.py | 10 | Critical | ✅ Done |
| DB timeout | database.py | 30 | Critical | ✅ Done |
| Log buffer | data_logging.py | 1000 | High | ✅ Done |
| Commit interval | data_logging.py | 30 | High | ✅ Done |
| Export batch | database.py | 1000 | High | ✅ Done |
| DQN cache | base_dqn.py | 100 | High | ✅ Done |
| Spawn amount | resource_manager.py | 5 | High | ✅ Done |
| **Phase 2 (Optional)** |
| Gradient clip | base_dqn.py | 1.0/0.5 | Medium | ❌ Not done |
| Worker timeout | database.py | 10.0 | Low | ❌ Not done |
| Watcher timeout | watcher.py | 5.0 | Low | ❌ Not done |
| CPU interval | parallel_runner.py | 0.1 | Low | ❌ Not done |
| Memory threshold | parallel_runner.py | 90.0 | Low | ❌ Not done |
| Cluster epsilon | social_behavior.py | 50.0 | Low | ❌ Not done |
| Window sizes | processors.py | 10/50 | Low | ❌ Not done |
| Metrics history | monitor.py | 1000 | Low | ❌ Not done |
| UI intervals | controllers | 0.1/1.0 | Very Low | ❌ Not done |

---

## Conclusion

The initial implementation successfully addressed **all critical and high priority items** (11 hardcoded values).

The second pass identified:
- **1 medium priority item** (gradient clipping - inconsistent values)
- **6 low priority items** (timeouts, monitoring, analysis params)
- **1 very low priority item** (UI intervals)

**Recommendation**: The current implementation is complete for high-priority needs. Consider implementing gradient clipping configuration in a future update to resolve the inconsistency between modules.