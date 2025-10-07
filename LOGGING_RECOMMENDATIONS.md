# General Logging Recommendations for AgentFarm Simulation Framework

## Executive Summary

This document outlines recommended logging additions to improve observability, debugging, and monitoring of the AgentFarm simulation framework. These are **general informational logs** (not error handling) that will help understand system behavior during runtime.

---

## 1. Environment Lifecycle Logging

### 1.1 Environment Initialization Complete
**Location:** `farm/core/environment.py` - End of `__init__` method

```python
logger.info(
    "environment_initialized",
    simulation_id=self.simulation_id,
    dimensions=(self.width, self.height),
    initial_agents=len(self.agents),
    initial_resources=len(self.resources),
    seed=self.seed_value,
    max_steps=self.max_steps,
    database_enabled=self.db is not None,
)
```

**Why:** Confirms successful environment setup with key parameters.

### 1.2 Simulation Step Milestones
**Location:** `farm/core/environment.py` - In `update()` method

```python
# Log every N steps (e.g., 100, 1000)
if self.time % 100 == 0:
    logger.info(
        "simulation_milestone",
        step=self.time,
        agents_alive=len(self.agents),
        total_resources=self.cached_total_resources,
        avg_agent_health=np.mean([a.current_health for a in self._agent_objects.values()]) if self._agent_objects else 0,
        elapsed_time_seconds=(time.time() - start_time) if 'start_time' in locals() else None,
    )
```

**Why:** Provides periodic snapshots of simulation state without excessive logging.

---

## 2. Agent Lifecycle Logging

### 2.1 Agent Added to Environment
**Location:** `farm/core/environment.py` - In `add_agent()` method

```python
logger.info(
    "agent_added",
    agent_id=agent.agent_id,
    agent_type=agent.__class__.__name__,
    position=agent.position,
    initial_resources=agent.resource_level,
    generation=getattr(agent, 'generation', 0),
    step=self.time,
    total_agents=len(self.agents),
)
```

**Why:** Track agent creation events (births, reproduction, initial population).

### 2.2 Agent Removed from Environment
**Location:** `farm/core/environment.py` - In `remove_agent()` method

```python
logger.info(
    "agent_removed",
    agent_id=agent_id,
    cause=getattr(agent, 'death_cause', 'unknown'),
    lifespan=self.time - getattr(agent, 'birth_time', 0),
    final_resources=getattr(agent, 'resource_level', 0),
    step=self.time,
    remaining_agents=len(self.agents) - 1,
)
```

**Why:** Track agent death events and population dynamics.

### 2.3 Population Milestones
**Location:** `farm/core/environment.py` - In `update()` method

```python
# Track population thresholds
if len(self.agents) in [1, 10, 50, 100, 500, 1000]:
    logger.info(
        "population_milestone",
        population=len(self.agents),
        step=self.time,
        agent_types={type(a).__name__: sum(1 for agent in self._agent_objects.values() if type(agent).__name__ == type(a).__name__) for a in self._agent_objects.values()},
    )
```

**Why:** Monitor population growth/decline patterns.

---

## 3. Resource Management Logging

### 3.1 Resources Initialized
**Location:** `farm/core/resource_manager.py` - In `initialize_resources()` method

```python
logger.info(
    "resources_initialized",
    count=len(self.resources),
    total_amount=sum(r.amount for r in self.resources),
    distribution_type=distribution_type,
    use_memmap=self._use_memmap,
    memmap_shape=self._memmap_shape if self._use_memmap else None,
)
```

**Why:** Confirms resource initialization parameters.

### 3.2 Resource Regeneration Cycle
**Location:** `farm/core/resource_manager.py` - In `update_resources()` method

```python
# Log regeneration every N steps
if current_time % 100 == 0 and regenerated > 0:
    logger.debug(
        "resource_regeneration_cycle",
        step=current_time,
        regenerated_count=regenerated,
        total_resources=sum(r.amount for r in self.resources),
        depleted_resources=self.depletion_events,
    )
```

**Why:** Monitor resource availability over time.

### 3.3 Resource Depletion Warning
**Location:** `farm/core/environment.py` - In `update()` method

```python
# Warn when resources are getting low
resource_ratio = self.cached_total_resources / initial_total_resources if initial_total_resources > 0 else 0
if resource_ratio < 0.1:
    logger.warning(
        "resources_depleted",
        remaining_ratio=resource_ratio,
        remaining_count=len([r for r in self.resources if r.amount > 0]),
        step=self.time,
    )
```

**Why:** Early warning of resource exhaustion conditions.

---

## 4. Performance & Optimization Logging

### 4.1 Spatial Index Rebuild
**Location:** `farm/core/spatial/index.py` - In `update()` method

```python
if rebuild_occurred:
    logger.debug(
        "spatial_index_rebuilt",
        agents_count=len(agents),
        resources_count=len(resources),
        rebuild_reason=rebuild_reason,  # 'dirty', 'threshold', 'forced'
        duration_ms=rebuild_duration_ms,
    )
```

**Why:** Monitor spatial index performance and rebuild frequency.

### 4.2 Batch Update Flush
**Location:** `farm/core/spatial/index.py` - In `_apply_batch_updates()` method

```python
if updates_applied > 0:
    logger.debug(
        "batch_updates_applied",
        update_count=updates_applied,
        duration_ms=duration_ms,
    )
```

**Why:** Track batch operation performance.

### 4.3 Database Batch Flush
**Location:** `farm/database/data_logging.py` - In `flush_all_buffers()` method

```python
logger.debug(
    "database_buffers_flushed",
    agents_flushed=len(agent_buffer),
    actions_flushed=len(action_buffer),
    states_flushed=len(state_buffer),
    duration_ms=duration_ms,
    step=step_number,
)
```

**Why:** Monitor database write performance and batch sizes.

### 4.4 Slow Step Warning
**Location:** `farm/core/simulation.py` - In main simulation loop

```python
step_duration = time.time() - step_start_time
if step_duration > 1.0:  # More than 1 second per step
    logger.warning(
        "slow_step_detected",
        step=current_step,
        duration_seconds=step_duration,
        agents_count=len(environment.agents),
        actions_processed=actions_this_step,
    )
```

**Why:** Identify performance bottlenecks in real-time.

---

## 5. Configuration & State Changes

### 5.1 Action Space Updates
**Location:** `farm/core/environment.py` - In `update_action_space()` method

```python
logger.info(
    "action_space_updated",
    enabled_actions=[a.name for a in enabled_actions],
    action_count=len(enabled_actions),
    step=self.time,
)
```

**Why:** Track dynamic action space changes (curriculum learning).

### 5.2 Configuration Reload
**Location:** `farm/config/watcher.py` - In `_handle_file_change()` method

```python
logger.info(
    "configuration_reloaded",
    filepath=filepath,
    config_hash=new_hash[:8],  # First 8 chars of hash
    changes_detected=True,
)
```

**Why:** Track runtime configuration changes (already partially implemented).

---

## 6. Decision & Learning Logging

### 6.1 Decision Algorithm Initialized
**Location:** `farm/core/decision/decision.py` - In algorithm init methods

```python
logger.info(
    "decision_algorithm_initialized",
    algorithm=self.algorithm_name,
    agent_id=self.agent_id,
    network_architecture=self.network_config if hasattr(self, 'network_config') else None,
)
```

**Why:** Track which learning algorithms are being used (already partially implemented).

### 6.2 Learning Progress
**Location:** `farm/core/agent.py` - In learning update methods

```python
# Log every N learning updates
if self.training_steps % 1000 == 0:
    logger.debug(
        "learning_progress",
        agent_id=self.agent_id,
        training_steps=self.training_steps,
        avg_reward=self.recent_avg_reward,
        loss=self.recent_loss if hasattr(self, 'recent_loss') else None,
    )
```

**Why:** Monitor learning progress without excessive logging.

---

## 7. Action Execution Logging

### 7.1 Significant Actions
**Location:** `farm/core/action.py` - In action execution methods

```python
# Log high-impact actions (reproduction, combat)
if action_type in ['reproduce', 'attack', 'defend']:
    logger.info(
        "significant_action",
        action=action_type,
        agent_id=agent.agent_id,
        success=result['success'],
        resources_cost=resources_before - resources_after,
        step=current_step,
    )
```

**Why:** Track important game events without logging every action.

### 7.2 Action Frequency Distribution
**Location:** `farm/core/environment.py` - Periodic summary

```python
# Log action distribution every N steps
if self.time % 1000 == 0:
    action_counts = {}  # Count actions from recent history
    logger.info(
        "action_distribution",
        step=self.time,
        action_counts=action_counts,
        most_common=max(action_counts, key=action_counts.get) if action_counts else None,
    )
```

**Why:** Understand emergent behavior patterns.

---

## 8. Database & Persistence Logging

### 8.1 Database Connection Status
**Location:** `farm/database/database.py` - In `__init__` method

```python
logger.info(
    "database_connected",
    db_path=self.db_path,
    in_memory=isinstance(self, InMemorySimulationDatabase),
    cache_size_mb=self.cache_size_mb,
    pragma_profile=self.pragma_profile,
)
```

**Why:** Confirm database setup and configuration.

### 8.2 Data Export
**Location:** `farm/database/database.py` - In `export_data()` method

```python
logger.info(
    "data_exported",
    format=export_format,
    output_path=filepath,
    tables_exported=len(tables),
    total_rows=total_rows,
    duration_seconds=export_duration,
)
```

**Why:** Track data export operations for analysis workflows.

---

## 9. Memory & Resource Management

### 9.1 Memory Usage Warning
**Location:** `farm/database/database.py` - In memory monitoring

```python
if memory_usage_mb > memory_limit_mb * 0.8:
    logger.warning(
        "high_memory_usage",
        current_mb=memory_usage_mb,
        limit_mb=memory_limit_mb,
        usage_ratio=memory_usage_mb / memory_limit_mb,
    )
```

**Why:** Early warning of memory constraints (already partially implemented).

### 9.2 Memmap Usage
**Location:** `farm/core/resource_manager.py` - After memmap initialization

```python
logger.info(
    "memmap_resources_enabled",
    path=self._memmap_path,
    shape=self._memmap_shape,
    dtype=self._memmap_dtype,
    size_mb=memmap_size_mb,
)
```

**Why:** Confirm memory-mapped resource grid setup.

---

## 10. Simulation Completion Logging

### 10.1 Simulation Summary
**Location:** `farm/core/simulation.py` - In `run_simulation()` at completion

```python
logger.info(
    "simulation_completed",
    simulation_id=simulation_id,
    total_steps=environment.time,
    final_population=len(environment.agents),
    total_births=birth_count,
    total_deaths=death_count,
    final_resources=environment.cached_total_resources,
    duration_seconds=total_duration,
    avg_step_time_ms=avg_step_time_ms,
)
```

**Why:** Comprehensive simulation run summary for analysis.

---

## Implementation Priority

### High Priority (Implement First)
1. Environment initialization complete
2. Agent added/removed
3. Simulation step milestones
4. Slow step warnings
5. Simulation completion summary

### Medium Priority
6. Resource initialization and regeneration
7. Population milestones
8. Database connection and flush operations
9. Memory usage warnings

### Low Priority (Nice to Have)
10. Spatial index rebuilds
11. Action frequency distribution
12. Learning progress
13. Configuration reloads

---

## Log Level Guidelines

| Level | Use Case | Examples |
|-------|----------|----------|
| **DEBUG** | Detailed diagnostic info, frequent events | Spatial index updates, batch flushes, individual resource regeneration |
| **INFO** | Normal operation milestones | Agent births/deaths, step milestones, initialization, completion |
| **WARNING** | Abnormal but recoverable conditions | Low resources, slow steps, high memory usage, deprecated features |
| **ERROR** | Error conditions requiring attention | Already well-covered in error logging pass |

---

## Performance Considerations

1. **Use log levels appropriately:** DEBUG for frequent events, INFO for milestones
2. **Add sampling for high-frequency logs:** Log every N occurrences, not every occurrence
3. **Lazy evaluation:** Use logger method that only evaluates if log level is enabled
4. **Avoid expensive computations:** Don't compute metrics unless logging is enabled
5. **Batch status logs:** Combine multiple metrics into single log entry

---

## Example: Complete Environment Logging

```python
class Environment(AECEnv):
    def __init__(self, ...):
        # ... initialization code ...
        
        # Log completion
        logger.info(
            "environment_initialized",
            simulation_id=self.simulation_id,
            dimensions=(self.width, self.height),
            agents=len(self.agents),
            resources=len(self.resources),
            max_steps=self.max_steps,
        )
    
    def add_agent(self, agent, ...):
        # ... add agent code ...
        
        logger.info(
            "agent_added",
            agent_id=agent.agent_id,
            type=agent.__class__.__name__,
            step=self.time,
            total=len(self.agents),
        )
    
    def remove_agent(self, agent):
        # ... remove agent code ...
        
        logger.info(
            "agent_removed",
            agent_id=agent.agent_id,
            lifespan=self.time - agent.birth_time,
            step=self.time,
            remaining=len(self.agents) - 1,
        )
    
    def update(self):
        # Milestone logging
        if self.time % 100 == 0:
            logger.info(
                "simulation_milestone",
                step=self.time,
                agents=len(self.agents),
                resources=self.cached_total_resources,
            )
        
        # ... update code ...
```

---

## Testing Recommendations

1. **Add log level tests:** Ensure critical logs are at correct level
2. **Test log sampling:** Verify high-frequency logs are properly sampled
3. **Performance benchmarks:** Measure overhead of logging (should be < 1%)
4. **Log format validation:** Ensure all structured logs have consistent format

---

## Benefits

✅ **Better Debugging:** Understand what's happening during simulation runs
✅ **Performance Monitoring:** Identify bottlenecks in real-time  
✅ **System Observability:** Track system health and behavior patterns
✅ **Troubleshooting:** Quick diagnosis of issues from log files
✅ **Analytics:** Rich data for post-simulation analysis
✅ **Compliance:** Audit trail for research reproducibility

---

## Next Steps

1. **Review and prioritize:** Decide which logs are most valuable
2. **Implement high-priority logs:** Start with environment lifecycle and milestones
3. **Test performance impact:** Measure overhead with benchmarks
4. **Document log format:** Create log field reference guide
5. **Add log analysis tools:** Scripts to parse and visualize logs
