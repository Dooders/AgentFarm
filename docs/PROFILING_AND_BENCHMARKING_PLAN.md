# Profiling & Benchmarking Plan for AgentFarm Simulation Engine

## Executive Summary

Your simulation engine is a complex multi-agent system with several performance-critical components:
1. **Core Simulation Loop** (agent.act() → environment.update())
2. **Spatial Indexing** (KD-tree, Quadtree, Spatial Hash)
3. **Observation/Perception System** (multi-channel observations)
4. **Database Logging** (SQLite with batch operations)
5. **Resource Management** (with optional memmap)
6. **Decision Making** (RL algorithms: DDQN, PPO)

---

## Requirements

### 1. Profiling Tools & Dependencies

```python
# Add to requirements.txt
cProfile          # Built-in - already available
pstats            # Built-in - already available
snakeviz>=2.2.0   # Interactive profiler visualization
line_profiler>=4.1.0  # Line-by-line profiling
memory_profiler>=0.61.0  # Memory usage profiling
py-spy>=0.3.14    # Sampling profiler (no code changes)
yappi>=1.4.0      # Multi-threaded profiling
psutil>=5.9.0     # System resource monitoring
```

### 2. Benchmark Infrastructure

Your existing benchmark framework in `/workspace/benchmarks/` is solid. We'll extend it with:
- **Micro-benchmarks** for individual components
- **End-to-end benchmarks** for full simulation runs
- **Regression tracking** to detect performance degradation
- **Comparative analysis** between optimization approaches

---

## Profiling Plan

### Phase 1: Macro-Level Profiling (Identify Major Bottlenecks)

**Goal:** Identify which major components consume the most time

#### 1.1 cProfile Baseline (Already Partially Implemented)

```bash
# Run with existing --perf-profile flag
python run_simulation.py --steps 1000 --agents 100 --perf-profile
```

**Expected Hotspots to Investigate:**
- `agent.act()` - Agent decision-making loop
- `environment.update()` - Environment state updates
- `environment._get_observation()` - Observation generation
- `spatial_index.update()` - Spatial index rebuilds
- `db.logger.log_*()` - Database logging operations

#### 1.2 Py-Spy Sampling Profile (No Code Changes)

```bash
# Install py-spy
pip install py-spy

# Run with py-spy (captures call stacks periodically)
py-spy record -o profile.svg -- python run_simulation.py --steps 1000 --agents 100

# For live monitoring during run
py-spy top -- python run_simulation.py --steps 5000 --agents 200
```

**Advantages:**
- No code instrumentation needed
- Low overhead (~1-3%)
- Interactive flame graphs
- Can attach to running processes

---

### Phase 2: Component-Level Profiling

#### 2.1 Spatial Index Profiling

**Create:** `/workspace/benchmarks/implementations/spatial_index_profiler.py`

```python
"""
Profile spatial index operations:
- KD-tree rebuild time vs. dataset size
- Query performance (get_nearby, get_nearest)
- Batch update performance
- Memory usage per index type
"""
```

**Key Metrics:**
- Time per index rebuild
- Query latency (p50, p95, p99)
- Memory footprint per 1000 entities
- Dirty region tracking overhead

#### 2.2 Observation/Perception Profiling

**Create:** `/workspace/benchmarks/implementations/observation_profiler.py`

```python
"""
Profile observation generation:
- Multi-channel observation build time
- Bilinear interpolation vs. nearest-neighbor
- Memmap resource window vs. spatial queries
- Perception system overhead
"""
```

**Key Metrics:**
- Time per observation generation
- Memory allocations per observation
- Resource layer build time
- Agent layer build time

#### 2.3 Database Logging Profiling

**Extend:** `/workspace/benchmarks/implementations/pragma_profile_benchmark.py`

```python
"""
Profile database operations:
- Batch vs. individual inserts
- Different buffer sizes (100, 500, 1000, 5000)
- SQLite PRAGMA configurations
- In-memory vs. disk performance
"""
```

**Key Metrics:**
- Inserts per second
- Flush time distribution
- Memory usage vs. buffer size
- Disk I/O patterns

#### 2.4 Decision Module Profiling

**Create:** `/workspace/benchmarks/implementations/decision_module_profiler.py`

```python
"""
Profile RL decision-making:
- Forward pass latency (DDQN, PPO)
- Experience replay overhead
- Training update frequency impact
- Memory buffer management
"""
```

**Key Metrics:**
- Decision latency per agent
- Neural network inference time
- Batch training throughput
- Memory usage per algorithm

#### 2.5 Resource Manager Profiling

**Create:** `/workspace/benchmarks/implementations/resource_manager_profiler.py`

```python
"""
Profile resource management:
- Memmap vs. list-based storage
- Regeneration algorithms
- Consumption patterns
- Grid updates
"""
```

**Key Metrics:**
- Resource regeneration time
- Memory usage (memmap vs. list)
- Query performance
- Grid update overhead

---

### Phase 3: Micro-Benchmarks (Line-Level Profiling)

#### 3.1 Line Profiler Setup

```python
# Add decorator to hot functions
from line_profiler import profile

@profile
def _get_observation(self, agent_id: str) -> np.ndarray:
    """Generate observation - line-by-line profiling"""
    # ... existing code
```

**Run with:**
```bash
kernprof -l -v run_simulation.py --steps 100 --agents 50
```

**Target Functions:**
- `environment._get_observation()`
- `agent.act()`
- `spatial_index.update()`
- `decision_module.decide_action()`
- `resource_manager.update_resources()`

#### 3.2 Memory Profiling

```python
# Add to hot functions
from memory_profiler import profile as memory_profile

@memory_profile
def _get_observation(self, agent_id: str) -> np.ndarray:
    # ... existing code
```

**Run with:**
```bash
python -m memory_profiler run_simulation.py --steps 100 --agents 50
```

---

### Phase 4: System-Level Profiling

#### 4.1 Resource Monitoring

**Create:** `/workspace/farm/utils/profiling_utils.py`

```python
"""
System resource monitoring:
- CPU usage per component
- Memory usage over time
- Disk I/O patterns
- Thread/process metrics
"""

import psutil
import time
from contextlib import contextmanager

@contextmanager
def monitor_resources(name: str):
    """Context manager for resource monitoring"""
    process = psutil.Process()
    start_cpu = process.cpu_percent()
    start_mem = process.memory_info().rss / 1024**2  # MB
    start_time = time.perf_counter()
    
    yield
    
    end_time = time.perf_counter()
    end_cpu = process.cpu_percent()
    end_mem = process.memory_info().rss / 1024**2
    
    print(f"{name}:")
    print(f"  Duration: {end_time - start_time:.3f}s")
    print(f"  CPU: {end_cpu:.1f}%")
    print(f"  Memory: {end_mem - start_mem:+.1f} MB")
```

---

## Benchmarking Plan

### Benchmark Suite Structure

```
benchmarks/
├── implementations/
│   ├── profiling/
│   │   ├── spatial_index_profiler.py
│   │   ├── observation_profiler.py
│   │   ├── database_profiler.py
│   │   ├── decision_module_profiler.py
│   │   └── resource_manager_profiler.py
│   ├── regression/
│   │   ├── performance_regression_suite.py
│   │   └── baseline_results.json
│   └── end_to_end/
│       ├── full_simulation_benchmark.py
│       └── scalability_benchmark.py
└── run_profiling_suite.py  # NEW: Master profiling script
```

### Key Benchmarks to Implement

#### 1. Scalability Benchmarks

Test how performance scales with:
- Number of agents (10, 50, 100, 500, 1000)
- Simulation steps (100, 500, 1000, 5000)
- Environment size (50x50, 100x100, 500x500)
- Resource density (sparse, medium, dense)

#### 2. Component Benchmarks

Individual component performance:
- Spatial index query latency at different entity counts
- Observation generation time with different radii
- Database insert throughput with different batch sizes
- Decision latency for different algorithms

#### 3. Configuration Comparison

Compare different configuration options:
- Disk DB vs. In-Memory DB
- Memmap resources vs. list-based
- Batch spatial updates on/off
- Different observation radii
- Different RL algorithms

#### 4. Optimization Verification

Before/after comparisons for optimizations:
- Baseline vs. optimized spatial queries
- Single vs. batched database operations
- Synchronous vs. async logging

---

## Expected Bottlenecks & Optimization Targets

Based on code analysis, likely bottlenecks:

### 1. Observation Generation (High Priority)

**Current Issues:**
- `_get_observation()` called for every agent every step
- Builds multi-channel tensors with spatial queries
- Bilinear interpolation for resource distribution

**Profiling Focus:**
- Time spent in spatial queries
- Bilinear interpolation overhead
- Tensor allocation/copy overhead
- Memmap window extraction performance

**Optimization Opportunities:**
- Cache observations for agents that haven't moved
- Pre-compute static observation components
- Vectorize bilinear interpolation
- Use memmap more aggressively

### 2. Spatial Index Updates (High Priority)

**Current Issues:**
- Rebuilds entire KD-tree when positions change
- Dirty region tracking adds overhead
- Batch updates may delay queries

**Profiling Focus:**
- Frequency of index rebuilds
- Cost of dirty region tracking
- Batch vs. immediate update trade-offs
- Query performance degradation with stale data

**Optimization Opportunities:**
- Tune batch sizes and flush policies
- Implement incremental KD-tree updates
- Use spatial hash for frequent updates
- Profile different index types per use case

### 3. Database Logging (Medium Priority)

**Current Issues:**
- Many small inserts per step
- Buffer flush timing impact
- Disk I/O blocking simulation

**Profiling Focus:**
- Flush frequency vs. performance
- Buffer size impact
- In-memory vs. disk latency
- Batch insert performance

**Optimization Opportunities:**
- Increase buffer sizes
- Use async logging thread
- Optimize PRAGMA settings
- Defer non-critical logging

### 4. Agent Decision Making (Medium Priority)

**Current Issues:**
- Neural network inference per agent
- Experience replay memory overhead
- Training frequency impact

**Profiling Focus:**
- Decision latency distribution
- Training vs. inference time split
- Memory usage growth
- Batch inference opportunities

**Optimization Opportunities:**
- Batch agent decisions
- Reduce training frequency
- Use smaller networks
- Cache decision states

### 5. Resource Management (Low-Medium Priority)

**Current Issues:**
- Regeneration every step
- Iteration over all resources
- Memmap synchronization

**Profiling Focus:**
- Regeneration algorithm cost
- Memmap vs. list performance
- Update frequency necessity

**Optimization Opportunities:**
- Skip regeneration for unchanged resources
- Batch resource updates
- Optimize memmap usage patterns

---

## Deliverables & Timeline

### Week 1: Setup & Baseline
- [ ] Install profiling tools
- [ ] Run cProfile baseline on 1000-step simulation
- [ ] Run py-spy sampling on 5000-step simulation
- [ ] Document top 10 hotspots

### Week 2: Component Profiling
- [ ] Implement spatial index profiler
- [ ] Implement observation profiler
- [ ] Implement database profiler
- [ ] Generate component-level reports

### Week 3: Deep Dive & Micro-Benchmarks
- [ ] Line-profile top 5 hotspots
- [ ] Memory-profile top 3 memory consumers
- [ ] Create micro-benchmarks for critical paths
- [ ] Document optimization opportunities

### Week 4: Optimization & Verification
- [ ] Implement top 3 optimizations
- [ ] Run regression benchmarks
- [ ] Validate performance improvements
- [ ] Document optimization results

---

## Metrics to Track

### Primary Metrics
1. **Steps per second** - Overall simulation throughput
2. **Time per step** - Average/p95/p99 latency
3. **Memory usage** - Peak and average RSS
4. **CPU utilization** - Percentage per core

### Component Metrics
1. **Observation generation time** (ms/agent)
2. **Spatial query latency** (μs/query)
3. **Database insert rate** (inserts/sec)
4. **Decision latency** (ms/agent)
5. **Index rebuild time** (ms)

### Scalability Metrics
1. **Time vs. agent count** (linear, quadratic?)
2. **Memory vs. agent count**
3. **Query performance vs. entity density**

---

## Recommendations

### Immediate Actions

1. **Run baseline cProfile:**
   ```bash
   python run_simulation.py --steps 1000 --agents 100 --perf-profile
   ```

2. **Install py-spy and generate flame graph:**
   ```bash
   pip install py-spy
   py-spy record -o profile.svg -- python run_simulation.py --steps 1000 --agents 100
   ```

3. **Review existing benchmark results in `/workspace/benchmarks/results/`**

4. **Profile with different configurations:**
   - In-memory DB vs. disk DB
   - Different agent counts (50, 100, 200, 500)
   - Different observation radii (5, 10, 20)

### Long-term Strategy

1. **Establish Performance Baseline:**
   - Document current performance across configurations
   - Create regression test suite
   - Set performance budgets per component

2. **Continuous Profiling:**
   - Integrate profiling into CI/CD
   - Track performance metrics over time
   - Alert on regressions

3. **Optimization Prioritization:**
   - Focus on high-impact, low-effort optimizations first
   - Validate each optimization with benchmarks
   - Document trade-offs and configuration recommendations

4. **Scalability Testing:**
   - Test with realistic workloads (1000+ agents, 10000+ steps)
   - Identify scaling bottlenecks
   - Optimize for target use cases

---

## Progress Tracking

### Phase 1: Macro-Level Profiling ✅ RUNNING
- [x] Documentation created
- [x] Profiling tools installed
- [x] Infrastructure implemented
- [x] cProfile baseline running
- [ ] py-spy sampling profile
- [ ] Analysis report generated

### Phase 2: Component-Level Profiling ✅ RUNNING
- [x] Documentation created
- [x] Infrastructure implemented
- [x] Spatial index profiler created
- [x] Observation profiler created
- [x] Database profiler created
- [x] Quick mode profiling running
- [ ] Analysis report generated

### Phase 3: Micro-Benchmarks ✅ RUNNING
- [x] Documentation created
- [x] Infrastructure implemented
- [x] Line profiler setup
- [x] Function-specific profilers created
- [x] Line profiling running
- [ ] Analysis report generated

### Phase 4: System-Level Profiling
- [ ] Not started (future work)
