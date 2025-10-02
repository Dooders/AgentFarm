# Phase 3: Micro-Level (Line-by-Line) Profiling - Complete ✅

## What Has Been Implemented

### ✅ Phase 3 Line Profiler

**Script**: `benchmarks/run_phase3_profiling.py`

Profiles specific functions at the line level to identify exact bottleneck lines.

### Target Functions

Based on expected Phase 1/2 bottlenecks:

1. **`observe`** - `environment._get_observation()`
   - Multi-channel observation generation
   - Resource layer building
   - Agent/ally/enemy detection
   - Bilinear interpolation

2. **`agent_act`** - `agent.act()`
   - Decision making
   - Action execution
   - State management
   - Learning updates

3. **`spatial_update`** - `spatial_index.update()`
   - KD-tree rebuilding
   - Dirty region processing
   - Position hashing

4. **`database_log`** - `db.logger.log_agent_action()`
   - Buffer management
   - SQL statement execution
   - Batch insertions

## Usage

### Profile All Functions

```bash
cd /workspace

# Run all line profiles
python3 benchmarks/run_phase3_profiling.py
```

### Profile Specific Function

```bash
# Profile observation generation only
python3 benchmarks/run_phase3_profiling.py --function observe

# Profile agent actions only
python3 benchmarks/run_phase3_profiling.py --function agent_act

# Profile spatial updates only
python3 benchmarks/run_phase3_profiling.py --function spatial_update

# Profile database logging only
python3 benchmarks/run_phase3_profiling.py --function database_log
```

### Memory Profiling

```bash
# Run memory profiler instead of line profiler
python3 benchmarks/run_phase3_profiling.py --memory

# Memory profile specific function
python3 benchmarks/run_phase3_profiling.py --function observe --memory
```

## Output Files

```
profiling_results/phase3/
├── line_profile_observe.txt          # Line-by-line timing for observation
├── line_profile_agent_act.txt        # Line-by-line timing for agent actions
├── line_profile_spatial_update.txt   # Line-by-line timing for spatial updates
├── line_profile_database_log.txt     # Line-by-line timing for database
├── profile_observe.py                # Generated profiling script
├── profile_agent_act.py              # Generated profiling script
├── profile_spatial_update.py         # Generated profiling script
├── profile_database_log.py           # Generated profiling script
├── phase3_summary.json               # Structured results
└── PHASE3_REPORT.md                  # Summary report
```

## Interpreting Results

### Line Profiler Output

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    45        100       1200.0     12.0      5.2      x = agent.position[0]
    46        100      15000.0    150.0     65.2      obs = self._build_observation(agent)
    47        100       6500.0     65.0     28.3      return self._normalize(obs)
    48        100        300.0      3.0      1.3      return obs
```

**What to look for:**
- **High % Time**: Lines consuming most of the function's time
- **High Hits + High Time**: Frequently executed slow operations
- **Unexpected slow lines**: May indicate algorithmic issues

**Example findings:**
```
Line 46: 65.2% of time in _build_observation
  → This is the bottleneck! Investigate _build_observation next

Line 47: 28.3% of time in _normalize
  → Second priority for optimization

Lines 45, 48: <10% combined
  → Not worth optimizing
```

### Memory Profiler Output

```
Line #    Mem usage    Increment  Occurrences   Line Contents
================================================================
    45      125.2 MiB     0.0 MiB          100   x = agent.position[0]
    46      425.8 MiB   300.6 MiB          100   obs = np.zeros((10, 100, 100))
    47      426.0 MiB     0.2 MiB          100   obs = self._normalize(obs)
```

**What to look for:**
- **Large Increments**: Big memory allocations
- **Repeated allocations**: Memory allocated in loops
- **No decrements**: Potential memory leaks

**Example findings:**
```
Line 46: 300.6 MiB allocated
  → Creating large numpy arrays - can we reuse?

Line 47: Small increment
  → _normalize is memory-efficient (good!)
```

## Expected Bottlenecks

### 1. Observation Generation (`observe`)

**Likely slow lines:**
- Spatial queries for nearby resources/agents
- Bilinear interpolation loops
- Tensor allocations/copies
- Channel stacking operations

**Optimization opportunities:**
- Cache observations for stationary agents
- Pre-allocate tensors
- Vectorize bilinear interpolation
- Use memmap for resource layer

### 2. Agent Actions (`agent_act`)

**Likely slow lines:**
- `create_decision_state()` - building state tensor
- `decide_action()` - neural network inference
- Action execution with environment queries
- Learning update (experience replay)

**Optimization opportunities:**
- Batch decision making
- Cache decision states
- Reduce training frequency
- Optimize action validation

### 3. Spatial Updates (`spatial_update`)

**Likely slow lines:**
- Position hashing/comparison
- KD-tree construction
- Dirty region iteration
- Reference list updates

**Optimization opportunities:**
- Incremental tree updates
- Optimize dirty region detection
- Better data structures
- Reduce update frequency

### 4. Database Logging (`database_log`)

**Likely slow lines:**
- SQL statement construction
- Buffer append operations
- Batch insert execution
- Transaction commits

**Optimization opportunities:**
- Larger buffers
- Pre-compiled statements
- Async logging
- Defer non-critical logs

## Analysis Workflow

### 1. Review Each Profile

```bash
# View line profiles
cat profiling_results/phase3/line_profile_observe.txt
cat profiling_results/phase3/line_profile_agent_act.txt
cat profiling_results/phase3/line_profile_spatial_update.txt
cat profiling_results/phase3/line_profile_database_log.txt

# View summary
cat profiling_results/phase3/PHASE3_REPORT.md
```

### 2. Identify Hot Lines

For each function:
- List lines with >20% time
- Note high-hit lines with significant time
- Flag unexpected slow operations

### 3. Analyze Root Causes

Ask for each hot line:
- **Why is it slow?**
  - Complex computation?
  - Inefficient algorithm?
  - Too many allocations?
  - Blocking I/O?

- **Can it be optimized?**
  - Different algorithm?
  - Caching?
  - Vectorization?
  - Pre-computation?

- **What's the expected improvement?**
  - 2x faster? 10x?
  - Worth the effort?

### 4. Plan Optimizations

Prioritize by impact × ease:

**High Impact, Easy:**
- Caching frequently computed values
- Removing unnecessary operations
- Using built-in functions

**High Impact, Medium:**
- Algorithmic improvements
- Data structure changes
- Vectorization

**High Impact, Hard:**
- Architecture changes
- Major refactoring
- C extensions

### 5. Implement & Validate

For each optimization:
1. Implement the change
2. Re-run line profiler
3. Compare before/after
4. Verify no regressions
5. Document the improvement

## Example Optimization Process

### Before: Observation Generation

```python
# Line 46: 65% of time, 100 hits
for resource in nearby_resources:
    rx, ry = discretize_position(resource.position)
    lx, ly = world_to_local(rx, ry, agent_position)
    if in_bounds(lx, ly, obs_size):
        obs[ly, lx] += resource.amount / max_amount
```

**Problem**: Python loop over many resources

**Solution**: Vectorize with NumPy

```python
# Extract positions and amounts
positions = np.array([r.position for r in nearby_resources])
amounts = np.array([r.amount for r in nearby_resources])

# Vectorized discretization
discrete_pos = np.floor(positions).astype(int)
local_pos = discrete_pos - agent_pos + R

# Vectorized bounds check
mask = ((local_pos >= 0) & (local_pos < obs_size)).all(axis=1)

# Vectorized assignment
obs[local_pos[mask, 1], local_pos[mask, 0]] += amounts[mask] / max_amount
```

### After: Observation Generation

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    46        100       2000.0     20.0     15.0      # Vectorized code
```

**Result**: 65% → 15% time (4.3x faster!)

## Integration with Previous Phases

### Cross-Reference with Phase 1 (Macro)

```
Phase 1 found: _get_observation takes 40% of total time
Phase 3 found: Line 46 in _get_observation takes 65% of function time
Conclusion: Optimizing Line 46 could improve overall performance by ~26%
```

### Cross-Reference with Phase 2 (Component)

```
Phase 2 found: Observation generation scales poorly with radius
Phase 3 found: Bilinear interpolation loop is the bottleneck
Conclusion: Vectorizing bilinear interpolation should fix scaling
```

## Next Steps

### After Phase 3 Completes

1. **Document Hot Lines**
   - Create spreadsheet/table of bottleneck lines
   - Include % time, hits, and current implementation

2. **Categorize Optimizations**
   - Quick wins (easy, high impact)
   - Medium term (moderate effort)
   - Long term (major refactoring)

3. **Implement Quick Wins First**
   - Start with easiest optimizations
   - Validate each with benchmarks
   - Build momentum with early successes

4. **Plan Medium-Term Optimizations**
   - Sketch implementation approaches
   - Estimate effort and impact
   - Get feedback from team

5. **Proceed to Phase 4**
   - System-level profiling
   - Scaling analysis
   - Production readiness testing

## Running Phase 3 Now

```bash
cd /workspace

# Start with single function to test
python3 benchmarks/run_phase3_profiling.py --function observe

# If successful, run all
python3 benchmarks/run_phase3_profiling.py
```

**Estimated time**: ~5-10 minutes for all functions

## Success Criteria

Phase 3 is complete when you have:

- [x] Phase 3 infrastructure implemented
- [ ] Line profiles generated for all target functions
- [ ] Hot lines identified (>20% time)
- [ ] Root causes analyzed
- [ ] Optimization strategies documented
- [ ] Quick wins implemented
- [ ] Before/after benchmarks run

## Tips for Success

### Line Profiler Best Practices

1. **Focus on % Time**: Don't optimize 1% lines
2. **Consider Hits**: 1000 hits × 1ms = optimize this!
3. **Look for loops**: Often the biggest opportunities
4. **Check assumptions**: Profile surprises are valuable

### Common Optimization Patterns

- **Caching**: Store computed values
- **Vectorization**: Replace loops with NumPy
- **Pre-allocation**: Reuse arrays/objects
- **Lazy evaluation**: Defer until needed
- **Early exit**: Skip unnecessary work

### Avoiding Premature Optimization

- ✅ Profile first, optimize second
- ✅ Focus on hot paths (>10% time)
- ✅ Validate with benchmarks
- ❌ Don't optimize cold code
- ❌ Don't sacrifice readability for 1% gain

---

**Status**: Phase 3 infrastructure complete and ready to run!

**Next**: Execute Phase 3 line profiling to identify exact bottleneck lines for optimization.
