# Profiling Results

This directory contains profiling results from the comprehensive performance analysis of the AgentFarm simulation engine.

## Directory Structure

```
profiling_results/
├── phase1/                    # Macro-level profiling (identify major bottlenecks)
│   ├── cprofile_*.prof       # Binary cProfile data (open with snakeviz)
│   ├── cprofile_*.profile.txt # Human-readable cProfile output
│   ├── cprofile_*.log        # Full execution logs
│   ├── pyspy_*.svg           # Flame graphs (open in browser)
│   ├── pyspy_*.speedscope.json # Speedscope format (import at speedscope.app)
│   ├── phase1_summary.json   # Structured profiling results
│   └── PHASE1_REPORT.md      # Human-readable report
├── phase2/                    # Component-level profiling
├── phase3/                    # Micro-benchmarks (line-level)
└── phase4/                    # System-level profiling
```

## How to Use These Results

### 1. View cProfile Results

**Interactive visualization (recommended):**
```bash
# Install snakeviz if not already installed
pip install snakeviz

# Open interactive flame graph
snakeviz profiling_results/phase1/cprofile_baseline_small.prof
```

**Command-line analysis:**
```bash
# Analyze with our custom tool
python benchmarks/analyze_cprofile.py profiling_results/phase1/cprofile_baseline_small.prof

# Or use Python's pstats directly
python -m pstats profiling_results/phase1/cprofile_baseline_small.prof
# Then type: sort cumulative
# Then type: stats 20
```

### 2. View py-spy Flame Graphs

**SVG flame graphs:**
```bash
# Open in browser
firefox profiling_results/phase1/pyspy_sampling_medium.svg
# or
chrome profiling_results/phase1/pyspy_sampling_medium.svg
```

**Speedscope (interactive):**
1. Go to https://www.speedscope.app/
2. Drag and drop the `.speedscope.json` file
3. Explore the interactive flame graph with zoom, search, and timeline features

### 3. Read Reports

```bash
# Phase 1 summary
cat profiling_results/phase1/PHASE1_REPORT.md

# Detailed logs
cat profiling_results/phase1/cprofile_baseline_small.log
```

## Understanding the Metrics

### cProfile Metrics

- **ncalls**: Number of times the function was called
- **tottime**: Total time spent in the function (excluding subfunctions)
- **percall**: `tottime / ncalls`
- **cumtime**: Cumulative time (including subfunctions)
- **percall**: `cumtime / ncalls`

**Focus on:**
- **cumtime** to find major bottlenecks (functions + their callees)
- **tottime** to find functions that are slow themselves
- High **ncalls** with significant time = optimization opportunity

### py-spy Flame Graphs

- **Width** = time spent (wider = more time)
- **Height** = call stack depth
- **Color** = just for differentiation (not meaningful)

**How to read:**
1. Start from the bottom (entry points)
2. Follow upward to see what's called
3. Wide bars at the top = hot code paths
4. Click to zoom in on specific call stacks

## Key Questions to Answer

### Phase 1 (Macro-Level)

- [ ] What are the top 5 functions by cumulative time?
- [ ] What percentage of time is spent in:
  - Agent decision-making (`agent.act()`)
  - Observation generation (`_get_observation()`)
  - Spatial indexing (`spatial_index.update()`)
  - Database logging (`db.logger.*`)
  - Environment updates (`environment.update()`)
- [ ] Are there any surprising bottlenecks?
- [ ] What's the overall steps-per-second throughput?

### Phase 2 (Component-Level)

- [ ] Which spatial index operations are slowest?
- [ ] How much time is spent building each observation channel?
- [ ] What's the database insert throughput?
- [ ] How long does each RL decision take?
- [ ] What's the resource regeneration overhead?

### Phase 3 (Micro-Level)

- [ ] Which specific lines in hot functions are the bottleneck?
- [ ] Are there unnecessary allocations or copies?
- [ ] Can loops be vectorized?
- [ ] Are there redundant computations?

## Common Patterns to Look For

### Performance Anti-Patterns

1. **Repeated Work**: Same computation multiple times
2. **Unnecessary Allocations**: Creating objects in hot loops
3. **Synchronous I/O**: Blocking database/file operations
4. **Inefficient Data Structures**: Wrong choice for access patterns
5. **Premature Optimization**: Complex code with no benefit

### Optimization Opportunities

1. **Caching**: Memoize expensive computations
2. **Batching**: Group operations together
3. **Lazy Evaluation**: Defer work until needed
4. **Vectorization**: Replace loops with numpy operations
5. **Algorithmic Improvements**: Better data structures or algorithms

## Analyzing Results Workflow

1. **Start with Phase 1 report**
   - Identify top 3-5 bottlenecks
   - Estimate potential impact of optimizing each

2. **Dive into specific components (Phase 2)**
   - Profile individual subsystems
   - Understand component interactions

3. **Line-by-line analysis (Phase 3)**
   - Focus on highest-impact functions
   - Identify exact bottleneck lines

4. **Validate improvements**
   - Run before/after benchmarks
   - Ensure no regressions
   - Document trade-offs

## Next Steps After Phase 1

Based on the Phase 1 results, you should:

1. **Document Top Bottlenecks**
   - List the top 10 functions by cumulative time
   - Categorize by component (spatial, observation, database, etc.)
   - Estimate percentage of total time for each

2. **Prioritize Components for Phase 2**
   - Focus on components with highest impact
   - Consider effort vs. potential gain
   - Plan component-specific profiling

3. **Update Profiling Plan**
   - Mark Phase 1 as complete
   - Document findings and insights
   - Plan Phase 2 activities

4. **Share Results**
   - Present findings to team
   - Discuss optimization priorities
   - Get feedback on approach

## Tips and Tricks

### Performance Profiling Best Practices

1. **Profile in production-like conditions**
   - Use realistic data volumes
   - Similar hardware configuration
   - Representative workloads

2. **Run multiple iterations**
   - Averages are more reliable
   - Warm up JIT/caches first
   - Watch for variance

3. **Profile incrementally**
   - One change at a time
   - Always compare to baseline
   - Document assumptions

4. **Focus on high-impact optimizations**
   - 80/20 rule: 20% of code = 80% of time
   - Optimize bottlenecks, not everything
   - Measure before and after

### Common Pitfalls

- ❌ Profiling in debug mode (slower, misleading)
- ❌ Too short profiling runs (noise dominates)
- ❌ Optimizing non-bottlenecks (wasted effort)
- ❌ Breaking functionality for speed (test!)
- ❌ Ignoring readability (maintainability matters)

### When to Stop Optimizing

- ✓ Performance meets requirements
- ✓ Diminishing returns (<5% improvement)
- ✓ Code complexity increases significantly
- ✓ Other priorities are more important

## Resources

- [cProfile documentation](https://docs.python.org/3/library/profile.html)
- [py-spy documentation](https://github.com/benfred/py-spy)
- [Speedscope](https://www.speedscope.app/)
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [Profiling Python Code](https://realpython.com/python-profiling/)

## Questions?

See the main profiling plan: `/workspace/docs/PROFILING_AND_BENCHMARKING_PLAN.md`
