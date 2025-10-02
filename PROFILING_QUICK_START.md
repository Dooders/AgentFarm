# Profiling Quick Start Guide

## What We've Set Up

✅ **Profiling infrastructure is ready!** Here's what's been created:

### 1. Comprehensive Plan
- **Location**: `docs/PROFILING_AND_BENCHMARKING_PLAN.md`
- **Contains**: 4-phase profiling strategy, expected bottlenecks, optimization recommendations

### 2. Automated Profiling Scripts
- **Phase 1 Runner**: `benchmarks/run_phase1_profiling.py`
- **cProfile Analyzer**: `benchmarks/analyze_cprofile.py`

### 3. Results Directory
- **Location**: `profiling_results/`
- **README**: `profiling_results/README.md` (detailed guide on interpreting results)

### 4. Tools Installed
- ✅ cProfile (built-in)
- ✅ snakeviz (interactive visualization)
- ✅ py-spy (sampling profiler)
- ✅ line_profiler (line-by-line profiling)
- ✅ memory_profiler (memory usage)
- ✅ psutil (system monitoring)

## Quick Start Commands

### Run Phase 1 Profiling (Quick Mode - ~5-10 min)

```bash
cd /workspace
python3 benchmarks/run_phase1_profiling.py --quick
```

**This will:**
- Run cProfile baseline (500 steps, 50 agents)
- Generate detailed reports
- Create flame graphs
- Identify top bottlenecks

### Run Phase 1 Profiling (Full Mode - ~30-60 min)

```bash
cd /workspace
python3 benchmarks/run_phase1_profiling.py
```

**This will:**
- Run multiple configurations (small/medium/large)
- Generate py-spy sampling profiles
- Create comprehensive analysis
- Provide optimization recommendations

### Analyze Results

```bash
# View the summary report
cat profiling_results/phase1/PHASE1_REPORT.md

# Analyze specific cProfile output
python3 benchmarks/analyze_cprofile.py profiling_results/phase1/cprofile_baseline_small.prof

# Open interactive visualization
snakeviz profiling_results/phase1/cprofile_baseline_small.prof

# View flame graph in browser
firefox profiling_results/phase1/pyspy_sampling_medium.svg
```

## What's Running Now

**Background processes started:**
1. Initial cProfile baseline (1000 steps) - check `profiling_results/phase1/cprofile_baseline_log.txt`
2. Phase 1 quick profiling suite

**To check status:**
```bash
# Check for running Python processes
ps aux | grep python | grep -E 'run_simulation|profiling'

# View real-time logs
tail -f profiling_results/phase1/*.log
```

## Next Steps

### After Phase 1 Completes

1. **Review Results**
   ```bash
   cat profiling_results/phase1/PHASE1_REPORT.md
   ```

2. **Identify Top Bottlenecks**
   - Look at cumulative time rankings
   - Categorize by component (spatial, observation, database, etc.)
   - Estimate impact of optimizing each

3. **Update the Plan**
   - Document findings in `docs/PROFILING_AND_BENCHMARKING_PLAN.md`
   - Mark Phase 1 as complete
   - Plan Phase 2 component-level profiling

4. **Start Phase 2** (Component-Level Profiling)
   - Focus on highest-impact components
   - Create detailed component benchmarks
   - Profile specific subsystems

## Phase Overview

### ✅ Phase 1: Macro-Level (IN PROGRESS)
**Goal**: Identify major bottlenecks across the entire simulation

**Key Questions:**
- Which components consume the most time?
- What's the performance breakdown by subsystem?
- Where should we focus optimization efforts?

**Deliverables:**
- cProfile baseline reports
- py-spy flame graphs
- Top 10 bottleneck list
- Optimization priority ranking

### Phase 2: Component-Level (NEXT)
**Goal**: Deep dive into specific components

**Components to Profile:**
1. Spatial Indexing (KD-tree, Quadtree, Spatial Hash)
2. Observation/Perception System
3. Database Logging
4. Decision Making (RL algorithms)
5. Resource Management

### Phase 3: Micro-Level
**Goal**: Line-by-line optimization of hot functions

**Tools:**
- line_profiler for specific functions
- memory_profiler for memory usage
- Manual code inspection

### Phase 4: System-Level
**Goal**: Overall system performance and scaling

**Focus:**
- CPU/Memory usage over time
- Scaling behavior (agents, steps, environment size)
- System resource utilization

## Expected Timeline

- **Phase 1**: 1-3 days (setup + baseline + analysis)
- **Phase 2**: 3-5 days (component profiling)
- **Phase 3**: 3-5 days (line-level optimization)
- **Phase 4**: 2-3 days (system-level analysis)

**Total**: ~2-3 weeks for comprehensive profiling and initial optimizations

## Key Metrics to Track

### Performance
- **Steps per second**: Overall throughput
- **Time per step**: Latency (avg, p95, p99)
- **Agent processing time**: Per-agent overhead

### Scalability
- **Time vs agents**: How does it scale?
- **Memory vs agents**: Memory growth rate
- **Query performance vs density**: Spatial index efficiency

### Components
- **Observation generation**: Time per observation
- **Spatial queries**: Query latency
- **Database inserts**: Insert throughput
- **Decision making**: Decision latency per agent

## Common Bottlenecks (Predictions)

Based on code analysis, we expect bottlenecks in:

1. **Observation Generation** (HIGH)
   - Multi-channel tensor creation
   - Spatial queries per agent
   - Bilinear interpolation

2. **Spatial Indexing** (HIGH)
   - KD-tree rebuilds
   - Batch update overhead
   - Query frequency

3. **Database Logging** (MEDIUM)
   - Buffer flush timing
   - Insert batching
   - I/O blocking

4. **Agent Decisions** (MEDIUM)
   - Neural network inference
   - Experience replay
   - Training overhead

5. **Resource Management** (LOW-MEDIUM)
   - Regeneration algorithms
   - Memmap operations

## Tips

### Getting Better Profiles

1. **Warm up first**: Run a few steps before profiling starts
2. **Disable visualization**: Use `--no-snakeviz` for automated runs
3. **Use benchmark mode**: `--environment benchmark --profile benchmark`
4. **Profile realistic workloads**: Use production-like agent counts and steps
5. **Run multiple times**: Average results for more reliable data

### Analyzing Results

1. **Start broad**: Look at overall time distribution first
2. **Drill down**: Focus on top 5-10 bottlenecks
3. **Understand context**: Why is this function called so much?
4. **Look for patterns**: Repeated work, unnecessary allocations, etc.
5. **Estimate impact**: What's the theoretical speedup?

### Optimization Strategy

1. **Measure first**: Always profile before optimizing
2. **One change at a time**: Isolate the impact of each optimization
3. **Benchmark after**: Verify the improvement
4. **Watch for regressions**: Ensure no other parts slow down
5. **Document tradeoffs**: Note any complexity or maintainability costs

## Getting Help

### Documentation
- Main plan: `docs/PROFILING_AND_BENCHMARKING_PLAN.md`
- Results guide: `profiling_results/README.md`
- Benchmark README: `benchmarks/README.md`

### Tools Documentation
- [cProfile](https://docs.python.org/3/library/profile.html)
- [py-spy](https://github.com/benfred/py-spy)
- [snakeviz](https://jiffyclub.github.io/snakeviz/)
- [line_profiler](https://github.com/pyutils/line_profiler)

### Questions?
Check the detailed plan and READMEs first. For specific issues:
1. Review the tool's documentation
2. Check existing benchmark implementations
3. Look at similar profiling examples online

---

**Ready to start!** The Phase 1 profiling is running in the background. Results will be available in `profiling_results/phase1/` when complete.
