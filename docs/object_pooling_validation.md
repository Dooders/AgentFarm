## Object Pooling Validation

### Acceptance Criteria (from PR)
- **Reuse without leaks**: Pools must reuse objects safely.
- **Benchmark improves creation speed and reduces GC pauses**.
- **Fallback when pool exhausted**: Must still create new objects.
- **Memory target**: 20–30% lower peak memory.

### What’s Implemented
- **Agent pool**: `farm/core/pool.py` (`AgentPool`, env toggle via `FARM_DISABLE_POOLING`).
- **BaseAgent reuse hooks**: `BaseAgent.reset(...)` and `BaseAgent.prepare_for_release()` to support pooling.
- **Lifecycle integration**:
  - `simulation.create_initial_agents(...)` uses the pool when enabled.
  - `environment.remove_agent(...)` releases agents back to the pool.
- **Action pooling**: `ActionPool` and `global_action_pool` used by `Genome.to_agent(...)` to avoid recreating `Action` instances.

### How to Run Benchmarks
- Synthetic (no torch required):
```bash
python3 scripts/benchmark_pooling_synthetic.py --objects 3000 --payload-kb 96 --max-pool 2000
```

- Full simulation (torch required):
```bash
# With pooling
python3 scripts/benchmark_object_pooling.py --agents 1200 --steps 50

# Without pooling
FARM_DISABLE_POOLING=1 python3 scripts/benchmark_object_pooling.py --agents 1200 --steps 50
```

Notes:
- The env toggle `FARM_DISABLE_POOLING=1` disables pooling paths for apples-to-apples comparisons.
- The simulation benchmark requires a torch-enabled environment.

### Current Results (Synthetic Benchmark)
- With pooling: time 1.315s, mem Δ 316.07 MB, peak RSS 333.53 MB, reused 1500, created 3000
- Without pooling: time 0.779s, mem Δ 473.86 MB, peak RSS 492.00 MB
- **Estimated improvement**: time −68.8% (slower), memory +33.3% better (meets target)

### Interpretation
- **Why time regressed in synthetic test**:
  - The benchmark’s `reset` still allocates large payloads, so reuse doesn’t avoid the most expensive work; Python-level pooling also adds overhead.
  - First wave still constructs all objects (pool starts empty), and only half are reused.
- **Why memory improved**:
  - Reuse reduces allocation churn and live-object pressure, lowering ΔRSS by ~33%.
- **Expectations for real simulation**:
  - `BaseAgent.reset` reuses the `DecisionModule` and avoids reconstructing heavy submodules, which should reduce creation time and GC pauses in practice.

### Real Simulation Benchmark (User Run)
```
With pooling   -> time: 145.42s, mem Δ: 25519.07 MB, peak RSS: 27130.10 MB
Without pooling-> time: 138.00s, mem Δ:  8869.12 MB, peak RSS: 27611.26 MB
```

Observed:
- Time: ~5.4% slower with pooling
- Memory: Δ memory significantly worse with pooling; peak RSS ~similar (slightly better without pooling)

Preliminary Diagnosis:
- Pool size was unbounded, causing retained agents to inflate Δ memory. We added:
  - `SimulationConfig.agent_pool_max_size` with auto-cap (~2× intended population, minimum 1024)
  - Environment now initializes `AgentPool(BaseAgent, max_size=...)` to prevent unbounded growth
- Creation path is now consistent: `simulation.create_initial_agents` uses the environment’s pool; `remove_agent` releases back.
- Memory subsystem init is guarded in `reset` to avoid heavy per-agent memory where not requested.

Next Validation Steps:
1. Set a cap explicitly if needed (example):
   ```yaml
   agent_pool_max_size: 2400  # ~2x of 1200 agents
   ```
2. Re-run the real benchmark:
   ```bash
   # With pooling
   python3 scripts/benchmark_object_pooling.py --agents 1200 --steps 50
   # Without pooling
   FARM_DISABLE_POOLING=1 python3 scripts/benchmark_object_pooling.py --agents 1200 --steps 50
   ```
3. Compare peak RSS and time; target: 20–30% lower peak memory, non-regressing time.

### Next Steps to Improve Timing
- **Make reset cheaper**: Avoid re-allocations in `BaseAgent.reset`; zero or reuse buffers/submodules where possible.
- **Pre-warm the pool** for large spawns to remove “first creation” cost spikes.
- **Validate with real sim benchmark** in a torch-enabled environment (commands above). We expect both memory and creation time to improve.

