# Verified spatial benchmark

**Generated file — do not edit numbers by hand.** Regenerate with `PYTHONHASHSEED=0 python benchmarks/implementations/spatial/comprehensive_spatial_benchmark.py --verified` from the repository root (after `pip install -e .`).

## Run metadata

- **generator**: `comprehensive_spatial_benchmark.py --verified`
- **git_revision**: `952c6b145a8922ea21ac6cb8b3c18f7188edf9f2`
- **platform**: `Linux-6.12.94+-x86_64-with-glibc2.39`
- **processor**: `x86_64`
- **python_version**: `3.12.3`
- **pythonhashseed**: `0`
- **query_workload**: `100 random radius queries + 50 nearest-neighbor queries per run`
- **random_seed**: `42`
- **test_iterations**: `3`
- **warmup_iterations**: `1`
- **world**: `1000x1000`

## Index build and radius-query timings

Each cell is the mean over measured iterations after warmup. AgentFarm timings include the `SpatialIndex` orchestration layer; SciPy / scikit-learn rows time their respective library trees directly.

### Distribution: `clustered`

| Implementation | Entities | Build (ms) | Avg radius query (µs) | Memory (MB) |
|------------------|----------|-------------|------------------------|-------------|
| AgentFarm KD-Tree | 100 | 0.282 | 4.58 | 0.007 |
| AgentFarm Quadtree | 100 | 1.152 | 3.25 | 0.040 |
| AgentFarm Spatial Hash | 100 | 0.466 | 2.75 | 0.030 |
| SciPy KD-Tree | 100 | 0.113 | 3.62 | 0.003 |
| Scikit-learn KD-Tree | 100 | 0.363 | 40.92 | 0.006 |
| Scikit-learn BallTree | 100 | 0.335 | 40.62 | 0.006 |
| AgentFarm KD-Tree | 500 | 0.804 | 4.92 | 0.035 |
| AgentFarm Quadtree | 500 | 5.196 | 5.53 | 0.165 |
| AgentFarm Spatial Hash | 500 | 2.011 | 3.12 | 0.135 |
| SciPy KD-Tree | 500 | 0.168 | 3.93 | 0.009 |
| Scikit-learn KD-Tree | 500 | 0.494 | 43.45 | 0.012 |
| Scikit-learn BallTree | 500 | 0.441 | 41.96 | 0.012 |
| AgentFarm KD-Tree | 1000 | 1.560 | 5.19 | 0.071 |
| AgentFarm Quadtree | 1000 | 11.587 | 6.97 | 0.337 |
| AgentFarm Spatial Hash | 1000 | 3.917 | 3.67 | 0.233 |
| SciPy KD-Tree | 1000 | 0.240 | 4.09 | 0.017 |
| Scikit-learn KD-Tree | 1000 | 0.522 | 42.87 | 0.020 |
| Scikit-learn BallTree | 1000 | 0.484 | 42.28 | 0.020 |
| AgentFarm KD-Tree | 2000 | 2.804 | 5.75 | 0.138 |
| AgentFarm Quadtree | 2000 | 23.444 | 10.41 | 0.670 |
| AgentFarm Spatial Hash | 2000 | 7.454 | 4.83 | 0.415 |
| SciPy KD-Tree | 2000 | 0.382 | 4.16 | 0.032 |
| Scikit-learn KD-Tree | 2000 | 0.614 | 43.33 | 0.035 |
| Scikit-learn BallTree | 2000 | 0.608 | 42.28 | 0.035 |

### Distribution: `uniform`

| Implementation | Entities | Build (ms) | Avg radius query (µs) | Memory (MB) |
|------------------|----------|-------------|------------------------|-------------|
| AgentFarm KD-Tree | 100 | 0.283 | 4.67 | 0.007 |
| AgentFarm Quadtree | 100 | 0.980 | 3.67 | 0.036 |
| AgentFarm Spatial Hash | 100 | 0.468 | 2.92 | 0.031 |
| SciPy KD-Tree | 100 | 0.111 | 3.65 | 0.003 |
| Scikit-learn KD-Tree | 100 | 0.366 | 40.78 | 0.006 |
| Scikit-learn BallTree | 100 | 0.343 | 40.44 | 0.006 |
| AgentFarm KD-Tree | 500 | 0.802 | 4.96 | 0.035 |
| AgentFarm Quadtree | 500 | 4.938 | 5.56 | 0.163 |
| AgentFarm Spatial Hash | 500 | 2.027 | 3.25 | 0.141 |
| SciPy KD-Tree | 500 | 0.165 | 3.97 | 0.009 |
| Scikit-learn KD-Tree | 500 | 0.406 | 42.54 | 0.012 |
| Scikit-learn BallTree | 500 | 0.415 | 43.44 | 0.012 |
| AgentFarm KD-Tree | 1000 | 1.518 | 5.38 | 0.071 |
| AgentFarm Quadtree | 1000 | 10.732 | 7.18 | 0.342 |
| AgentFarm Spatial Hash | 1000 | 3.933 | 3.66 | 0.270 |
| SciPy KD-Tree | 1000 | 0.257 | 4.33 | 0.017 |
| Scikit-learn KD-Tree | 1000 | 0.499 | 43.39 | 0.020 |
| Scikit-learn BallTree | 1000 | 0.501 | 43.19 | 0.020 |
| AgentFarm KD-Tree | 2000 | 2.780 | 5.48 | 0.138 |
| AgentFarm Quadtree | 2000 | 22.389 | 10.93 | 0.670 |
| AgentFarm Spatial Hash | 2000 | 7.636 | 4.73 | 0.505 |
| SciPy KD-Tree | 2000 | 0.404 | 4.28 | 0.032 |
| Scikit-learn KD-Tree | 2000 | 0.671 | 43.74 | 0.035 |
| Scikit-learn BallTree | 2000 | 0.677 | 43.28 | 0.035 |

## Batch vs immediate position updates (microbenchmark)

For each population, 10% of entities receive a new position. **Batch path**: queue moves then `process_batch_updates(force=True)`. **Immediate path**: batching disabled, `update_entity_position` per move. This is not an end-to-end simulation profile; it isolates update mechanics.

| Entities | Batch path (ms) | Immediate path (ms) | Immediate / batch |
|----------|-----------------|----------------------|-------------------|
| 100 | 0.434 | 0.011 | 0.03× |
| 500 | 1.257 | 0.033 | 0.03× |
| 1000 | 1.781 | 0.053 | 0.03× |

## Interleaved step workload (simulation-style)

Interleaved simulation steps on a single KD-tree named index: each step applies random moves then random radius queries. Batch mode queues moves (large flush thresholds so moves are not auto-flushed mid-step); immediate mode calls update_entity_position per move. Compares total wall time over identical schedules.

**Parameters:**
- `index_type`: `kdtree`
- `moves_per_step`: `18`
- `num_steps`: `35`
- `queries_per_step`: `25`
- `timed_iterations`: `5`
- `warmup_iterations`: `2`

| Entities | Batch mean (s) | Immediate mean (s) | Immediate / batch | ms/step batch | ms/step immediate |
|----------|----------------|-------------------|-------------------|---------------|-------------------|
| 500 | 0.0162 | 0.0114 | 0.70× | 0.462 | 0.326 |
| 1000 | 0.0232 | 0.0182 | 0.78× | 0.663 | 0.520 |

Interpretation: **Immediate / batch** is `immediate_mean_s / batch_mean_s`. Values **less than 1** mean immediate updates finished with **lower** total wall time for the same schedule (immediate wins here). Values **greater than 1** mean batching amortized work better overall on this harness.

