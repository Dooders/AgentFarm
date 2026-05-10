# Verified spatial benchmark

**Generated file — do not edit numbers by hand.** Regenerate with `PYTHONHASHSEED=0 python benchmarks/implementations/spatial/comprehensive_spatial_benchmark.py --verified` from the repository root (after `pip install -e .`).

## Run metadata

- **generator**: `comprehensive_spatial_benchmark.py --verified`
- **git_revision**: `1b7bd2220ccd5f2ead28c785082cc20aa2081dd8`
- **platform**: `Linux-6.12.58+-x86_64-with-glibc2.39`
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
| AgentFarm KD-Tree | 100 | 0.290 | 4.81 | 0.007 |
| AgentFarm Quadtree | 100 | 1.164 | 3.35 | 0.040 |
| AgentFarm Spatial Hash | 100 | 0.507 | 2.85 | 0.030 |
| SciPy KD-Tree | 100 | 0.110 | 3.78 | 0.003 |
| Scikit-learn KD-Tree | 100 | 0.306 | 23.26 | 0.006 |
| Scikit-learn BallTree | 100 | 0.291 | 23.12 | 0.006 |
| AgentFarm KD-Tree | 500 | 0.851 | 4.95 | 0.035 |
| AgentFarm Quadtree | 500 | 5.287 | 5.65 | 0.165 |
| AgentFarm Spatial Hash | 500 | 2.140 | 3.22 | 0.135 |
| SciPy KD-Tree | 500 | 0.182 | 3.99 | 0.009 |
| Scikit-learn KD-Tree | 500 | 0.413 | 24.26 | 0.012 |
| Scikit-learn BallTree | 500 | 0.391 | 24.19 | 0.012 |
| AgentFarm KD-Tree | 1000 | 1.500 | 5.09 | 0.071 |
| AgentFarm Quadtree | 1000 | 11.689 | 6.77 | 0.337 |
| AgentFarm Spatial Hash | 1000 | 3.896 | 3.64 | 0.233 |
| SciPy KD-Tree | 1000 | 0.225 | 4.08 | 0.017 |
| Scikit-learn KD-Tree | 1000 | 0.440 | 24.73 | 0.020 |
| Scikit-learn BallTree | 1000 | 0.421 | 24.58 | 0.020 |
| AgentFarm KD-Tree | 2000 | 2.780 | 5.42 | 0.138 |
| AgentFarm Quadtree | 2000 | 23.356 | 10.51 | 0.670 |
| AgentFarm Spatial Hash | 2000 | 7.522 | 5.04 | 0.415 |
| SciPy KD-Tree | 2000 | 0.362 | 4.17 | 0.032 |
| Scikit-learn KD-Tree | 2000 | 0.627 | 25.27 | 0.035 |
| Scikit-learn BallTree | 2000 | 0.556 | 25.00 | 0.035 |

### Distribution: `uniform`

| Implementation | Entities | Build (ms) | Avg radius query (µs) | Memory (MB) |
|------------------|----------|-------------|------------------------|-------------|
| AgentFarm KD-Tree | 100 | 0.316 | 4.81 | 0.007 |
| AgentFarm Quadtree | 100 | 1.021 | 3.72 | 0.036 |
| AgentFarm Spatial Hash | 100 | 0.490 | 2.99 | 0.031 |
| SciPy KD-Tree | 100 | 0.103 | 3.75 | 0.003 |
| Scikit-learn KD-Tree | 100 | 0.313 | 23.55 | 0.006 |
| Scikit-learn BallTree | 100 | 0.306 | 23.42 | 0.006 |
| AgentFarm KD-Tree | 500 | 0.831 | 4.98 | 0.035 |
| AgentFarm Quadtree | 500 | 5.018 | 5.64 | 0.163 |
| AgentFarm Spatial Hash | 500 | 2.123 | 3.28 | 0.141 |
| SciPy KD-Tree | 500 | 0.170 | 4.16 | 0.009 |
| Scikit-learn KD-Tree | 500 | 0.390 | 24.66 | 0.012 |
| Scikit-learn BallTree | 500 | 0.391 | 24.21 | 0.012 |
| AgentFarm KD-Tree | 1000 | 1.493 | 5.23 | 0.071 |
| AgentFarm Quadtree | 1000 | 10.712 | 7.22 | 0.342 |
| AgentFarm Spatial Hash | 1000 | 3.986 | 3.79 | 0.270 |
| SciPy KD-Tree | 1000 | 0.247 | 4.22 | 0.017 |
| Scikit-learn KD-Tree | 1000 | 0.474 | 25.01 | 0.020 |
| Scikit-learn BallTree | 1000 | 0.440 | 24.43 | 0.020 |
| AgentFarm KD-Tree | 2000 | 2.748 | 5.49 | 0.138 |
| AgentFarm Quadtree | 2000 | 22.359 | 10.38 | 0.670 |
| AgentFarm Spatial Hash | 2000 | 7.784 | 4.77 | 0.505 |
| SciPy KD-Tree | 2000 | 0.385 | 4.33 | 0.032 |
| Scikit-learn KD-Tree | 2000 | 0.633 | 25.64 | 0.035 |
| Scikit-learn BallTree | 2000 | 0.553 | 24.87 | 0.035 |

## Batch vs immediate position updates (microbenchmark)

For each population, 10% of entities receive a new position. **Batch path**: queue moves then `process_batch_updates(force=True)`. **Immediate path**: batching disabled, `update_entity_position` per move. This is not an end-to-end simulation profile; it isolates update mechanics.

| Entities | Batch path (ms) | Immediate path (ms) | Immediate / batch |
|----------|-----------------|----------------------|-------------------|
| 100 | 0.452 | 0.011 | 0.02× |
| 500 | 1.273 | 0.027 | 0.02× |
| 1000 | 1.794 | 0.053 | 0.03× |

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
| 500 | 0.0161 | 0.0113 | 0.70× | 0.459 | 0.322 |
| 1000 | 0.0233 | 0.0180 | 0.77× | 0.665 | 0.515 |

Interpretation: **Immediate / batch** is `immediate_mean_s / batch_mean_s`. Values **less than 1** mean immediate updates finished with **lower** total wall time for the same schedule (immediate wins here). Values **greater than 1** mean batching amortized work better overall on this harness.

