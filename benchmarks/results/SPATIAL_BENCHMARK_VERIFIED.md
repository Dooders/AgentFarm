# Verified spatial benchmark

**Generated file — do not edit numbers by hand.** Regenerate with `PYTHONHASHSEED=0 python benchmarks/implementations/spatial/comprehensive_spatial_benchmark.py --verified` from the repository root (after `pip install -e .`).

## Run metadata

- **generator**: `comprehensive_spatial_benchmark.py --verified`
- **git_revision**: `e340c890f7478d0ce73b03fdf8e8de9d190ce6a0`
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
| AgentFarm KD-Tree | 100 | 0.350 | 5.10 | 0.007 |
| AgentFarm Quadtree | 100 | 1.195 | 3.33 | 0.040 |
| AgentFarm Spatial Hash | 100 | 0.534 | 2.78 | 0.030 |
| SciPy KD-Tree | 100 | 0.148 | 3.97 | 0.003 |
| Scikit-learn KD-Tree | 100 | 0.360 | 23.21 | 0.006 |
| Scikit-learn BallTree | 100 | 0.353 | 43.12 | 0.006 |
| AgentFarm KD-Tree | 500 | 0.937 | 5.08 | 0.035 |
| AgentFarm Quadtree | 500 | 6.210 | 6.44 | 0.165 |
| AgentFarm Spatial Hash | 500 | 2.184 | 3.17 | 0.135 |
| SciPy KD-Tree | 500 | 0.220 | 4.26 | 0.009 |
| Scikit-learn KD-Tree | 500 | 0.445 | 24.53 | 0.012 |
| Scikit-learn BallTree | 500 | 0.465 | 25.84 | 0.012 |
| AgentFarm KD-Tree | 1000 | 1.621 | 5.36 | 0.071 |
| AgentFarm Quadtree | 1000 | 11.561 | 6.70 | 0.337 |
| AgentFarm Spatial Hash | 1000 | 4.180 | 3.70 | 0.233 |
| SciPy KD-Tree | 1000 | 0.290 | 4.38 | 0.017 |
| Scikit-learn KD-Tree | 1000 | 0.563 | 27.22 | 0.020 |
| Scikit-learn BallTree | 1000 | 0.524 | 24.55 | 0.020 |
| AgentFarm KD-Tree | 2000 | 2.972 | 5.56 | 0.138 |
| AgentFarm Quadtree | 2000 | 24.268 | 11.22 | 0.670 |
| AgentFarm Spatial Hash | 2000 | 8.473 | 5.51 | 0.415 |
| SciPy KD-Tree | 2000 | 0.432 | 4.56 | 0.032 |
| Scikit-learn KD-Tree | 2000 | 0.686 | 26.46 | 0.035 |
| Scikit-learn BallTree | 2000 | 0.660 | 25.51 | 0.035 |

### Distribution: `uniform`

| Implementation | Entities | Build (ms) | Avg radius query (µs) | Memory (MB) |
|------------------|----------|-------------|------------------------|-------------|
| AgentFarm KD-Tree | 100 | 0.336 | 4.98 | 0.007 |
| AgentFarm Quadtree | 100 | 1.033 | 3.61 | 0.036 |
| AgentFarm Spatial Hash | 100 | 0.507 | 2.92 | 0.031 |
| SciPy KD-Tree | 100 | 0.125 | 3.90 | 0.003 |
| Scikit-learn KD-Tree | 100 | 0.326 | 23.78 | 0.006 |
| Scikit-learn BallTree | 100 | 0.315 | 23.56 | 0.006 |
| AgentFarm KD-Tree | 500 | 0.861 | 5.05 | 0.035 |
| AgentFarm Quadtree | 500 | 5.038 | 5.57 | 0.163 |
| AgentFarm Spatial Hash | 500 | 2.339 | 3.31 | 0.141 |
| SciPy KD-Tree | 500 | 0.246 | 4.25 | 0.009 |
| Scikit-learn KD-Tree | 500 | 0.496 | 24.57 | 0.012 |
| Scikit-learn BallTree | 500 | 0.436 | 24.27 | 0.012 |
| AgentFarm KD-Tree | 1000 | 1.628 | 5.32 | 0.071 |
| AgentFarm Quadtree | 1000 | 11.039 | 7.19 | 0.342 |
| AgentFarm Spatial Hash | 1000 | 4.149 | 3.62 | 0.270 |
| SciPy KD-Tree | 1000 | 0.300 | 4.50 | 0.017 |
| Scikit-learn KD-Tree | 1000 | 0.548 | 24.90 | 0.020 |
| Scikit-learn BallTree | 1000 | 0.535 | 24.35 | 0.020 |
| AgentFarm KD-Tree | 2000 | 2.972 | 5.73 | 0.138 |
| AgentFarm Quadtree | 2000 | 26.512 | 11.16 | 0.670 |
| AgentFarm Spatial Hash | 2000 | 8.278 | 5.14 | 0.505 |
| SciPy KD-Tree | 2000 | 0.444 | 4.67 | 0.032 |
| Scikit-learn KD-Tree | 2000 | 0.722 | 25.63 | 0.035 |
| Scikit-learn BallTree | 2000 | 0.759 | 33.58 | 0.035 |

## Batch vs immediate position updates (microbenchmark)

For each population, 10% of entities receive a new position. **Batch path**: queue moves then `process_batch_updates(force=True)`. **Immediate path**: batching disabled, `update_entity_position` per move. This is not an end-to-end simulation profile; it isolates update mechanics.

| Entities | Batch path (ms) | Immediate path (ms) | Immediate / batch |
|----------|-----------------|----------------------|-------------------|
| 100 | 0.453 | 0.013 | 0.03× |
| 500 | 1.292 | 0.028 | 0.02× |
| 1000 | 1.773 | 0.053 | 0.03× |

