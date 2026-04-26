# Intrinsic Evolution — Analysis Summary

## Run
- **num_steps_completed**: 600
- **snapshot_interval**: 25
- **seed**: 42

## Population
- **initial_population**: 30
- **peak_population**: 77
- **mean_population**: 35.45
- **final_population_observed**: 28
- **birth_rate_mean**: 0.002346
- **death_rate_mean**: 0.002369

## Gene means (initial → final)
| gene | initial mean | final mean | shift |
| --- | --- | --- | --- |
| `learning_rate` | 0.2601 | 0.2767 | 0.0166 |
| `gamma` | 0.8089 | 0.795 | -0.01392 |
| `epsilon_decay` | 0.8459 | 0.8474 | 0.001441 |
| `memory_size` | 2000 | 2000 | 0 |

## Speciation
- **final**: 0.2552
- **max**: 0.565
- **mean**: 0.2714
- **unique clusters tracked**: 10

## Lineages
- **founders at start**: 30
- **surviving founders at end**: 17
- **max lineage depth (final snapshot)**: 1
- **mean lineage depth (final snapshot)**: 0.4286

## Niche correlation (final snapshot)
### Cluster 1
- size=28, mean_x=None, mean_y=None, mean_energy=None
