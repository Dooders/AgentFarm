# Intrinsic Evolution — Analysis Summary

## Run
- **num_steps_completed**: 10000
- **snapshot_interval**: 200
- **seed**: 42

## Population
- **initial_population**: 30
- **peak_population**: 77
- **mean_population**: 27.54
- **final_population_observed**: 28
- **birth_rate_mean**: 0.0007783
- **death_rate_mean**: 0.0007554

## Gene means (initial → final)
| gene | initial mean | final mean | shift |
| --- | --- | --- | --- |
| `learning_rate` | 0.2601 | 0.2529 | -0.007205 |
| `gamma` | 0.8089 | 0.8464 | 0.03752 |
| `epsilon_decay` | 0.8459 | 0.8298 | -0.01612 |
| `memory_size` | 2000 | 2000 | 0 |

## Speciation
- **final**: 0.4844
- **max**: 0.6025
- **mean**: 0.4637
- **unique clusters tracked**: 10

## Lineages
- **founders at start**: 30
- **surviving founders at end**: 7
- **max lineage depth (final snapshot)**: 4
- **mean lineage depth (final snapshot)**: 1.286

## Niche correlation (final snapshot)
### Cluster 4
- size=6, mean_x=None, mean_y=None, mean_energy=None
- size=4, mean_x=None, mean_y=None, mean_energy=None
- size=6, mean_x=None, mean_y=None, mean_energy=None
- size=12, mean_x=None, mean_y=None, mean_energy=None
