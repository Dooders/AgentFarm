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
| `learning_rate` | 0.19 | 0.1591 | -0.03082 |
| `gamma` | 0.8114 | 0.8669 | 0.05552 |
| `epsilon_decay` | 0.7841 | 0.7425 | -0.04166 |
| `memory_size` | 2.538e+05 | 2.131e+05 | -4.078e+04 |

## Speciation
- **final**: 0.7747
- **max**: 0.8443
- **mean**: 0.7057
- **unique clusters tracked**: 8

## Lineages
- **founders at start**: 30
- **surviving founders at end**: 7
- **max lineage depth (final snapshot)**: 4
- **mean lineage depth (final snapshot)**: 1.286

## Niche correlation (final snapshot)
### Cluster 4
- size=6, mean_x=None, mean_y=None, mean_energy=None
- size=11, mean_x=None, mean_y=None, mean_energy=None
- size=6, mean_x=None, mean_y=None, mean_energy=None
- size=5, mean_x=None, mean_y=None, mean_energy=None
