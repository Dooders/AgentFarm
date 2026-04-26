# Intrinsic Evolution — Analysis Summary

## Run
- **num_steps_completed**: 5000
- **snapshot_interval**: 100
- **seed**: 42

## Population
- **initial_population**: 30
- **peak_population**: 77
- **mean_population**: 28.09
- **final_population_observed**: 27
- **birth_rate_mean**: 0.0009749
- **death_rate_mean**: 0.0009585

## Gene means (initial → final)
| gene | initial mean | final mean | shift |
| --- | --- | --- | --- |
| `learning_rate` | 0.2601 | 0.2047 | -0.05547 |
| `gamma` | 0.8089 | 0.8266 | 0.01763 |
| `epsilon_decay` | 0.8459 | 0.8184 | -0.02751 |
| `memory_size` | 2000 | 2000 | 0 |

## Speciation
- **final**: 0.3769
- **max**: 0.5978
- **mean**: 0.4265
- **unique clusters tracked**: 12

## Lineages
- **founders at start**: 30
- **surviving founders at end**: 9
- **max lineage depth (final snapshot)**: 2
- **mean lineage depth (final snapshot)**: 0.8148

## Niche correlation (final snapshot)
### Cluster 4
- size=4, mean_x=None, mean_y=None, mean_energy=None
- size=13, mean_x=None, mean_y=None, mean_energy=None
- size=8, mean_x=None, mean_y=None, mean_energy=None
- size=2, mean_x=None, mean_y=None, mean_energy=None
