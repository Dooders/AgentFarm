# Multi-Seed Cohort Runner

Single evolution runs are noisy.  A configuration that looks best in one run
can appear significantly worse (or better) in another simply because of random
seed variance.  The **cohort runner** eliminates this ambiguity by executing
the same evolution configuration over *N* independent random seeds and
aggregating the results into a single summary with mean, standard deviation,
and convergence statistics.

---

## Quick start

```bash
source venv/bin/activate
python scripts/run_cohort_experiment.py \
  --preset stable_hyper_evo \
  --generations 8 \
  --population-size 10 \
  --steps-per-candidate 80 \
  --num-seeds 5 \
  --base-seed 0 \
  --output-dir experiments/cohort_smoke
```

This runs 5 seeds (0, 1, 2, 3, 4) and writes three artifacts to
`experiments/cohort_smoke/`:

| File | Contents |
|------|----------|
| `cohort_manifest.json` | Resolved configuration snapshot (written before the run) |
| `cohort_aggregate.json` | Per-seed detail + aggregate statistics |
| `cohort_aggregate.csv` | One row per seed (notebook-ready) |
| `seed_<N>/` | Full per-seed evolution artifacts (same layout as `run_evolution_experiment.py`) |

---

## Command-line flags

All evolution flags from `run_evolution_experiment.py` are available plus
two cohort-specific flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--num-seeds` | `3` | Number of seeds to run |
| `--base-seed` | `0` | Seeds are `[base_seed, …, base_seed+num_seeds-1]` |

Every other flag (generations, population-size, preset, adaptive-mutation,
convergence, etc.) applies identically to each seed run.

---

## Artifact schema

### `cohort_aggregate.json`

```json
{
  "config": { ... },
  "num_seeds": 5,
  "seeds": [0, 1, 2, 3, 4],

  "best_fitness_mean": 7.8,
  "best_fitness_std":  1.2,
  "best_fitness_min":  6.0,
  "best_fitness_max":  9.5,

  "convergence_rate": 0.4,
  "convergence_reason_counts": {
    "fitness_plateau": 2
  },
  "mean_generation_of_convergence": 5.5,
  "std_generation_of_convergence":  0.7,

  "lower_bound_occupancy_mean": 0.125,
  "lower_bound_occupancy_std":  0.05,

  "mean_elapsed_seconds": 12.3,
  "total_elapsed_seconds": 61.5,

  "seed_results": [
    {
      "seed": 0,
      "best_fitness": 8.0,
      "num_generations_completed": 8,
      "converged": true,
      "convergence_reason": "fitness_plateau",
      "generation_of_convergence": 6,
      "elapsed_seconds": 11.9,
      "lower_bound_occupancy": 0.125
    }
  ]
}
```

**Field reference**

| Field | Type | Description |
|-------|------|-------------|
| `config` | object | Serialised `EvolutionExperimentConfig` template (seed field is the template value before per-seed override) |
| `num_seeds` | int | Total seeds executed |
| `seeds` | list[int] | Seed values in execution order |
| `best_fitness_mean` | float | Mean of per-seed best fitness values |
| `best_fitness_std` | float | Population standard deviation of best fitness |
| `best_fitness_min` | float | Minimum best fitness across seeds |
| `best_fitness_max` | float | Maximum best fitness across seeds |
| `convergence_rate` | float | Fraction (0–1) of seeds that satisfied a convergence criterion |
| `convergence_reason_counts` | object | Mapping of `ConvergenceReason` value → count |
| `mean_generation_of_convergence` | float\|null | Mean 0-based generation index at convergence (converged seeds only); `null` when no seed converged |
| `std_generation_of_convergence` | float\|null | Standard deviation of the same; `null` when fewer than 2 seeds converged |
| `lower_bound_occupancy_mean` | float\|null | Mean fraction of generations where the best chromosome's `learning_rate` was at its lower boundary |
| `lower_bound_occupancy_std` | float\|null | Standard deviation of the same |
| `mean_elapsed_seconds` | float | Average wall-clock seconds per seed |
| `total_elapsed_seconds` | float | Total wall-clock seconds for the cohort |
| `seed_results` | list | One entry per seed (see below) |

**`seed_results` entry**

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Seed used for this run |
| `best_fitness` | float | Best fitness observed across all generations |
| `num_generations_completed` | int | Generations that ran (may be less than budget when `early_stop=True`) |
| `converged` | bool | Whether a convergence criterion was satisfied |
| `convergence_reason` | str\|null | `"fitness_plateau"`, `"diversity_collapse"`, `"budget_exhausted"`, or `null` |
| `generation_of_convergence` | int\|null | 0-based generation of first convergence event |
| `elapsed_seconds` | float | Wall-clock seconds for this seed |
| `lower_bound_occupancy` | float\|null | Fraction of generations the best chromosome hit the `learning_rate` lower boundary |

### `cohort_aggregate.csv`

One row per seed with the same columns as the `seed_results` entries above.
Load directly into pandas:

```python
import pandas as pd
df = pd.read_csv("experiments/cohort_smoke/cohort_aggregate.csv")
print(df[["seed", "best_fitness", "converged", "lower_bound_occupancy"]])
```

---

## Notebook ingestion

A minimal loading snippet for `notebooks/hyperparameter_evolution_results.ipynb`
or any new notebook:

```python
import json, pandas as pd

with open("experiments/cohort_smoke/cohort_aggregate.json") as f:
    cohort = json.load(f)

# Top-level aggregates
print(f"best_fitness  mean={cohort['best_fitness_mean']:.3f} "
      f"± {cohort['best_fitness_std']:.3f}  "
      f"[{cohort['best_fitness_min']:.3f}, {cohort['best_fitness_max']:.3f}]")
print(f"convergence_rate={cohort['convergence_rate']:.0%}")
print(f"lower_bound_occupancy mean={cohort['lower_bound_occupancy_mean']}")

# Per-seed DataFrame
df = pd.DataFrame(cohort["seed_results"])
df.plot(x="seed", y="best_fitness", marker="o", title="Best fitness per seed")
```

---

## Interpreting results with statistical confidence

### Using mean ± std for fitness comparisons

When comparing two configurations A and B:

- If `best_fitness_mean_A − best_fitness_std_A > best_fitness_mean_B + best_fitness_std_B`,
  configuration A is reliably superior.
- Overlapping 1-σ bands indicate that the apparent winner may invert on
  different seeds.  Run more seeds (`--num-seeds 10+`) or use a paired
  t-test on the `seed_results` values.

### Convergence rate

A high `convergence_rate` (> 0.7) combined with a low
`std_generation_of_convergence` means the search reliably converges in a
predictable number of generations.  A low rate suggests the configuration
rarely escapes the search space within the generation budget — consider
increasing `--generations` or adjusting mutation parameters.

### Lower-bound occupancy

`lower_bound_occupancy_mean` close to 1.0 indicates that winning candidates
consistently collapse to the minimum `learning_rate` boundary.  This is
a sign of **lower-bound collapse** — the optimizer is effectively not
searching the learning-rate space.  Mitigations:

- Switch to `--boundary-mode reflect` (part of the `stable_hyper_evo`
  preset) to let gene values bounce off boundaries instead of sticking.
- Enable `--boundary-penalty-enabled` to add a soft fitness penalty near
  boundaries.
- Widen the `learning_rate` gene's minimum bound in the chromosome config.

### Comparing multiple configurations

Run each configuration as a separate cohort and store results in separate
`--output-dir` directories:

```bash
# Config A
python scripts/run_cohort_experiment.py \
  --preset stable_hyper_evo --num-seeds 10 --base-seed 0 \
  --output-dir experiments/cohort_A

# Config B
python scripts/run_cohort_experiment.py \
  --selection-method roulette --num-seeds 10 --base-seed 0 \
  --output-dir experiments/cohort_B
```

Then load both `cohort_aggregate.json` files in a notebook and compare the
`best_fitness_mean` / `best_fitness_std` side-by-side:

```python
import json, pandas as pd

configs = {"A": "experiments/cohort_A", "B": "experiments/cohort_B"}
rows = []
for name, path in configs.items():
    with open(f"{path}/cohort_aggregate.json") as f:
        d = json.load(f)
    rows.append({
        "config": name,
        "mean": d["best_fitness_mean"],
        "std": d["best_fitness_std"],
        "convergence_rate": d["convergence_rate"],
        "lb_occupancy": d["lower_bound_occupancy_mean"],
    })
comparison = pd.DataFrame(rows)
print(comparison)
```

---

## Programmatic API

The `CohortRunner` class is exported from `farm.runners` and can be used
directly without the CLI:

```python
from farm.config import SimulationConfig
from farm.runners import (
    AdaptiveMutationConfig,
    CohortRunner,
    ConvergenceCriteria,
    EvolutionExperimentConfig,
    EvolutionFitnessMetric,
    EvolutionSelectionMethod,
)
from farm.core.hyperparameter_chromosome import BoundaryMode

base_config = SimulationConfig.from_centralized_config(environment="development")
template = EvolutionExperimentConfig(
    num_generations=8,
    population_size=10,
    num_steps_per_candidate=80,
    selection_method=EvolutionSelectionMethod.TOURNAMENT,
    boundary_mode=BoundaryMode.REFLECT,
    adaptive_mutation=AdaptiveMutationConfig(enabled=True),
    convergence_criteria=ConvergenceCriteria(enabled=True, early_stop=True),
    seed=None,  # overridden per seed
)

runner = CohortRunner(
    base_config=base_config,
    experiment_config_template=template,
    seeds=list(range(5)),          # seeds 0..4
    output_dir="experiments/cohort_api",
)
aggregate = runner.run()

print(f"fitness  {aggregate.best_fitness_mean:.3f} ± {aggregate.best_fitness_std:.3f}")
print(f"converged {aggregate.convergence_rate:.0%} of seeds")
```

The `CohortAggregateResult` and `CohortSeedResult` dataclasses are also
exported from `farm.runners` if you need to type-annotate your own analysis
code.
