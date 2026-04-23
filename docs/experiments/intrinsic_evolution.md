# Intrinsic Evolution Experiment

The intrinsic evolution runner treats hyperparameter selection as an
*emergent* property of a single simulation rather than as an outer-loop
search over independent runs. Each agent carries its own
`HyperparameterChromosome`; offspring inherit it, optionally crossed with a
co-parent and mutated; selection happens implicitly because agents must
survive and reproduce in the shared resource environment.

This is a complement to, not a replacement for,
[`EvolutionExperiment`](./hyperparameter_evolution_convergence.md). The two
runners answer different questions:

| Question | Use |
| --- | --- |
| "What learning rate works best?" (clean, controlled, replicated) | `EvolutionExperiment` |
| "How does the LR distribution evolve under intra-population competition?" | `IntrinsicEvolutionExperiment` |

## Architecture

```
IntrinsicEvolutionExperiment.run()
        |
        +-> seed_population_diversity(env, policy)
        |
        +-> run_simulation loop (single sim)
        |       |
        |       +-> AgentCore.reproduce()
        |             |
        |             +-> _select_coparent() (optional)
        |             +-> crossover_chromosomes (optional)
        |             +-> mutate_chromosome
        |
        +-> GeneTrajectoryLogger.snapshot(env)
                |
                +-> intrinsic_gene_trajectory.jsonl  (every step, aggregate)
                +-> intrinsic_gene_snapshots.jsonl   (every N steps, full)
```

## When to use

Reach for the intrinsic runner when you care about:

- **Frequency-dependent dynamics.** The "best" learning rate may depend on
  what other agents are doing. Outer-loop GAs evaluate each candidate in a
  monoculture, so they can't see this.
- **Non-stationarity.** Resource depletion or population shifts can change
  the optimal hyperparameters mid-run; an in-situ GA tracks that.
- **Cost.** One simulation instead of `generations x population_size`
  separate simulations for a comparable amount of selection pressure.
- **Biological realism.** The environment itself becomes the fitness
  function rather than a human-chosen scalar.

Stick with `EvolutionExperiment` when you need clean per-candidate
counterfactuals, statistical replication across seeds, or production
hyperparameter selection.

## Behavior change vs. previous code

Before this runner existed, `AgentCore.reproduce()` *always* mutated the
parent's chromosome at a hardcoded rate of 0.1, which silently affected the
outer-loop `EvolutionExperiment`'s fitness signal.

The new contract is:

- When `environment.intrinsic_evolution_policy` is `None` or disabled,
  children inherit the parent's chromosome unchanged. This is the default
  for `run_simulation`, `EvolutionExperiment`, and any other path that does
  not explicitly opt in.
- When the policy is attached and enabled (as the
  `IntrinsicEvolutionExperiment` runner does), reproduction runs optional
  crossover with a co-parent, then mutation, using the policy's knobs.

This means the outer-loop GA is now strictly cleaner: each candidate
simulation observes a homogeneous starting population governed by the
candidate's chromosome, with no intra-sim drift contaminating the fitness
signal.

## Components

| Module | Purpose |
| --- | --- |
| [`farm/runners/intrinsic_evolution_experiment.py`](../../farm/runners/intrinsic_evolution_experiment.py) | `IntrinsicEvolutionPolicy`, `IntrinsicEvolutionExperimentConfig`, `IntrinsicEvolutionResult`, `seed_population_diversity()`, `IntrinsicEvolutionExperiment` |
| [`farm/runners/gene_trajectory_logger.py`](../../farm/runners/gene_trajectory_logger.py) | `GeneTrajectoryLogger`: writes per-step aggregates and periodic full snapshots |
| [`farm/core/agent/core.py`](../../farm/core/agent/core.py) | `AgentCore._derive_child_chromosome` and `_select_coparent` (called from `reproduce`) |
| [`farm/core/simulation.py`](../../farm/core/simulation.py) | `run_simulation(..., on_environment_ready, on_step_end)` callback hooks the runner uses |

## Quick start

```python
from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    CrossoverMode,
    MutationMode,
)
from farm.runners.intrinsic_evolution_experiment import (
    IntrinsicEvolutionExperiment,
    IntrinsicEvolutionExperimentConfig,
    IntrinsicEvolutionPolicy,
)

base_config = SimulationConfig.from_centralized_config(environment="development")

policy = IntrinsicEvolutionPolicy(
    enabled=True,
    seed_initial_diversity=True,
    seed_mutation_rate=1.0,
    seed_mutation_scale=0.2,
    mutation_rate=0.1,
    mutation_scale=0.1,
    mutation_mode=MutationMode.GAUSSIAN,
    boundary_mode=BoundaryMode.CLAMP,
    crossover_enabled=True,
    crossover_mode=CrossoverMode.UNIFORM,
    coparent_strategy="nearest_alive_same_type",
    coparent_max_radius=10.0,
)

config = IntrinsicEvolutionExperimentConfig(
    num_steps=2000,
    snapshot_interval=100,
    policy=policy,
    output_dir="experiments/intrinsic_evolution_smoke",
    seed=42,
)

result = IntrinsicEvolutionExperiment(base_config, config).run()
print(f"Final population: {result.final_population}")
print(f"Final mean LR: {result.final_gene_statistics['learning_rate']['mean']}")
```

## `IntrinsicEvolutionPolicy` reference

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `True` | Master switch. When `False`, reproduction inherits chromosomes unchanged. |
| `seed_initial_diversity` | `True` | Mutate every initial agent's chromosome once before the loop starts so the starting population is not a monoculture. |
| `seed_mutation_rate` | `1.0` | Per-gene mutation probability for the seed pass. |
| `seed_mutation_scale` | `0.2` | Per-gene scale for the seed pass. Larger spreads the population further. |
| `mutation_rate` | `0.1` | Per-gene mutation probability at each reproduction event. |
| `mutation_scale` | `0.1` | Per-gene scale at each reproduction event. |
| `mutation_mode` | `MutationMode.GAUSSIAN` | `gaussian` or `multiplicative`. See [`HyperparameterChromosome` docs](../design/hyperparameter_chromosome.md). |
| `boundary_mode` | `BoundaryMode.CLAMP` | `clamp`, `reflect`, or `interior_biased`. Controls out-of-bounds handling after mutation. |
| `interior_bias_fraction` | `1e-3` | Used only when `boundary_mode = interior_biased`. |
| `crossover_enabled` | `False` | When `True`, pick a co-parent and run `crossover_chromosomes` before mutation. |
| `crossover_mode` | `CrossoverMode.UNIFORM` | Crossover operator (`single_point`, `uniform`, `blend`, `multi_point`). |
| `blend_alpha` | `0.5` | BLX-alpha extent for blend crossover. |
| `num_crossover_points` | `2` | Pivot count for multi-point crossover. |
| `coparent_strategy` | `"nearest_alive_same_type"` | Either `nearest_alive_same_type` or `random_alive_same_type`. Both filter to alive agents of the same `agent_type`. |
| `coparent_max_radius` | `None` | Optional spatial cap on the co-parent search; `None` is unbounded. |
| `seed` | `None` | Optional RNG seed. Falls back to `IntrinsicEvolutionExperimentConfig.seed`. |

If no eligible co-parent exists when crossover is enabled, reproduction
silently falls back to mutation-only inheritance. This is the only way
crossover can become asexual at runtime; everything else is policy-driven.

## `IntrinsicEvolutionExperimentConfig` reference

| Field | Default | Meaning |
| --- | --- | --- |
| `num_steps` | `2000` | Length of the simulation. |
| `snapshot_interval` | `100` | Cadence of full per-agent chromosome snapshots. Per-step aggregates are always recorded. |
| `policy` | `IntrinsicEvolutionPolicy()` | See above. |
| `output_dir` | `None` | When set, the runner writes JSONL trajectory and metadata files here (and forwards `path=output_dir` to `run_simulation` for its own artifacts). |
| `seed` | `None` | Top-level RNG seed propagated to `run_simulation` and the policy if the policy seed is unset. |

## Output artifacts

When `output_dir` is set, three new files are written alongside whatever
`run_simulation` produces (config, database, etc.):

### `intrinsic_gene_trajectory.jsonl`

One record per step. Compact and safe to write at every step.

```json
{
  "step": 0,
  "n_alive": 30,
  "n_with_chromosome": 30,
  "gene_stats": {
    "learning_rate": {
      "mean": 0.012,
      "median": 0.011,
      "std": 0.004,
      "min": 0.005,
      "max": 0.022,
      "at_min_count": 0.0,
      "at_max_count": 0.0,
      "boundary_fraction": 0.0
    },
    "gamma": { ... },
    "epsilon_decay": { ... }
  }
}
```

The schema matches the per-generation gene statistics produced by
`EvolutionExperiment`, so downstream tooling can consume both.

### `intrinsic_gene_snapshots.jsonl`

One record every `snapshot_interval` steps (always at step 0). Heavier;
used for lineage / per-agent analysis.

```json
{
  "step": 0,
  "agents": [
    {
      "agent_id": "0",
      "agent_type": "system",
      "generation": 0,
      "parent_ids": ["seed"],
      "chromosome": { "learning_rate": 0.012, "gamma": 0.99, ... }
    },
    ...
  ]
}
```

### `intrinsic_evolution_metadata.json`

A single object summarizing the run. Includes the resolved policy with
enums serialized to plain strings so it round-trips cleanly:

```json
{
  "num_steps_completed": 2000,
  "num_steps_configured": 2000,
  "snapshot_interval": 100,
  "final_population": 47,
  "final_gene_statistics": { ... },
  "policy": {
    "enabled": true,
    "mutation_mode": "gaussian",
    "boundary_mode": "clamp",
    "crossover_mode": "uniform",
    ...
  },
  "seed": 42
}
```

## Caveats and known limitations

- **Confounded fitness.** Survival and reproduction in a shared environment
  are noisy. An agent might "win" because of position, lineage timing, or
  social context rather than its hyperparameters. Replicate runs across
  multiple seeds and look at distributions, not single trajectories.
- **Population_size = 1 per genotype.** Each unique chromosome is initially
  represented by a single agent. Statistical power is low until lineages
  expand. Seed diversity is intentionally aggressive (`mutation_rate=1.0`
  by default) to spread the initial population.
- **No selection-pressure knob yet.** Reproduction cost is fixed by the
  agent's `ReproductionConfig`; density-dependent costs and explicit
  speciation are out of scope for v1.
- **Crossover requires same `agent_type`.** Cross-type pollination is not
  supported. Co-parent selection always filters to identical
  `agent_type`.

## Out of scope (for now)

- Density-dependent reproduction cost / explicit selection-pressure knobs.
- Lineage tree visualization. The snapshot file makes this a notebook job.
- Speciation / niche detection.

## Testing

- [`tests/runners/test_intrinsic_evolution_experiment.py`](../../tests/runners/test_intrinsic_evolution_experiment.py): policy / config validation, `seed_population_diversity`, runner orchestration with mocked `run_simulation`, artifact persistence.
- [`tests/core/agent/test_reproduce_chromosome_policy.py`](../../tests/core/agent/test_reproduce_chromosome_policy.py): no-policy passthrough, mutation path, co-parent selection (nearest, random, radius, type-filter, alive-filter), crossover end-to-end.
- [`tests/test_agent_reproduction_hyperparameters.py`](../../tests/test_agent_reproduction_hyperparameters.py): updated to assert the new contract (no policy = inheritance unchanged).
