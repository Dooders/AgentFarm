# Initial Genotype Diversity

AgentFarm can seed each starting agent with a slightly different
hyperparameter chromosome before the simulation loop begins. This produces a
non-monoculture starting population so subsequent learning, reproduction, and
selection have a richer search space to work with from step zero.

This is a **platform-wide feature**: every simulation can opt in via a single
config block; the intrinsic-evolution runner installs a non-trivial mode by
default. Off by default for ordinary simulations so existing baselines remain
behaviorally identical.

## Configuration

Driven by `SimulationConfig.initial_diversity`, which is an
[`InitialDiversityConfig`](../farm/core/initial_diversity.py).

```python
from farm.config import SimulationConfig
from farm.core.initial_diversity import InitialDiversityConfig, SeedingMode

config = SimulationConfig.from_centralized_config(environment="development")
config.initial_diversity = InitialDiversityConfig(
    mode=SeedingMode.UNIQUE,
    mutation_rate=1.0,
    mutation_scale=0.2,
    max_retries_per_agent=32,
    seed=42,  # optional; defaults to the simulation seed
)
```

You can also set the same fields from the CLI when running
`python run_simulation.py`:

```bash
python run_simulation.py \
  --initial-diversity-mode unique \
  --initial-diversity-mutation-rate 1.0 \
  --initial-diversity-mutation-scale 0.2 \
  --initial-diversity-max-retries 32 \
  --seed 42
```

## Modes

| Mode | Behavior | Guarantees |
|---|---|---|
| `none` | Skip seeding; chromosomes stay at their config defaults. | Backwards compatible; zero-cost. |
| `independent_mutation` | Mutate every agent independently using `mutation_rate` / `mutation_scale`. | Best-effort diversity; collisions possible at small populations. |
| `unique` | Independent mutation with bounded retries until each chromosome is novel at the gene encoding precision. | Distinct chromosomes when `max_retries_per_agent` allows; otherwise records `fallbacks > 0`. |
| `min_distance` | Independent mutation with bounded retries until each chromosome is at least `min_distance` away (normalized Euclidean over evolvable genes) from every previously accepted chromosome. | Pairwise distance lower bound when satisfiable; otherwise records `fallbacks > 0`. |

Strict modes (`unique`, `min_distance`) **always terminate**: once
`max_retries_per_agent` candidates have been drawn for an agent, the latest
candidate is accepted and the metrics report a fallback. Choose
`mutation_scale` and `min_distance` together to keep fallbacks low.

## Telemetry

Every seeding pass returns an
[`InitialDiversityMetrics`](../farm/core/initial_diversity.py) instance,
attached to the environment as `environment.initial_diversity_metrics`:

| Field | Description |
|---|---|
| `mode` | Seeding mode that produced the report. |
| `agents_processed` | Number of agents whose chromosomes were considered. |
| `unique_count` | Distinct chromosome signatures observed (encoding precision). |
| `collision_count` | Accepted chromosomes whose signature matched a prior accepted one. |
| `retries_used` | Total failed candidate draws across all agents. |
| `fallbacks` | Number of agents accepted under fallback (retries exhausted). |
| `min_pairwise_distance` | Smallest normalized Euclidean distance between accepted chromosomes. |
| `mean_pairwise_distance` | Mean normalized Euclidean distance across all accepted pairs. |
| `notes` | Human-readable diagnostics surfaced during seeding. |

When `mode != none`, `run_simulation` also:

- Emits a structured `initial_diversity_seeded` log event carrying the same fields.
- Writes `<output_dir>/initial_diversity_metadata.json` next to the simulation database.

The intrinsic-evolution runner additionally embeds the metrics in
`intrinsic_evolution_metadata.json` under the `initial_diversity_metrics` key
and the resolved config under `initial_diversity`.

## Determinism

Seeding uses `random.Random(seed)` with `seed = cfg.seed if cfg.seed is not
None else simulation_seed`. Two runs with the same simulation seed (and
unchanged config) produce identical seeded chromosomes and identical metrics.

## Intrinsic evolution defaults

`IntrinsicEvolutionExperiment.run()` sets
`base_config.initial_diversity = InitialDiversityConfig(mode=independent_mutation, mutation_rate=1.0, mutation_scale=0.2, ...)`
when the caller leaves `mode=none`. Pass an explicit `InitialDiversityConfig`
on `base_config` to opt into a stricter mode (e.g. `unique`) for an
intrinsic-evolution run. See
[`docs/experiments/intrinsic_evolution/intrinsic_evolution.md`](experiments/intrinsic_evolution/intrinsic_evolution.md)
for runner-specific guidance.

## Pluggable diversity sources

`apply_initial_diversity(environment, cfg, rng, source=...)` accepts any
implementation of the [`InitialDiversitySource`](../farm/core/initial_diversity.py)
`Protocol`:

```python
class InitialDiversitySource(Protocol):
    def seed(
        self,
        environment: Any,
        cfg: InitialDiversityConfig,
        rng: random.Random,
    ) -> InitialDiversityMetrics: ...
```

The default implementation is `ChromosomeDiversitySource`, which mutates each
agent's `hyperparameter_chromosome`. Future scopes (per-agent-type
parameters, decision-module weights, spatial layouts) can ship as additional
sources without changing the orchestration contract.
