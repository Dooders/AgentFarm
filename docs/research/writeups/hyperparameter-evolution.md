---
layout: experiment
title: Hyperparameter evolution convergence
subtitle: Generational search over learning priors, and what actually converges
status: complete
updated: 2026-04-18
category: Evolutionary dynamics
excerpt: >-
  A classical GA over HyperparameterChromosome values, scored by short
  simulation rollouts. Tournament plus moderate mutation can raise fitness and
  still collapse learning rate to the bound; reflection plus a stable preset
  is what made the search usable.
variants:
  - id: smoke
    title: Pipeline smoke
  - id: closure
    title: Tournament vs roulette
  - id: boundary
    title: Clamp vs reflect vs penalty
  - id: penalty-strength
    title: Penalty-strength sensitivity
  - id: stable-preset
    title: stable_hyper_evo preset
---

This experiment is the *extra-population* counterpart of
[intrinsic evolution](intrinsic-evolution.md). A generational genetic
algorithm proposes chromosomes, each candidate is scored by a short
simulation, and selection / mutation / crossover produce the next generation.
There is an external fitness function (`final_population`, `total_births`, or
`final_resources`). Intrinsic evolution has none of that — selection there is
survival in a shared world.

Protocol and CLI details:
[Hyperparameter Evolution Convergence](../experiments/hyperparameter_evolution_convergence.md).

## Shared protocol

- Runner: `scripts/run_evolution_experiment.py` (`EvolutionExperiment`)
- Generational metrics: `evolution_generation_summaries.json`
- Lineage: `evolution_lineage.json`
- Typical search controls: tournament or roulette; mutation rate / scale;
  boundary mode `clamp` or `reflect`; optional soft boundary penalty;
  optional adaptive mutation.

Plot a run with:

```bash
python scripts/plot_hyperparameter_evolution.py \
  --summary-json experiments/evolution_convergence/evolution_generation_summaries.json \
  --output experiments/evolution_convergence/hyperparameter_evolution.png
```

## Pipeline smoke {#smoke}

Early encoding check: a short run to confirm chromosomes mutate, crossover,
and persist, not to claim optimization.

Observed in the smoke summaries: learning-rate contraction toward smaller
values across one generation, with **flat fitness**. No selection gradient —
the contraction is initialization, mutation, and elitism. Useful as a
pipeline test; not evidence of convergence.

## Tournament vs roulette {#closure}

Two six-generation runs, population 8, 40 steps per candidate, persisted
under `experiments/evolution_convergence`.

| Run | Selection | Mutation | Seed | Best fitness | Learning-rate story |
|---|---|---|---|---|---|
| `run_tournament_mut020_g6` | tournament | 0.20 / 0.20 | 42 | 68 → 72 | winner `0.001 → 1e-06`; **partial optimization with boundary collapse** |
| `run_roulette_mut040_g6` | roulette | 0.40 / 0.35 | 99 | 72 → 72 (flat) | mean `0.145 → 0.265`, spread up; **mutation-dominated oscillation** |

Lower mutation plus tournament produced a fitness gain and then parked the
winning learning rate on the lower bound. Higher mutation plus roulette kept
diversity and did not improve fitness. `learning_rate` is a sensitive, noisy
control on `final_population`; exploitation risks collapse, exploration risks
stagnation.

## Clamp vs reflect vs penalty {#boundary}

Same seed (42), six generations, three boundary treatments.

| Run | Best fitness | Best learning rate | Min-bound occupancy (of 8) |
|---|---|---|---|
| clamp baseline | 72 → 69 | `0.001 → 1e-06` | `[2, 3, 3, 5, 8, 7]` |
| reflect | 74 → 75 | `0.377` held | `[0, 0, 0, 0, 0, 0]` |
| clamp + penalty | 76 → 74 (adjusted) | stuck at `1e-06` | `[2, 4, 5, 6, 8, 7]` |

Reflective mutation is the treatment that actually prevented lower-bound
occupancy in this comparison. Clamp variants piled onto `1e-06`. A soft
penalty on clamp did not save the search — occupancy still climbed.

## Penalty-strength sensitivity {#penalty-strength}

Follow-up on the clamp+penalty arm: vary penalty strength rather than
inventing a new selection rule. The comparison is documented in the
[experiment note](../experiments/hyperparameter_evolution_convergence.md#penalty-strength-sensitivity-completed).
The practical takeaway that survived into the preset is: do not rely on
penalty strength alone to unstick a collapsed locus; change the boundary
geometry (`reflect`) and keep mutation from exploding.

## stable_hyper_evo preset {#stable-preset}

The configuration that encoded those closure findings as the default way to
run the experiment:

```bash
python scripts/run_evolution_experiment.py \
  --preset stable_hyper_evo \
  --generations 8 \
  --population-size 10 \
  --steps-per-candidate 80 \
  --output-dir experiments/evolution_smoke
```

The preset sets tournament selection, `--boundary-mode reflect`, mutation
rate 0.20, scale 0.15, and enables adaptive mutation. Explicit flags override
the preset. Adaptive mutation grows rate/scale on stall or diversity
collapse and shrinks them when fitness clearly improves; telemetry is
written per generation (`adaptive_event`, multipliers, measured diversity).

Single evolution runs are still noisy. Wrap this experiment in the
[multi-seed cohort runner](../experiments/multi_seed_cohort.md) when a
configuration comparison has to survive seed variance.

## Synthesis

The runner works. Naive clamp + elitist search can look like "convergence"
while it is really bound collapse. Reflective boundaries plus a moderate
tournament mutation schedule are what made fitness gains *and* non-collapsed
learning rates coexist. Report a run as: settings, one convergence figure,
and a one-line verdict (converged / oscillated / collapsed).

## Reproduce and artifacts

```bash
python scripts/run_evolution_experiment.py --preset stable_hyper_evo \
  --generations 8 --population-size 10 --steps-per-candidate 80 \
  --output-dir experiments/evolution_smoke
```

- Runner: `farm/runners/evolution_experiment.py`
- [Experiment note](../experiments/hyperparameter_evolution_convergence.md)
- [Catalog entry](../experiments-catalog.md)
- [Multi-seed cohort](../experiments/multi_seed_cohort.md)
