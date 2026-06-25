---
layout: page
title: "DNA-Style Hyperparameter Evolution Results"
---

# DNA-Style Hyperparameter Evolution Results

This devlog covers the design and initial outcomes of the genetics-inspired hyperparameter evolution work in AgentFarm.

Related tracking issue: [Issue #22](https://github.com/Dooders/AgentFarm/issues/22)

## What Changed

The core design shift was to model tunable parameters as explicit genes inside a typed chromosome instead of using free-floating config values.

- Added `HyperparameterChromosome` and `HyperparameterGene` with explicit bounds and validation.
- Implemented encode/decode support for genes, including quantized and log-scale handling for learning-rate style ranges.
- Added genetic operators for selection, crossover, and mutation.
- Added a generation-based evolution runner that persists lineage and summaries for reproducibility.

Design details:

- [Hyperparameter Chromosome Design](../design/hyperparameter_chromosome.md)
- [Convergence Experiment Notes](../experiments/hyperparameter_evolution_convergence.md)

## Why This Design

Using an explicit chromosome structure gives us:

- a clear contract for what can evolve
- deterministic, auditable experiment artifacts
- safer mutation/crossover behavior with bounded genes
- easier extension to additional parameters in later iterations

This keeps the evolutionary logic intentional instead of ad hoc.

## What the Initial Runs Showed

Two comparison runs were executed using different selection and mutation settings:

1. Tournament + lower mutation pressure
2. Roulette + higher mutation pressure

High-level outcome:

- Tournament with lower mutation improved best fitness, but pushed the winning learning rate to the lower bound.
- Roulette with higher mutation preserved broader exploration but showed flat best fitness.

This is the expected exploration/exploitation trade-off showing up clearly in the data.

## Overall Interpretation

The pipeline now works end-to-end and responds to different evolutionary settings in meaningful ways.

Current behavior suggests:

- lower mutation + stronger selection can find useful signal, but risks boundary collapse
- higher mutation + softer selection can keep diversity, but may wash out optimization progress

So the next practical tuning direction is:

- keep tournament-style selection pressure
- lower mutation pressure further
- add a soft lower-bound guard/penalty so improvements do not depend on boundary collapse

## Artifacts

The current comparison outputs live under:

- `experiments/evolution_convergence/run_tournament_mut020_g6`
- `experiments/evolution_convergence/run_roulette_mut040_g6`

Each run includes:

- `evolution_generation_summaries.json`
- `evolution_lineage.json`
- `run_manifest.json`
- `hyperparameter_evolution.png`

## Closing Note

This is a strong foundation milestone: the representation, operators, and experiment loop are in place and reproducible. The next stage is focused tuning and adding more evolvable genes with the same discipline.
