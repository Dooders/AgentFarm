---
layout: experiment
title: One of a Kind
subtitle: Initial conditions, not agent type, decide dominance
status: complete
updated: 2026-01-01
category: Emergent behavior
excerpt: >-
  500 iterations of System, Independent, and Control agents on a shared
  resource grid. Who "wins" depends on the success metric — and the strongest
  predictor across metrics is where each type started relative to food.
variants:
  - id: case-study
    title: 500-iteration case study
  - id: positioning
    title: Initial positioning
  - id: equal-access
    title: Equal resource access
  - id: advantaged
    title: Advantaged positioning
  - id: distribution
    title: Resource distribution patterns
---

A case study of dominance dynamics between three agent types in a
resource-constrained world. The question is not "which architecture is best?"
but **which factors most significantly determine agent dominance**, and
whether that answer survives a change in how dominance is defined.

Findings are from the published case-study corpus under
[`docs/research/experiments/one_of_a_kind/`](../experiments/one_of_a_kind/Findings.md)
and should be read as a completed observational study plus a small set of
controlled positioning variants, not as a modern seed-matched matrix.

## Shared world

- 100×100 grid, dynamically replenishing resources, randomized placement
- Three types with randomized starting positions: **System** (cooperation and
  sharing), **Independent** (individual survival and acquisition), **Control**
  (balanced)
- Reproduction without trait inheritance between generations
- 3,000 steps per iteration — long enough for population patterns to settle
- Telemetry: population by type, resources, reproduction events, mortality,
  spatial relationship to resources

## 500-iteration case study {#case-study}

500 independent iterations with randomized initial positions.

| Dominance definition | System | Control | Independent |
|---|---|---|---|
| Population (final count) | 45.4% | 33.2% | 21.4% |
| Survival (longevity) | 22.8% | 27.6% | 49.6% |
| Comprehensive composite | 45.6% | 33.6% | 20.8% |

The composite includes persistence (AUC), recency-weighted AUC, dominance
duration, late growth trend, and final population ratio. Details:
[Measures](../experiments/one_of_a_kind/Measures.md).

The headline is the split: Independents win *survival*, Systems win
*population* and the composite. How you define success changes which
architecture looks effective. The simple final-count ranking is almost
identical to the composite, so the extra machinery mostly confirms the
population result rather than overturning it.

Paths to dominance also split. Systems expand through reproduction when
resource access is favorable. Independents stay small and durable, especially
in scarcity. Controls stay in the middle. See
[Reproduction](../experiments/one_of_a_kind/Reproduction.md),
[Competition](../experiments/one_of_a_kind/Competition.md),
[Cooperation](../experiments/one_of_a_kind/Cooperation.md).

## Initial positioning {#positioning}

The strongest predictor across the 500 runs was **starting proximity to
resources**, not type. The first ~100 steps set a trajectory that usually
held. Agents with an early resource lead invested in reproduction and
compounded; types that started far from food rarely caught up on population
metrics.

Time-series annotations in the case study consistently mark initial resource
advantage as the fork, not a mid-run strategy switch. Full metric list:
[Positioning metrics](../experiments/one_of_a_kind/PositioningMetrics.md),
[Initial positioning](../experiments/one_of_a_kind/InitialPositioning.md).

## Equal resource access {#equal-access}

Controlled variant: equalize proximity so no type starts closer to food.

Intrinsic architecture then matters more. Independents' survival edge and
Systems' reproductive edge show through instead of being swamped by spawn
luck. This is the check that positioning was not merely correlated with type.

## Advantaged positioning {#advantaged}

Controlled variant: give one type a deliberate proximity advantage,
regardless of architecture.

That type consistently achieves dominance. The advantage is portable across
System, Independent, and Control — evidence that the case-study ranking is
mostly "who spawned on the buffet," not an invariant architecture ordering.

## Resource distribution patterns {#distribution}

Controlled variant: clustered vs uniform vs random resource fields, holding
agent-type mix fixed.

Distribution systematically shifts dominance probabilities. Clustered
resources amplify the first-mover spawn effect (one lucky neighborhood feeds
a lineage). Uniform fields reduce that lottery and give architecture more
room. This is the environmental half of the same claim as the positioning
variants.

## Synthesis

In this design, initial spatial luck dominates architecture. Independents are
better at not dying; Systems are better at filling the map when they get
food; neither result should be quoted without the metric and the spawn
condition. Later AgentFarm work (intrinsic evolution, veil ceiling) treats
seed and spatial confounds as first-class design problems — this case study
is why.

## Reproduce and artifacts

Case-study writeups and figures live with the experiment docs, not a modern
CLI runner:

- [Findings](../experiments/one_of_a_kind/Findings.md)
- [Dominance](../experiments/one_of_a_kind/Dominance.md)
- [Catalog entry](../experiments-catalog.md)
