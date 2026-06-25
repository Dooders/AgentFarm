# Stable Profile Comparison — Intrinsic Evolution

This document compares three intrinsic-evolution runs that share every
evolutionary policy and selection-pressure setting and differ only in their
`stable` initial-conditions profile (resource level, count, and regen rate).
The goal is to isolate the effect of the *resource buffer* on what gets
selected, not just on how many agents survive.

The three runs live under:

- [experiments/intrinsic_evolution_stable_conservative](../../../experiments/intrinsic_evolution_stable_conservative)
- [experiments/intrinsic_evolution_stable_balanced](../../../experiments/intrinsic_evolution_stable_balanced)
- [experiments/intrinsic_evolution_stable_buffered](../../../experiments/intrinsic_evolution_stable_buffered)

## Setup

All three use the runner described in
[intrinsic_evolution.md](intrinsic_evolution.md), invoked through
`scripts/run_intrinsic_evolution_experiment.py` with identical arguments
except the `stable` profile manual overrides. From each
`intrinsic_evolution_metadata.json` the resolved configuration is:

### What changed

| Variant | `initial_agent_resource_level` | `initial_resource_count` | `resource_regen_rate` | `resource_regen_amount` |
| --- | --- | --- | --- | --- |
| conservative | 8 | 32 | 0.14 | 3 |
| balanced | 10 | 34 | 0.15 | 3 |
| buffered | 12 | 36 | 0.16 | 3 |

### What was held fixed

| Setting | Value |
| --- | --- |
| Steps logged | 1000 |
| Warmup steps | 200 |
| Snapshot interval | 50 |
| Founders (post-seeding) | 29–30 |
| Mutation | gaussian, rate 0.15, scale 0.10, reflect boundary |
| Crossover | disabled |
| `selection_pressure` | `low` (local density coef 0.5, no carrying-cap) |
| Initial diversity seeding | `INDEPENDENT_MUTATION`, rate 1.0, scale 0.25 |
| Speciation | GMM, max k = 4, scaler `none` |
| Seed | 42 |

## Population dynamics

`initial_population` here is the post-warmup population at step 1 of logged
steps. Birth/death rates are the per-step means across all 1000 logged steps.

| Variant | Initial (post-warmup) | Peak (step) | Mean | Final | Birth rate (mean) | Death rate (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| conservative | 135 | 137 (step 5) | 87.8 | 88 | 0.00487 | 0.00521 |
| balanced | 100 | 120 (step 45) | 96.4 | 92 | 0.00557 | 0.00558 |
| buffered | 165 | 171 (step 11) | 111.1 | 99 | 0.00427 | 0.00473 |

The post-warmup populations track the resource buffer, but all three relax
toward a similar steady state (88–99 alive at step 1000). Conservative pulls
its population down to ~88 within the first 100 logged steps; buffered drops
roughly twice as far in absolute terms (171 → ~99) and takes most of the run
to bleed the surplus off. The balanced run is the only one whose
death rate does *not* meaningfully exceed its birth rate on average, which
matches its smaller post-warmup overshoot.

## Speciation

`speciation_index` is the silhouette-based score described in
[intrinsic_evolution.md](intrinsic_evolution.md). Higher means clusters in
chromosome space are better separated.

| Variant | Final | Max | Mean | Unique clusters tracked | Index at step 50 / 500 / 1000 |
| --- | --- | --- | --- | --- | --- |
| conservative | 0.684 | 0.765 | 0.678 | 75 | 0.732 / 0.713 / 0.684 |
| balanced | 0.711 | 0.817 | 0.676 | 63 | 0.708 / 0.654 / 0.711 |
| buffered | 0.753 | 0.785 | 0.692 | 76 | 0.653 / 0.729 / 0.753 |

The trajectory shape is the cleanest signal in the comparison:

- **Conservative**: monotonically declining — clusters merge over the run.
- **Balanced**: V-shape — splits, partially reconverges, splits again.
- **Buffered**: monotonically rising — clusters separate further over time.

Conservative and buffered have a comparable number of unique clusters
*tracked* over the whole run (75 vs. 76), but their endpoints are different
because conservative's clusters keep dying off and being replaced, while
buffered's persist and pull apart.

## Lineages

| Variant | Founders at start | Surviving founders at end | Max depth (final) | Mean depth (final) |
| --- | --- | --- | --- | --- |
| conservative | 30 | 15 | 3 | 1.011 |
| balanced | 29 | 15 | 3 | 1.315 |
| buffered | 30 | 18 | 3 | 1.172 |

All three runs hit the same maximum lineage depth of 3 (great-grandchildren),
which is consistent with similar reproduction rates and run length. Buffered
keeps the most founder lines alive (60% vs. 50%); conservative has the
flattest tree (mean depth ~1.0 indicates many surviving founders have no
descendants beyond themselves at the final snapshot).

## Gene shifts

Means at step 0 → step 1000, computed from `per_gene_initial_mean` and
`per_gene_final_mean` in each `analysis_summary.json`. Percent shift is
relative to the initial mean.

### Convergent shifts (all three runs agree on direction)

| Gene | conservative | balanced | buffered | Direction |
| --- | --- | --- | --- | --- |
| `attack_weight` | -20.6% | -25.0% | -9.3% | down |
| `share_weight` | -5.8% | -27.7% | -25.6% | down |
| `attack_mult_desperate` | -3.7% | -13.6% | -14.7% | down |
| `move_mult_no_resources` | -20.6% | -22.2% | -15.4% | down |
| `memory_size` | -27.8% | -8.5% | -24.4% | down |
| `dqn_hidden_size` | -13.9% | -21.9% | -12.3% | down |
| `epsilon_start` | -5.1% | -6.2% | -8.1% | down |
| `per_alpha` | +12.3% | +11.1% | +3.6% | up |
| `target_update_freq` | +21.9% | +33.5% | +5.1% | up |

The cross-variant agreement on these is the most robust pattern in the
comparison. Cheaper, less-aggressive, less-sharing, less-wandering, smaller
networks; more prioritised replay and slower target syncing.

### Direction-flipping shifts (resource regime decides the sign)

| Gene | conservative | balanced | buffered |
| --- | --- | --- | --- |
| `learning_rate` | -8.3% | -6.0% | **+23.1%** |
| `ensemble_size` | **-25.9%** | -8.8% | +2.6% |
| `reproduce_mult_wealthy` | -4.6% | +0.4% | **+8.4%** |
| `reproduce_mult_poor` | **+21.2%** | +0.8% | -5.2% |
| `gamma` | -3.0% | +0.3% | -2.7% |

`learning_rate` is the clearest case: with more food, the population selects
*for* faster learners; with less food, selection drifts toward slower, more
conservative updates. `ensemble_size` collapses by ~26% only in the
conservative run — a "cheap brain under stress" signature. The reproduce
multipliers also tell a consistent story: buffered raises the bonus for
breeding when wealthy and lowers it when poor (selective reproduction
when surplus exists), while conservative does the opposite (breed whenever
you can, especially when poor).

## Per-variant artifacts

Each run has the standard analysis bundle. Quick links to the most useful
plots:

| Variant | Population | Genes | Speciation | Lineage |
| --- | --- | --- | --- | --- |
| conservative | [population_dynamics.png](../../../experiments/intrinsic_evolution_stable_conservative/analysis/population_dynamics.png) | [gene_trajectories.png](../../../experiments/intrinsic_evolution_stable_conservative/analysis/gene_trajectories.png) | [speciation_index.png](../../../experiments/intrinsic_evolution_stable_conservative/analysis/speciation_index.png) | [intrinsic_lineage_tree.png](../../../experiments/intrinsic_evolution_stable_conservative/analysis/intrinsic_lineage_tree.png) |
| balanced | [population_dynamics.png](../../../experiments/intrinsic_evolution_stable_balanced/analysis/population_dynamics.png) | [gene_trajectories.png](../../../experiments/intrinsic_evolution_stable_balanced/analysis/gene_trajectories.png) | [speciation_index.png](../../../experiments/intrinsic_evolution_stable_balanced/analysis/speciation_index.png) | [intrinsic_lineage_tree.png](../../../experiments/intrinsic_evolution_stable_balanced/analysis/intrinsic_lineage_tree.png) |
| buffered | [population_dynamics.png](../../../experiments/intrinsic_evolution_stable_buffered/analysis/population_dynamics.png) | [gene_trajectories.png](../../../experiments/intrinsic_evolution_stable_buffered/analysis/gene_trajectories.png) | [speciation_index.png](../../../experiments/intrinsic_evolution_stable_buffered/analysis/speciation_index.png) | [intrinsic_lineage_tree.png](../../../experiments/intrinsic_evolution_stable_buffered/analysis/intrinsic_lineage_tree.png) |

## Interpretation

1. **Resource buffer scales the carrying-capacity transient, not the steady
   state.** Post-warmup populations track the buffer (135 / 100 / 165) but
   all three relax to 88–99 alive by step 1000. Bigger buffers buy a longer,
   slower decay rather than a higher floor.
2. **Resource buffer tunes whether speciation strengthens or collapses.**
   Buffered keeps splitting clusters apart (final 0.753, rising). Conservative
   merges them (final 0.684, falling). Balanced is bistable (V-shape, ends at
   0.711). With low selection pressure and abundant food, sub-populations
   have room to drift apart; under tighter food, the survivors converge on
   whatever happens to be working.
3. **Selection moves the same direction on cheap-and-cautious traits
   regardless of the buffer.** Attack, share, wandering-when-empty, memory
   size, and network width all drop in every variant. Prioritised replay
   `per_alpha` and `target_update_freq` go up. This is a robust convergent
   signal across the resource regime.
4. **Learning rate is the cleanest direction-flipping locus.** Buffered
   selects for faster learners (+23%); conservative and balanced select for
   slower (-8% / -6%). `ensemble_size` is a similar story: only the resource-
   tight run compresses it dramatically. There appears to be a real gradient
   between resource availability and the aggressiveness of the learning
   priors that survive.
5. **Reproduction strategy splits along the buffer.** Buffered selects for
   *selective* reproduction (breed when wealthy, hold off when poor).
   Conservative selects for *opportunistic* reproduction (the wealthy
   multiplier drops slightly while the poor multiplier rises by +21%).
   Balanced sits between them with neither shift exceeding 1%.

## Caveats

- **Single seed each.** Every run uses `seed=42`. The qualitative pattern is
  consistent across the three buffer levels, but a multi-seed sweep is
  needed before treating any specific number as representative.
- **No crossover.** All three runs are mutation-only inheritance. The
  speciation pattern in particular may look very different with crossover
  enabled, since it provides another mechanism for clusters to mix or stay
  apart.
- **Same warmup length.** The 200-step warmup runs the simulation under each
  resource profile *before* logging starts, which is part of why the
  post-warmup populations differ so much. The buffered profile produces a
  larger founder cohort (because more agents survive warmup), and that
  initial size affects the first ~100 logged steps. None of the steady-state
  conclusions depend on the post-warmup population, but the early-window
  metrics do.

## Suggested next steps

- Replicate each profile across a small seed sweep (e.g. seeds 42, 7, 19,
  101) to confirm the speciation-trajectory direction is a property of the
  buffer rather than a single-seed artefact.  See
  `scripts/run_stable_profile_seed_sweep.py` for the runner and
  `scripts/analyze_stable_profile_seed_sweep.py` for aggregation.
- Extend conservative to ~5000 steps to see whether the cluster-merging
  trend completes (single dominant cluster) or stabilises around k=2.
- ~~Re-run the buffered profile with crossover enabled~~ — done (2026-05-18);
  see [crossover_rerun.md](crossover_rerun.md) (Issue
  [#845](https://github.com/Dooders/AgentFarm/issues/845)). Rising
  speciation trajectories survive both crossover arms (6/6 seeds diverging;
  no robust collapse of final index or slope). Note: **conservative** profiles
  *do* show robust speciation collapse under crossover in the same sweep.
- Add a `stress` and `legacy` profile to the comparison for a wider
  spread on the carrying-capacity axis.
