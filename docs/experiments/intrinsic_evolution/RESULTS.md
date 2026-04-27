# Intrinsic Evolution Experiment — Results (10,000 steps)

This run exercises
`[IntrinsicEvolutionExperiment](../../farm/runners/intrinsic_evolution_experiment.py)`
end-to-end: a single 10000-step simulation in which every agent carries its
own `[HyperparameterChromosome](../../farm/core/hyperparameter_chromosome.py)`,
crossover with a co-parent is enabled, and selection emerges from the shared
resource environment under the `"low"` density-dependent reproduction-cost
preset.

## Configuration


| Setting                   | Value                                             |
| ------------------------- | ------------------------------------------------- |
| Environment               | `development`                                     |
| Steps                     | 10000                                             |
| Snapshot interval         | 200                                               |
| Seed                      | 42                                                |
| Crossover                 | uniform, nearest alive same-type co-parent        |
| Mutation                  | gaussian, rate 0.15, scale 0.10, reflect boundary |
| Initial diversity seeding | rate 1.0, scale 0.25                              |
| Selection pressure        | `low` (local density coef = 0.5, no carrying-cap) |
| Speciation tracking       | GMM, max k = 4                                    |


CLI:

```bash
PYTHONHASHSEED=0 python scripts/run_intrinsic_evolution_experiment.py \
    --num-steps 10000 --snapshot-interval 200 \
    --output-dir experiments/intrinsic_evolution \
    --crossover --selection-pressure low --seed 42
```

Wall-clock: 581 s end-to-end (≈17 steps/s).

## Headline results

- **Population**: 30 → peak 77 (around step 100) → settled to a noisy
steady state of ~28 alive (mean 27.5, final 28).
- **Births / deaths**: ~0.78 birth / death per 1000 steps (mean rates
≈ 7.8e-4 and 7.6e-4); roughly balanced turnover.
- **Surviving founder lineages**: 30 → **7** (77 % founder extinction —
vs. 70 % at 5000 steps and 43 % at 600 steps; selection keeps
compounding).
- **Gene means (initial → final)**:
  - `learning_rate` 0.260 → 0.253 (-0.007; back near origin after a dip
  around step 3000)
  - `gamma` 0.809 → **0.846** (+0.038; sustained directional rise)
  - `epsilon_decay` 0.846 → 0.830 (-0.016)
  - `memory_size` 2000 → 2000 (locked, evolvable=False)
- **Speciation index**: peaked ~0.60 around steps 2400 and 6000; mean
0.46; **final 0.48**. The polymorphism is durable, not a transient.
- **Niches**: GMM detects **k = 4** clusters at step 10000 with sizes
{12, 6, 6, 4} and silhouette 0.48 — the cleanest cluster structure
produced by any run length so far.
- **Lineage depth**: max **4**, mean **1.29** at the final snapshot —
great-great-grandchildren survive.

## Visualisations

All plots are produced by
`python scripts/analyze_intrinsic_evolution.py experiments/intrinsic_evolution`.

### Population dynamics

population dynamics

Same early boom-and-crash pattern as the 600/5000-step runs (population
briefly hits 77 around step 100, then collapses to ~28). The remaining
~9900 steps are a long stationary turnover regime around 25–35 alive.
The selection-strength CV (bottom panel) shows recurring bursts up to
0.40 (notably around step 6000), well above the steady-state ~0.15 — a
sign that density-dependent reproduction cost is non-trivially structured
across the population.

### Gene trajectories (per-step mean ± std)

gene trajectories

`gamma` exhibits a transient dip between steps ~1500 and ~3300 (a
high-`gamma` cluster temporarily expands then dies out, briefly tightening
the std band) followed by a sustained rise to 0.85 — the cleanest
directional gene-mean signal in the run. `learning_rate` mean cycles
between 0.20 and 0.30 without converging. `epsilon_decay` drifts modestly
downward.

### Per-snapshot gene-value distributions

gene distribution history

`gamma` violins narrow significantly after step ~3300; `learning_rate`
keeps a wide bimodal distribution throughout the run, consistent with
the persistent multi-cluster structure.

### Speciation index over time

speciation index

After the early boom-crash dip (steps 300–700), the index climbs steadily
and stays in the 0.40–0.60 band for the remaining ~~9000 steps. Two clear
peaks (~~step 2400 and ~step 6000) reach 0.60. The population maintains
4 niches throughout — this is the durable polymorphic equilibrium the
runner is designed to expose.

### Cluster persistence

cluster persistence

Many clusters appear, peak, and decline; `c2`, `c6`, `c7`, `c8` are the
durable survivors that coexist for the second half of the run. Cluster
turnover is itself a feature: some niches are stable, others displace
each other.

### Chromosome-space scatter (GMM clustering)

Step 0 (post seed-mutation pass):
cluster step 0

Step 5000 (mid-run):
cluster step 5000

Step 10000 (final):
cluster step 10000

The final scatter shows 4 well-separated clusters (silhouette 0.48)
along PC1 (59 % of variance) and PC2 (32 % of variance) — **a
qualitatively cleaner separation than at 5000 steps** (0.38).

### Lineage tree (coloured by `learning_rate`)

lineage tree

DAG (because crossover gives two parents) over 197 unique agents. The
tree extends to depth 4 — at least one great-great-grandchild
reproduced — with a sparse but visible tail of depth-2 and depth-3
lineages.

### Lineage summary

lineage summary

Surviving founder lineages decline monotonically from 30 to 7 — the
extinction curve flattens after step ~5500. Mean lineage depth grows
steadily from 0 to 1.3, and the max depth ratchets up from 2 to 4 as the
run progresses.

## Interpretation

- **Selection compounds with run length.** Founder extinction was 43 % at
600 steps, 70 % at 5000 steps, and 77 % at 10000 steps. The cleanest
signal is the founder-survival curve.
- **Polymorphism is durable, not a transient.** Speciation index sits in
0.40–0.60 for ~9000 steps; GMM-BIC consistently picks k = 4. The
population is genuinely structured into stable sub-populations.
- `**gamma` shows the strongest directional drift.** Mean `gamma` rises
~0.04 across the run — small in absolute terms but consistent with
weak directional selection on top of strong frequency-dependent
pressure.
- **Lineage depth is meaningful.** The DAG reaches depth 4, meaning
multi-generation lineages are surviving and reproducing — not just
founder F1 children. Mean depth 1.29 vs 0.81 at 5000 steps shows
generation turnover is real.
- **The 600/5000/10000-step trio gives a useful runtime/insight curve.**
The 600-step run was barely informative; 5000 was clearly polymorphic;
10000 makes the structure visually clean and provides the deepest
lineage signal.

For longer / heavier-pressure runs, increase `--num-steps` further or
pass `--selection-pressure high` (or a numeric scale) to apply stronger
density-dependent costs. See
`[docs/experiments/intrinsic_evolution.md](../../docs/experiments/intrinsic_evolution.md)`
for the runner reference.

## Artifacts on disk

```
experiments/intrinsic_evolution/
├── analysis/
│   ├── analysis_summary.json
│   ├── analysis_summary.md
│   ├── cluster_lineage_sizes.png
│   ├── gene_distribution_history.png
│   ├── gene_trajectories.png
│   ├── intrinsic_lineage_tree.png
│   ├── lineage_summary.png
│   ├── population_dynamics.png
│   ├── speciation_clusters_step0.png
│   ├── speciation_clusters_step5000.png
│   ├── speciation_clusters_step10000.png
│   └── speciation_index.png
├── cluster_lineage.jsonl
├── intrinsic_evolution_metadata.json
├── intrinsic_gene_snapshots.jsonl
├── intrinsic_gene_trajectory.jsonl
├── run_manifest.json
└── run_summary.json
```