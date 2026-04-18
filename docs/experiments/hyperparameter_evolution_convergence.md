# Hyperparameter Evolution Convergence

This note documents how to capture and interpret learning-rate convergence from the hyperparameter chromosome evolution runner.

## Methodology

- Runner: `scripts/run_evolution_experiment.py`
- Generational metrics source: `evolution_generation_summaries.json`
- Lineage source: `evolution_lineage.json`
- Typical search controls:
  - parent selection: tournament or roulette
  - mutation rate: `--mutation-rate` (default `0.25`)
  - mutation scale: `--mutation-scale` (default `0.2`)
  - boundary mode: `--boundary-mode` (`clamp` or `reflect`, default `clamp`)
  - optional soft boundary penalty:
    - `--boundary-penalty-enabled`
    - `--boundary-penalty-strength` (default `0.01`)
    - `--boundary-penalty-threshold` (default `0.05`)
  - fitness metric: `final_population`, `total_births`, or `final_resources`

Example:

```bash
source venv/bin/activate
python scripts/run_evolution_experiment.py \
  --generations 8 \
  --population-size 10 \
  --steps-per-candidate 80 \
  --selection-method tournament \
  --mutation-rate 0.25 \
  --mutation-scale 0.2 \
  --boundary-mode clamp \
  --fitness-metric final_population \
  --output-dir experiments/evolution_convergence
```

## Visualization

Use the plotting helper to generate a convergence figure from persisted summaries:

```bash
source venv/bin/activate
python scripts/plot_hyperparameter_evolution.py \
  --summary-json experiments/evolution_convergence/evolution_generation_summaries.json \
  --output experiments/evolution_convergence/hyperparameter_evolution.png
```

The chart contains:
- best fitness per generation
- per-gene mean and `+-1 std` trend over generations

## Adaptive Mutation

Static mutation rates have a trade-off between boundary collapse (too
exploitative) and stagnation (too exploratory). The runner supports an
opt-in adaptive mutation schedule driven by
``AdaptiveMutationConfig``.  See ``farm/runners/adaptive_mutation.py`` for
the controller implementation.

When ``--adaptive-mutation`` is passed, each generation's best fitness and
population diversity are observed and used to adjust the mutation rate and
scale that will produce the **next** generation:

- **Fitness adaptation** (``use_fitness_adaptation``): if best fitness did
  not improve over the trailing ``stall_window`` generations by more than
  ``improvement_threshold``, the rate/scale multipliers are grown by
  ``stall_multiplier``.  When fitness improves clearly, the multipliers are
  shrunk by ``improve_multiplier``.
- **Diversity adaptation** (``use_diversity_adaptation``): if the mean
  normalized gene standard deviation falls at or below
  ``diversity_threshold``, the rate/scale multipliers are boosted by
  ``diversity_multiplier`` to escape a collapsed population.
- **Per-gene multipliers** (``per_gene_rate_multipliers`` /
  ``per_gene_scale_multipliers``, also exposed as
  ``--adaptive-per-gene-rate`` / ``--adaptive-per-gene-scale``): constant
  weights applied to specific loci to give individual genes stronger or
  weaker mutation pressure.

Multipliers are always clamped to ``[min_*_multiplier, max_*_multiplier]``,
and the effective rate is clamped to ``[0, 1]``.  When a clamp actually
moves a value, the controller adds a ``rate_clamped`` or ``scale_clamped``
tag to ``adaptive_event`` so saturation is visible in telemetry.

### Telemetry

Every entry in ``evolution_generation_summaries.json`` records the mutation
parameters that **produced this generation's population** (not the ones
that will produce the next), along with the multipliers in force, the
measured diversity of this generation, and a short string describing which
adaptation rules fired:

```json
{
  "generation": 2,
  "best_fitness": 72.0,
  "mutation_rate_used": 0.30,
  "mutation_scale_used": 0.18,
  "mutation_rate_multiplier": 1.5,
  "mutation_scale_multiplier": 1.5,
  "diversity": 0.034,
  "adaptive_event": "stalled+diversity_collapse"
}
```

Because generation 0 is seeded by ``EvolutionExperiment._initialize_population``
(which uses ``mutation_rate=1.0`` to spread seed candidates) rather than by
the adaptive controller, its ``mutation_rate_used`` /
``mutation_scale_used`` / ``mutation_*_multiplier`` fields are ``null`` and
its ``adaptive_event`` is ``"initial_seeding"``.  ``diversity`` is still
recorded for every generation since it describes the evaluated population.

### Example

```bash
source venv/bin/activate
python scripts/run_evolution_experiment.py \
  --generations 12 \
  --population-size 10 \
  --steps-per-candidate 80 \
  --selection-method tournament \
  --mutation-rate 0.2 \
  --mutation-scale 0.15 \
  --adaptive-mutation \
  --adaptive-stall-window 3 \
  --adaptive-stall-multiplier 1.5 \
  --adaptive-improve-multiplier 0.8 \
  --adaptive-improve-threshold 1e-6 \
  --adaptive-diversity-threshold 0.05 \
  --adaptive-diversity-multiplier 1.5 \
  --adaptive-per-gene-rate learning_rate=0.5 \
  --fitness-metric final_population \
  --output-dir experiments/evolution_adaptive
```

### Tuning guidance

- Start with **fitness adaptation only** (``--adaptive-disable-diversity``)
  on short runs to confirm the stall/improve rules fire as expected in your
  fitness regime.
- Keep ``stall_multiplier`` close to ``1.5`` and ``improve_multiplier``
  close to ``0.8``: larger values can cause oscillation between exploit
  and explore regimes.
- ``stall_window`` should be **at least 2** but smaller than the number of
  generations; a value of ``3`` is a good default for 8-20 generation runs.
- Set ``diversity_threshold`` after inspecting the diversity values logged
  by a non-adaptive baseline run.  Typical collapsing populations report
  ``diversity < 0.05`` on normalized gene ranges.
- Use ``per_gene_rate_multipliers`` (or
  ``--adaptive-per-gene-rate learning_rate=0.5``) to mute mutation on genes
  that are known to be sensitive while letting coarser knobs keep exploring.
- Watch for ``rate_clamped`` / ``scale_clamped`` in ``adaptive_event``: if
  these appear repeatedly the multiplier is saturating against
  ``max_*_multiplier``.  Either widen the bound or rebalance
  ``stall_multiplier`` and ``improve_multiplier`` so their geometric mean
  is close to ``1`` (defaults of ``1.5`` and ``0.8`` net-grow over equal
  numbers of stalls and improvements).

## Interpreting Results

When reviewing `learning_rate` convergence:

- tightening standard deviation over generations suggests convergence pressure
- unstable or growing spread suggests mutation pressure dominates selection
- rising best fitness with shrinking spread usually indicates useful convergence

## Artifact Refresh for Multi-Gene Reporting (2026-04-18)

To close the multi-gene acceptance criteria, all checked-in convergence artifacts
under `experiments/evolution_convergence` were regenerated from the current
chromosome schema and runner wiring.

What changed in persisted outputs:

- every `evolution_generation_summaries.json` now includes per-gene stats for
  all loci in the active schema: `learning_rate`, `gamma`, `epsilon_decay`,
  and `memory_size`
- every `best_chromosome` snapshot now includes `gamma` and `epsilon_decay`
  alongside `learning_rate`
- `evolution_lineage.json` remains intentionally compact and still stores
  top-level `learning_rate` + metadata (full per-gene stats live in summaries)

Current final-generation snapshots from regenerated runs:

- `run_clamp_baseline_g6`: final best fitness `76.0`; best chromosome
  (`learning_rate=1e-06`, `gamma=1.0`, `epsilon_decay=0.7241990478892087`)
- `run_clamp_penalty_g6`: final best fitness `66.99`; best chromosome
  (`learning_rate=0.20027491749951848`, `gamma=1.0`, `epsilon_decay=0.7592339511733013`)
- `run_clamp_penalty002_g6`: final best fitness `76.98`; best chromosome
  (`learning_rate=1e-06`, `gamma=0.95`, `epsilon_decay=0.9320406841297989`)
- `run_clamp_penalty005_g6`: final best fitness `68.95`; best chromosome
  (`learning_rate=1e-06`, `gamma=0.8369660526170754`, `epsilon_decay=0.8288524477063025`)
- `run_clamp_penalty010_g6`: final best fitness `72.9`; best chromosome
  (`learning_rate=1e-06`, `gamma=0.8618623974764232`, `epsilon_decay=0.5743829534957358`)
- `run_reflect_g6`: final best fitness `78.0`; best chromosome
  (`learning_rate=0.19271257710715683`, `gamma=0.9751036551406521`, `epsilon_decay=0.8970057808457064`)
- `run_roulette_mut040_g6`: final best fitness `76.0`; best chromosome
  (`learning_rate=1e-06`, `gamma=1.0`, `epsilon_decay=1.0`)
- `run_tournament_mut020_g6`: final best fitness `71.0`; best chromosome
  (`learning_rate=0.03602658902654344`, `gamma=0.9751036551406521`, `epsilon_decay=0.9688639803415231`)
- `run_tournament_mut025`: final best fitness `71.0`; best chromosome
  (`learning_rate=0.2269703882742677`, `gamma=1.0`, `epsilon_decay=1.0`)

## Findings From Current Smoke Run

Using the checked-in artifacts in `experiments/evolution_smoke`:

- Generations evaluated: `2` (`generation=0` and `generation=1`)
- Candidates evaluated: `8` total (`4` per generation)
- Fitness metadata present in lineage: `final_population`
- Fitness behavior: flat (`min=mean=max=6.0` in both generations)
- Learning-rate spread (computed from `evolution_lineage.json`):
  - generation 0: mean `0.0013067191`, std `0.0017334593`, min `1e-06`, max `0.0042248765`
  - generation 1: mean `0.00025075`, std `0.0004325797`, min `1e-06`, max `0.001`

Interpretation:

- The run shows **learning-rate contraction toward smaller values** across one generation.
- Because fitness is completely flat, there is **no meaningful selection gradient** in this run; observed contraction is likely driven by initialization/mutation + elitism dynamics rather than clear fitness improvement.
- This is acceptable as a smoke validation of the pipeline (encoding, mutation/crossover, lineage persistence), but it is **not yet evidence of optimization convergence**.

## Closure Run Comparison (Completed)

Two multi-generation runs were executed and persisted under `experiments/evolution_convergence`:

1. `run_tournament_mut020_g6`
   - selection: `tournament`
   - mutation: rate `0.20`, scale `0.2`
   - seed: `42`
   - settings: `--generations 6 --population-size 8 --steps-per-candidate 40`
2. `run_roulette_mut040_g6`
   - selection: `roulette`
   - mutation: rate `0.40`, scale `0.35`
   - seed: `99`
   - settings: `--generations 6 --population-size 8 --steps-per-candidate 40`

Reproducibility manifests were saved at:

- `experiments/evolution_convergence/run_tournament_mut020_g6/run_manifest.json`
- `experiments/evolution_convergence/run_roulette_mut040_g6/run_manifest.json`

Generated convergence figures:

- `experiments/evolution_convergence/run_tournament_mut020_g6/hyperparameter_evolution.png`
- `experiments/evolution_convergence/run_roulette_mut040_g6/hyperparameter_evolution.png`

Observed outcomes from persisted summaries:

- Tournament (`run_tournament_mut020_g6`)
  - best fitness: `68.0 -> 72.0`
  - learning-rate mean: `0.0557 -> 0.0519`
  - learning-rate std: `0.0612 -> 0.1372`
  - best-candidate learning rate: `0.001 -> 1e-06`
  - narrative: **partial optimization with boundary collapse** (fitness improved, but the winning learning rate moved to the lower bound and spread increased).
- Roulette (`run_roulette_mut040_g6`)
  - best fitness: `72.0 -> 72.0` (flat)
  - learning-rate mean: `0.1449 -> 0.2654`
  - learning-rate std: `0.1443 -> 0.2526`
  - best-candidate learning rate: `0.2100 -> 0.6895`
  - narrative: **oscillation / mutation-dominated regime** (no fitness gain, increasing spread and drift toward larger rates).

Overall interpretation:

- The pipeline is working end-to-end and can change population-level behavior under different evolutionary settings.
- Lower mutation pressure with tournament selection produced better optimization signal (fitness improvement), but also collapsed the winning `learning_rate` to the lower bound, which indicates an over-strong attractor at the boundary.
- Higher mutation pressure with roulette selection maintained diversity but did not improve fitness, consistent with exploration overpowering selection.
- Across both runs, `learning_rate` appears to be a sensitive but noisy control variable for `final_population`; current settings show trade-offs between exploitation (collapse risk) and exploration (stagnation risk).
- Practical next tuning step: keep tournament selection, reduce mutation pressure further, and add a soft lower-bound guard or penalty so improvement does not depend on boundary collapse.

Commands used:

```bash
source venv/bin/activate
python scripts/run_evolution_experiment.py \
  --generations 6 --population-size 8 --steps-per-candidate 40 \
  --selection-method tournament --mutation-rate 0.20 --mutation-scale 0.2 \
  --fitness-metric final_population --seed 42 \
  --output-dir experiments/evolution_convergence/run_tournament_mut020_g6

python scripts/run_evolution_experiment.py \
  --generations 6 --population-size 8 --steps-per-candidate 40 \
  --selection-method roulette --mutation-rate 0.40 --mutation-scale 0.35 \
  --fitness-metric final_population --seed 99 \
  --output-dir experiments/evolution_convergence/run_roulette_mut040_g6
```

For issue closure write-ups, include:
- selected mutation/selection settings
- one generated convergence figure
- a short narrative of whether convergence, oscillation, or collapse occurred

## Boundary-Handling Comparison Plan

To specifically evaluate boundary-collapse risk, keep all settings identical and
toggle only boundary handling. Suggested A/B matrix:

1. clamp baseline
2. reflect mutation
3. clamp + soft boundary penalty

Use a shared seed and identical generation/population settings:

```bash
source venv/bin/activate

# A) Clamp baseline
python scripts/run_evolution_experiment.py \
  --generations 6 --population-size 8 --steps-per-candidate 40 \
  --selection-method tournament --mutation-rate 0.20 --mutation-scale 0.2 \
  --boundary-mode clamp \
  --fitness-metric final_population --seed 42 \
  --output-dir experiments/evolution_convergence/run_clamp_baseline_g6

# B) Reflective mutation
python scripts/run_evolution_experiment.py \
  --generations 6 --population-size 8 --steps-per-candidate 40 \
  --selection-method tournament --mutation-rate 0.20 --mutation-scale 0.2 \
  --boundary-mode reflect \
  --fitness-metric final_population --seed 42 \
  --output-dir experiments/evolution_convergence/run_reflect_g6

# C) Clamp + soft boundary penalty
python scripts/run_evolution_experiment.py \
  --generations 6 --population-size 8 --steps-per-candidate 40 \
  --selection-method tournament --mutation-rate 0.20 --mutation-scale 0.2 \
  --boundary-mode clamp \
  --boundary-penalty-enabled \
  --boundary-penalty-strength 0.01 \
  --boundary-penalty-threshold 0.05 \
  --fitness-metric final_population --seed 42 \
  --output-dir experiments/evolution_convergence/run_clamp_penalty_g6
```

Compare each run's `evolution_generation_summaries.json` and lineage outputs for:

- best-candidate learning-rate trajectory (especially boundary hits)
- learning-rate min/max and standard deviation trends
- fitness gains relative to boundary occupancy

## Boundary-Handling Comparison Results (Completed)

Executed on 2026-04-18 with the exact commands above and shared seed (`42`).
Outputs were written to:

- `experiments/evolution_convergence/run_clamp_baseline_g6`
- `experiments/evolution_convergence/run_reflect_g6`
- `experiments/evolution_convergence/run_clamp_penalty_g6`

Observed outcomes from persisted summaries + lineage:

- Clamp baseline (`run_clamp_baseline_g6`)
  - best fitness: `72.0 -> 69.0`
  - best-candidate learning rate: `0.001 -> 1e-06`
  - learning-rate std: `0.0612 -> 0.1372`
  - exact min-boundary occupancy by generation: `[2, 3, 3, 5, 8, 7]` (out of 8 candidates)
- Reflect (`run_reflect_g6`)
  - best fitness: `74.0 -> 75.0`
  - best-candidate learning rate: `0.3771 -> 0.3771`
  - learning-rate std: `0.1143 -> 0.1638`
  - exact min-boundary occupancy by generation: `[0, 0, 0, 0, 0, 0]`
- Clamp + penalty (`run_clamp_penalty_g6`)
  - adjusted best fitness: `75.99 -> 73.99`
  - raw best fitness (metadata): `76.0` (max over lineage)
  - best-candidate learning rate: `1e-06 -> 1e-06`
  - exact min-boundary occupancy by generation: `[2, 4, 5, 6, 8, 7]`
  - mean boundary penalty across candidates: `0.00764` (max `0.01`)

Interpretation:

- Reflective mutation clearly reduced boundary-collapse risk in this comparison:
  no candidates landed exactly at the lower bound, while clamp variants showed
  repeated and increasing lower-bound occupancy.
- Reflect also preserved/improved optimization signal (best fitness reached
  `75.0`) without relying on boundary-hugging winners.
- The small soft-penalty setting (`0.01`, threshold `0.05`) was not strong
  enough to dislodge clamp dynamics in this setup; it reduced adjusted fitness
  but did not prevent lower-bound collapse. Increasing penalty strength and/or
  threshold is the next tuning step if clamp must be retained.

## Penalty Strength Sensitivity (Completed)

A follow-up sweep kept clamp mode fixed and varied only
`boundary_penalty_strength` (`threshold=0.05`, same seed/settings):

- `experiments/evolution_convergence/run_clamp_penalty002_g6`
- `experiments/evolution_convergence/run_clamp_penalty005_g6`
- `experiments/evolution_convergence/run_clamp_penalty010_g6`

Observed outcomes:

- `strength=0.02` (`run_clamp_penalty002_g6`)
  - adjusted best fitness: `70.0 -> 79.0`
  - best-candidate learning rate: `0.1594 -> 0.1594`
  - min-boundary hits by generation: `[2, 3, 2, 2, 1, 0]`
  - mean penalty: `0.00530` (max `0.02`)
- `strength=0.05` (`run_clamp_penalty005_g6`)
  - adjusted best fitness: `68.951 -> 67.95`
  - best-candidate learning rate: `0.001 -> 1e-06`
  - min-boundary hits by generation: `[2, 3, 0, 1, 7, 7]`
  - mean penalty: `0.03262` (max `0.05`)
- `strength=0.10` (`run_clamp_penalty010_g6`)
  - adjusted best fitness: `74.902 -> 73.0`
  - best-candidate learning rate: `0.001 -> 0.1594`
  - min-boundary hits by generation: `[2, 3, 2, 4, 2, 2]`
  - mean penalty: `0.04303` (max `0.10`)

Sensitivity takeaway:

- Penalty impact is non-monotonic in this stochastic setting.
- `0.02` and `0.10` reduced lower-bound collapse relative to prior clamp runs,
  while `0.05` still collapsed late.
- A higher penalty (`0.10`) prevented the final winner from collapsing to
  `1e-06`, but reflective mutation remains the most consistent anti-collapse
  strategy in this set of experiments.
