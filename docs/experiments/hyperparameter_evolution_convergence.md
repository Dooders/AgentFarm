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

## Interpreting Results

When reviewing `learning_rate` convergence:

- tightening standard deviation over generations suggests convergence pressure
- unstable or growing spread suggests mutation pressure dominates selection
- rising best fitness with shrinking spread usually indicates useful convergence

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
