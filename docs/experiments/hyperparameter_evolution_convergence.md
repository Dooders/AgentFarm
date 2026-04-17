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

Recommended next closure run (to strengthen issue evidence):

- Increase signal: more generations and longer candidate evaluations (for example `--generations 8 --population-size 10 --steps-per-candidate 80`).
- Compare at least two mutation settings and selection methods.
- Attach one generated plot and summarize whether behavior is convergence, oscillation, or collapse.
- Persist a small run manifest (CLI args + seed) with artifacts so findings are fully reproducible.

For issue closure write-ups, include:
- selected mutation/selection settings
- one generated convergence figure
- a short narrative of whether convergence, oscillation, or collapse occurred
