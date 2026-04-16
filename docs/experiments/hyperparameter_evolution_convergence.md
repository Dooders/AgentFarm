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

For issue closure write-ups, include:
- selected mutation/selection settings
- one generated convergence figure
- a short narrative of whether convergence, oscillation, or collapse occurred
