# Baldwinian vs Lamarckian A/B (Issue #849)

This protocol runs matched intrinsic-evolution sweeps under two inheritance
modes to quantify when Lamarckian warm-starting is worth potential stability
costs.

Issue: [#849](https://github.com/Dooders/AgentFarm/issues/849)

## Arms

- `baldwinian`: offspring inherit only the hyperparameter chromosome and start
  with a fresh policy.
- `lamarckian`: offspring still inherit chromosome dynamics, and additionally
  attempt compatible policy warm-start from the parent.

## Matrix

- Profiles: `conservative`, `balanced`, `buffered`
- Seeds: `42 7 19 101 137 256`
- Total: `2 × 3 × 6 = 36` runs

## Held fixed

- `num_steps=1000`, `warmup_steps=200`, `snapshot_interval=50`
- Mutation: gaussian, `mutation_rate=0.15`, `mutation_scale=0.10`,
  boundary mode reflect
- Crossover disabled (to isolate inheritance mode)
- `selection_pressure=low`
- Initial diversity: independent mutation (`rate=1.0`, `scale=0.25`)
- Speciation: GMM, `max_k=4`

## Run commands

**Note:** Results under `experiments/inheritance_ab_pre_fix/` were collected
before a decision-path fix (2026-05-22) that prevented policy weights from
influencing actions. That aggregate is invalid; use a fresh output directory
after the fix lands.

### 1) Run both arms

```bash
PYTHONHASHSEED=0 python scripts/run_inheritance_mode_ab.py \
  --output-dir experiments/inheritance_ab \
  --disk-database \
  --resume
```

### 2) Compare paired deltas (treatment vs Baldwinian baseline)

```bash
python scripts/compare_inheritance_arms.py \
  --baseline-dir experiments/inheritance_ab/baldwinian \
  --baseline-label baldwinian \
  --treatment-dir experiments/inheritance_ab/lamarckian \
  --arm-labels lamarckian \
  --output-dir experiments/inheritance_ab/aggregate
```

## Outputs

- `experiments/inheritance_ab/inheritance_ab_manifest.json`
- Per-arm `sweep_manifest.json` and stable-profile analysis artifacts
- `experiments/inheritance_ab/aggregate/inheritance_ab_summary.json`
- `experiments/inheritance_ab/aggregate/inheritance_ab_summary.md`
- `experiments/inheritance_ab/aggregate/paired_delta_heatmap.png`
- `experiments/inheritance_ab/aggregate/speciation_trajectories_with_arms.png`
- `experiments/inheritance_ab/aggregate/startup_transient_comparison.png`

## Acceptance criteria

Use paired-seed deltas (`treatment - baseline`) with:

- 95% CI excluding zero, and
- sign agreement >= 75%

Primary readouts:

- Performance: `population_mean`, `population_final`
- Stability: `startup_transient.peak_death_rate`,
  `startup_transient.oscillation_amplitude`, `lineage.churn_rate`
- Diversity impact: `speciation_final`, `speciation_slope`
- Mechanism coverage: `lamarckian_warmstart_rate`

Per-profile recommendation categories (label-aware):

- `net recommend <treatment-label>` (e.g. `net recommend lamarckian`)
- `net recommend <baseline-label>` (e.g. `net recommend baldwinian`)
- `performance win + stability loss`
- `speciation collapse risk`
- `no robust effect`

The classifier consults the metric groups documented above:

- Performance: `population_mean`, `population_final`
- Stability loss: `startup_transient.peak_death_rate`,
  `startup_transient.oscillation_amplitude`, `lineage.churn_rate`
- Speciation collapse: robust negative delta on `speciation_slope`
