---
layout: page
title: "Baldwinian vs Lamarckian A/B harness landed (with first smoke run)"
---

Issue [#849](https://github.com/Dooders/AgentFarm/issues/849) asked for an explicit,
matched Baldwinian-vs-Lamarckian experiment path so we can quantify when
offspring warm-starting helps and when it destabilizes runs.

This post is the implementation log plus a first smoke-run snapshot. The full
36-run matrix is intentionally deferred.

## What shipped

The core inheritance toggle is now implemented in the intrinsic evolution path:

- `IntrinsicEvolutionPolicy` now supports `inheritance_mode` with:
  - `baldwinian` (default): child gets fresh decision policy
  - `lamarckian`: child attempts policy warm-start from parent
- Reproduction now applies warm-start only when `inheritance_mode=lamarckian`.
- Warm-start telemetry is persisted in metadata:
  - `policy_inheritance_metrics.lamarckian_warmstart_applied`
  - `policy_inheritance_metrics.lamarckian_warmstart_skipped`

New experiment scripts:

- `scripts/run_inheritance_mode_ab.py` (arm orchestrator)
- `scripts/compare_inheritance_arms.py` (paired treatment-vs-baseline analyzer)

Runner wiring updates:

- `scripts/run_stable_profile_seed_sweep.py` now accepts
  `--inheritance-mode {baldwinian,lamarckian}` and records it in manifests.

Protocol doc:

- `docs/experiments/intrinsic_evolution/inheritance_mode_ab.md`

## First smoke run (initial result, not a conclusion)

I ran a minimal A/B smoke to verify behavior and data plumbing:

- profiles: `balanced`
- seeds: `42`
- steps: `200` (with warmup `200`)
- total runs: `2` (one per arm)
- output dir: `experiments/inheritance_ab_smoke/`

Command:

```bash
PYTHONHASHSEED=0 python scripts/run_inheritance_mode_ab.py \
  --profiles balanced --seeds 42 --num-steps 200 \
  --output-dir experiments/inheritance_ab_smoke \
  --disk-database --resume
```

Paired compare command:

```bash
python scripts/compare_inheritance_arms.py \
  --baseline-dir experiments/inheritance_ab_smoke/baldwinian \
  --treatment-dir experiments/inheritance_ab_smoke/lamarckian \
  --arm-labels lamarckian \
  --output-dir experiments/inheritance_ab_smoke/aggregate
```

### Observed smoke outputs

- Both runs completed successfully (`1/1` per arm).
- Runtime:
  - Baldwinian arm: `341.965s`
  - Lamarckian arm: `362.731s`
- Lamarckian inheritance telemetry (seed 42, balanced):
  - `applied = 182`
  - `skipped = 39`
  - warm-start attempt success among recorded outcomes: `~82.4%`
- Final population at step 200:
  - Baldwinian: `101`
  - Lamarckian: `101`
- Startup transient metrics at this smoke scale were identical in this single
  seed run (`peak_birth_rate=0.054`, `peak_death_rate=0.0`,
  `oscillation_amplitude=41`).

## How to read this

This smoke run is only a harness check:

- It confirms Lamarckian wiring executes in live reproduction and records
  non-zero warm-start counts.
- It does **not** support any performance/stability claim yet.
- Single-seed, single-profile parity is expected to be noisy and underpowered.

## Next step

Run the full matched matrix described in the experiment doc:

- 2 arms × 3 profiles × 6 seeds = 36 runs
- then use `scripts/compare_inheritance_arms.py` on the full outputs.

That full run is where we decide whether warm-starting offspring is worth the
stability trade-off by regime.
