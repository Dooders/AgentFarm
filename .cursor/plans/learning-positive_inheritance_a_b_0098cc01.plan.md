---
name: Learning-positive inheritance A/B
overview: "Run the #904 P0–P4 inheritance-ladder A/B in the learning-positive regime validated by the precondition gate (small population, 3000-step horizon), by extending the existing A/B harness with population knobs and grading on early-age net RL reward."
todos:
  - id: population-knobs
    content: Add --population/--max-population to run_inheritance_mode_ab.py and thread through the sweep runner via a shared regime-config helper
    status: completed
  - id: multi-arm-early-life
    content: Extend analyze_early_life_fitness.py to grade multiple treatment arms against the P0 baseline
    status: completed
  - id: tests
    content: Add unit tests for population plumbing and multi-arm early-life aggregation
    status: completed
  - id: pilot
    content: "Run pilot: balanced profile x 2 seeds x 5 arms at 3000 steps; verify telemetry, offspring counts, wall time"
    status: completed
  - id: full-sweep
    content: Run full 90-run sweep (5 arms x 3 profiles x 6 seeds) with resume + disk DB
    status: pending
  - id: aggregate
    content: "Aggregate: compare_inheritance_arms + early-life verdict; apply decision rule"
    status: pending
  - id: record
    content: "Write results summary doc and update issue #904"
    status: pending
isProject: false
---

# Learning-positive inheritance-ladder A/B (#904)

## Goal

Run the matched P0–P4 experiment in the regime the precondition gate validated — not the default config where within-life learning is ~null. The gate established the honest effect size (~+15–30 net early reward), so the A/B is graded on **early-age net RL reward at ages {10, 25, 50}** with the project robustness gate (paired per-seed 95% CI excludes zero AND sign agreement >= 0.75).

## Regime (decided)

- **Arms (5):** `baldwinian` (P0 baseline), `lamarckian` (P1), `p2`, `p3`, `p4` — one matched sweep, default warm-start knobs (damping 0.5, replay limit 256, blend 0.5, gate 1.0)
- **Population:** 8 independent agents only (0 system/control), `max_population=32`. Unlike the gate, **reproduction stays ON** (inheritance happens at reproduction), so the cap prevents the colony explosion the gate avoided by blocking `reproduce`
- **Horizon:** `--num-steps 3000`, `--warmup-steps 200` (unchanged; offspring for early-life scoring are born after warmup), `--snapshot-interval 100`
- **Matrix:** 3 profiles (`conservative`/`balanced`/`buffered`) x 6 seeds (`42 7 19 101 137 256`) x 5 arms = **90 runs**, `--disk-database` (required by early-life analysis) + `--resume`, `PYTHONHASHSEED=0`

## Gap 1: harness has no population knobs

[scripts/run_inheritance_mode_ab.py](scripts/run_inheritance_mode_ab.py) exposes `--num-steps/--profiles/--seeds` but population comes from the environment YAML (development = 30 mixed agents, cap 50). Add `--population` and `--max-population` flags, threaded through `run_stable_profile_seed_sweep._execute_sweep` into the `SimulationConfig`, mirroring `build_regime_config()` in [scripts/measure_transferable_signal.py](scripts/measure_transferable_signal.py) (independent-only, zero out system/control/order/chaos). Extract that config shaping into one shared helper so the gate and the A/B use the identical regime definition (DRY).

## Gap 2: early-life analysis is 2-arm only

[scripts/analyze_early_life_fitness.py](scripts/analyze_early_life_fitness.py) computes exactly the graded metrics (`rl_reward_at_age` at 10/25/50, survival, decision success) but compares one baseline vs one treatment. Extend it to accept multiple treatment arms (P1–P4 each paired against P0), reusing its existing per-seed pairing and `_verdict` logic. Output one summary table: arm x profile x metric with robust flags.

## Execution phases

**Phase A — pilot (before burning ~90 long runs):** 1 profile (`balanced`) x 2 seeds x all 5 arms at full 3000 steps. Verify: (1) independent-only population works with the intrinsic-evolution runner (speciation/selection machinery), (2) warm-start telemetry shows `warmstart_rate > 0` for P1–P4 and a sensible P4 `gate_hit_rate`, (3) enough post-warmup offspring exist for early-life scoring, (4) per-run wall time is acceptable (pin torch threads like the gate runner if needed).

**Phase B — full sweep:**

```bash
PYTHONHASHSEED=0 python scripts/run_inheritance_mode_ab.py \
  --arms baldwinian lamarckian p2 p3 p4 \
  --population 8 --max-population 32 \
  --num-steps 3000 --snapshot-interval 100 \
  --output-dir experiments/inheritance_ab_learning_positive \
  --disk-database --resume
```

**Phase C — aggregate and grade:**
1. `compare_inheritance_arms.py` (P0 baseline, P1–P4 treatments) — population/speciation/stability/mechanism readouts
2. Extended `analyze_early_life_fitness.py` — the **primary verdict**: net RL reward at ages {10, 25, 50} per arm, sized against the ~+15–30 budget the gate measured
3. Apply the decision rule: keep a richer payload only if it robustly beats P0 on early-life net-RL-reward without degrading stability

**Phase D — record:** results summary under `docs/research/experiments/intrinsic_evolution/`, check off the checklist item on #904 with a comment. (Cross-ecology transfer test is the *next* checklist item — out of scope here, but the harvested runs should be kept for it.)

## Tests

Unit tests for the new population-override plumbing and the multi-arm early-life aggregation (extend `tests/scripts/`), following the existing `test_measure_transferable_signal.py` pattern.