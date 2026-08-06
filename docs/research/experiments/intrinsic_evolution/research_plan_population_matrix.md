# Research Plan: Is intrinsic evolution non-uniform, and does larger population size sharpen the signal?

This plan turns a feasibility question — *would a long intrinsic-evolution run
be valuable, or return a mostly uniform result, and would a larger grid with
more agents help?* — into a replicated, evidence-gated experiment. It reuses the
existing runner
([`IntrinsicEvolutionExperiment`](../../../../farm/runners/intrinsic_evolution_experiment.py))
and analyzers rather than adding new simulation logic.

The orchestrator is
[`scripts/run_intrinsic_evolution_matrix.py`](../../../../scripts/run_intrinsic_evolution_matrix.py),
which fans the matrix across CPU cores by calling
[`scripts/run_stable_profile_seed_sweep.py`](../../../../scripts/run_stable_profile_seed_sweep.py)
once per (cell, seed).

## 1. Motivation

The recorded 10,000-step run ([RESULTS.md](RESULTS.md)) already shows durable
polymorphism (a sustained speciation index of 0.4–0.6, four coexisting niches,
77% founder-lineage extinction). So the system is not obviously uniform. But the
project's own methods note
([seed-sweep reality check](../../devlog/2026-05-12-seed-sweep-reality-check.md))
established that single-seed, per-gene claims do not replicate, because
within-seed genetic drift variance exceeds the between-condition differences.

That reframes the question. The scientific target is **not** "run one simulation
longer." It is "raise the effective population size (`Ne`) so drift stops
dominating, and replicate across enough seeds to pass the evidence gate." A
single 30-day run is `n = 1`; the same compute buys a whole replicated factorial.

## 2. Hypotheses

Directions and the null are pre-registered so the analysis is confirmatory, not
a search for a story.

- **H1 — non-uniformity.** With initial-diversity seeding on and selection
  pressure at least `low`, per-gene span-normalized diversity at the end stays
  near its ~0.28 starting value and does not collapse toward the drift-only
  floor (~0.02). The `pressure = none` arm is the drift-only null.
- **H2 — `Ne` sharpens the signal.** Moving from the `simulation` profile
  (~300 cap) to the `research` profile (~500 cap) narrows the across-seed 95% CIs
  on per-gene shifts and raises within-condition sign agreement, without flipping
  the direction of robust shifts.
- **H3 — selection shapes structure.** Higher pressure raises founder-lineage
  extinction and effective-selection-strength telemetry, and drives no single
  gene to a monoculture — consistent with the frequency-dependent reading in
  [selection pressure and intrinsic goals](../../devlog/2026-07-29-selection-pressure-and-intrinsic-goals.md).
- **H4 — gene flow.** Crossover-on changes speciation-index variance and/or mode
  entropy relative to mutation-only (unresolved for the buffered profile;
  conservative showed a real reduction in
  [gene flow and the buffer](../../devlog/2026-05-18-gene-flow-and-the-buffer.md)).

Primary claims are restricted to the loci that moved robustly before
(`gamma`, `learning_rate`, `attack_weight`, `share_weight`, `attack_mult_stable`,
`per_alpha`). The remaining ~25 genes are treated as near-neutral secondary
readouts, since prior work shows most gene variance is behaviorally near-neutral.

## 3. Design

A replicated factorial over three axes.

| Axis | Levels | Rationale |
| --- | --- | --- |
| Selection pressure | `none`, `low`, `high` | `none` is the drift-only null for H1/H3 |
| Gene flow | `mutation`, `crossover` | Tests H4 |
| Population / `Ne` | `sim` (100×100, cap 300), `research` (150×150, cap 500) | Tests H2 — the core lever |

That is 3 × 2 × 2 = 12 cells × **8 seeds** = **96 runs** at 10,000 logged steps
after a 200-step warmup. Eight seeds clears the project's ≥5-seed gate and
tightens the CIs.

Population is set by the environment profile plus `max_population`, **not** by
resource supply. A VM probe confirmed a hard ceiling of 50 agents in the
`development` environment regardless of a 6× resource boost
([`development.yaml`](../../../../farm/config/environments/development.yaml)). A
bigger grid alone spreads agents thinner and *increases* drift, so the
population axis must move the environment profile (grid + starting agents +
`max_population`) together.

### Phases

- **Phase 1 — calibration (≈4–6 runs).** Pilot one `sim` and one `research`
  cell to (1) re-measure seconds/step on the chosen VM, (2) confirm no arm goes
  extinct before ~2,000 steps at `high` pressure, and (3) confirm the population
  stabilizes above ~150 (i.e., `max_population`, not resource scarcity, is the
  effective ceiling). Gate: proceed only if both hold; otherwise scale resource
  regeneration up and re-pilot.
- **Phase 2 — confirmatory matrix (96 runs).** The full factorial above.
- **Phase 3 — depth (≈6 runs, exploratory).** Extend the most polymorphic
  Phase-2 cell to 50,000 steps for 6 seeds to test whether niches keep turning
  over (interesting) or eventually fix (uniform-at-last). This is the only place
  long horizons are justified.

## 4. Compute budget

Planning constant from the VM probe and the
[GCP spot-VM guide](../../../guides/gcp-spot-vm-sweep.md): ~0.02 s per alive
agent per step, roughly linear in population.

| Population | Steps | Core-hours / run |
| --- | ---: | ---: |
| ~300 (`sim`) | 10,000 | ~17 |
| ~500 (`research`) | 10,000 | ~28 |
| ~300 (depth) | 50,000 | ~83 |

Phase 2 ≈ (48 × 17) + (48 × 28) ≈ 2,160 core-hours. Phase 3 ≈ 500 core-hours.
With a pilot and ~20% Spot-preemption margin, the total is **~3,300 core-hours**.
A single 8-vCPU Spot VM over 30 days provides 30 × 24 × 8 ≈ 5,760 core-hours, so
the plan fits with ~40% headroom (a larger VM finishes proportionally sooner).
Runs are single-threaded with BLAS pinned to one thread, so parallelism is one
process per vCPU. `--dry-run` prints an upper-bound estimate; re-measure with the
Phase-1 pilot before trusting the total for scheduling.

## 5. Execution

Run the whole matrix with the orchestrator; each (cell, seed) is an independent,
resumable subprocess:

```bash
source venv/bin/activate
python scripts/run_intrinsic_evolution_matrix.py \
    --populations sim research \
    --pressures none low high \
    --gene-flow mutation crossover \
    --seeds 42 7 19 101 137 256 512 999 \
    --num-steps 10000 --warmup-steps 200 --snapshot-interval 200 \
    --jobs 8 --disk-database --resume \
    --output-dir experiments/intrinsic_matrix
```

Operational rules (from the GCP guide): run detached so it survives SSH drops;
`PYTHONHASHSEED=0` and BLAS thread-pinning are set for every child automatically;
disk-backed SQLite for the long runs; `--resume` so a Spot preemption redoes at
most the interrupted runs; delete the VM at teardown. Preview first with
`--dry-run`.

## 6. Analysis and evidence gates

- Per-run plots and lineage/speciation artifacts:
  [`scripts/analyze_intrinsic_evolution.py`](../../../../scripts/analyze_intrinsic_evolution.py).
- Per-cell aggregation (mean, variance, 95% Student-t CI, speciation slope and
  direction, sign agreement):
  [`scripts/analyze_stable_profile_seed_sweep.py`](../../../../scripts/analyze_stable_profile_seed_sweep.py),
  once per cell directory.
- Cross-condition mode/transition classification with conservative gates:
  [`scripts/analyze_transition_regime.py`](../../../../scripts/analyze_transition_regime.py).

A directional claim counts only if its 95% CI excludes zero **and**
within-condition sign agreement is ≥ 0.75 across the 8 seeds — the project's
existing gate. Between-condition claims (e.g. "`Ne` sharpens the signal") require
the CI on the *difference* to exclude zero, not merely each arm on its own. H1's
"not uniform" is operationalized as: per-gene span-normalized end diversity stays
≥ ~0.20 in the selection arms while the `none` arm trends toward the ~0.02 drift
floor.

## 7. Deliverables and success criteria

- `experiments/intrinsic_matrix/` with per-cell run artifacts, `matrix_manifest.json`,
  aggregated `seed_sweep_summary.{json,md}` per cell, and comparison plots.
- A devlog-style writeup (matching `docs/research/devlog/`) answering the
  headline question with CIs and the robustness gate applied.
- Success = a defensible, replicated statement of the form: "under pressure `P`
  and population `Ne`, the system sustains `k` niches with speciation index in
  `[a, b]`, founder survival `f`, and robust gene shifts in {…}; the `none` arm
  collapses to drift" — or a documented null if the gates fail.

## 8. Risk register

| Risk | Mitigation |
| --- | --- |
| Early extinction at `high` pressure × sparse resources | Phase-1 gate; scale resource regeneration with grid before Phase 2 |
| Spot preemption mid-run | `--resume` + disk DB; at most the interrupted runs are redone |
| Near-neutral genes limit richness | Pre-register primary claims on the ~6 loci that moved before; rest are secondary |
| DB disk growth over long/depth runs | Snapshot interval 200; disk DB with periodic pruning |
| Reproducibility | Fixed seed list, `PYTHONHASHSEED=0`, per-cell `run_manifest.json` + `matrix_manifest.json` |

## Related docs

- [Intrinsic evolution experiment doc](intrinsic_evolution.md)
- [Results (10,000 steps)](RESULTS.md)
- [Transition-regime workflow](transition_regime.md)
- [Seed-sweep reality check (methods)](../../devlog/2026-05-12-seed-sweep-reality-check.md)
- [GCP spot-VM sweep runbook](../../../guides/gcp-spot-vm-sweep.md)
