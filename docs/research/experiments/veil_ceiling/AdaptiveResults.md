# Adaptive monitor — results

**Design:** [Adaptive.md](Adaptive.md) · **Full report:**
[`experiments/veil_ceiling/adaptive_results/ADAPTIVE_REPORT.md`](../../../../experiments/veil_ceiling/adaptive_results/ADAPTIVE_REPORT.md)
· **Raw record:** [`experiments/veil_ceiling/adaptive_results/`](../../../../experiments/veil_ceiling/adaptive_results)
· **Parent:** [RESULTS.md](RESULTS.md)

- 840 runs: 14 conditions (C1, C3_f0.7, C3_f0.9, C2, C4, and three policies
  at `f ∈ {0.7, 0.9, 1.0}`) × 2 inheritance modes × 30 seed-matched runs.
  Coverage `c = 0.5` and penalty `p = 6` are unchanged from the confirmatory
  design. No extinctions.
- Simulation code commit `00fdcdad` (recorded in `manifest.json`). Blind
  chases occupancy (agent-ticks on the current true map); cells chase caught
  defections; move chases observed movement. Laplace smoothing is 1.0.
- All intervals are 95 % percentile bootstraps over seeds (2000 replicates).
- Exploratory relative to the original pre-registration; HA1–HA5 were stated
  before this run in [Adaptive.md](Adaptive.md).

## Verdicts

| Hypothesis | Baldwinian | Lamarckian | Note |
|---|---|---|---|
| HA1 cells reduces Δ_cue vs static C2 | inconclusive | inconclusive | Δ diffs +0.003 [−0.002, 0.009] and −0.006 [−0.025, 0.012] |
| HA2 cells raises realised enforcement | **supported** | inconclusive | +0.012 [0.003, 0.022] Baldwinian; +0.001 [−0.005, 0.007] Lamarckian |
| HA3 move reduces Δ more than cells | **falsified** | **falsified** | move does not beat cells or static on Δ_cue |
| HA4 cells beats blind (not crowding) | inconclusive | inconclusive | cells − blind Δ CI crosses 0; occupancy is the stronger reallocater |
| HA5 C1-calibrated evaluator recovers | inconclusive | **falsified** | AUC 0.663 vs C2 0.664 Baldwinian; 0.485 [0.466, 0.503] Lamarckian |

The pre-registered-style claims about *chasing caught defections* do not hold
in the Lamarckian arm that produced concealment. A C1-calibrated evaluator
stays at chance. The interesting result is in the control.

## What the data show

### Caught-defection chasing barely reallocates

`adaptive_cells` at `f = 1` has mean weight KL 0.019 (Baldwinian) and 0.011
(Lamarckian) and mask overlap 0.51 — almost the uniform redraw of static C2
(KL 0, overlap 0.50). Caught defections per epoch are sparse relative to
Laplace smoothing over 320 training cells, so the next map is nearly uniform.
Δ_cue stays inside the static C2 interval (Baldwinian 0.082 vs 0.079;
Lamarckian 0.353 vs 0.359). Realised enforcement in Lamarckian C2 was 0.141;
cells leaves it at 0.142.

That is why HA1 is inconclusive and HA5 is falsified in the Lamarckian arm:
the policy that was supposed to catch concealment does not move the monitor
enough to change the behaviour the evaluator sees.

### Occupancy concentrates the map and raises enforcement

`adaptive_blind` (the busy-cell control) has weight KL 0.73 / 0.67 and overlap
0.58. Occupancy is counted every agent-tick on the current true map, so the
residual is dense enough to concentrate. At `f = 1`:

| Mode | Policy | Δ_cue | Realised enforcement | vs static enforcement |
|---|---|---|---|---|
| Baldwinian | static C2 | 0.079 [0.074, 0.083] | 0.352 [0.346, 0.357] | — |
| Baldwinian | cells | 0.082 [0.077, 0.087] | 0.364 [0.356, 0.372] | +0.012 [0.003, 0.022] |
| Baldwinian | occupancy | 0.086 [0.081, 0.091] | 0.430 [0.420, 0.441] | +0.079 [0.069, 0.089] (30/30 seeds) |
| Lamarckian | static C2 | 0.359 [0.343, 0.375] | 0.141 [0.135, 0.147] | — |
| Lamarckian | cells | 0.353 [0.341, 0.365] | 0.142 [0.136, 0.148] | +0.001 [−0.005, 0.007] |
| Lamarckian | occupancy | 0.335 [0.316, 0.354] | 0.184 [0.176, 0.192] | +0.043 [0.036, 0.051] (30/30 seeds) |

Occupancy also lowers Lamarckian Δ_cue vs static (−0.024 [−0.042, −0.006])
and held-out Δ_cue (−0.087 [−0.148, −0.029]). It was specified as a control
for crowding, not as a treatment; on the operational metrics it is the only
policy that clearly reallocates the budget.

HA4 asked whether cells beats occupancy on Δ_cue. It does not. Cells minus
occupancy is −0.004 [−0.010, 0.003] Baldwinian and +0.018 [−0.008, 0.044]
Lamarckian. Occupancy *raises* realised enforcement relative to cells
(−0.067 and −0.042 on the cells−blind contrast, sign agreement 0.97 / 0.93).

### Movement chasing is not the actionable leak

`adaptive_move` reallocates more than cells (KL 0.07–0.08, overlap 0.54) but
not like occupancy. It does not reduce Δ_cue relative to cells or to static
C2 (HA3 falsified in both modes). The H2 side-channel (`move_share`) is
predictive for a classifier; using it as a sampling weight for the next
monitor map does not close the concealment gap.

### Evaluator recovery fails at f = 1

A logistic evaluator fitted on C1 and applied unchanged:

| Cell | Baldwinian AUC | Lamarckian AUC |
|---|---|---|
| C4 (decoy) | 0.785 [0.775, 0.795] | 0.772 [0.758, 0.786] |
| occupancy f = 1 | 0.673 [0.660, 0.688] | 0.492 [0.475, 0.510] |
| cells f = 1 | 0.663 [0.647, 0.678] | 0.485 [0.466, 0.503] |
| move f = 1 | 0.651 [0.637, 0.665] | 0.474 [0.453, 0.494] |
| static C2 | 0.664 [0.647, 0.680] | 0.480 [0.460, 0.501] |

HA5 required AUC ≥ 0.7, or at least 0.05 above static C2. Neither happens.
Occupancy's enforcement gain is not enough to restore a sealed-world ranking
once the cue is perfect. At `f = 0.7` every adaptive cell still has
C1-calibrated AUC ≥ 0.76, matching the parent dose-response: concealment that
breaks evaluation is a high-fidelity phenomenon.

## Takeaway

A fixed coverage budget cannot catch this concealment by chasing the last
place a defection was *caught*. That residual is too sparse to reallocate
the map. Chasing occupancy does reallocate and raises realised enforcement
in every seed, and in the Lamarckian arm it trims Δ_cue, but a C1-calibrated
evaluator remains at chance at `f = 1`. Closing the veil with the same `c`
and `p` would need a denser, or differently targeted, residual than caught
defections — and even a residual that concentrates the map is not enough
once the cue is perfect.

## Limitations

- Occupancy was the HA4 control, not a pre-stated treatment. The enforcement
  and Δ_cue gains for blind are exploratory relative to HA1–HA5.
- Laplace smoothing 1.0 over 320 cells is part of why cells stays near-uniform;
  a lower floor, or a count that is not per-cell, could make defection-chasing
  concentrate. That would be a different policy, not this run.
- Baldwinian within-lifetime learning remains weak (same world as the parent).
- The held-out cue confound from the parent experiment is unchanged.

## Reproduce

```bash
PYTHONHASHSEED=0 python scripts/run_veil_ceiling_adaptive.py        # ~48 min on 4 cores
python scripts/run_veil_ceiling_adaptive.py --analyze-only
```
