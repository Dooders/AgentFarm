# Adaptive monitor follow-up

**Status:** Implemented (`farm.experiments.veil_ceiling.adaptive`)
**CLI:** [`scripts/run_veil_ceiling_adaptive.py`](../../../../scripts/run_veil_ceiling_adaptive.py)
**Parent experiment:** [Design.md](Design.md), [RESULTS.md](RESULTS.md)

The confirmatory Veil Ceiling run held enforcement fixed (`c = 0.5`, `p = 6`)
and varied cue fidelity. A leaked cue let agents defect where they were not
watched, so realised enforcement in Lamarckian C2 fell to 0.14. The honest
evaluator (calibrated on C1) then ranked those agents at chance.

This follow-up asks the policy question that setup implies: **can the same
coverage budget catch concealment by reallocating where it looks**, without
raising `c` or `p`?

## Constraints

- Coverage per epoch is exact: `round(c · n_training_cells)` cells, as in the
  original design. The monitor cannot buy more eyes.
- Penalty, epoch length, world, learner and seeds are unchanged.
- The decoy map is still drawn uniformly, so a leaked cue of the *true* map
  is not confounded with a leaked cue of an adaptive decoy.
- Seed `s` still shares the world layout and the *first* epoch's true map
  with static C2. Adaptive draws consume extra RNG only after epoch 0.

## Policies

Each epoch the monitor records an observed residual on the current true map
and uses it as sampling weights (Laplace-smoothed) for the next epoch.

| Policy | Residual | What it tests |
|---|---|---|
| `adaptive_cells` | observed defections | Chase the last place defection was *caught*. |
| `adaptive_move` | observed movement | Act on the side channel that survived the H2 ablation. |
| `adaptive_blind` | observed occupancy (agent-ticks on the current true map) | Control: is the monitor just chasing busy cells? |

Static C2 / C3_f0.9 / C3_f0.7 of the same seed are the matched baselines.

## Cells

`ADAPTIVE_MATRIX_CONDITION_ORDER`: C1, C3_f0.7, C3_f0.9, C2, C4, and the three
policies at `f ∈ {0.7, 0.9, 1.0}`. Crossed with Baldwinian and Lamarckian.
30 seeds. No penalty-robustness cells (the original run already showed Δ is
stable in `p`).

## Hypotheses

| | Claim | Test at `f = 1`, seed-paired |
|---|---|---|
| HA1 | `adaptive_cells` reduces Δ_cue vs static C2 | CI of difference < 0 |
| HA2 | `adaptive_cells` raises realised enforcement vs static C2 | CI of difference > 0 |
| HA3 | `adaptive_move` reduces Δ_cue more than `adaptive_cells` | move − cells CI < 0; else inconclusive if it still beats static |
| HA4 | `adaptive_cells` beats `adaptive_blind` | cells − blind Δ_cue CI < 0 (not just crowding) |
| HA5 | A C1-calibrated evaluator recovers on `adaptive_cells` | AUC ≥ 0.7, or at least clearly above static C2 |

## Outputs

Raw per-cell ledgers under `experiments/veil_ceiling/adaptive_results/`, plus
`ADAPTIVE_REPORT.md`, analysis tables and figures. Reproduce with:

```bash
PYTHONHASHSEED=0 python scripts/run_veil_ceiling_adaptive.py
python scripts/run_veil_ceiling_adaptive.py --analyze-only
```
