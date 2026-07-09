---
layout: page
title: "Sparse ecology doesn't save the ladder: warm-start still loses"
---

The [07-08 saturated A/B](2026-07-08-inheritance-ladder-warm-start-clamps-offspring.md)
left one confound open: every arm ran at full population saturation (final count
32/32 in essentially every cell), while the transferable-signal budget the
ladder is graded against was measured in a sparse, fixed-8, no-reproduction
regime. This post closes that confound — the decisive follow-up that re-runs
the identical inheritance-ladder A/B with `--low-churn` (`max_population = 8`)
so density matches the precondition gate.

**Result: warm-start still loses.** Zero robust positives, zero positive mean
deltas across all 36 (profile, arm, age) cells. Absolute reward is higher in
the sparse regime — cold-start captures that gain — but every warm-start
payload still earns roughly half of what Baldwinian cold-start earns. Saturation
was a confound for levels, not ranking. The barrier is real for this design.

## The regime and what ran

Identical to the [07-08 saturated sweep](2026-07-08-inheritance-ladder-warm-start-clamps-offspring.md)
except for the population cap:

- **Arms (5):** `baldwinian` (P0), `lamarckian` (P1, weights), `p2` (plasticity
  damping), `p3` (optimizer state + replay slice), `p4` (fitness-gated θ blend).
  Default warm-start knobs (damping 0.5, replay limit 256, blend 0.5, gate 1.0).
- **Population:** 8 independent agents, **`max_population = 8`** (`--low-churn`).
  Reproduction replaces rather than expands the colony — density matches the
  precondition gate. Warm-start fires at reproduction as before.
- **Horizon:** 3000 steps, 200-step warmup, snapshot every 100.
- **Matrix:** 3 profiles (`conservative`/`balanced`/`buffered`) × 6 seeds
  (`42 7 19 101 137 256`) × 5 arms = **90 runs**.

Offspring cohorts are smaller than in the saturated sweep — the colony stays
near its cap of 8 so reproduction events are less frequent — but cohort sizes
are still adequate for the robustness gate (paired 95% CI excludes zero **and**
within-profile sign agreement ≥ 0.75).

## Headline result: still entirely negative

Across all 36 (profile, arm, age) cells, **zero cells are robustly positive.
Zero positive mean deltas.** The profile and arm ordering is preserved exactly
from the saturated sweep: `buffered` is the most damaging ecology for warm-start,
P4 (the richest payload) is the worst arm in every profile. No payload wins, at
any age, in any profile.

This is the same dose-response seen in the saturated sweep: the richer the
payload imposed on the offspring, the larger the loss. If the problem were
"insufficient transferred signal," enriching the payload would help. It does the
opposite.

## Absolute reward jumps — cold-start takes it

The sparse regime raises absolute reward for everyone: cold-start (`baldwinian`)
offspring earn substantially more than their saturated-regime counterparts,
especially in `conservative` and `balanced`. The warm-start arms do share in the
absolute gain — but they capture roughly half of what cold-start earns, the same
fraction as in the saturated run. Reducing density did not shrink the relative
gap; it just moved the floor up. The inherited policy is still calibrated to
parent conditions that differ from what the offspring encounters, and the
offspring still pays for it.

## The confound is closed

Saturation was the most natural explanation for the 07-08 result: offspring born
into a crowded colony face a different resource landscape than the parent, so a
parent-calibrated policy would be systematically wrong. Reduce density so that
conditions at birth better match conditions during the parent's life, and the
mismatch should shrink.

It does not shrink. The ranking is identical, the robustness pattern is
identical, and the dose-response is identical. Crowding inflated the *scale* of
the loss; it was not the cause.

Together with the
[07-08 saturated A/B](2026-07-08-inheritance-ladder-warm-start-clamps-offspring.md),
the honest claim is: *Lamarckian-style policy inheritance does not beat
Baldwinian cold-start on early-life reward in either the saturated or the sparse
learning-positive regime, and richer payloads make it worse.*

## What it means for #904

The decision rule — keep a richer payload only if it robustly beats P0 on
early-life net reward without degrading stability — is not met in either regime.
**None of P1–P4 advance.** Baldwinian cold-start stays the default. Richer
payloads don't help and usually hurt, worst of all in resource-rich ecologies.

The saturation open question (first bullet in the 07-08 post) is now closed.
The remaining open questions concern *different* warm-start designs, not
re-testing this one under yet another density condition.

## Reproduce

Low-churn sweep (resume-safe; skips completed cells):

```bash
PYTHONHASHSEED=0 python scripts/run_inheritance_mode_ab.py \
  --arms baldwinian lamarckian p2 p3 p4 \
  --population 8 --max-population 8 \
  --num-steps 3000 --snapshot-interval 100 \
  --warmstart-replay-buffer-limit 256 \
  --output-dir experiments/inheritance_ab_lowchurn \
  --disk-database --resume
```

Grade it (same analysis scripts as the saturated sweep):

```bash
python scripts/analyze_early_life_fitness.py \
  --ab-dir experiments/inheritance_ab_lowchurn \
  --baseline-arm baldwinian --treatment-arms lamarckian p2 p3 p4

python scripts/compare_inheritance_arms.py \
  --baseline-dir experiments/inheritance_ab_lowchurn/baldwinian \
  --baseline-label baldwinian \
  --treatment-dir experiments/inheritance_ab_lowchurn/lamarckian \
  --treatment-dir experiments/inheritance_ab_lowchurn/p2 \
  --treatment-dir experiments/inheritance_ab_lowchurn/p3 \
  --treatment-dir experiments/inheritance_ab_lowchurn/p4 \
  --arm-labels lamarckian p2 p3 p4 \
  --output-dir experiments/inheritance_ab_lowchurn/aggregate
```

Outputs land in `experiments/inheritance_ab_lowchurn/early_life/`
(`early_life_ladder_summary.json` / `.md`) and `.../aggregate/`.

## Open questions

- **Does the clamp relax with a weaker blend?** P4 at blend 0.5 imposes half the
  parent policy; sweeping the blend toward 0 should interpolate back to P0. If
  even a small blend hurts in `buffered`, the imposition itself is the problem
  regardless of density.
- **Per-offspring warm-start contrast.** Coverage is ~100%, so the arm-level
  comparison can't separate "warm-started" from "born via the treatment path."
  Per-offspring applied/skipped telemetry would let us compare warm-started and
  cold offspring *within* the same run (see the `incompatible_state` skip issue).
- **Different transfer designs.** Distilled priors, delayed fitness-gated
  transfer, or partial-layer warm-start are untested; the barrier established
  here is for *this* warm-start-on-reproduction ladder.

## Related docs

- [The saturated A/B this settles](2026-07-08-inheritance-ladder-warm-start-clamps-offspring.md)
- [Implement inherited-payload ladder (P2-P4) and run the #848 experiment (#904)](https://github.com/Dooders/AgentFarm/issues/904)
- [Pilot early-life results: no RL-reward gain vs P0 (#964)](https://github.com/Dooders/AgentFarm/issues/964)
- [The transferable-signal gate: do learned policies beat their own init?](2026-06-20-transferable-signal-budget.md)
- [Inherited payload design](../../design/inherited_payload_design.md)
