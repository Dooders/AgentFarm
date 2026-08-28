---
title: "Consensus experiment: normalize the winner platform before the 0.72/0.28 blend"
type: Bug
labels: [Experiment]
---

## Context

`allocate` mixes a directed benefit mean (rows of a simplex, so the mix sums
to 1) with an unconstrained clipped platform:

```python
# farm/experiments/consensus/allocation.py
raw = lam * dir_supporters + (1.0 - lam) * dir_all   # sums to 1
raw = 0.72 * raw + 0.28 * np.clip(winner_platform, 0.0, None)
return raw / raw.sum()
```

`clip(platform, 0, ∞)` has no scale constraint. Platforms are cluster centers
plus `N(0, 0.35)` noise. On a typical 8-candidate draw, clipped L1 norms
ranged **1.13–2.82**. After the final normalization the platform's effective
share was **30–52%**, not 28%.

Extreme platforms (larger clipped norm) mechanically get more allocation
weight regardless of λ. Party vs individual can therefore differ through
platform norm, not loyalty — a spurious treatment effect.

The 0.72/0.28 weights are documented as if they were a convex combination.
They are not.

## Goal

Put `raw` and the platform term on the same scale before blending, so the
weights mean what they say.

Suggested fix (keep the existing weights unless a new spec is written):

1. `plat = clip(winner_platform, 0, ∞)`
2. If `plat.sum() == 0`, skip the platform term (directed-only).
3. `plat = plat / plat.sum()`
4. `raw = 0.72 * directed + 0.28 * plat`
5. Normalize (now a no-op aside from numerical noise).

Add a unit test: for any finite platform, the platform term's mass after
blending equals `BLEND_PLATFORM` when the directed vector is a probability
vector.

## Acceptance

- [ ] `allocate` normalizes the clipped platform (or uses another documented,
      scale-invariant map) before mixing.
- [ ] A test asserts the effective platform share equals `BLEND_PLATFORM`
      ± 1e-9 for several platform norms, including near-zero and large-norm
      platforms.
- [ ] README / report methods line no longer implies `0.72/0.28` is a blend
      of unmatched units.
- [ ] Re-run the default cell and record whether party-vs-individual
      allocation gaps shrink once norm is controlled.

## Files

- `farm/experiments/consensus/allocation.py`
- `farm/runners/consensus_paradigms_experiment.py` (`_allocate` — same bug)
- `tests/experiments/test_consensus_invariants.py`
- `farm/experiments/consensus/README.md`
- `farm/experiments/consensus/report.py` (methods paragraph)
