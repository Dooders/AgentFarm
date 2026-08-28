---
title: "Consensus experiment: replace tautological invariants with property tests"
type: Task
labels: [Experiment]
---

## Context

`tests/experiments/test_consensus_invariants.py` currently checks:

- allocations sum to 1 and are nonnegative (guaranteed by the normalize line)
- utilities are finite
- party `loser_share` is near 0.5 under two equal clusters (the generator
  forces 50/50 cluster counts; parties sit at those cluster means — a
  tautology restated as validation)
- identical seed ⇒ identical summary
- `constrained_individual` respects the λ cap
- zero supporters falls back to the everyone-direction
- benefits rows sum to 1

Those are useful smoke tests. They do not test the scientific claims.

Worth testing instead (from the critique, still unimplemented):

1. **Monotonicity.** Holding population, platform, and supporter mask fixed,
   higher λ ⇒ weakly lower welfare for the complement of that mask.
2. **λ=0 maximizes total welfare in the allocation family.** For a fixed
   platform blend, `allocate(..., lam=0)` should have total welfare ≥
   `allocate(..., lam=1)` when the platform term is held constant (or
   document the cases where the platform term can violate this).
3. **Metamorphic invariance.** Permuting candidate order does not change the
   elected platform / λ (ties already break to lowest index — document and
   test after a stable tie-break, or test that the winner's *platform and λ*
   match). Permuting project columns consistently in prefs, benefits, and
   platforms leaves welfare scalars unchanged.
4. **Scale invariance of the platform blend** (companion to the normalize
   issue): `allocate(..., k * platform)` equals `allocate(..., platform)` for
   `k > 0` after clip, once the platform is normalized.
5. **Paired identity.** Two paradigms on the same trial share
   `n_voters`, and party vs individual can elect different winners.

## Goal

Keep the smoke tests. Add property tests for the claims the experiment
actually needs to be true.

## Acceptance

- [ ] Monotonicity test: complement welfare is nonincreasing in λ, ceteris
      paribus, on a two_cluster draw.
- [ ] In-family total-welfare test at λ=0 vs λ=1 with a documented allowance
      for the platform term.
- [ ] At least one metamorphic test (candidate relabel or project permutation).
- [ ] Party loser-share test is commented as a generator/sanity check, not as
      proof the party rule is correct.

## Files

- `tests/experiments/test_consensus_invariants.py`
- `farm/experiments/consensus/allocation.py` (if monotonicity requires a
  cleaner platform term)
