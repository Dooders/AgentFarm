---
title: "Consensus experiment: use the paired design — CIs, tests, effect sizes, correction"
type: Feature
labels: [Experiment]
---

## Context

`run_trials` already draws one population and one candidate slate per trial
and runs every paradigm on that draw. That is the right design. The analysis
throws the pairing away:

- `summarize` is independent group mean ± std.
- `report.py` computes `lambda_sem` and never displays it.
- No confidence intervals, no paired test, no effect size.
- 4 paradigms × ~6 metrics × 4 populations is on the order of 100 comparisons
  with no correction.
- Hypothesis verdicts use hardcoded thresholds
  (`LAMBDA_UNCHANGED_EPS = 0.02`, `WELFARE_FLAT_REL_EPS = 0.005`,
  `LOSER_SHARE_LARGE_RISE = 0.15`) instead of uncertainty.

With 250 paired trials, a Wilcoxon (or paired t) on
`λ_party - λ_individual` would have had plenty of power to confirm the
default-condition null. It is not run.

## Goal

Analyze the experiment as a paired design.

Per (population, n_candidates) cell, for each non-party paradigm vs party,
on the **same trial index**:

- paired difference mean, SD, and 95% CI (bootstrap or t)
- paired test (Wilcoxon signed-rank is enough; document the choice)
- effect size (e.g. paired Cohen's d, or rank-biserial)

Apply a documented multiple-comparison correction (Holm or Benjamini–Hochberg)
across the pre-registered primary endpoints. Primary endpoints should be few
— suggested, once the companion issues land:

1. Δλ_winner (only if the λ-selection hypothesis is kept)
2. Δ minority-cluster (or cluster-B) welfare
3. Δ total welfare

Everything else is exploratory.

Write these into `summary.csv` / a new `contrasts.csv` and into `REPORT.md`.
Stop using raw threshold constants as the sole verdict machine.

## Acceptance

- [ ] A contrasts table exists with paired Δ, CI, p, adjusted p, and effect
      size for the pre-registered endpoints.
- [ ] `REPORT.md` quotes those, not only mean ± std of unpaired groups.
- [ ] Multiple-comparison policy is written in the README and applied.
- [ ] A test on a tiny seeded run checks that pairing uses matching `trial`
      ids and that CIs are finite.

## Files

- `farm/experiments/consensus/experiment.py`
- `farm/experiments/consensus/report.py`
- `farm/experiments/consensus/README.md`
- `tests/experiments/test_consensus_invariants.py`
