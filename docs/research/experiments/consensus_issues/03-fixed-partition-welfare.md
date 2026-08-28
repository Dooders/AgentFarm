---
title: "Consensus experiment: report welfare on a fixed partition, with baselines and tails"
type: Feature
labels: [Experiment]
---

## Context

"Supporters" is a different estimand in every treatment:

| Paradigm | How supporters are defined | Typical loser_share (two_cluster, M=8) |
|---|---|---|
| party | voters for the winning brand | ~0.50 |
| individual | plurality ballots for the winner | ~0.70 |
| score | voters whose top score is the winner | ~0.82 |
| latent_match | nearest-candidate back-derivation; nobody voted this way | ~0.84 |

`gap` and `loser_welfare` therefore compare non-comparable quantities.
`cluster_ids` already exist on `Population` and are never written to metrics.

The spec names this as a falsifier ("gap shrinks only because the winning
bloc is mushier") but the check in `report.py` only fires if the gap shrinks
**and** loser welfare does not rise. In the default run loser welfare *does*
rise — because the loser set now includes near-median voters. The auto-report
then says "not triggered," which is exactly the artifact.

There is also no scale: no random-winner floor, no utilitarian-optimal
ceiling, no egalitarian benchmark. A gap of 0.15 vs 0.05 is uninterpretable.

Only group means are stored. "Treats non-supporters better" is a
distributional claim; min / 10th percentile / Gini are missing.

`rural_town` (70/30) reintroduces the size confound: party `loser_share` is
0.30 in every sweep cell by construction, so cross-population gap comparisons
mix treatment with bloc size. A fixed cluster partition removes that.

## Goal

Hold the welfare partition **constant across paradigms**, and give every
welfare number a (random, optimal) bracket plus tail stats.

### Fixed partition

For every trial, record welfare for generator clusters (and, for
`one_cluster`, the PCA split already used to place party brands):

- `cluster_k_welfare` for each cluster k
- `minority_welfare` / `majority_welfare` using the two largest clusters
- keep today's supporter/loser numbers, but label them as
  *election-endogenous* and do not use them as the primary contrast

Primary hypothesis contrast becomes: does the selected winner's allocation
raise **cluster-B** (or the minority cluster) welfare relative to party,
holding the set of people fixed?

### Baselines (same population + candidate draw)

- `random_winner`: pick a candidate uniformly; allocate with that candidate's
  λ and with supporters = that candidate's nearest-pref voters (or with the
  fixed partition only — pick one and document it).
- `utilitarian`: allocation `a` on the simplex maximizing `mean(benefits @ a)`
  (here that is the mean benefit direction itself, i.e. `dir_all`, optionally
  still blended with a chosen platform — document the family).
- `egalitarian`: maximize the minimum voter utility, or a documented
  approximation (e.g. project that maximizes the 10th percentile).

Normalize reported welfare to `(metric - random) / (optimal - random)` when
the denominator is nonzero.

### Distributional stats

Per trial, on the **fixed** partition and on all voters:

- minimum utility
- 10th percentile
- Gini of utilities

## Acceptance

- [ ] `trials.csv` includes cluster-level welfare (or equivalent fixed-group
      columns) for every paradigm × trial.
- [ ] Auto-report's primary Δloser / Δgap lines are replaced or supplemented
      by fixed-partition contrasts.
- [ ] Random-winner and utilitarian (and documented egalitarian) baselines
      appear in `summary.csv` and `REPORT.md`.
- [ ] Min, p10, and Gini are computed; "treats non-supporters better" is not
      claimed from means alone.
- [ ] `rural_town` commentary no longer compares party `loser_share` across
      populations as if it were a treatment effect.
- [ ] The mushy-bloc falsifier uses the fixed partition (gap on election
      supporters can shrink while cluster-B welfare does not rise).

## Files

- `farm/experiments/consensus/metrics.py`
- `farm/experiments/consensus/experiment.py` (`WELFARE_COLUMNS`, `summarize`)
- `farm/experiments/consensus/report.py`
- `farm/experiments/consensus/plots.py`
- `tests/experiments/test_consensus_invariants.py`
