---
title: "Consensus experiment: make project names true, and persist synthetic ballots"
type: Bug
labels: [Experiment]
---

## Context

### Project names overclaim

`README.md` and the auto-report still describe:

- `outgroup_repair` as helping "people unlike the winner"
- `buffer_reserve` as insurance that "mainly protects electoral losers"

What is implemented (`farm/experiments/consensus/population.py`):

- `outgroup_repair` is **cluster-B pork** (center weight 0.95 on B, −0.35 on
  A). If cluster B wins, this project helps the *winner's* bloc.
- `buffer_reserve` is boosted for voters far from the *population* center.
  Peripheral voters are only likelier losers; the column is not a function of
  electoral loser status or of distance to the winner.

Cluster loading is real (A mean benefit on `outgroup_repair` ≈ 0.06 vs B ≈
0.48). The *names* are still fiction relative to the winner. The auto-written
`REPORT.md` lists those names as if the semantics exist.

### Ballots are discarded under a privacy story

`metrics.py` says individual ballots and voter-level utilities never leave
the module. `REPORT.md` calls this "Privacy." These are synthetic voters.
Not saving the ballot matrix / supporter mask means the supporter
classification — the crux of the incomparable-estimand problem — cannot be
audited from `trials.csv`.

Privacy language belongs to the political design being modeled, not to the
simulation of it. Official notary policy (#983: do not notarize voter-level
choices) can stay: persist locally, keep them off the stamped record.

`cluster_ids` are already on `Population` and are also dropped.

## Goal

### Names

Either:

- **Implement the names.** After the winner is known, set column 3's benefit
  (or a post-allocation transfer) as an increasing function of distance from
  the winner's platform, and set `buffer_reserve` as a function of
  loser-status or of distance-to-winner; **or**
- **Strip the names.** Relabel to `project_0`…`project_4`, or to honest
  cluster tags (`majority_pork`, `minority_pork`, `public_good`, …). Update
  README, report, overview video, and animation captions.

Do not keep winner-relative English on cluster-static columns.

### Audit artifacts

Write, per trial (or behind `--persist-ballots`, default **on** for local
runs):

- supporter mask per paradigm (N bools), or the ballot / score matrix
- `cluster_ids`
- optionally per-voter utilities

Notarize policy unchanged: official record stays `summary.csv` + `trials.csv`
aggregates. Document that split.

## Acceptance

- [ ] No remaining docstring, README, report, or video caption claims
      `outgroup_repair` helps people unlike the winner unless that is
      implemented as a function of the winner.
- [ ] Same for `buffer_reserve` vs electoral losers.
- [ ] A local run can reconstruct who counted as a supporter for each
      paradigm (test opens the artifact and checks party supporter share
      ≈ 0.5 under two_cluster).
- [ ] Methods text does not call discarding ballots "privacy."

## Files

- `farm/experiments/consensus/population.py`
- `farm/experiments/consensus/metrics.py`
- `farm/experiments/consensus/experiment.py` (`write_outputs`)
- `farm/experiments/consensus/report.py`
- `farm/experiments/consensus/README.md`
- `farm/experiments/consensus/overview_video.py`
- `farm/experiments/consensus/animate.py`
