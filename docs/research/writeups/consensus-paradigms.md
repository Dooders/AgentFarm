---
layout: experiment
title: Consensus paradigms
subtitle: Does ballot format move minority-cluster welfare relative to party?
status: complete
updated: 2026-09-12
category: Collective choice
excerpt: >-
  Four selection rules see the same voters and candidate slates. Individual,
  score, and latent-match all lower minority-cluster welfare versus party;
  total welfare barely moves. Election-endogenous "loser" gaps shrink without
  helping the fixed minority cluster.
variants:
  - id: party
    title: Party
    note: baseline
  - id: individual
    title: Individual plurality
  - id: score
    title: Score voting
  - id: latent-match
    title: Latent match
  - id: appendix
    title: Appendix cells
    note: λ-correlated, reelection, λ cap
---

The default cell asks whether ballot format changes the winner's allocation in
a way that raises **minority-cluster** (fixed-partition) welfare and/or total
welfare relative to party, holding λ's marginal fixed. It is a selection-rule
comparison with exogenous types, not a test of loyalty formation.

Voters never see λ in the default generator, so `E[λ_winner]` is the Beta mean
under every rule by construction. A flat λ profile is not a finding.

Official artifacts can be stamped with FarmNotary; synthetic ballots stay
under `private/` off the record. Package README:
[`farm/experiments/consensus/README.md`](../../../farm/experiments/consensus/README.md).
Protocol notes: [consensus paradigms](../experiments/consensus_paradigms.md).

## Shared protocol

- **Population:** 400 voters per trial, latent preference vectors in R⁵ from
  cluster centers plus Gaussian noise (σ = 0.35). Official cell: `two_cluster`.
- **Candidates:** 8 per trial, platforms from the same cluster structure.
  Loyalty trait λ ~ Beta(2.2, 2.2), independent of platform and of anything
  voters see.
- **Allocation:** `directed = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`,
  then mixed with a renormalized platform simplex.
- **Trials:** 250 per cell, base seed 0. Every paradigm sees the identical
  population and candidate slate within a trial.
- **Primary endpoints:** paired Δ minority-cluster welfare and Δ total
  welfare vs party, Holm-corrected. Wilcoxon signed-rank; 95% Student-t CIs;
  paired Cohen's d.
- **Baselines on the same draws:** random winner, utilitarian vertex,
  egalitarian maximin.

```bash
python run_experiment.py --trials 250 --voters 400 --candidates 8 \
    --population two_cluster --seed 0 --no-persist-ballots \
    --out experiments/consensus_paradigms/results
python scripts/notarize_run.py --run-dir experiments/consensus_paradigms/results --runner consensus_paradigms
```

Official stamp:
[`experiments/consensus_paradigms/results/REPORT.md`](../../../experiments/consensus_paradigms/results/REPORT.md)
(aggregates only).

## Party {#party}

Vote nearest of two party brands; the nominee of the winning brand allocates.
Supporters are voters for that brand.

| | total welfare | minority welfare | majority welfare | gini |
|---|---|---|---|---|
| party | 0.2313 ± 0.0051 | 0.2549 ± 0.0653 | 0.2076 ± 0.0643 | 0.1841 ± 0.0424 |

Party is the contrast baseline. Mean winner allocation puts more on
`minority_pork` (0.327) than `majority_pork` (0.215) in this generator —
party brands can land in the minority cluster. Loser share is ~0.50 by
construction of two brands, not a treatment effect.

## Individual plurality {#individual}

Vote nearest candidate; winner is plurality. Supporters are voters who picked
the winner.

Versus party, same-trial:

| endpoint | Δ mean | 95% CI | Holm p | paired d |
|---|---|---|---|---|
| minority welfare | −0.0244 | [−0.0361, −0.0127] | 0.0017 | −0.259 |
| total welfare | −0.0004 | [−0.0012, +0.0003] | 0.811 | −0.073 |

Individual-centered ballots do **not** raise the fixed minority cluster's
welfare. Total welfare is a wash. Election-endogenous loser share jumps to
~0.70 because "loser" now means everyone who did not pick the winner — a
different estimand, not a welfare gain.

## Score voting {#score}

Rate all candidates; highest mean wins. Supporters are voters whose top score
is the winner.

| endpoint | Δ mean vs party | 95% CI | Holm p | paired d |
|---|---|---|---|---|
| minority welfare | −0.0280 | [−0.0387, −0.0174] | < 0.001 | −0.327 |
| total welfare | −0.0025 | [−0.0036, −0.0014] | 0.0019 | −0.288 |

Score is the largest minority-welfare loss of the three treatments, with a
small but significant total-welfare loss. Gini is lower than party (0.143 vs
0.184): a flatter individual distribution that still does not help the
pre-registered minority cluster.

## Latent match {#latent-match}

Candidate closest to the mean preference vector wins. Supporters are voters
whose nearest candidate is the winner.

| endpoint | Δ mean vs party | 95% CI | Holm p | paired d |
|---|---|---|---|---|
| minority welfare | −0.0262 | [−0.0364, −0.0160] | < 0.001 | −0.321 |
| total welfare | −0.0034 | [−0.0045, −0.0023] | < 0.001 | −0.373 |

Closest to "what the mean voter wants," and the worst total-welfare contrast
versus party. Public-good share of the winner's allocation is highest here
(0.287) and in score (0.282) versus party (0.267), which is not the same
thing as raising the minority cluster's welfare.

## Appendix cells {#appendix}

These are not in the official 250×400×8 two-cluster stamp. They exist as
flags on the same runner so the writeup can name them without pretending they
are the default cell.

- **`--lambda-correlated`** — robustness appendix where λ is no longer
  independent of what voters see.
- **`--mechanism reelection`** — incentive cell in which winners *choose* λ.
  The default cell is one-shot with exogenous types.
- **`constrained_individual`** — same election as individual with winner λ
  capped. It is a constitutional overlay, not a voting rule, and is excluded
  from hypothesis contrasts.

See [`farm/experiments/consensus/README.md`](../../../farm/experiments/consensus/README.md)
for how to run them.

## Synthesis

Holding people, projects, and λ's marginal fixed, switching from party to
individual, score, or latent-match *lowers* minority-cluster welfare. Total
welfare is flat or slightly down. The election-endogenous supporter/loser gap
shrinks because the loser *set* changes with the rule — that is not the
primary estimand. Mushy-bloc check: gap shrinkage without minority-cluster
gains.

## Reproduce and artifacts

```bash
python run_experiment.py --trials 250 --voters 400 --candidates 8 \
    --population two_cluster --seed 0 --no-persist-ballots \
    --out experiments/consensus_paradigms/results
```

- Wrapper: `ConsensusParadigmsExperiment` → `farm.experiments.consensus.experiment.run_trials`
- [Experiment doc](../experiments/consensus_paradigms.md)
- [Catalog entry](../experiments-catalog.md)
- [FarmNotary guide](../../guides/farm-notary.md)
- Official report: `experiments/consensus_paradigms/results/REPORT.md`
