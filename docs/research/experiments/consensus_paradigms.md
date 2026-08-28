# Consensus paradigms

**Status:** Implemented  
**Package:** `farm.experiments.consensus`  
**CLI:** `python run_experiment.py`  
**Wrapper:** `ConsensusParadigmsExperiment` (`farm/runners/consensus_paradigms_experiment.py`) calls `run_trials`  
**Notary:** [FarmNotary](https://github.com/Dooders/FarmNotary) via `farm.provenance.notary`

## Question

The **default** cell asks whether ballot format changes the winner's allocation
in a way that raises **minority-cluster** (fixed-partition) welfare and/or
total welfare relative to party, holding λ's marginal fixed. It is a
selection-rule comparison with exogenous types, not a test of loyalty
formation.

Voters never see λ in the default generator, so `E[λ_winner]` is the Beta mean
under every rule by construction. Do not treat a flat λ profile as a finding.

`--lambda-correlated` is a robustness appendix. `--mechanism reelection` is
the incentive cell in which winners choose λ. See
[`farm/experiments/consensus/README.md`](../../../farm/experiments/consensus/README.md).

## Paradigms

| Key | Role | Supporters are |
|-----|------|----------------|
| `party` | Vote nearest of two party brands; nominee of winning brand | Voters for that brand |
| `individual` | Vote nearest candidate | Voters who picked the winner |
| `score` | Rate all candidates; highest mean wins | Voters whose top score is the winner |
| `latent_match` | Candidate closest to mean preference vector | Voters whose nearest candidate is the winner |
| `constrained_individual` | Same election as individual; winner `λ` capped | Same as individual — **not a voting rule**; excluded from hypothesis contrasts |

## Metrics

Primary (paired vs party, Holm-corrected): minority-cluster welfare, total welfare.

Also reported: election-endogenous supporter / loser welfare and gap (different
estimand per rule); min / p10 / Gini; random-winner and utilitarian / egalitarian
brackets; `lambda_winner` (exploratory in the default cell).

## Official record

Notarize `summary.csv`, `trials.csv` aggregates, and `contrasts.csv`. Do not
notarize `private/` (synthetic ballots / supporter masks). Discarding those
locally is an official-record split, not a privacy claim about synthetic voters.

## Quick start

```bash
python run_experiment.py --trials 20 --voters 80 --candidates 6 --out results/consensus
python scripts/notarize_run.py --run-dir results/consensus --runner consensus
```

The wrapper CLI still works and calls the same `run_trials`:

```bash
python scripts/run_consensus_paradigms_experiment.py --trials 20 --voters 80 --candidates 6
```
