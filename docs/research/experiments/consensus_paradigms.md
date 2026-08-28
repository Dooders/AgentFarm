# Consensus paradigms

**Status:** Scaffold  
**Runner:** `ConsensusParadigmsExperiment` (`farm/runners/consensus_paradigms_experiment.py`)  
**CLI:** `scripts/run_consensus_paradigms_experiment.py`  
**Notary:** [FarmNotary](https://github.com/Dooders/FarmNotary) via `farm.provenance.notary`

## Question

Does selecting a *person* instead of a *party* produce stewards who treat electoral non-supporters better?

After selection, one steward allocates a fixed budget across five projects. Loyalty `λ ∈ [0,1]` mixes “serve supporters” vs “serve everyone.” Ballot format is the treatment. `λ` of the winner is an outcome, not an input to the voting rule.

## Paradigms

| Key | Rule | Supporters are |
|-----|------|----------------|
| `party` | Vote nearest of two party brands; nominee of winning brand | Voters for that brand |
| `individual` | Vote nearest candidate | Voters who picked the winner |
| `score` | Rate all candidates; highest mean wins | Voters whose top score is the winner |
| `latent_match` | Candidate closest to mean preference vector | Voters whose nearest candidate is the winner |
| `constrained_individual` | Same as individual, winner `λ` capped | Same as individual |

## Metrics

- total / supporter / loser welfare
- gap = supporter − loser
- `lambda_winner`
- `loser_share`

Hypothesis to test (do not assume): individual / score / latent_match select lower `λ` and raise loser welfare without merely inflating `loser_share`.

Prototype orientation (two clusters, 250 trials): total welfare and `λ_winner` were nearly flat across rules; party had ~50% losers and a larger gap; non-party rules had ~70–75% loser share and a smaller gap.

## Open design issues

Drafts for remaining design flaws (headline λ hypothesis, welfare estimands,
scaffold vs package split) are in
[consensus_issues/](consensus_issues/README.md).

## Official record

Notarize `summary.csv` and `trials.csv`. Do not notarize per-voter choices or per-winner allocations.

## Quick start

```bash
python scripts/run_consensus_paradigms_experiment.py --trials 20 --voters 80 --candidates 6
python scripts/notarize_run.py --run-dir experiments/consensus_paradigms/results --runner consensus_paradigms
```
