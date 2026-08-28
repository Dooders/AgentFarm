# Political consensus experiment — auto-generated report

Comparison of selection paradigms on a post-election budget-allocation task: does individual-centered selection (no parties) produce stewards who treat non-supporters better than party selection does?

## Methods

- **Population**: 400 voters per trial, latent preference vectors in R^5 drawn from cluster centers plus Gaussian noise (σ = 0.35); benefit vectors are a softmax (temperature 0.55) of preferences, so rows are nonnegative and sum to 1. Population types run here: `two_cluster`.
- **Projects**: `core_services`, `coalition_club`, `outgroup_repair`, `prestige_project`, `buffer_reserve`.
- **Candidates**: 8 per trial (sweeps may vary this), platforms drawn from the same cluster structure; loyalty trait λ ~ Beta(2.2, 2.2), rank-coupled to platform extremity (`--lambda-correlated`).
- **Allocation**: `raw = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`, then `raw = 0.72·raw + 0.28·clip(platform, 0, ∞)`, normalized to sum to 1. Zero supporters ⇒ all-voter direction only.
- **Paradigms**: `party`, `individual`, `score`, `latent_match`, `constrained_individual`. `constrained_individual` caps the winner's effective λ at 0.25.
- **Trials**: 250 per cell, base seed 0; every paradigm sees the identical population and candidate slate within a trial.
- **Privacy**: only aggregates and winner allocations are persisted; no individual ballots.

## Results

Mean ± std over trials, by population, candidate count, and paradigm:

| population | n_candidates | paradigm | total_welfare | supporter_welfare | loser_welfare | gap | lambda_winner | loser_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | party | 0.2329 ± 0.0066 | 0.3026 ± 0.0214 | 0.1626 ± 0.0155 | 0.1400 ± 0.0349 | 0.4305 ± 0.1893 | 0.4978 ± 0.0022 |
| two_cluster | 8 | individual | 0.2324 ± 0.0078 | 0.3094 ± 0.0337 | 0.1984 ± 0.0175 | 0.1110 ± 0.0433 | 0.4398 ± 0.1954 | 0.7008 ± 0.0766 |
| two_cluster | 8 | score | 0.2304 ± 0.0078 | 0.2677 ± 0.0297 | 0.2202 ± 0.0108 | 0.0475 ± 0.0342 | 0.2394 ± 0.1247 | 0.8198 ± 0.0849 |
| two_cluster | 8 | latent_match | 0.2295 ± 0.0079 | 0.2612 ± 0.0264 | 0.2220 ± 0.0098 | 0.0393 ± 0.0288 | 0.2047 ± 0.1040 | 0.8395 ± 0.0835 |
| two_cluster | 8 | constrained_individual | 0.2324 ± 0.0074 | 0.2966 ± 0.0252 | 0.2040 ± 0.0133 | 0.0927 ± 0.0306 | 0.4398 ± 0.1954 | 0.7008 ± 0.0766 |

### Mean winner allocations

| population | n_candidates | paradigm | core_services | coalition_club | outgroup_repair | prestige_project | buffer_reserve |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | party | 0.2756 | 0.2120 | 0.3332 | 0.0902 | 0.0890 |
| two_cluster | 8 | individual | 0.2728 | 0.2669 | 0.2778 | 0.0925 | 0.0900 |
| two_cluster | 8 | score | 0.2866 | 0.2607 | 0.2577 | 0.0997 | 0.0952 |
| two_cluster | 8 | latent_match | 0.2889 | 0.2538 | 0.2570 | 0.1015 | 0.0988 |
| two_cluster | 8 | constrained_individual | 0.2740 | 0.2687 | 0.2745 | 0.0932 | 0.0896 |

## Hypothesis evaluation

Hypothesis: individual-centered rules (`individual`, `score`, `latent_match`) select lower-λ winners than `party` and raise loser welfare without merely inflating loser share.

### two_cluster, 8 candidates

- `individual` vs `party`: Δλ_winner = +0.0093, Δloser_welfare = +0.03580, Δloser_share = +0.2030, Δgap = -0.02899, Δtotal_welfare = -0.00047 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = -0.1911, Δloser_welfare = +0.05754, Δloser_share = +0.3220, Δgap = -0.09249, Δtotal_welfare = -0.00250 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = -0.2258, Δloser_welfare = +0.05933, Δloser_share = +0.3417, Δgap = -0.10069, Δtotal_welfare = -0.00342 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.2258 → not triggered.
- total welfare flat (max relative change < 0.5%): max relative change = 1.4675% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.3417 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

## Limitations

- Voters and candidates live in a stylized 5-dimensional project space; real preference structures are higher-dimensional and partly unobservable.
- λ is exogenous by default; strategic candidate behavior, campaigning, and repeated elections are out of scope.
- Party structure is idealized as two brands at cluster means with loyal nomination; real parties select through noisy primaries.
- Supporter definitions differ across paradigms by design (that is part of the treatment), so loser-share differences should be read together with loser welfare, not alone.

## Reproduce

```
python run_experiment.py --trials 250 --voters 400 --candidates 8 --population two_cluster --seed 0 --lambda-correlated --include-constrained --out results/consensus_lambda_correlated
```

Deterministic given identical parameters and seed (per-trial streams are derived from `numpy.random.default_rng([seed, population, candidates, trial])`).
