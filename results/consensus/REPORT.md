# Political consensus experiment — auto-generated report

Comparison of selection paradigms on a post-election budget-allocation task: does individual-centered selection (no parties) produce stewards who treat non-supporters better than party selection does?

## Methods

- **Population**: 400 voters per trial, latent preference vectors in R^5 drawn from cluster centers plus Gaussian noise (σ = 0.35); benefit vectors are a softmax (temperature 0.55) of preferences, so rows are nonnegative and sum to 1. Population types run here: `two_cluster`.
- **Projects**: `core_services`, `coalition_club`, `outgroup_repair`, `prestige_project`, `buffer_reserve`.
- **Candidates**: 8 per trial (sweeps may vary this), platforms drawn from the same cluster structure; loyalty trait λ ~ Beta(2.2, 2.2), independent of the platform.
- **Allocation**: `raw = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`, then `raw = 0.72·raw + 0.28·clip(platform, 0, ∞)`, normalized to sum to 1. Zero supporters ⇒ all-voter direction only.
- **Paradigms**: `party`, `individual`, `score`, `latent_match`.
- **Trials**: 250 per cell, base seed 0; every paradigm sees the identical population and candidate slate within a trial.
- **Privacy**: only aggregates and winner allocations are persisted; no individual ballots.

## Results

Mean ± std over trials, by population, candidate count, and paradigm:

| population | n_candidates | paradigm | total_welfare | supporter_welfare | loser_welfare | gap | lambda_winner | loser_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | party | 0.2329 ± 0.0067 | 0.3073 ± 0.0200 | 0.1580 ± 0.0149 | 0.1493 ± 0.0326 | 0.5145 ± 0.2211 | 0.4978 ± 0.0022 |
| two_cluster | 8 | individual | 0.2323 ± 0.0081 | 0.3124 ± 0.0304 | 0.1969 ± 0.0169 | 0.1154 ± 0.0385 | 0.5111 ± 0.2155 | 0.7008 ± 0.0766 |
| two_cluster | 8 | score | 0.2297 ± 0.0088 | 0.2763 ± 0.0323 | 0.2171 ± 0.0122 | 0.0592 ± 0.0373 | 0.5080 ± 0.2230 | 0.8198 ± 0.0849 |
| two_cluster | 8 | latent_match | 0.2286 ± 0.0090 | 0.2707 ± 0.0306 | 0.2186 ± 0.0119 | 0.0521 ± 0.0344 | 0.5098 ± 0.2210 | 0.8395 ± 0.0835 |

### Mean winner allocations

| population | n_candidates | paradigm | core_services | coalition_club | outgroup_repair | prestige_project | buffer_reserve |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | party | 0.2756 | 0.2100 | 0.3353 | 0.0902 | 0.0889 |
| two_cluster | 8 | individual | 0.2731 | 0.2709 | 0.2724 | 0.0930 | 0.0905 |
| two_cluster | 8 | score | 0.2893 | 0.2593 | 0.2517 | 0.1023 | 0.0974 |
| two_cluster | 8 | latent_match | 0.2931 | 0.2483 | 0.2525 | 0.1042 | 0.1018 |

## Hypothesis evaluation

Hypothesis: individual-centered rules (`individual`, `score`, `latent_match`) select lower-λ winners than `party` and raise loser welfare without merely inflating loser share.

### two_cluster, 8 candidates

- `individual` vs `party`: Δλ_winner = -0.0034, Δloser_welfare = +0.03897, Δloser_share = +0.2030, Δgap = -0.03385, Δtotal_welfare = -0.00064 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = -0.0065, Δloser_welfare = +0.05915, Δloser_share = +0.3220, Δgap = -0.09007, Δtotal_welfare = -0.00326 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = -0.0047, Δloser_welfare = +0.06067, Δloser_share = +0.3417, Δgap = -0.09720, Δtotal_welfare = -0.00438 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0065 → **triggered**.
- total welfare flat (max relative change < 0.5%): max relative change = 1.8798% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.3417 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

## Limitations

- Voters and candidates live in a stylized 5-dimensional project space; real preference structures are higher-dimensional and partly unobservable.
- λ is exogenous by default; strategic candidate behavior, campaigning, and repeated elections are out of scope.
- Party structure is idealized as two brands at cluster means with loyal nomination; real parties select through noisy primaries.
- Supporter definitions differ across paradigms by design (that is part of the treatment), so loser-share differences should be read together with loser welfare, not alone.

## Reproduce

```
python run_experiment.py --trials 250 --voters 400 --candidates 8 --population two_cluster --seed 0 --out results/consensus
```

Deterministic given identical parameters and seed (per-trial streams are derived from `numpy.random.default_rng([seed, population, candidates, trial])`).
