# Political consensus experiment — auto-generated report

Robustness appendix (not the primary cell): when high λ is rank-coupled to platform extremity, do centrist-favoring rules select lower-λ winners, and does that raise minority-cluster welfare?

## Methods

- **Population**: 400 voters per trial, latent preference vectors in R^5 drawn from cluster centers plus Gaussian noise (σ = 0.35); benefit vectors are a softmax (temperature 0.55) of preferences, so rows are nonnegative and sum to 1. Population types run here: `two_cluster`.
- **Projects** (generator tags, not winner-relative): `public_good`, `majority_pork`, `minority_pork`, `prestige`, `periphery_buffer`.
- **Candidates**: 8 per trial (sweeps may vary this), platforms drawn from the same cluster structure; loyalty trait λ ~ Beta(2.2, 2.2), rank-coupled to platform extremity (`--lambda-correlated`). This is a robustness appendix, not the primary cell.
- **Allocation**: `directed = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`, then `raw = 0.72·directed + 0.28·simplex(clip(platform, 0, ∞))`. The platform is renormalized before mixing so the weights are a convex combination of matched units. Zero supporters ⇒ all-voter direction only. A platform with no positive mass is dropped.
- **Paradigms (selection treatments)**: `party`, `individual`, `score`, `latent_match`. `constrained_individual` is a constitutional λ cap, not a voting rule, and is reported separately below (cap 0.25).
- **Voting**: `sincere`. Sincere plurality is the default baseline; `abandon_trailing` is a Duverger-style heuristic on individual (and the cap overlay).
- **Mechanism**: `oneshot`. The default one-shot cell is a selection-rule comparison with exogenous types, not a test of loyalty formation.
- **Trials**: 250 per cell, base seed 0; every paradigm sees the identical population and candidate slate within a trial.
- **Primary endpoints**: Δ minority-cluster welfare and Δ total welfare vs party, plus Δλ_winner. Holm correction across that family. Wilcoxon signed-rank on paired trial-level differences; 95% CIs are Student-t; effect size is paired Cohen's d.
- **Fixed partition**: welfare is reported on generator clusters (PCA split for `one_cluster`). Election-endogenous supporter/loser numbers are kept and labeled as such; they are not the primary contrast. `rural_town` party `loser_share` equals the minority bloc size by construction (~0.30) and is not a treatment effect.
- **Baselines** (same population + candidate draw): `random_winner` (uniform candidate, nearest-pref supporters), `utilitarian` (simplex vertex maximizing mean utility), `egalitarian` (maximin LP). Normalized welfare is `(metric − random) / (utilitarian − random)` when the denominator is nonzero.
- **Audit**: synthetic ballots, supporter masks, and cluster ids are written under `private/` when `--persist-ballots` is on (the default). They are not a privacy claim and they are not notarized. Official record: `summary.csv`, `trials.csv` aggregates, `contrasts.csv`.

## Results

Mean ± std over trials for **selection rules**. `loser_share` / `loser_welfare` / `gap` are election-endogenous (the loser *set* changes with the rule) and are not the primary contrast.

| population | n_candidates | paradigm | total_welfare | minority_welfare | majority_welfare | min_utility | p10_utility | gini_utility | lambda_winner | loser_share | loser_welfare | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | party | 0.2312 ± 0.0051 | 0.2541 ± 0.0601 | 0.2084 ± 0.0589 | 0.1225 ± 0.0182 | 0.1497 ± 0.0179 | 0.1701 ± 0.0420 | 0.4305 ± 0.1893 | 0.4978 ± 0.0022 | 0.1699 ± 0.0155 | 0.1222 ± 0.0341 |
| two_cluster | 8 | individual | 0.2310 ± 0.0063 | 0.2330 ± 0.0663 | 0.2289 ± 0.0658 | 0.1195 ± 0.0209 | 0.1475 ± 0.0202 | 0.1749 ± 0.0482 | 0.4398 ± 0.1954 | 0.7008 ± 0.0766 | 0.2010 ± 0.0162 | 0.0980 ± 0.0411 |
| two_cluster | 8 | score | 0.2295 ± 0.0064 | 0.2290 ± 0.0417 | 0.2300 ± 0.0404 | 0.1414 ± 0.0136 | 0.1751 ± 0.0153 | 0.1128 ± 0.0373 | 0.2394 ± 0.1247 | 0.8198 ± 0.0849 | 0.2209 ± 0.0095 | 0.0400 ± 0.0301 |
| two_cluster | 8 | latent_match | 0.2289 ± 0.0065 | 0.2296 ± 0.0365 | 0.2281 ± 0.0361 | 0.1447 ± 0.0124 | 0.1793 ± 0.0129 | 0.1029 ± 0.0323 | 0.2047 ± 0.1040 | 0.8395 ± 0.0835 | 0.2226 ± 0.0082 | 0.0327 ± 0.0248 |

### Allocation baselines (not selection treatments)

Random-winner floor and utilitarian / egalitarian brackets on the same draws. Normalized columns live on every row of `trials.csv`.

| population | n_candidates | paradigm | total_welfare | minority_welfare | majority_welfare | min_utility | p10_utility | gini_utility | lambda_winner | loser_share | loser_welfare | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | random_winner | 0.2314 ± 0.0090 | 0.2315 ± 0.0726 | 0.2314 ± 0.0722 | 0.1129 ± 0.0277 | 0.1437 ± 0.0281 | 0.1842 ± 0.0698 | 0.5140 ± 0.2258 | 0.8747 ± 0.1038 | 0.2196 ± 0.0153 | 0.0980 ± 0.0599 |
| two_cluster | 8 | utilitarian | 0.2678 ± 0.0047 | 0.2697 ± 0.2092 | 0.2660 ± 0.2109 | 0.0065 ± 0.0047 | 0.0266 ± 0.0164 | 0.4958 ± 0.0439 | nan | nan | nan | nan |
| two_cluster | 8 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |

### Constitutional cap (not a selection treatment)

`constrained_individual` reuses the `individual` election and caps `λ_effective`. It manipulates the outcome function, not the selection rule, and is excluded from hypothesis contrasts and from 'which rule wins' language.

| population | n_candidates | paradigm | total_welfare | minority_welfare | majority_welfare | min_utility | p10_utility | gini_utility | lambda_winner | loser_share | loser_welfare | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | constrained_individual | 0.2309 ± 0.0058 | 0.2316 ± 0.0503 | 0.2303 ± 0.0501 | 0.1363 ± 0.0094 | 0.1653 ± 0.0067 | 0.1391 ± 0.0236 | 0.4398 ± 0.1954 | 0.7008 ± 0.0766 | 0.2077 ± 0.0107 | 0.0758 ± 0.0252 |

### Mean winner allocations

| population | n_candidates | paradigm | public_good | majority_pork | minority_pork | prestige | periphery_buffer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | party | 0.2669 | 0.2174 | 0.3248 | 0.0949 | 0.0961 |
| two_cluster | 8 | individual | 0.2646 | 0.2663 | 0.2764 | 0.0958 | 0.0970 |
| two_cluster | 8 | score | 0.2790 | 0.2611 | 0.2585 | 0.1012 | 0.1003 |
| two_cluster | 8 | latent_match | 0.2815 | 0.2549 | 0.2584 | 0.1024 | 0.1028 |
| two_cluster | 8 | random_winner | 0.2623 | 0.2741 | 0.2742 | 0.1001 | 0.0893 |
| two_cluster | 8 | utilitarian | 0.0440 | 0.4720 | 0.4840 | 0.0000 | 0.0000 |
| two_cluster | 8 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |

## Paired contrasts vs party

Same-trial differences. Primary endpoints carry Holm-adjusted p-values. Threshold constants are not used as a verdict machine.

### two_cluster, 8

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | -0.0210 | -0.0320 | -0.0101 | 0.0028 | 0.0110 | -0.2389 |
| individual | total_welfare | -0.0002 | -0.0010 | 0.0005 | 0.8308 | 1.0000 | -0.0434 |
| individual | lambda_winner | 0.0093 | -0.0166 | 0.0351 | 0.6653 | 1.0000 | 0.0447 |
| score | minority_welfare | -0.0250 | -0.0344 | -0.0157 | 0.0000 | 0.0000 | -0.3339 |
| score | total_welfare | -0.0017 | -0.0027 | -0.0008 | 0.0116 | 0.0349 | -0.2237 |
| score | lambda_winner | -0.1911 | -0.2151 | -0.1671 | 0.0000 | 0.0000 | -0.9924 |
| latent_match | minority_welfare | -0.0244 | -0.0333 | -0.0155 | 0.0000 | 0.0000 | -0.3403 |
| latent_match | total_welfare | -0.0024 | -0.0034 | -0.0014 | 0.0002 | 0.0012 | -0.3002 |
| latent_match | lambda_winner | -0.2258 | -0.2485 | -0.2031 | 0.0000 | 0.0000 | -1.2388 |

- `individual` vs `party` on `minority_welfare`: Δ = -0.02102 (95% CI [-0.03198, -0.01006]), paired Cohen's d = -0.239, Wilcoxon p = 0.002755, Holm-adjusted p = 0.01102.
- `individual` vs `party` on `total_welfare`: Δ = -0.00025 (95% CI [-0.00096, +0.00046]), paired Cohen's d = -0.043, Wilcoxon p = 0.8308, Holm-adjusted p = 1.
- `individual` vs `party` on `lambda_winner`: Δ = +0.00927 (95% CI [-0.01656, +0.03511]), paired Cohen's d = +0.045, Wilcoxon p = 0.6653, Holm-adjusted p = 1.
- `score` vs `party` on `minority_welfare`: Δ = -0.02504 (95% CI [-0.03438, -0.01570]), paired Cohen's d = -0.334, Wilcoxon p = 2.922e-07, Holm-adjusted p = 2.046e-06.
- `score` vs `party` on `total_welfare`: Δ = -0.00172 (95% CI [-0.00268, -0.00076]), paired Cohen's d = -0.224, Wilcoxon p = 0.01164, Holm-adjusted p = 0.03492.
- `score` vs `party` on `lambda_winner`: Δ = -0.19108 (95% CI [-0.21506, -0.16710]), paired Cohen's d = -0.992, Wilcoxon p = 1.113e-30, Holm-adjusted p = 8.902e-30.
- `latent_match` vs `party` on `minority_welfare`: Δ = -0.02441 (95% CI [-0.03335, -0.01547]), paired Cohen's d = -0.340, Wilcoxon p = 3.648e-07, Holm-adjusted p = 2.189e-06.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00238 (95% CI [-0.00337, -0.00139]), paired Cohen's d = -0.300, Wilcoxon p = 0.0002343, Holm-adjusted p = 0.001171.
- `latent_match` vs `party` on `lambda_winner`: Δ = -0.22577 (95% CI [-0.24848, -0.20307]), paired Cohen's d = -1.239, Wilcoxon p = 1.063e-36, Holm-adjusted p = 9.567e-36.

Mushy-bloc check (fixed partition):
- election-endogenous gap shrank while minority-cluster welfare did not rise for `individual`, `score`, `latent_match`.

## Limitations

- Voters and candidates live in a stylized 5-dimensional project space; real preference structures are higher-dimensional and partly unobservable.
- The default one-shot cell has exogenous λ. It compares selection rules with drawn types; it does not test citizen-candidate entry or core- vs swing-voter targeting. Use `--mechanism reelection` for an incentive-based λ cell (still not an entry model).
- Party structure is idealized as two brands at cluster means with loyal nomination; real parties select through noisy primaries.
- Supporter definitions differ across paradigms by design. Primary welfare contrasts therefore use the fixed cluster partition, not the election-endogenous loser set.
- `--lambda-correlated` is a researcher degree of freedom that can flip λ rankings. It is labeled a robustness appendix whenever it is the condition being reported.

## Reproduce

```
python run_experiment.py --trials 250 --voters 400 --candidates 8 --population two_cluster --seed 0 --lambda-correlated --include-constrained --no-persist-ballots --out '{run_dir}'
```

Deterministic given identical parameters and seed (per-trial streams are derived from `numpy.random.default_rng([seed, population, candidates, trial])`).
