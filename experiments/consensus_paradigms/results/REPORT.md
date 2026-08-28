# Political consensus experiment — auto-generated report

Holding λ's marginal distribution fixed and holding the set of people fixed, do individual-centered ballot formats change the winner's allocation in a way that raises minority-cluster welfare and/or total welfare relative to party? Election-endogenous loser welfare is reported but is not the primary contrast: 'supporters' is a different estimand under every rule.

## Methods

- **Population**: 400 voters per trial, latent preference vectors in R^5 drawn from cluster centers plus Gaussian noise (σ = 0.35); benefit vectors are a softmax (temperature 0.55) of preferences, so rows are nonnegative and sum to 1. Population types run here: `two_cluster`.
- **Projects** (generator tags, not winner-relative): `public_good`, `majority_pork`, `minority_pork`, `prestige`, `periphery_buffer`.
- **Candidates**: 8 per trial (sweeps may vary this), platforms drawn from the same cluster structure; loyalty trait λ ~ Beta(2.2, 2.2), independent of the platform and of anything voters see. No rule reads λ, so E[λ_winner] is the Beta mean under every paradigm by construction — not an empirical finding.
- **Allocation**: `directed = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`, then `raw = 0.72·directed + 0.28·simplex(clip(platform, 0, ∞))`. The platform is renormalized before mixing so the weights are a convex combination of matched units. Zero supporters ⇒ all-voter direction only. A platform with no positive mass is dropped.
- **Paradigms (selection treatments)**: `party`, `individual`, `score`, `latent_match`. `constrained_individual` is a constitutional λ cap, not a voting rule; it is not in this run.
- **Voting**: `sincere`. Sincere plurality is the default baseline; `abandon_trailing` is a Duverger-style heuristic on individual (and the cap overlay).
- **Mechanism**: `oneshot`. The default one-shot cell is a selection-rule comparison with exogenous types, not a test of loyalty formation.
- **Trials**: 250 per cell, base seed 0; every paradigm sees the identical population and candidate slate within a trial.
- **Primary endpoints**: Δ minority-cluster welfare and Δ total welfare vs party. Holm correction across that family. Wilcoxon signed-rank on paired trial-level differences; 95% CIs are Student-t; effect size is paired Cohen's d.
- **Fixed partition**: welfare is reported on generator clusters (PCA split for `one_cluster`). Election-endogenous supporter/loser numbers are kept and labeled as such; they are not the primary contrast. `rural_town` party `loser_share` equals the minority bloc size by construction (~0.30) and is not a treatment effect.
- **Baselines** (same population + candidate draw): `random_winner` (uniform candidate, nearest-pref supporters), `utilitarian` (simplex vertex maximizing mean utility), `egalitarian` (maximin LP). Normalized welfare is `(metric − random) / (utilitarian − random)` when the denominator is nonzero.
- **Audit**: synthetic ballots, supporter masks, and cluster ids are written under `private/` when `--persist-ballots` is on (the default). They are not a privacy claim and they are not notarized. Official record: `summary.csv`, `trials.csv` aggregates, `contrasts.csv`.

## Results

Mean ± std over trials for **selection rules**. `loser_share` / `loser_welfare` / `gap` are election-endogenous (the loser *set* changes with the rule) and are not the primary contrast.

| population | n_candidates | paradigm | total_welfare | minority_welfare | majority_welfare | min_utility | p10_utility | gini_utility | lambda_winner | loser_share | loser_welfare | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | party | 0.2313 ± 0.0051 | 0.2549 ± 0.0653 | 0.2076 ± 0.0643 | 0.1150 ± 0.0204 | 0.1427 ± 0.0190 | 0.1841 ± 0.0424 | 0.5145 ± 0.2211 | 0.4978 ± 0.0022 | 0.1644 ± 0.0161 | 0.1331 ± 0.0342 |
| two_cluster | 8 | individual | 0.2308 ± 0.0066 | 0.2306 ± 0.0697 | 0.2311 ± 0.0692 | 0.1135 ± 0.0213 | 0.1422 ± 0.0196 | 0.1851 ± 0.0441 | 0.5111 ± 0.2155 | 0.7008 ± 0.0766 | 0.1993 ± 0.0156 | 0.1031 ± 0.0367 |
| two_cluster | 8 | score | 0.2287 ± 0.0076 | 0.2269 ± 0.0537 | 0.2305 ± 0.0527 | 0.1275 ± 0.0203 | 0.1593 ± 0.0218 | 0.1425 ± 0.0479 | 0.5080 ± 0.2230 | 0.8198 ± 0.0849 | 0.2174 ± 0.0111 | 0.0534 ± 0.0344 |
| two_cluster | 8 | latent_match | 0.2279 ± 0.0078 | 0.2287 ± 0.0503 | 0.2270 ± 0.0498 | 0.1305 ± 0.0197 | 0.1621 ± 0.0218 | 0.1350 ± 0.0480 | 0.5098 ± 0.2210 | 0.8395 ± 0.0835 | 0.2189 ± 0.0107 | 0.0472 ± 0.0319 |

### Allocation baselines (not selection treatments)

Random-winner floor and utilitarian / egalitarian brackets on the same draws. Normalized columns live on every row of `trials.csv`.

| population | n_candidates | paradigm | total_welfare | minority_welfare | majority_welfare | min_utility | p10_utility | gini_utility | lambda_winner | loser_share | loser_welfare | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | random_winner | 0.2309 ± 0.0088 | 0.2298 ± 0.0688 | 0.2321 ± 0.0693 | 0.1139 ± 0.0244 | 0.1449 ± 0.0240 | 0.1806 ± 0.0584 | 0.5266 ± 0.2261 | 0.8747 ± 0.1038 | 0.2189 ± 0.0154 | 0.0941 ± 0.0489 |
| two_cluster | 8 | utilitarian | 0.2678 ± 0.0047 | 0.2697 ± 0.2092 | 0.2660 ± 0.2109 | 0.0065 ± 0.0047 | 0.0266 ± 0.0164 | 0.4958 ± 0.0439 | nan | nan | nan | nan |
| two_cluster | 8 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |

### Mean winner allocations

| population | n_candidates | paradigm | public_good | majority_pork | minority_pork | prestige | periphery_buffer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 8 | party | 0.2669 | 0.2153 | 0.3269 | 0.0949 | 0.0960 |
| two_cluster | 8 | individual | 0.2649 | 0.2710 | 0.2702 | 0.0964 | 0.0975 |
| two_cluster | 8 | score | 0.2822 | 0.2601 | 0.2511 | 0.1040 | 0.1026 |
| two_cluster | 8 | latent_match | 0.2865 | 0.2493 | 0.2530 | 0.1053 | 0.1059 |
| two_cluster | 8 | random_winner | 0.2667 | 0.2733 | 0.2679 | 0.1014 | 0.0908 |
| two_cluster | 8 | utilitarian | 0.0440 | 0.4720 | 0.4840 | 0.0000 | 0.0000 |
| two_cluster | 8 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |

## Paired contrasts vs party

Same-trial differences. Primary endpoints carry Holm-adjusted p-values. Threshold constants are not used as a verdict machine.

λ_winner is **not** a primary endpoint in this cell. Voters never see λ, so a flat λ profile is implied by the generator.

### two_cluster, 8

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | -0.0244 | -0.0361 | -0.0127 | 0.0006 | 0.0017 | -0.2592 |
| individual | total_welfare | -0.0004 | -0.0012 | 0.0003 | 0.8111 | 0.8111 | -0.0731 |
| score | minority_welfare | -0.0280 | -0.0387 | -0.0174 | 0.0000 | 0.0000 | -0.3268 |
| score | total_welfare | -0.0025 | -0.0036 | -0.0014 | 0.0010 | 0.0019 | -0.2877 |
| latent_match | minority_welfare | -0.0262 | -0.0364 | -0.0160 | 0.0000 | 0.0000 | -0.3206 |
| latent_match | total_welfare | -0.0034 | -0.0045 | -0.0023 | 0.0000 | 0.0000 | -0.3729 |

- `individual` vs `party` on `minority_welfare`: Δ = -0.02440 (95% CI [-0.03612, -0.01267]), paired Cohen's d = -0.259, Wilcoxon p = 0.0005735, Holm-adjusted p = 0.001721.
- `individual` vs `party` on `total_welfare`: Δ = -0.00043 (95% CI [-0.00117, +0.00031]), paired Cohen's d = -0.073, Wilcoxon p = 0.8111, Holm-adjusted p = 0.8111.
- `score` vs `party` on `minority_welfare`: Δ = -0.02805 (95% CI [-0.03874, -0.01736]), paired Cohen's d = -0.327, Wilcoxon p = 3.452e-07, Holm-adjusted p = 2.071e-06.
- `score` vs `party` on `total_welfare`: Δ = -0.00253 (95% CI [-0.00363, -0.00144]), paired Cohen's d = -0.288, Wilcoxon p = 0.000965, Holm-adjusted p = 0.00193.
- `latent_match` vs `party` on `minority_welfare`: Δ = -0.02624 (95% CI [-0.03643, -0.01605]), paired Cohen's d = -0.321, Wilcoxon p = 4.869e-07, Holm-adjusted p = 2.434e-06.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00340 (95% CI [-0.00453, -0.00226]), paired Cohen's d = -0.373, Wilcoxon p = 4.637e-06, Holm-adjusted p = 1.855e-05.

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
python run_experiment.py --trials 250 --voters 400 --candidates 8 --population two_cluster --seed 0 --no-persist-ballots --out {run_dir}
```

Deterministic given identical parameters and seed (per-trial streams are derived from `numpy.random.default_rng([seed, population, candidates, trial])`).
