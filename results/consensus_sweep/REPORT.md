# Political consensus experiment — auto-generated report

Holding λ's marginal distribution fixed and holding the set of people fixed, do individual-centered ballot formats change the winner's allocation in a way that raises minority-cluster welfare and/or total welfare relative to party? Election-endogenous loser welfare is reported but is not the primary contrast: 'supporters' is a different estimand under every rule.

## Methods

- **Population**: 400 voters per trial, latent preference vectors in R^5 drawn from cluster centers plus Gaussian noise (σ = 0.35); benefit vectors are a softmax (temperature 0.55) of preferences, so rows are nonnegative and sum to 1. Population types run here: `two_cluster`, `three_cluster`, `rural_town`.
- **Projects** (generator tags, not winner-relative): `public_good`, `majority_pork`, `minority_pork`, `prestige`, `periphery_buffer`.
- **Candidates**: 6 per trial (sweeps may vary this), platforms drawn from the same cluster structure; loyalty trait λ ~ Beta(2.2, 2.2), independent of the platform and of anything voters see. No rule reads λ, so E[λ_winner] is the Beta mean under every paradigm by construction — not an empirical finding.
- **Allocation**: `directed = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`, then `raw = 0.72·directed + 0.28·simplex(clip(platform, 0, ∞))`. The platform is renormalized before mixing so the weights are a convex combination of matched units. Zero supporters ⇒ all-voter direction only. A platform with no positive mass is dropped.
- **Paradigms (selection treatments)**: `party`, `individual`, `score`, `latent_match`. `constrained_individual` is a constitutional λ cap, not a voting rule; it is not in this run.
- **Voting**: `sincere`. Sincere plurality is the default baseline; `abandon_trailing` is a Duverger-style heuristic on individual (and the cap overlay).
- **Mechanism**: `oneshot`. The default one-shot cell is a selection-rule comparison with exogenous types, not a test of loyalty formation.
- **Trials**: 100 per cell, base seed 0; every paradigm sees the identical population and candidate slate within a trial.
- **Primary endpoints**: Δ minority-cluster welfare and Δ total welfare vs party. Holm correction across that family. Wilcoxon signed-rank on paired trial-level differences; 95% CIs are Student-t; effect size is paired Cohen's d.
- **Fixed partition**: welfare is reported on generator clusters (PCA split for `one_cluster`). Election-endogenous supporter/loser numbers are kept and labeled as such; they are not the primary contrast. `rural_town` party `loser_share` equals the minority bloc size by construction (~0.30) and is not a treatment effect.
- **Baselines** (same population + candidate draw): `random_winner` (uniform candidate, nearest-pref supporters), `utilitarian` (simplex vertex maximizing mean utility), `egalitarian` (maximin LP). Normalized welfare is `(metric − random) / (utilitarian − random)` when the denominator is nonzero.
- **Audit**: synthetic ballots, supporter masks, and cluster ids are written under `private/` when `--persist-ballots` is on (the default). They are not a privacy claim and they are not notarized. Official record: `summary.csv`, `trials.csv` aggregates, `contrasts.csv`.

## Results

Mean ± std over trials for **selection rules**. `loser_share` / `loser_welfare` / `gap` are election-endogenous (the loser *set* changes with the rule) and are not the primary contrast.

| population | n_candidates | paradigm | total_welfare | minority_welfare | majority_welfare | min_utility | p10_utility | gini_utility | lambda_winner | loser_share | loser_welfare | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 6 | party | 0.2311 ± 0.0052 | 0.2555 ± 0.0596 | 0.2067 ± 0.0607 | 0.1204 ± 0.0169 | 0.1481 ± 0.0168 | 0.1740 ± 0.0391 | 0.4580 ± 0.1789 | 0.4978 ± 0.0021 | 0.1683 ± 0.0151 | 0.1251 ± 0.0316 |
| two_cluster | 6 | individual | 0.2307 ± 0.0062 | 0.2149 ± 0.0663 | 0.2466 ± 0.0682 | 0.1152 ± 0.0217 | 0.1432 ± 0.0199 | 0.1836 ± 0.0462 | 0.5114 ± 0.2119 | 0.6348 ± 0.0771 | 0.1887 ± 0.0198 | 0.1117 ± 0.0418 |
| two_cluster | 6 | score | 0.2298 ± 0.0065 | 0.2390 ± 0.0508 | 0.2206 ± 0.0513 | 0.1276 ± 0.0201 | 0.1610 ± 0.0225 | 0.1398 ± 0.0480 | 0.5215 ± 0.2208 | 0.7766 ± 0.1036 | 0.2151 ± 0.0136 | 0.0570 ± 0.0354 |
| two_cluster | 6 | latent_match | 0.2293 ± 0.0070 | 0.2431 ± 0.0484 | 0.2155 ± 0.0480 | 0.1305 ± 0.0193 | 0.1630 ± 0.0218 | 0.1349 ± 0.0475 | 0.5012 ± 0.2189 | 0.7709 ± 0.1146 | 0.2140 ± 0.0156 | 0.0550 ± 0.0367 |
| two_cluster | 8 | party | 0.2314 ± 0.0050 | 0.2601 ± 0.0625 | 0.2027 ± 0.0617 | 0.1166 ± 0.0199 | 0.1436 ± 0.0184 | 0.1820 ± 0.0421 | 0.5052 ± 0.2108 | 0.4979 ± 0.0020 | 0.1653 ± 0.0159 | 0.1317 ± 0.0342 |
| two_cluster | 8 | individual | 0.2310 ± 0.0068 | 0.2323 ± 0.0678 | 0.2297 ± 0.0684 | 0.1158 ± 0.0208 | 0.1441 ± 0.0193 | 0.1806 ± 0.0433 | 0.4979 ± 0.2127 | 0.7002 ± 0.0752 | 0.2007 ± 0.0139 | 0.0997 ± 0.0356 |
| two_cluster | 8 | score | 0.2298 ± 0.0071 | 0.2249 ± 0.0539 | 0.2346 ± 0.0531 | 0.1268 ± 0.0207 | 0.1593 ± 0.0219 | 0.1434 ± 0.0475 | 0.5226 ± 0.2264 | 0.8064 ± 0.0941 | 0.2165 ± 0.0122 | 0.0585 ± 0.0361 |
| two_cluster | 8 | latent_match | 0.2282 ± 0.0075 | 0.2251 ± 0.0525 | 0.2313 ± 0.0500 | 0.1307 ± 0.0182 | 0.1617 ± 0.0214 | 0.1373 ± 0.0492 | 0.5193 ± 0.2186 | 0.8312 ± 0.0938 | 0.2180 ± 0.0112 | 0.0513 ± 0.0341 |
| two_cluster | 12 | party | 0.2309 ± 0.0047 | 0.2584 ± 0.0598 | 0.2034 ± 0.0589 | 0.1175 ± 0.0218 | 0.1462 ± 0.0189 | 0.1760 ± 0.0402 | 0.4815 ± 0.2292 | 0.4976 ± 0.0023 | 0.1675 ± 0.0156 | 0.1262 ± 0.0316 |
| two_cluster | 12 | individual | 0.2307 ± 0.0051 | 0.2182 ± 0.0679 | 0.2433 ± 0.0665 | 0.1156 ± 0.0220 | 0.1436 ± 0.0201 | 0.1817 ± 0.0474 | 0.4972 ± 0.2192 | 0.7598 ± 0.0648 | 0.2072 ± 0.0130 | 0.0945 ± 0.0412 |
| two_cluster | 12 | score | 0.2290 ± 0.0065 | 0.2335 ± 0.0528 | 0.2245 ± 0.0506 | 0.1257 ± 0.0198 | 0.1584 ± 0.0194 | 0.1416 ± 0.0425 | 0.4838 ± 0.2177 | 0.8566 ± 0.0743 | 0.2209 ± 0.0088 | 0.0496 ± 0.0255 |
| two_cluster | 12 | latent_match | 0.2281 ± 0.0070 | 0.2319 ± 0.0477 | 0.2242 ± 0.0469 | 0.1283 ± 0.0191 | 0.1621 ± 0.0198 | 0.1315 ± 0.0415 | 0.4808 ± 0.2047 | 0.8756 ± 0.0790 | 0.2215 ± 0.0106 | 0.0432 ± 0.0248 |
| three_cluster | 6 | party | 0.2287 ± 0.0050 | 0.2258 ± 0.0069 | 0.2231 ± 0.0566 | 0.1245 ± 0.0209 | 0.1557 ± 0.0194 | 0.1456 ± 0.0407 | 0.4683 ± 0.2328 | 0.4909 ± 0.0079 | 0.1805 ± 0.0164 | 0.0947 ± 0.0318 |
| three_cluster | 6 | individual | 0.2294 ± 0.0053 | 0.2251 ± 0.0076 | 0.2329 ± 0.0564 | 0.1262 ± 0.0207 | 0.1586 ± 0.0213 | 0.1411 ± 0.0484 | 0.4674 ± 0.2082 | 0.6517 ± 0.0722 | 0.2035 ± 0.0151 | 0.0749 ± 0.0422 |
| three_cluster | 6 | score | 0.2275 ± 0.0070 | 0.2268 ± 0.0089 | 0.2267 ± 0.0414 | 0.1380 ± 0.0186 | 0.1727 ± 0.0200 | 0.1066 ± 0.0443 | 0.4893 ± 0.2227 | 0.7382 ± 0.1008 | 0.2176 ± 0.0121 | 0.0345 ± 0.0317 |
| three_cluster | 6 | latent_match | 0.2273 ± 0.0069 | 0.2269 ± 0.0086 | 0.2297 ± 0.0378 | 0.1404 ± 0.0170 | 0.1752 ± 0.0182 | 0.1010 ± 0.0393 | 0.4741 ± 0.2188 | 0.7339 ± 0.1087 | 0.2180 ± 0.0120 | 0.0309 ± 0.0268 |
| three_cluster | 8 | party | 0.2289 ± 0.0045 | 0.2264 ± 0.0066 | 0.2376 ± 0.0600 | 0.1207 ± 0.0180 | 0.1516 ± 0.0156 | 0.1533 ± 0.0324 | 0.4995 ± 0.2144 | 0.4912 ± 0.0063 | 0.1776 ± 0.0129 | 0.1008 ± 0.0249 |
| three_cluster | 8 | individual | 0.2288 ± 0.0056 | 0.2253 ± 0.0070 | 0.2221 ± 0.0603 | 0.1212 ± 0.0220 | 0.1540 ± 0.0206 | 0.1489 ± 0.0455 | 0.4791 ± 0.2344 | 0.7048 ± 0.0553 | 0.2059 ± 0.0113 | 0.0777 ± 0.0391 |
| three_cluster | 8 | score | 0.2274 ± 0.0063 | 0.2267 ± 0.0087 | 0.2242 ± 0.0397 | 0.1377 ± 0.0153 | 0.1732 ± 0.0186 | 0.1060 ± 0.0386 | 0.5138 ± 0.2185 | 0.7908 ± 0.0749 | 0.2201 ± 0.0085 | 0.0307 ± 0.0258 |
| three_cluster | 8 | latent_match | 0.2268 ± 0.0067 | 0.2265 ± 0.0085 | 0.2253 ± 0.0380 | 0.1388 ± 0.0155 | 0.1746 ± 0.0183 | 0.1027 ± 0.0388 | 0.5250 ± 0.2167 | 0.7985 ± 0.0794 | 0.2202 ± 0.0094 | 0.0287 ± 0.0260 |
| three_cluster | 12 | party | 0.2293 ± 0.0044 | 0.2264 ± 0.0059 | 0.2225 ± 0.0628 | 0.1193 ± 0.0171 | 0.1492 ± 0.0143 | 0.1590 ± 0.0303 | 0.5127 ± 0.2071 | 0.4901 ± 0.0082 | 0.1754 ± 0.0120 | 0.1058 ± 0.0225 |
| three_cluster | 12 | individual | 0.2288 ± 0.0061 | 0.2240 ± 0.0088 | 0.2294 ± 0.0647 | 0.1175 ± 0.0238 | 0.1485 ± 0.0201 | 0.1633 ± 0.0459 | 0.5147 ± 0.2144 | 0.7839 ± 0.0503 | 0.2099 ± 0.0107 | 0.0857 ± 0.0423 |
| three_cluster | 12 | score | 0.2256 ± 0.0082 | 0.2260 ± 0.0099 | 0.2252 ± 0.0357 | 0.1422 ± 0.0173 | 0.1776 ± 0.0174 | 0.0940 ± 0.0379 | 0.5047 ± 0.2214 | 0.8646 ± 0.0482 | 0.2221 ± 0.0075 | 0.0232 ± 0.0234 |
| three_cluster | 12 | latent_match | 0.2250 ± 0.0083 | 0.2258 ± 0.0107 | 0.2221 ± 0.0322 | 0.1428 ± 0.0175 | 0.1793 ± 0.0165 | 0.0901 ± 0.0352 | 0.5249 ± 0.2244 | 0.8714 ± 0.0428 | 0.2222 ± 0.0075 | 0.0200 ± 0.0201 |
| rural_town | 6 | party | 0.2655 ± 0.0084 | 0.1519 ± 0.0108 | 0.3142 ± 0.0148 | 0.0968 ± 0.0159 | 0.1365 ± 0.0110 | 0.1819 ± 0.0227 | 0.5229 ± 0.2159 | 0.3020 ± 0.0037 | 0.1522 ± 0.0108 | 0.1623 ± 0.0237 |
| rural_town | 6 | individual | 0.2533 ± 0.0234 | 0.1773 ± 0.0484 | 0.2859 ± 0.0530 | 0.1072 ± 0.0264 | 0.1463 ± 0.0240 | 0.1613 ± 0.0496 | 0.5322 ± 0.2088 | 0.6423 ± 0.0926 | 0.2233 ± 0.0256 | 0.0811 ± 0.0462 |
| rural_town | 6 | score | 0.2588 ± 0.0141 | 0.1606 ± 0.0206 | 0.3009 ± 0.0269 | 0.1059 ± 0.0211 | 0.1448 ± 0.0184 | 0.1634 ± 0.0407 | 0.5454 ± 0.2136 | 0.6953 ± 0.1271 | 0.2372 ± 0.0210 | 0.0601 ± 0.0454 |
| rural_town | 6 | latent_match | 0.2550 ± 0.0155 | 0.1680 ± 0.0269 | 0.2923 ± 0.0317 | 0.1111 ± 0.0241 | 0.1513 ± 0.0238 | 0.1499 ± 0.0495 | 0.5580 ± 0.2115 | 0.7161 ± 0.1340 | 0.2372 ± 0.0209 | 0.0500 ± 0.0461 |
| rural_town | 8 | party | 0.2651 ± 0.0078 | 0.1539 ± 0.0104 | 0.3128 ± 0.0142 | 0.1002 ± 0.0147 | 0.1389 ± 0.0112 | 0.1793 ± 0.0217 | 0.4754 ± 0.2155 | 0.3014 ± 0.0034 | 0.1540 ± 0.0104 | 0.1591 ± 0.0232 |
| rural_town | 8 | individual | 0.2488 ± 0.0265 | 0.1911 ± 0.0581 | 0.2735 ± 0.0618 | 0.1104 ± 0.0230 | 0.1467 ± 0.0203 | 0.1635 ± 0.0424 | 0.4884 ± 0.2083 | 0.7116 ± 0.0583 | 0.2271 ± 0.0296 | 0.0764 ± 0.0430 |
| rural_town | 8 | score | 0.2593 ± 0.0122 | 0.1631 ± 0.0199 | 0.3005 ± 0.0243 | 0.1089 ± 0.0171 | 0.1473 ± 0.0159 | 0.1620 ± 0.0340 | 0.4795 ± 0.1900 | 0.7723 ± 0.0897 | 0.2475 ± 0.0106 | 0.0475 ± 0.0333 |
| rural_town | 8 | latent_match | 0.2558 ± 0.0127 | 0.1683 ± 0.0223 | 0.2934 ± 0.0261 | 0.1141 ± 0.0188 | 0.1521 ± 0.0176 | 0.1514 ± 0.0371 | 0.4778 ± 0.2009 | 0.7889 ± 0.0979 | 0.2471 ± 0.0111 | 0.0367 ± 0.0307 |
| rural_town | 12 | party | 0.2664 ± 0.0079 | 0.1507 ± 0.0109 | 0.3160 ± 0.0144 | 0.0972 ± 0.0155 | 0.1355 ± 0.0115 | 0.1851 ± 0.0225 | 0.5225 ± 0.2196 | 0.3017 ± 0.0029 | 0.1509 ± 0.0108 | 0.1654 ± 0.0237 |
| rural_town | 12 | individual | 0.2505 ± 0.0278 | 0.1845 ± 0.0611 | 0.2788 ± 0.0650 | 0.1054 ± 0.0240 | 0.1401 ± 0.0202 | 0.1749 ± 0.0452 | 0.5258 ± 0.2186 | 0.7757 ± 0.0627 | 0.2317 ± 0.0309 | 0.0818 ± 0.0429 |
| rural_town | 12 | score | 0.2609 ± 0.0112 | 0.1611 ± 0.0145 | 0.3037 ± 0.0198 | 0.1067 ± 0.0154 | 0.1454 ± 0.0137 | 0.1638 ± 0.0307 | 0.5071 ± 0.2095 | 0.8293 ± 0.0811 | 0.2517 ± 0.0105 | 0.0458 ± 0.0306 |
| rural_town | 12 | latent_match | 0.2570 ± 0.0138 | 0.1669 ± 0.0194 | 0.2956 ± 0.0257 | 0.1110 ± 0.0193 | 0.1502 ± 0.0165 | 0.1521 ± 0.0361 | 0.5205 ± 0.2264 | 0.8514 ± 0.0773 | 0.2504 ± 0.0123 | 0.0368 ± 0.0276 |

### Allocation baselines (not selection treatments)

Random-winner floor and utilitarian / egalitarian brackets on the same draws. Normalized columns live on every row of `trials.csv`.

| population | n_candidates | paradigm | total_welfare | minority_welfare | majority_welfare | min_utility | p10_utility | gini_utility | lambda_winner | loser_share | loser_welfare | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 6 | random_winner | 0.2311 ± 0.0080 | 0.2272 ± 0.0660 | 0.2350 ± 0.0664 | 0.1166 ± 0.0210 | 0.1475 ± 0.0224 | 0.1748 ± 0.0537 | 0.5086 ± 0.2109 | 0.8185 ± 0.1251 | 0.2120 ± 0.0215 | 0.0927 ± 0.0475 |
| two_cluster | 6 | utilitarian | 0.2680 ± 0.0044 | 0.2247 ± 0.2060 | 0.3113 ± 0.2061 | 0.0063 ± 0.0049 | 0.0264 ± 0.0151 | 0.4955 ± 0.0420 | nan | nan | nan | nan |
| two_cluster | 6 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |
| two_cluster | 8 | random_winner | 0.2300 ± 0.0094 | 0.2283 ± 0.0679 | 0.2317 ± 0.0664 | 0.1161 ± 0.0245 | 0.1472 ± 0.0242 | 0.1747 ± 0.0597 | 0.5241 ± 0.2306 | 0.8817 ± 0.1024 | 0.2192 ± 0.0140 | 0.0875 ± 0.0464 |
| two_cluster | 8 | utilitarian | 0.2682 ± 0.0046 | 0.2546 ± 0.2098 | 0.2818 ± 0.2115 | 0.0065 ± 0.0047 | 0.0265 ± 0.0159 | 0.4956 ± 0.0422 | nan | nan | nan | nan |
| two_cluster | 8 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |
| two_cluster | 12 | random_winner | 0.2307 ± 0.0103 | 0.2333 ± 0.0667 | 0.2280 ± 0.0658 | 0.1182 ± 0.0227 | 0.1499 ± 0.0239 | 0.1713 ± 0.0597 | 0.4619 ± 0.2203 | 0.9247 ± 0.0672 | 0.2231 ± 0.0115 | 0.0944 ± 0.0540 |
| two_cluster | 12 | utilitarian | 0.2676 ± 0.0049 | 0.2825 ± 0.2041 | 0.2527 ± 0.2040 | 0.0078 ± 0.0062 | 0.0306 ± 0.0231 | 0.4847 ± 0.0605 | nan | nan | nan | nan |
| two_cluster | 12 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |
| three_cluster | 6 | random_winner | 0.2296 ± 0.0077 | 0.2237 ± 0.0104 | 0.2173 ± 0.0619 | 0.1188 ± 0.0219 | 0.1517 ± 0.0224 | 0.1586 ± 0.0541 | 0.5199 ± 0.2181 | 0.8210 ± 0.1162 | 0.2148 ± 0.0130 | 0.0843 ± 0.0486 |
| three_cluster | 6 | utilitarian | 0.2648 ± 0.0062 | 0.2998 ± 0.0435 | 0.2441 ± 0.0952 | 0.0250 ± 0.0129 | 0.0890 ± 0.0322 | 0.3286 ± 0.0784 | nan | nan | nan | nan |
| three_cluster | 6 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |
| three_cluster | 8 | random_winner | 0.2280 ± 0.0089 | 0.2239 ± 0.0106 | 0.2351 ± 0.0605 | 0.1198 ± 0.0228 | 0.1557 ± 0.0238 | 0.1477 ± 0.0552 | 0.5054 ± 0.1991 | 0.8805 ± 0.0934 | 0.2197 ± 0.0106 | 0.0767 ± 0.0483 |
| three_cluster | 8 | utilitarian | 0.2655 ± 0.0062 | 0.2929 ± 0.0492 | 0.2715 ± 0.1161 | 0.0224 ± 0.0130 | 0.0830 ± 0.0362 | 0.3457 ± 0.0880 | nan | nan | nan | nan |
| three_cluster | 8 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |
| three_cluster | 12 | random_winner | 0.2291 ± 0.0085 | 0.2245 ± 0.0121 | 0.2305 ± 0.0658 | 0.1166 ± 0.0252 | 0.1511 ± 0.0247 | 0.1584 ± 0.0584 | 0.5127 ± 0.2252 | 0.9162 ± 0.0563 | 0.2226 ± 0.0092 | 0.0837 ± 0.0549 |
| three_cluster | 12 | utilitarian | 0.2654 ± 0.0055 | 0.2968 ± 0.0446 | 0.2661 ± 0.1079 | 0.0219 ± 0.0113 | 0.0860 ± 0.0342 | 0.3389 ± 0.0836 | nan | nan | nan | nan |
| three_cluster | 12 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |
| rural_town | 6 | random_winner | 0.2452 ± 0.0275 | 0.1910 ± 0.0558 | 0.2684 ± 0.0613 | 0.1116 ± 0.0226 | 0.1476 ± 0.0200 | 0.1581 ± 0.0463 | 0.4916 ± 0.1885 | 0.8438 ± 0.0982 | 0.2334 ± 0.0294 | 0.0717 ± 0.0445 |
| rural_town | 6 | utilitarian | 0.3533 ± 0.0067 | 0.0538 ± 0.0033 | 0.4817 ± 0.0096 | 0.0065 ± 0.0018 | 0.0318 ± 0.0025 | 0.3889 ± 0.0050 | nan | nan | nan | nan |
| rural_town | 6 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |
| rural_town | 8 | random_winner | 0.2498 ± 0.0277 | 0.1906 ± 0.0608 | 0.2752 ± 0.0643 | 0.1090 ± 0.0233 | 0.1455 ± 0.0238 | 0.1677 ± 0.0552 | 0.4886 ± 0.2057 | 0.8828 ± 0.0825 | 0.2408 ± 0.0278 | 0.0819 ± 0.0525 |
| rural_town | 8 | utilitarian | 0.3512 ± 0.0063 | 0.0532 ± 0.0037 | 0.4789 ± 0.0089 | 0.0061 ± 0.0017 | 0.0312 ± 0.0027 | 0.3908 ± 0.0056 | nan | nan | nan | nan |
| rural_town | 8 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |
| rural_town | 12 | random_winner | 0.2514 ± 0.0288 | 0.1850 ± 0.0604 | 0.2798 ± 0.0651 | 0.1064 ± 0.0285 | 0.1437 ± 0.0257 | 0.1696 ± 0.0591 | 0.4875 ± 0.2308 | 0.9158 ± 0.0778 | 0.2436 ± 0.0289 | 0.0872 ± 0.0599 |
| rural_town | 12 | utilitarian | 0.3527 ± 0.0066 | 0.0528 ± 0.0035 | 0.4813 ± 0.0093 | 0.0064 ± 0.0019 | 0.0311 ± 0.0028 | 0.3900 ± 0.0052 | nan | nan | nan | nan |
| rural_town | 12 | egalitarian | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 ± 0.0000 | nan | nan | nan | nan |

### Mean winner allocations

| population | n_candidates | paradigm | public_good | majority_pork | minority_pork | prestige | periphery_buffer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 6 | party | 0.2654 | 0.2142 | 0.3299 | 0.0986 | 0.0919 |
| two_cluster | 6 | individual | 0.2640 | 0.3080 | 0.2341 | 0.0988 | 0.0950 |
| two_cluster | 6 | score | 0.2967 | 0.2315 | 0.2748 | 0.1006 | 0.0964 |
| two_cluster | 6 | latent_match | 0.2961 | 0.2192 | 0.2850 | 0.1014 | 0.0982 |
| two_cluster | 6 | random_winner | 0.2677 | 0.2788 | 0.2618 | 0.0986 | 0.0931 |
| two_cluster | 6 | utilitarian | 0.0400 | 0.5800 | 0.3800 | 0.0000 | 0.0000 |
| two_cluster | 6 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| two_cluster | 8 | party | 0.2701 | 0.2030 | 0.3387 | 0.0956 | 0.0926 |
| two_cluster | 8 | individual | 0.2672 | 0.2658 | 0.2737 | 0.0985 | 0.0948 |
| two_cluster | 8 | score | 0.2884 | 0.2684 | 0.2444 | 0.0981 | 0.1008 |
| two_cluster | 8 | latent_match | 0.2857 | 0.2603 | 0.2448 | 0.1010 | 0.1082 |
| two_cluster | 8 | random_winner | 0.2687 | 0.2705 | 0.2631 | 0.1034 | 0.0944 |
| two_cluster | 8 | utilitarian | 0.0400 | 0.5100 | 0.4500 | 0.0000 | 0.0000 |
| two_cluster | 8 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| two_cluster | 12 | party | 0.2714 | 0.2026 | 0.3335 | 0.0938 | 0.0987 |
| two_cluster | 12 | individual | 0.2664 | 0.3006 | 0.2399 | 0.0904 | 0.1027 |
| two_cluster | 12 | score | 0.3005 | 0.2375 | 0.2594 | 0.1025 | 0.1000 |
| two_cluster | 12 | latent_match | 0.3053 | 0.2340 | 0.2523 | 0.1063 | 0.1022 |
| two_cluster | 12 | random_winner | 0.2624 | 0.2655 | 0.2781 | 0.1060 | 0.0880 |
| two_cluster | 12 | utilitarian | 0.0900 | 0.4200 | 0.4900 | 0.0000 | 0.0000 |
| two_cluster | 12 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| three_cluster | 6 | party | 0.2846 | 0.2425 | 0.2724 | 0.1038 | 0.0966 |
| three_cluster | 6 | individual | 0.2911 | 0.2632 | 0.2512 | 0.0974 | 0.0971 |
| three_cluster | 6 | score | 0.3126 | 0.2358 | 0.2408 | 0.1101 | 0.1007 |
| three_cluster | 6 | latent_match | 0.3130 | 0.2425 | 0.2317 | 0.1090 | 0.1038 |
| three_cluster | 6 | random_winner | 0.2767 | 0.2322 | 0.2980 | 0.1072 | 0.0859 |
| three_cluster | 6 | utilitarian | 0.8000 | 0.0700 | 0.1300 | 0.0000 | 0.0000 |
| three_cluster | 6 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| three_cluster | 8 | party | 0.2878 | 0.2774 | 0.2388 | 0.0987 | 0.0973 |
| three_cluster | 8 | individual | 0.2866 | 0.2393 | 0.2753 | 0.1018 | 0.0970 |
| three_cluster | 8 | score | 0.3057 | 0.2349 | 0.2499 | 0.1107 | 0.0988 |
| three_cluster | 8 | latent_match | 0.3080 | 0.2362 | 0.2428 | 0.1115 | 0.1015 |
| three_cluster | 8 | random_winner | 0.2819 | 0.2715 | 0.2431 | 0.1127 | 0.0908 |
| three_cluster | 8 | utilitarian | 0.7200 | 0.1700 | 0.1100 | 0.0000 | 0.0000 |
| three_cluster | 8 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| three_cluster | 12 | party | 0.2831 | 0.2432 | 0.2791 | 0.0958 | 0.0988 |
| three_cluster | 12 | individual | 0.2758 | 0.2621 | 0.2635 | 0.0924 | 0.1061 |
| three_cluster | 12 | score | 0.3103 | 0.2332 | 0.2344 | 0.1135 | 0.1086 |
| three_cluster | 12 | latent_match | 0.3127 | 0.2242 | 0.2369 | 0.1176 | 0.1086 |
| three_cluster | 12 | random_winner | 0.2752 | 0.2643 | 0.2640 | 0.1010 | 0.0956 |
| three_cluster | 12 | utilitarian | 0.7600 | 0.1400 | 0.1000 | 0.0000 | 0.0000 |
| three_cluster | 12 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| rural_town | 6 | party | 0.2701 | 0.4594 | 0.0822 | 0.1006 | 0.0876 |
| rural_town | 6 | individual | 0.2707 | 0.3921 | 0.1413 | 0.1043 | 0.0916 |
| rural_town | 6 | score | 0.2750 | 0.4247 | 0.0991 | 0.1073 | 0.0939 |
| rural_town | 6 | latent_match | 0.2813 | 0.4014 | 0.1134 | 0.1079 | 0.0959 |
| rural_town | 6 | random_winner | 0.2741 | 0.3484 | 0.1727 | 0.1132 | 0.0916 |
| rural_town | 6 | utilitarian | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| rural_town | 6 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| rural_town | 8 | party | 0.2722 | 0.4582 | 0.0870 | 0.0950 | 0.0876 |
| rural_town | 8 | individual | 0.2721 | 0.3642 | 0.1741 | 0.0962 | 0.0933 |
| rural_town | 8 | score | 0.2842 | 0.4227 | 0.1021 | 0.0985 | 0.0926 |
| rural_town | 8 | latent_match | 0.2876 | 0.4036 | 0.1120 | 0.1009 | 0.0959 |
| rural_town | 8 | random_winner | 0.2714 | 0.3698 | 0.1746 | 0.0953 | 0.0888 |
| rural_town | 8 | utilitarian | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| rural_town | 8 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| rural_town | 12 | party | 0.2672 | 0.4664 | 0.0825 | 0.0952 | 0.0887 |
| rural_town | 12 | individual | 0.2601 | 0.3812 | 0.1664 | 0.0948 | 0.0975 |
| rural_town | 12 | score | 0.2888 | 0.4272 | 0.0965 | 0.0995 | 0.0880 |
| rural_town | 12 | latent_match | 0.2989 | 0.4027 | 0.1048 | 0.1043 | 0.0893 |
| rural_town | 12 | random_winner | 0.2660 | 0.3820 | 0.1650 | 0.1026 | 0.0843 |
| rural_town | 12 | utilitarian | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| rural_town | 12 | egalitarian | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |

## Paired contrasts vs party

Same-trial differences. Primary endpoints carry Holm-adjusted p-values. Threshold constants are not used as a verdict machine.

λ_winner is **not** a primary endpoint in this cell. Voters never see λ, so a flat λ profile is implied by the generator.

### two_cluster, 6

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | -0.0406 | -0.0575 | -0.0237 | 0.0001 | 0.0006 | -0.4773 |
| individual | total_welfare | -0.0004 | -0.0014 | 0.0007 | 0.5292 | 0.5292 | -0.0655 |
| score | minority_welfare | -0.0165 | -0.0315 | -0.0015 | 0.0057 | 0.0285 | -0.2186 |
| score | total_welfare | -0.0013 | -0.0029 | 0.0003 | 0.0875 | 0.1843 | -0.1628 |
| latent_match | minority_welfare | -0.0124 | -0.0278 | 0.0030 | 0.0214 | 0.0857 | -0.1600 |
| latent_match | total_welfare | -0.0018 | -0.0035 | -0.0002 | 0.0614 | 0.1843 | -0.2166 |

- `individual` vs `party` on `minority_welfare`: Δ = -0.04063 (95% CI [-0.05752, -0.02374]), paired Cohen's d = -0.477, Wilcoxon p = 0.0001008, Holm-adjusted p = 0.0006046.
- `individual` vs `party` on `total_welfare`: Δ = -0.00036 (95% CI [-0.00144, +0.00073]), paired Cohen's d = -0.065, Wilcoxon p = 0.5292, Holm-adjusted p = 0.5292.
- `score` vs `party` on `minority_welfare`: Δ = -0.01652 (95% CI [-0.03151, -0.00152]), paired Cohen's d = -0.219, Wilcoxon p = 0.005703, Holm-adjusted p = 0.02851.
- `score` vs `party` on `total_welfare`: Δ = -0.00130 (95% CI [-0.00288, +0.00028]), paired Cohen's d = -0.163, Wilcoxon p = 0.08748, Holm-adjusted p = 0.1843.
- `latent_match` vs `party` on `minority_welfare`: Δ = -0.01242 (95% CI [-0.02782, +0.00298]), paired Cohen's d = -0.160, Wilcoxon p = 0.02143, Holm-adjusted p = 0.08574.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00180 (95% CI [-0.00346, -0.00015]), paired Cohen's d = -0.217, Wilcoxon p = 0.06142, Holm-adjusted p = 0.1843.

Mushy-bloc check (fixed partition):
- election-endogenous gap shrank while minority-cluster welfare did not rise for `individual`, `score`, `latent_match`.

### two_cluster, 8

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | -0.0279 | -0.0449 | -0.0109 | 0.0088 | 0.0264 | -0.3252 |
| individual | total_welfare | -0.0004 | -0.0016 | 0.0007 | 0.9616 | 0.9616 | -0.0778 |
| score | minority_welfare | -0.0352 | -0.0512 | -0.0193 | 0.0000 | 0.0002 | -0.4385 |
| score | total_welfare | -0.0017 | -0.0033 | -0.0000 | 0.2342 | 0.4684 | -0.2000 |
| latent_match | minority_welfare | -0.0350 | -0.0502 | -0.0198 | 0.0000 | 0.0001 | -0.4565 |
| latent_match | total_welfare | -0.0032 | -0.0049 | -0.0015 | 0.0059 | 0.0235 | -0.3683 |

- `individual` vs `party` on `minority_welfare`: Δ = -0.02786 (95% CI [-0.04486, -0.01086]), paired Cohen's d = -0.325, Wilcoxon p = 0.008793, Holm-adjusted p = 0.02638.
- `individual` vs `party` on `total_welfare`: Δ = -0.00045 (95% CI [-0.00159, +0.00069]), paired Cohen's d = -0.078, Wilcoxon p = 0.9616, Holm-adjusted p = 0.9616.
- `score` vs `party` on `minority_welfare`: Δ = -0.03522 (95% CI [-0.05116, -0.01928]), paired Cohen's d = -0.439, Wilcoxon p = 3.86e-05, Holm-adjusted p = 0.000193.
- `score` vs `party` on `total_welfare`: Δ = -0.00166 (95% CI [-0.00332, -0.00001]), paired Cohen's d = -0.200, Wilcoxon p = 0.2342, Holm-adjusted p = 0.4684.
- `latent_match` vs `party` on `minority_welfare`: Δ = -0.03501 (95% CI [-0.05023, -0.01979]), paired Cohen's d = -0.456, Wilcoxon p = 1.921e-05, Holm-adjusted p = 0.0001153.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00319 (95% CI [-0.00491, -0.00147]), paired Cohen's d = -0.368, Wilcoxon p = 0.005885, Holm-adjusted p = 0.02354.

Mushy-bloc check (fixed partition):
- election-endogenous gap shrank while minority-cluster welfare did not rise for `individual`, `score`, `latent_match`.

### two_cluster, 12

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | -0.0402 | -0.0569 | -0.0235 | 0.0000 | 0.0000 | -0.4781 |
| individual | total_welfare | -0.0002 | -0.0012 | 0.0008 | 0.3938 | 0.3938 | -0.0337 |
| score | minority_welfare | -0.0249 | -0.0416 | -0.0083 | 0.0010 | 0.0030 | -0.2977 |
| score | total_welfare | -0.0019 | -0.0035 | -0.0004 | 0.0145 | 0.0290 | -0.2455 |
| latent_match | minority_welfare | -0.0265 | -0.0408 | -0.0122 | 0.0001 | 0.0006 | -0.3678 |
| latent_match | total_welfare | -0.0028 | -0.0044 | -0.0012 | 0.0004 | 0.0017 | -0.3523 |

- `individual` vs `party` on `minority_welfare`: Δ = -0.04022 (95% CI [-0.05692, -0.02353]), paired Cohen's d = -0.478, Wilcoxon p = 3.946e-06, Holm-adjusted p = 2.367e-05.
- `individual` vs `party` on `total_welfare`: Δ = -0.00017 (95% CI [-0.00120, +0.00085]), paired Cohen's d = -0.034, Wilcoxon p = 0.3938, Holm-adjusted p = 0.3938.
- `score` vs `party` on `minority_welfare`: Δ = -0.02494 (95% CI [-0.04156, -0.00832]), paired Cohen's d = -0.298, Wilcoxon p = 0.001012, Holm-adjusted p = 0.003037.
- `score` vs `party` on `total_welfare`: Δ = -0.00194 (95% CI [-0.00351, -0.00037]), paired Cohen's d = -0.246, Wilcoxon p = 0.0145, Holm-adjusted p = 0.029.
- `latent_match` vs `party` on `minority_welfare`: Δ = -0.02651 (95% CI [-0.04081, -0.01221]), paired Cohen's d = -0.368, Wilcoxon p = 0.000116, Holm-adjusted p = 0.0005802.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00283 (95% CI [-0.00442, -0.00124]), paired Cohen's d = -0.352, Wilcoxon p = 0.0004192, Holm-adjusted p = 0.001677.

Mushy-bloc check (fixed partition):
- election-endogenous gap shrank while minority-cluster welfare did not rise for `individual`, `score`, `latent_match`.

### three_cluster, 6

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | -0.0007 | -0.0021 | 0.0006 | 0.1546 | 0.5023 | -0.1112 |
| individual | total_welfare | 0.0008 | -0.0002 | 0.0017 | 0.0914 | 0.5023 | 0.1598 |
| score | minority_welfare | 0.0010 | -0.0011 | 0.0031 | 0.5589 | 0.9195 | 0.0959 |
| score | total_welfare | -0.0012 | -0.0027 | 0.0004 | 0.1243 | 0.5023 | -0.1507 |
| latent_match | minority_welfare | 0.0010 | -0.0010 | 0.0031 | 0.4598 | 0.9195 | 0.0997 |
| latent_match | total_welfare | -0.0014 | -0.0028 | 0.0001 | 0.0837 | 0.5023 | -0.1841 |

- `individual` vs `party` on `minority_welfare`: Δ = -0.00074 (95% CI [-0.00207, +0.00058]), paired Cohen's d = -0.111, Wilcoxon p = 0.1546, Holm-adjusted p = 0.5023.
- `individual` vs `party` on `total_welfare`: Δ = +0.00076 (95% CI [-0.00018, +0.00170]), paired Cohen's d = +0.160, Wilcoxon p = 0.09137, Holm-adjusted p = 0.5023.
- `score` vs `party` on `minority_welfare`: Δ = +0.00100 (95% CI [-0.00107, +0.00307]), paired Cohen's d = +0.096, Wilcoxon p = 0.5589, Holm-adjusted p = 0.9195.
- `score` vs `party` on `total_welfare`: Δ = -0.00117 (95% CI [-0.00271, +0.00037]), paired Cohen's d = -0.151, Wilcoxon p = 0.1243, Holm-adjusted p = 0.5023.
- `latent_match` vs `party` on `minority_welfare`: Δ = +0.00103 (95% CI [-0.00102, +0.00308]), paired Cohen's d = +0.100, Wilcoxon p = 0.4598, Holm-adjusted p = 0.9195.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00137 (95% CI [-0.00284, +0.00011]), paired Cohen's d = -0.184, Wilcoxon p = 0.08372, Holm-adjusted p = 0.5023.

Mushy-bloc check (fixed partition):
- election-endogenous gap shrank while minority-cluster welfare did not rise for `individual`.

### three_cluster, 8

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | -0.0011 | -0.0025 | 0.0003 | 0.1153 | 0.4613 | -0.1539 |
| individual | total_welfare | -0.0001 | -0.0012 | 0.0010 | 0.3072 | 0.9215 | -0.0198 |
| score | minority_welfare | 0.0003 | -0.0015 | 0.0022 | 0.9973 | 1.0000 | 0.0359 |
| score | total_welfare | -0.0016 | -0.0029 | -0.0002 | 0.0183 | 0.0917 | -0.2290 |
| latent_match | minority_welfare | 0.0002 | -0.0018 | 0.0021 | 0.8635 | 1.0000 | 0.0158 |
| latent_match | total_welfare | -0.0021 | -0.0035 | -0.0007 | 0.0026 | 0.0156 | -0.2952 |

- `individual` vs `party` on `minority_welfare`: Δ = -0.00110 (95% CI [-0.00252, +0.00032]), paired Cohen's d = -0.154, Wilcoxon p = 0.1153, Holm-adjusted p = 0.4613.
- `individual` vs `party` on `total_welfare`: Δ = -0.00011 (95% CI [-0.00119, +0.00097]), paired Cohen's d = -0.020, Wilcoxon p = 0.3072, Holm-adjusted p = 0.9215.
- `score` vs `party` on `minority_welfare`: Δ = +0.00033 (95% CI [-0.00151, +0.00217]), paired Cohen's d = +0.036, Wilcoxon p = 0.9973, Holm-adjusted p = 1.
- `score` vs `party` on `total_welfare`: Δ = -0.00158 (95% CI [-0.00294, -0.00021]), paired Cohen's d = -0.229, Wilcoxon p = 0.01834, Holm-adjusted p = 0.0917.
- `latent_match` vs `party` on `minority_welfare`: Δ = +0.00016 (95% CI [-0.00179, +0.00210]), paired Cohen's d = +0.016, Wilcoxon p = 0.8635, Holm-adjusted p = 1.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00209 (95% CI [-0.00349, -0.00068]), paired Cohen's d = -0.295, Wilcoxon p = 0.002596, Holm-adjusted p = 0.01557.

Mushy-bloc check (fixed partition):
- election-endogenous gap shrank while minority-cluster welfare did not rise for `individual`.

### three_cluster, 12

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | -0.0024 | -0.0039 | -0.0008 | 0.0023 | 0.0092 | -0.2997 |
| individual | total_welfare | -0.0006 | -0.0015 | 0.0004 | 0.4982 | 1.0000 | -0.1136 |
| score | minority_welfare | -0.0003 | -0.0025 | 0.0018 | 0.5799 | 1.0000 | -0.0325 |
| score | total_welfare | -0.0037 | -0.0055 | -0.0019 | 0.0007 | 0.0037 | -0.4134 |
| latent_match | minority_welfare | -0.0006 | -0.0029 | 0.0017 | 0.5822 | 1.0000 | -0.0524 |
| latent_match | total_welfare | -0.0043 | -0.0061 | -0.0026 | 0.0000 | 0.0002 | -0.4879 |

- `individual` vs `party` on `minority_welfare`: Δ = -0.00235 (95% CI [-0.00391, -0.00080]), paired Cohen's d = -0.300, Wilcoxon p = 0.00229, Holm-adjusted p = 0.00916.
- `individual` vs `party` on `total_welfare`: Δ = -0.00056 (95% CI [-0.00153, +0.00042]), paired Cohen's d = -0.114, Wilcoxon p = 0.4982, Holm-adjusted p = 1.
- `score` vs `party` on `minority_welfare`: Δ = -0.00035 (95% CI [-0.00247, +0.00178]), paired Cohen's d = -0.033, Wilcoxon p = 0.5799, Holm-adjusted p = 1.
- `score` vs `party` on `total_welfare`: Δ = -0.00369 (95% CI [-0.00547, -0.00192]), paired Cohen's d = -0.413, Wilcoxon p = 0.0007343, Holm-adjusted p = 0.003672.
- `latent_match` vs `party` on `minority_welfare`: Δ = -0.00060 (95% CI [-0.00286, +0.00167]), paired Cohen's d = -0.052, Wilcoxon p = 0.5822, Holm-adjusted p = 1.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00431 (95% CI [-0.00606, -0.00256]), paired Cohen's d = -0.488, Wilcoxon p = 3.177e-05, Holm-adjusted p = 0.0001906.

Mushy-bloc check (fixed partition):
- election-endogenous gap shrank while minority-cluster welfare did not rise for `individual`, `score`, `latent_match`.

### rural_town, 6

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | 0.0254 | 0.0161 | 0.0346 | 0.0065 | 0.0065 | 0.5431 |
| individual | total_welfare | -0.0122 | -0.0166 | -0.0078 | 0.0002 | 0.0004 | -0.5484 |
| score | minority_welfare | 0.0087 | 0.0047 | 0.0126 | 0.0001 | 0.0004 | 0.4349 |
| score | total_welfare | -0.0067 | -0.0095 | -0.0039 | 0.0000 | 0.0001 | -0.4681 |
| latent_match | minority_welfare | 0.0161 | 0.0109 | 0.0213 | 0.0000 | 0.0000 | 0.6170 |
| latent_match | total_welfare | -0.0105 | -0.0136 | -0.0073 | 0.0000 | 0.0000 | -0.6591 |

- `individual` vs `party` on `minority_welfare`: Δ = +0.02537 (95% CI [+0.01610, +0.03464]), paired Cohen's d = +0.543, Wilcoxon p = 0.006534, Holm-adjusted p = 0.006534.
- `individual` vs `party` on `total_welfare`: Δ = -0.01216 (95% CI [-0.01655, -0.00776]), paired Cohen's d = -0.548, Wilcoxon p = 0.000176, Holm-adjusted p = 0.000406.
- `score` vs `party` on `minority_welfare`: Δ = +0.00867 (95% CI [+0.00471, +0.01262]), paired Cohen's d = +0.435, Wilcoxon p = 0.0001353, Holm-adjusted p = 0.000406.
- `score` vs `party` on `total_welfare`: Δ = -0.00670 (95% CI [-0.00954, -0.00386]), paired Cohen's d = -0.468, Wilcoxon p = 2.275e-05, Holm-adjusted p = 9.101e-05.
- `latent_match` vs `party` on `minority_welfare`: Δ = +0.01609 (95% CI [+0.01092, +0.02127]), paired Cohen's d = +0.617, Wilcoxon p = 4.405e-08, Holm-adjusted p = 2.202e-07.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.01046 (95% CI [-0.01361, -0.00731]), paired Cohen's d = -0.659, Wilcoxon p = 3.708e-09, Holm-adjusted p = 2.225e-08.

Mushy-bloc check (fixed partition):
- not triggered — no rule shrank the election-endogenous gap without a rise in minority-cluster welfare.

### rural_town, 8

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | 0.0372 | 0.0258 | 0.0486 | 0.0000 | 0.0000 | 0.6471 |
| individual | total_welfare | -0.0164 | -0.0214 | -0.0113 | 0.0000 | 0.0000 | -0.6425 |
| score | minority_welfare | 0.0092 | 0.0051 | 0.0133 | 0.0000 | 0.0000 | 0.4453 |
| score | total_welfare | -0.0058 | -0.0082 | -0.0034 | 0.0000 | 0.0000 | -0.4799 |
| latent_match | minority_welfare | 0.0144 | 0.0098 | 0.0189 | 0.0000 | 0.0000 | 0.6248 |
| latent_match | total_welfare | -0.0093 | -0.0119 | -0.0066 | 0.0000 | 0.0000 | -0.6984 |

- `individual` vs `party` on `minority_welfare`: Δ = +0.03720 (95% CI [+0.02579, +0.04861]), paired Cohen's d = +0.647, Wilcoxon p = 1.184e-06, Holm-adjusted p = 2.417e-06.
- `individual` vs `party` on `total_welfare`: Δ = -0.01635 (95% CI [-0.02140, -0.01130]), paired Cohen's d = -0.642, Wilcoxon p = 8.056e-07, Holm-adjusted p = 2.417e-06.
- `score` vs `party` on `minority_welfare`: Δ = +0.00919 (95% CI [+0.00509, +0.01328]), paired Cohen's d = +0.445, Wilcoxon p = 2.845e-07, Holm-adjusted p = 1.138e-06.
- `score` vs `party` on `total_welfare`: Δ = -0.00583 (95% CI [-0.00825, -0.00342]), paired Cohen's d = -0.480, Wilcoxon p = 1.124e-06, Holm-adjusted p = 2.417e-06.
- `latent_match` vs `party` on `minority_welfare`: Δ = +0.01438 (95% CI [+0.00981, +0.01894]), paired Cohen's d = +0.625, Wilcoxon p = 3.707e-11, Holm-adjusted p = 2.224e-10.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00929 (95% CI [-0.01192, -0.00665]), paired Cohen's d = -0.698, Wilcoxon p = 7.578e-11, Holm-adjusted p = 3.789e-10.

Mushy-bloc check (fixed partition):
- not triggered — no rule shrank the election-endogenous gap without a rise in minority-cluster welfare.

### rural_town, 12

| paradigm | endpoint | delta_mean | ci_low | ci_high | pvalue | pvalue_adjusted | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | minority_welfare | 0.0339 | 0.0219 | 0.0459 | 0.0003 | 0.0004 | 0.5602 |
| individual | total_welfare | -0.0159 | -0.0214 | -0.0104 | 0.0002 | 0.0004 | -0.5765 |
| score | minority_welfare | 0.0104 | 0.0075 | 0.0134 | 0.0000 | 0.0000 | 0.7024 |
| score | total_welfare | -0.0055 | -0.0077 | -0.0033 | 0.0000 | 0.0000 | -0.4927 |
| latent_match | minority_welfare | 0.0162 | 0.0121 | 0.0203 | 0.0000 | 0.0000 | 0.7878 |
| latent_match | total_welfare | -0.0094 | -0.0121 | -0.0067 | 0.0000 | 0.0000 | -0.6934 |

- `individual` vs `party` on `minority_welfare`: Δ = +0.03386 (95% CI [+0.02187, +0.04586]), paired Cohen's d = +0.560, Wilcoxon p = 0.0002607, Holm-adjusted p = 0.000398.
- `individual` vs `party` on `total_welfare`: Δ = -0.01589 (95% CI [-0.02136, -0.01042]), paired Cohen's d = -0.577, Wilcoxon p = 0.000199, Holm-adjusted p = 0.000398.
- `score` vs `party` on `minority_welfare`: Δ = +0.01041 (95% CI [+0.00747, +0.01335]), paired Cohen's d = +0.702, Wilcoxon p = 5.196e-10, Holm-adjusted p = 2.598e-09.
- `score` vs `party` on `total_welfare`: Δ = -0.00550 (95% CI [-0.00772, -0.00329]), paired Cohen's d = -0.493, Wilcoxon p = 1.453e-05, Holm-adjusted p = 4.358e-05.
- `latent_match` vs `party` on `minority_welfare`: Δ = +0.01619 (95% CI [+0.01211, +0.02027]), paired Cohen's d = +0.788, Wilcoxon p = 2.069e-13, Holm-adjusted p = 1.241e-12.
- `latent_match` vs `party` on `total_welfare`: Δ = -0.00941 (95% CI [-0.01211, -0.00672]), paired Cohen's d = -0.693, Wilcoxon p = 1.466e-09, Holm-adjusted p = 5.866e-09.

Mushy-bloc check (fixed partition):
- not triggered — no rule shrank the election-endogenous gap without a rise in minority-cluster welfare.

## Limitations

- Voters and candidates live in a stylized 5-dimensional project space; real preference structures are higher-dimensional and partly unobservable.
- The default one-shot cell has exogenous λ. It compares selection rules with drawn types; it does not test citizen-candidate entry or core- vs swing-voter targeting. Use `--mechanism reelection` for an incentive-based λ cell (still not an entry model).
- Party structure is idealized as two brands at cluster means with loyal nomination; real parties select through noisy primaries.
- Supporter definitions differ across paradigms by design. Primary welfare contrasts therefore use the fixed cluster partition, not the election-endogenous loser set.
- `--lambda-correlated` is a researcher degree of freedom that can flip λ rankings. It is labeled a robustness appendix whenever it is the condition being reported.

## Reproduce

```
python run_experiment.py sweep --trials 100 --populations two_cluster,three_cluster,rural_town --candidates 6,8,12 --no-persist-ballots --out '{run_dir}'
```

Deterministic given identical parameters and seed (per-trial streams are derived from `numpy.random.default_rng([seed, population, candidates, trial])`).
