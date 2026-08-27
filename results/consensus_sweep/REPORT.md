# Political consensus experiment — auto-generated report

Comparison of selection paradigms on a post-election budget-allocation task: does individual-centered selection (no parties) produce stewards who treat non-supporters better than party selection does?

## Methods

- **Population**: 400 voters per trial, latent preference vectors in R^5 drawn from cluster centers plus Gaussian noise (σ = 0.35); benefit vectors are a softmax (temperature 0.55) of preferences, so rows are nonnegative and sum to 1. Population types run here: `two_cluster`, `three_cluster`, `rural_town`.
- **Projects**: `core_services`, `coalition_club`, `outgroup_repair`, `prestige_project`, `buffer_reserve`.
- **Candidates**: 6 per trial (sweeps may vary this), platforms drawn from the same cluster structure; loyalty trait λ ~ Beta(2.2, 2.2), independent of the platform.
- **Allocation**: `raw = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`, then `raw = 0.72·raw + 0.28·clip(platform, 0, ∞)`, normalized to sum to 1. Zero supporters ⇒ all-voter direction only.
- **Paradigms**: `party`, `individual`, `score`, `latent_match`.
- **Trials**: 100 per cell, base seed 0; every paradigm sees the identical population and candidate slate within a trial.
- **Privacy**: only aggregates and winner allocations are persisted; no individual ballots.

## Results

Mean ± std over trials, by population, candidate count, and paradigm:

| population | n_candidates | paradigm | total_welfare | supporter_welfare | loser_welfare | gap | lambda_winner | loser_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 6 | party | 0.2327 ± 0.0067 | 0.3034 ± 0.0194 | 0.1614 ± 0.0149 | 0.1420 ± 0.0319 | 0.4580 ± 0.1789 | 0.4978 ± 0.0021 |
| two_cluster | 6 | individual | 0.2321 ± 0.0079 | 0.3095 ± 0.0303 | 0.1856 ± 0.0209 | 0.1239 ± 0.0441 | 0.5114 ± 0.2119 | 0.6348 ± 0.0771 |
| two_cluster | 6 | score | 0.2312 ± 0.0075 | 0.2779 ± 0.0291 | 0.2149 ± 0.0148 | 0.0629 ± 0.0377 | 0.5215 ± 0.2208 | 0.7766 ± 0.1036 |
| two_cluster | 6 | latent_match | 0.2306 ± 0.0080 | 0.2741 ± 0.0280 | 0.2139 ± 0.0169 | 0.0603 ± 0.0387 | 0.5012 ± 0.2189 | 0.7709 ± 0.1146 |
| two_cluster | 8 | party | 0.2330 ± 0.0065 | 0.3063 ± 0.0204 | 0.1591 ± 0.0153 | 0.1472 ± 0.0337 | 0.5052 ± 0.2108 | 0.4979 ± 0.0020 |
| two_cluster | 8 | individual | 0.2324 ± 0.0084 | 0.3105 ± 0.0315 | 0.1984 ± 0.0151 | 0.1121 ± 0.0372 | 0.4979 ± 0.2127 | 0.7002 ± 0.0752 |
| two_cluster | 8 | score | 0.2310 ± 0.0083 | 0.2814 ± 0.0326 | 0.2162 ± 0.0134 | 0.0652 ± 0.0392 | 0.5226 ± 0.2264 | 0.8064 ± 0.0941 |
| two_cluster | 8 | latent_match | 0.2289 ± 0.0088 | 0.2746 ± 0.0321 | 0.2175 ± 0.0125 | 0.0571 ± 0.0369 | 0.5193 ± 0.2186 | 0.8312 ± 0.0938 |
| two_cluster | 12 | party | 0.2324 ± 0.0064 | 0.3030 ± 0.0178 | 0.1610 ± 0.0148 | 0.1420 ± 0.0301 | 0.4815 ± 0.2292 | 0.4976 ± 0.0023 |
| two_cluster | 12 | individual | 0.2321 ± 0.0066 | 0.3116 ± 0.0335 | 0.2059 ± 0.0136 | 0.1057 ± 0.0419 | 0.4972 ± 0.2192 | 0.7598 ± 0.0648 |
| two_cluster | 12 | score | 0.2299 ± 0.0079 | 0.2763 ± 0.0263 | 0.2208 ± 0.0098 | 0.0554 ± 0.0277 | 0.4838 ± 0.2177 | 0.8566 ± 0.0743 |
| two_cluster | 12 | latent_match | 0.2287 ± 0.0081 | 0.2693 ± 0.0244 | 0.2215 ± 0.0115 | 0.0479 ± 0.0268 | 0.4808 ± 0.2047 | 0.8756 ± 0.0790 |
| three_cluster | 6 | party | 0.2301 ± 0.0067 | 0.2830 ± 0.0184 | 0.1754 ± 0.0173 | 0.1076 ± 0.0330 | 0.4683 ± 0.2328 | 0.4909 ± 0.0079 |
| three_cluster | 6 | individual | 0.2310 ± 0.0069 | 0.2861 ± 0.0349 | 0.2019 ± 0.0165 | 0.0842 ± 0.0473 | 0.4674 ± 0.2082 | 0.6517 ± 0.0722 |
| three_cluster | 6 | score | 0.2281 ± 0.0079 | 0.2552 ± 0.0265 | 0.2173 ± 0.0130 | 0.0379 ± 0.0339 | 0.4893 ± 0.2227 | 0.7382 ± 0.1008 |
| three_cluster | 6 | latent_match | 0.2279 ± 0.0079 | 0.2520 ± 0.0225 | 0.2177 ± 0.0130 | 0.0343 ± 0.0290 | 0.4741 ± 0.2188 | 0.7339 ± 0.1087 |
| three_cluster | 8 | party | 0.2309 ± 0.0062 | 0.2875 ± 0.0149 | 0.1721 ± 0.0138 | 0.1154 ± 0.0260 | 0.4995 ± 0.2144 | 0.4912 ± 0.0063 |
| three_cluster | 8 | individual | 0.2305 ± 0.0070 | 0.2925 ± 0.0337 | 0.2046 ± 0.0124 | 0.0879 ± 0.0419 | 0.4791 ± 0.2344 | 0.7048 ± 0.0553 |
| three_cluster | 8 | score | 0.2285 ± 0.0072 | 0.2545 ± 0.0245 | 0.2204 ± 0.0093 | 0.0341 ± 0.0289 | 0.5138 ± 0.2185 | 0.7908 ± 0.0749 |
| three_cluster | 8 | latent_match | 0.2280 ± 0.0075 | 0.2521 ± 0.0241 | 0.2207 ± 0.0102 | 0.0315 ± 0.0286 | 0.5250 ± 0.2167 | 0.7985 ± 0.0794 |
| three_cluster | 12 | party | 0.2312 ± 0.0057 | 0.2902 ± 0.0138 | 0.1699 ± 0.0124 | 0.1203 ± 0.0235 | 0.5127 ± 0.2071 | 0.4901 ± 0.0082 |
| three_cluster | 12 | individual | 0.2303 ± 0.0075 | 0.3049 ± 0.0390 | 0.2091 ± 0.0117 | 0.0958 ± 0.0460 | 0.5147 ± 0.2144 | 0.7839 ± 0.0503 |
| three_cluster | 12 | score | 0.2263 ± 0.0091 | 0.2480 ± 0.0262 | 0.2223 ± 0.0084 | 0.0257 ± 0.0250 | 0.5047 ± 0.2214 | 0.8646 ± 0.0482 |
| three_cluster | 12 | latent_match | 0.2254 ± 0.0093 | 0.2443 ± 0.0239 | 0.2224 ± 0.0084 | 0.0220 ± 0.0211 | 0.5249 ± 0.2244 | 0.8714 ± 0.0428 |
| rural_town | 6 | party | 0.2692 ± 0.0103 | 0.3217 ± 0.0169 | 0.1477 ± 0.0112 | 0.1740 ± 0.0250 | 0.5229 ± 0.2159 | 0.3020 ± 0.0037 |
| rural_town | 6 | individual | 0.2559 ± 0.0261 | 0.3113 ± 0.0451 | 0.2236 ± 0.0294 | 0.0878 ± 0.0524 | 0.5322 ± 0.2088 | 0.6423 ± 0.0926 |
| rural_town | 6 | score | 0.2609 ± 0.0159 | 0.3024 ± 0.0407 | 0.2379 ± 0.0228 | 0.0645 ± 0.0488 | 0.5454 ± 0.2136 | 0.6953 ± 0.1271 |
| rural_town | 6 | latent_match | 0.2567 ± 0.0170 | 0.2912 ± 0.0421 | 0.2378 ± 0.0226 | 0.0534 ± 0.0496 | 0.5580 ± 0.2115 | 0.7161 ± 0.1340 |
| rural_town | 8 | party | 0.2691 ± 0.0087 | 0.3206 ± 0.0149 | 0.1497 ± 0.0104 | 0.1710 ± 0.0227 | 0.4754 ± 0.2155 | 0.3014 ± 0.0034 |
| rural_town | 8 | individual | 0.2510 ± 0.0292 | 0.3117 ± 0.0415 | 0.2270 ± 0.0334 | 0.0847 ± 0.0473 | 0.4884 ± 0.2083 | 0.7116 ± 0.0583 |
| rural_town | 8 | score | 0.2620 ± 0.0136 | 0.3007 ± 0.0371 | 0.2493 ± 0.0113 | 0.0514 ± 0.0356 | 0.4795 ± 0.1900 | 0.7723 ± 0.0897 |
| rural_town | 8 | latent_match | 0.2580 ± 0.0138 | 0.2884 ± 0.0344 | 0.2486 ± 0.0117 | 0.0398 ± 0.0329 | 0.4778 ± 0.2009 | 0.7889 ± 0.0979 |
| rural_town | 12 | party | 0.2708 ± 0.0094 | 0.3247 ± 0.0160 | 0.1462 ± 0.0109 | 0.1784 ± 0.0245 | 0.5225 ± 0.2196 | 0.3017 ± 0.0029 |
| rural_town | 12 | individual | 0.2527 ± 0.0313 | 0.3230 ± 0.0433 | 0.2317 ± 0.0353 | 0.0913 ± 0.0457 | 0.5258 ± 0.2186 | 0.7757 ± 0.0627 |
| rural_town | 12 | score | 0.2640 ± 0.0123 | 0.3035 ± 0.0349 | 0.2540 ± 0.0111 | 0.0495 ± 0.0326 | 0.5071 ± 0.2095 | 0.8293 ± 0.0811 |
| rural_town | 12 | latent_match | 0.2595 ± 0.0149 | 0.2923 ± 0.0353 | 0.2524 ± 0.0129 | 0.0399 ± 0.0292 | 0.5205 ± 0.2264 | 0.8514 ± 0.0773 |

### Mean winner allocations

| population | n_candidates | paradigm | core_services | coalition_club | outgroup_repair | prestige_project | buffer_reserve |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_cluster | 6 | party | 0.2739 | 0.2089 | 0.3384 | 0.0950 | 0.0838 |
| two_cluster | 6 | individual | 0.2725 | 0.3125 | 0.2310 | 0.0964 | 0.0876 |
| two_cluster | 6 | score | 0.3066 | 0.2290 | 0.2772 | 0.0976 | 0.0895 |
| two_cluster | 6 | latent_match | 0.3065 | 0.2155 | 0.2879 | 0.0988 | 0.0913 |
| two_cluster | 8 | party | 0.2792 | 0.1968 | 0.3473 | 0.0918 | 0.0849 |
| two_cluster | 8 | individual | 0.2760 | 0.2648 | 0.2762 | 0.0957 | 0.0873 |
| two_cluster | 8 | score | 0.2984 | 0.2672 | 0.2445 | 0.0946 | 0.0953 |
| two_cluster | 8 | latent_match | 0.2954 | 0.2594 | 0.2412 | 0.0989 | 0.1050 |
| two_cluster | 12 | party | 0.2799 | 0.1953 | 0.3427 | 0.0892 | 0.0928 |
| two_cluster | 12 | individual | 0.2754 | 0.3049 | 0.2368 | 0.0864 | 0.0965 |
| two_cluster | 12 | score | 0.3086 | 0.2347 | 0.2608 | 0.1011 | 0.0947 |
| two_cluster | 12 | latent_match | 0.3132 | 0.2303 | 0.2532 | 0.1056 | 0.0977 |
| three_cluster | 6 | party | 0.2925 | 0.2446 | 0.2727 | 0.1016 | 0.0886 |
| three_cluster | 6 | individual | 0.3019 | 0.2646 | 0.2502 | 0.0935 | 0.0899 |
| three_cluster | 6 | score | 0.3190 | 0.2343 | 0.2402 | 0.1096 | 0.0968 |
| three_cluster | 6 | latent_match | 0.3195 | 0.2414 | 0.2305 | 0.1081 | 0.1005 |
| three_cluster | 8 | party | 0.2971 | 0.2813 | 0.2391 | 0.0928 | 0.0897 |
| three_cluster | 8 | individual | 0.2971 | 0.2364 | 0.2789 | 0.0981 | 0.0895 |
| three_cluster | 8 | score | 0.3122 | 0.2340 | 0.2518 | 0.1086 | 0.0934 |
| three_cluster | 8 | latent_match | 0.3142 | 0.2367 | 0.2438 | 0.1086 | 0.0966 |
| three_cluster | 12 | party | 0.2902 | 0.2444 | 0.2841 | 0.0903 | 0.0910 |
| three_cluster | 12 | individual | 0.2825 | 0.2659 | 0.2633 | 0.0872 | 0.1012 |
| three_cluster | 12 | score | 0.3166 | 0.2339 | 0.2317 | 0.1124 | 0.1054 |
| three_cluster | 12 | latent_match | 0.3167 | 0.2247 | 0.2354 | 0.1169 | 0.1063 |
| rural_town | 6 | party | 0.2772 | 0.4740 | 0.0693 | 0.0987 | 0.0807 |
| rural_town | 6 | individual | 0.2778 | 0.4009 | 0.1323 | 0.1044 | 0.0846 |
| rural_town | 6 | score | 0.2816 | 0.4329 | 0.0881 | 0.1080 | 0.0895 |
| rural_town | 6 | latent_match | 0.2876 | 0.4074 | 0.1041 | 0.1088 | 0.0921 |
| rural_town | 8 | party | 0.2805 | 0.4735 | 0.0743 | 0.0912 | 0.0806 |
| rural_town | 8 | individual | 0.2823 | 0.3689 | 0.1690 | 0.0928 | 0.0870 |
| rural_town | 8 | score | 0.2919 | 0.4321 | 0.0929 | 0.0955 | 0.0877 |
| rural_town | 8 | latent_match | 0.2960 | 0.4100 | 0.1038 | 0.0989 | 0.0914 |
| rural_town | 12 | party | 0.2751 | 0.4839 | 0.0695 | 0.0902 | 0.0815 |
| rural_town | 12 | individual | 0.2666 | 0.3874 | 0.1634 | 0.0913 | 0.0914 |
| rural_town | 12 | score | 0.2977 | 0.4377 | 0.0861 | 0.0970 | 0.0814 |
| rural_town | 12 | latent_match | 0.3086 | 0.4100 | 0.0957 | 0.1028 | 0.0830 |

## Hypothesis evaluation

Hypothesis: individual-centered rules (`individual`, `score`, `latent_match`) select lower-λ winners than `party` and raise loser welfare without merely inflating loser share.

### two_cluster, 6 candidates

- `individual` vs `party`: Δλ_winner = +0.0534, Δloser_welfare = +0.02414, Δloser_share = +0.1370, Δgap = -0.01810, Δtotal_welfare = -0.00066 → partial support.
- `score` vs `party`: Δλ_winner = +0.0635, Δloser_welfare = +0.05352, Δloser_share = +0.2788, Δgap = -0.07909, Δtotal_welfare = -0.00158 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = +0.0432, Δloser_welfare = +0.05245, Δloser_share = +0.2731, Δgap = -0.08174, Δtotal_welfare = -0.00216 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0635 → not triggered.
- total welfare flat (max relative change < 0.5%): max relative change = 0.9282% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.2788 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

### two_cluster, 8 candidates

- `individual` vs `party`: Δλ_winner = -0.0072, Δloser_welfare = +0.03927, Δloser_share = +0.2023, Δgap = -0.03508, Δtotal_welfare = -0.00061 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = +0.0174, Δloser_welfare = +0.05707, Δloser_share = +0.3085, Δgap = -0.08198, Δtotal_welfare = -0.00204 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = +0.0141, Δloser_welfare = +0.05842, Δloser_share = +0.3333, Δgap = -0.09016, Δtotal_welfare = -0.00412 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0174 → **triggered**.
- total welfare flat (max relative change < 0.5%): max relative change = 1.7673% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.3333 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

### two_cluster, 12 candidates

- `individual` vs `party`: Δλ_winner = +0.0157, Δloser_welfare = +0.04485, Δloser_share = +0.2622, Δgap = -0.03630, Δtotal_welfare = -0.00024 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = +0.0023, Δloser_welfare = +0.05980, Δloser_share = +0.3590, Δgap = -0.08659, Δtotal_welfare = -0.00248 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = -0.0007, Δloser_welfare = +0.06043, Δloser_share = +0.3780, Δgap = -0.09416, Δtotal_welfare = -0.00363 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0157 → **triggered**.
- total welfare flat (max relative change < 0.5%): max relative change = 1.5604% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.3780 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

### three_cluster, 6 candidates

- `individual` vs `party`: Δλ_winner = -0.0009, Δloser_welfare = +0.02652, Δloser_share = +0.1608, Δgap = -0.02337, Δtotal_welfare = +0.00090 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = +0.0210, Δloser_welfare = +0.04196, Δloser_share = +0.2474, Δgap = -0.06974, Δtotal_welfare = -0.00203 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = +0.0058, Δloser_welfare = +0.04236, Δloser_share = +0.2430, Δgap = -0.07332, Δtotal_welfare = -0.00222 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0210 → not triggered.
- total welfare flat (max relative change < 0.5%): max relative change = 0.9652% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.2474 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

### three_cluster, 8 candidates

- `individual` vs `party`: Δλ_winner = -0.0204, Δloser_welfare = +0.03252, Δloser_share = +0.2137, Δgap = -0.02755, Δtotal_welfare = -0.00034 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = +0.0144, Δloser_welfare = +0.04825, Δloser_share = +0.2996, Δgap = -0.08130, Δtotal_welfare = -0.00239 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = +0.0255, Δloser_welfare = +0.04858, Δloser_share = +0.3073, Δgap = -0.08396, Δtotal_welfare = -0.00286 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0255 → not triggered.
- total welfare flat (max relative change < 0.5%): max relative change = 1.2399% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.3073 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

### three_cluster, 12 candidates

- `individual` vs `party`: Δλ_winner = +0.0020, Δloser_welfare = +0.03917, Δloser_share = +0.2938, Δgap = -0.02447, Δtotal_welfare = -0.00096 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = -0.0080, Δloser_welfare = +0.05240, Δloser_share = +0.3745, Δgap = -0.09458, Δtotal_welfare = -0.00497 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = +0.0122, Δloser_welfare = +0.05246, Δloser_share = +0.3813, Δgap = -0.09830, Δtotal_welfare = -0.00579 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0122 → **triggered**.
- total welfare flat (max relative change < 0.5%): max relative change = 2.5042% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.3813 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

### rural_town, 6 candidates

- `individual` vs `party`: Δλ_winner = +0.0092, Δloser_welfare = +0.07583, Δloser_share = +0.3404, Δgap = -0.08623, Δtotal_welfare = -0.01329 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = +0.0224, Δloser_welfare = +0.09017, Δloser_share = +0.3933, Δgap = -0.10945, Δtotal_welfare = -0.00826 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = +0.0351, Δloser_welfare = +0.09008, Δloser_share = +0.4142, Δgap = -0.12064, Δtotal_welfare = -0.01247 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0351 → not triggered.
- total welfare flat (max relative change < 0.5%): max relative change = 4.9365% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.4142 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

### rural_town, 8 candidates

- `individual` vs `party`: Δλ_winner = +0.0130, Δloser_welfare = +0.07735, Δloser_share = +0.4102, Δgap = -0.08625, Δtotal_welfare = -0.01807 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = +0.0041, Δloser_welfare = +0.09964, Δloser_share = +0.4709, Δgap = -0.11960, Δtotal_welfare = -0.00708 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = +0.0024, Δloser_welfare = +0.09891, Δloser_share = +0.4875, Δgap = -0.13115, Δtotal_welfare = -0.01108 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0130 → **triggered**.
- total welfare flat (max relative change < 0.5%): max relative change = 6.7140% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.4875 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

### rural_town, 12 candidates

- `individual` vs `party`: Δλ_winner = +0.0033, Δloser_welfare = +0.08544, Δloser_share = +0.4740, Δgap = -0.08716, Δtotal_welfare = -0.01815 → raises loser welfare but mostly by inflating loser share (partial support).
- `score` vs `party`: Δλ_winner = -0.0154, Δloser_welfare = +0.10777, Δloser_share = +0.5276, Δgap = -0.12892, Δtotal_welfare = -0.00688 → raises loser welfare but mostly by inflating loser share (partial support).
- `latent_match` vs `party`: Δλ_winner = -0.0020, Δloser_welfare = +0.10617, Δloser_share = +0.5497, Δgap = -0.13853, Δtotal_welfare = -0.01137 → raises loser welfare but mostly by inflating loser share (partial support).

Falsifier checks:
- λ_winner unchanged across rules (max |Δλ| < 0.02): max |Δλ| = 0.0154 → **triggered**.
- total welfare flat (max relative change < 0.5%): max relative change = 6.7009% → not triggered.
- loser_share rises a lot (> +0.15): max rise = +0.5497 → **triggered**.
- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): not triggered — every gap reduction coincides with higher loser welfare.

## Limitations

- Voters and candidates live in a stylized 5-dimensional project space; real preference structures are higher-dimensional and partly unobservable.
- λ is exogenous by default; strategic candidate behavior, campaigning, and repeated elections are out of scope.
- Party structure is idealized as two brands at cluster means with loyal nomination; real parties select through noisy primaries.
- Supporter definitions differ across paradigms by design (that is part of the treatment), so loser-share differences should be read together with loser welfare, not alone.

## Reproduce

```
python run_experiment.py sweep --trials 100 --populations two_cluster,three_cluster,rural_town --candidates 6,8,12
```

Deterministic given identical parameters and seed (per-trial streams are derived from `numpy.random.default_rng([seed, population, candidates, trial])`).
