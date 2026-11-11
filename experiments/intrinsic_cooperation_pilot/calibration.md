# Intrinsic cooperation calibration

Steps 3000, seeds 1, arms ['baseline', 'placebo', 'cooperation', 'cooperation_no_motive'], cap 300, selection medium, temperature 0.1, initial diversity None, frozen A/C True, commit `13ffe91e44`.

## Gates

| Gate | Result | Detail |
| --- | --- | --- |
| learning_L1 | pass | Policies prefer gather over uniform late in the run (L1 > 0) in every baseline replicate. L1 per replicate: [0.157, 0.173] |
| learning_L2 | pass | Late gather share exceeds the innate prior (L2 > 0) in every baseline replicate. L2 per replicate: [0.205, 0.22] |
| units_form | pass | Units form in the cooperation arm (mean coverage > 0.05 and units_formed > 0 in every replicate). mean coverage [0.5, 0.514], units formed [874, 978] |
| generations | pass | Selection has material: max_generation >= 3 in every cooperation replicate. max generation [3, 3] |
| exit_cost_bounded | pass | Enforced exits cost something finite and bounded (0 < forfeits per enforced exit < 10). forfeits per enforced exit [0.654, 0.384] |
| not_pinned | pass | The population is regulated below the cap: mean_population < 0.95 x max_population in every arm. mean population / cap: baseline 0.11, placebo 0.12, cooperation 0.15, cooperation_no_motive 0.17 |

## Learning positive controls

- **L1** (`policy_gather_lift_late`, baseline): [0.157, 0.173]
- **L2** (`gather_share_over_prior_late`, baseline): [0.205, 0.22]

| Arm | gather lift (policy − uniform) | gather share − prior |
| --- | --- | --- |
| baseline | 0.165 ± 0.011 | 0.213 ± 0.011 |
| placebo | 0.162 ± 0.010 | 0.219 ± 0.007 |
| cooperation | 0.099 ± 0.000682 | 0.163 ± 0.007 |
| cooperation_no_motive | 0.093 ± 0.016 | 0.156 ± 0.017 |

## Regime (mean ± sd over replicates)

| Metric | baseline | placebo | cooperation | cooperation_no_motive |
| --- | --- | --- | --- | --- |
| `mean_population` | 33.224 ± 0.957 | 35.228 ± 0.821 | 45.345 ± 2.239 | 49.949 ± 2.727 |
| `peak_population` | 70.000 ± 2.828 | 67.500 ± 2.121 | 72.000 ± 7.071 | 73.000 ± 5.657 |
| `final_population` | 32.500 ± 3.536 | 38.500 ± 4.950 | 48.000 ± 1.414 | 50.000 ± 7.071 |
| `total_births` | 183.000 ± 12.728 | 234.000 ± 74.953 | 164.500 ± 60.104 | 205.500 ± 75.660 |
| `total_deaths` | 180.500 ± 9.192 | 225.500 ± 79.903 | 146.500 ± 58.690 | 185.500 ± 68.589 |
| `starvation_deaths` | 179.500 ± 7.778 | 225.500 ± 79.903 | 146.500 ± 58.690 | 185.500 ± 68.589 |
| `combat_deaths` | 1.000 ± 1.414 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| `mean_lifespan_at_death` | 167.596 ± 0.783 | 156.694 ± 40.575 | 241.134 ± 2.331 | 230.204 ± 39.464 |
| `max_generation` | 2.000 ± 0 | 2.500 ± 0.707 | 3.000 ± 0 | 3.000 ± 0 |
| `mean_coverage` | 0 ± 0 | 0 ± 0 | 0.507 ± 0.009 | 0.568 ± 0.023 |
| `final_coverage` | 0 ± 0 | 0 ± 0 | 0.564 ± 0.076 | 0.587 ± 0.044 |
| `units_formed` | 0 ± 0 | 0 ± 0 | 926.000 ± 73.539 | 1322.000 ± 213.546 |
| `mean_unit_lifetime` | 0 ± 0 | 0 ± 0 | 33.943 ± 1.688 | 32.239 ± 10.105 |
| `forfeits_per_enforced_exit` | nan ± nan | nan ± nan | 0.519 ± 0.190 | 0.605 ± 0.279 |
| `action_share[bond]` | 0 ± 0 | 0.082 ± 0.027 | 0.109 ± 0.019 | 0.116 ± 0.022 |
| `action_share[leave]` | 0 ± 0 | 0 ± 0 | 0.016 ± 0.006 | 0.016 ± 0.007 |
| `policy_bond_prob_late` | 0 ± 0 | 0.094 ± 0.016 | 0.098 ± 0.000631 | 0.103 ± 0.009 |
| `acting_bond_prob_late` | nan ± nan | 0.078 ± 0.015 | 0.115 ± 0.026 | 0.117 ± 0.021 |
| `bond_rank_late` | nan ± nan | 5.074 ± 0.630 | 5.142 ± 0.066 | 4.982 ± 0.336 |
| `policy_peak_late` | 0.389 ± 0.005 | 0.367 ± 0.009 | 0.309 ± 0.007 | 0.295 ± 0.002 |

## Locus drift (run-long change in the population mean)

| Locus | placebo mean ± sd | cooperation mean ± sd | paired diff mean ± sd |
| --- | --- | --- | --- |
| `change_coop_scope` | -0.009 ± 0.017 | -0.019 ± 0.044 | -0.009 ± 0.061 |
| `change_coop_exit_cost` | 0.022 ± 0.022 | -0.004 ± 0.021 | -0.027 ± 0.043 |
| `change_coop_recognition` | 0.016 ± 0.064 | 0.057 ± 0.007 | 0.041 ± 0.071 |
| `change_coop_symmetry` | 0.001 ± 0.012 | -0.036 ± 0.081 | -0.037 ± 0.069 |
| `change_coop_exclusivity` | 0.015 ± 0.018 | 0.002 ± 0.014 | -0.013 ± 0.005 |
| `change_coop_max_members` | 0.130 ± 0.225 | 0.149 ± 0.010 | 0.019 ± 0.234 |
| `change_motive_political` | -0.011 ± 0.077 | 0.003 ± 0.087 | 0.014 ± 0.010 |
| `change_motive_economic` | 0.015 ± 0.048 | 0.018 ± 0.071 | 0.004 ± 0.023 |
| `change_motive_emotional` | 0.048 ± 0.032 | -0.022 ± 0.019 | -0.070 ± 0.013 |
| `change_motive_hedonist` | -0.069 ± 0.032 | -0.096 ± 0.010 | -0.027 ± 0.042 |
| `change_motive_religious` | 0.024 ± 0.050 | -0.054 ± 0.060 | -0.077 ± 0.009 |
| `change_bond_weight` | -0.008 ± 0.011 | 0.020 ± 0.025 | 0.027 ± 0.014 |
| `change_leave_weight` | -0.029 ± 0.017 | -0.030 ± 0.000261 | -0.00096 ± 0.018 |
| `change_expected_exit_payment` | 0.006 ± 0.021 | 0.017 ± 0.019 | 0.012 ± 0.002 |

## Power at n = 20 (paired, Holm strictest alpha = 0.007143, family of 7)

| Key | Metric | Direction | Pilot mean diff | Pilot sd | dz (toward hypothesis) | Power | MDE dz | n for 80% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2 | `policy_bond_prob_late` | greater | 0.004 | 0.015 | 0.265 | 0.083 | 0.822 | 165 |
| S3 | `mean_population` | greater | 10.117 | 3.060 | 3.306 | 1.000 | 0.822 | 5 |
| S3b | `mean_lifespan_at_death` | greater | 84.440 | 42.905 | 1.968 | 1.000 | 0.822 | 7 |
| S4_scope | `change_coop_scope` | two-sided | -0.009 | 0.061 | -0.154 | 0.019 | 0.899 | 552 |
| S4_exit_cost | `change_coop_exit_cost` | two-sided | -0.027 | 0.043 | -0.618 | 0.399 | 0.899 | 38 |
| S4_recognition | `change_coop_recognition` | two-sided | 0.041 | 0.071 | 0.580 | 0.343 | 0.899 | 43 |
| S5 | `change_expected_exit_payment` | greater | 0.012 | 0.002 | 6.463 | 1.000 | 0.822 | 4 |

Power uses a paired t at the Wilcoxon asymptotic relative efficiency (0.955 n); pilot effect sizes from three seeds are rough and the MDE column is the number to plan against.
