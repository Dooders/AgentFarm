# Union emergence (first_glance)

| world | arm | synergy | energy | paired | duration | Δ commit | Δ fidelity | extraction | lineages |
|---|---|---|---|---|---|---|---|---|---|
| baseline | solo_only | — | 7.8 | 0.000 | — | -0.003 | 0.015 | — | 34.0 |
| baseline | promiscuous | — | 8.4 | 0.000 | — | -0.027 | 0.028 | — | 31.0 |
| baseline | optional_union | 1.174 | 12.0 | 0.842 | 18.6 | -0.001 | 0.017 | 0.090 | 45.5 |
| baseline | forced_union | 2.471 | 3.1 | 0.825 | 8.3 | 0.011 | 0.059 | 0.083 | 29.0 |
| cheap_exit_cheap_bond | solo_only | — | 7.8 | 0.000 | — | -0.003 | 0.015 | — | 34.0 |
| cheap_exit_cheap_bond | promiscuous | — | 8.4 | 0.000 | — | -0.027 | 0.028 | — | 31.0 |
| cheap_exit_cheap_bond | optional_union | 1.128 | 15.9 | 0.877 | 20.3 | 0.007 | -0.001 | 0.067 | 48.0 |
| cheap_exit_cheap_bond | forced_union | 1.843 | 15.8 | 0.971 | 19.2 | 0.005 | -0.001 | 0.065 | 48.0 |
| no_courtship | solo_only | — | 7.8 | 0.000 | — | -0.003 | 0.015 | — | 34.0 |
| no_courtship | promiscuous | — | 8.4 | 0.000 | — | -0.027 | 0.028 | — | 31.0 |
| no_courtship | optional_union | 1.174 | 12.5 | 0.841 | 20.2 | -0.015 | 0.010 | 0.077 | 46.5 |
| no_courtship | forced_union | 2.002 | 8.9 | 0.893 | 13.3 | 0.010 | 0.014 | 0.079 | 41.5 |
| wide_neighborhood | solo_only | — | 7.9 | 0.000 | — | -0.027 | 0.020 | — | 32.5 |
| wide_neighborhood | promiscuous | — | 8.8 | 0.000 | — | -0.022 | 0.020 | — | 34.0 |
| wide_neighborhood | optional_union | 1.422 | 11.7 | 0.897 | 19.9 | 0.003 | 0.014 | 0.072 | 47.0 |
| wide_neighborhood | forced_union | 3.350 | 1.5 | 0.825 | 6.4 | 0.017 | 0.029 | 0.067 | 20.0 |

Win-condition direction checks: 6/9.
- PASS: baseline_forced_synergy_gt_1
- PASS: baseline_optional_synergy_gt_1
- FAIL: cheap_exit_does_not_lengthen_bonds
- PASS: cheap_exit_drops_or_holds_optional_synergy
- FAIL: forced_extraction_gt_optional
- FAIL: no_courtship_inflates_optional_synergy
- PASS: optional_commitment_does_not_climb
- PASS: optional_fidelity_rises_on_baseline
- PASS: wide_neighborhood_flattens_fidelity
