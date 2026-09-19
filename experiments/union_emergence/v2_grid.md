# Union emergence (v2)

| world | arm | synergy | energy | paired | duration | Δ commit | Δ fidelity | extraction | lineages |
|---|---|---|---|---|---|---|---|---|---|
| baseline | solo_only | — | 15.0 | 0.000 | — | -0.022 | 0.007 | — | 29.8 |
| baseline | promiscuous | — | 17.3 | 0.000 | — | -0.009 | 0.011 | — | 29.5 |
| baseline | optional_union | 1.153 | 27.9 | 0.875 | 23.5 | 0.007 | 0.012 | 0.081 | 45.8 |
| baseline | forced_union | 2.580 | 18.7 | 0.906 | 16.3 | 0.021 | 0.078 | 0.065 | 31.2 |
| cheap_exit_cheap_bond | solo_only | — | 15.0 | 0.000 | — | -0.022 | 0.007 | — | 29.8 |
| cheap_exit_cheap_bond | promiscuous | — | 17.3 | 0.000 | — | -0.009 | 0.011 | — | 29.5 |
| cheap_exit_cheap_bond | optional_union | 1.031 | 36.4 | 0.888 | 24.9 | 0.007 | -0.001 | 0.071 | 48.0 |
| cheap_exit_cheap_bond | forced_union | 2.231 | 36.3 | 0.983 | 23.4 | 0.006 | -0.004 | 0.056 | 48.0 |
| no_courtship | solo_only | — | 15.0 | 0.000 | — | -0.022 | 0.007 | — | 29.8 |
| no_courtship | promiscuous | — | 17.3 | 0.000 | — | -0.009 | 0.011 | — | 29.5 |
| no_courtship | optional_union | 1.139 | 28.2 | 0.869 | 23.7 | -0.002 | 0.000 | 0.082 | 47.0 |
| no_courtship | forced_union | 2.625 | 24.2 | 0.934 | 19.2 | 0.023 | 0.014 | 0.065 | 40.2 |
| wide_neighborhood | solo_only | — | 15.1 | 0.000 | — | -0.023 | 0.008 | — | 29.2 |
| wide_neighborhood | promiscuous | — | 16.5 | 0.000 | — | -0.019 | 0.010 | — | 30.5 |
| wide_neighborhood | optional_union | 1.364 | 27.1 | 0.915 | 23.5 | 0.000 | 0.014 | 0.073 | 45.5 |
| wide_neighborhood | forced_union | 5.446 | 9.5 | 0.893 | 10.9 | 0.051 | 0.128 | 0.063 | 19.5 |

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
