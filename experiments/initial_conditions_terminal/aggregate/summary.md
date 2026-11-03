# Initial conditions and the terminal population

Baseline arm only. Food budget pinned at 60 resource nodes, medium selection pressure, population cap 300, 3,000 steps, `policy_temperature` 0.1, platform genes frozen. 48 finished runs.

The late-window mean is the primary endpoint. A directional claim counts only when its 95% interval excludes zero and the sign agrees in at least 5 of 6 seeds.

## Per-cell means

| Founder | Offspring | Peak | Peak step | Overshoot | Early amplitude | Final | Late-window mean | Starvation deaths | Lifespan | Max generation | Births |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 5 | 71.0 [62.3, 79.7] | 114 [104, 123] | 40.0 [31.8, 48.1] | 41.0 [32.3, 49.7] | 29.7 [28.0, 31.4] | 31.0 [29.6, 32.5] | 212 [171, 253] | 169.1 [141.4, 196.9] | 3.17 [2.74, 3.60] | 212 [169, 256] |
| 5 | 8 | 76.3 [70.1, 82.5] | 123 [112, 135] | 43.7 [37.1, 50.2] | 46.3 [40.1, 52.5] | 35.3 [29.7, 41.0] | 32.7 [31.2, 34.1] | 196 [178, 214] | 191.1 [179.4, 202.7] | 3.50 [2.93, 4.07] | 202 [182, 222] |
| 8 | 5 | 74.0 [63.9, 84.1] | 117 [97, 137] | 41.8 [32.3, 51.2] | 44.0 [33.9, 54.1] | 32.3 [28.4, 36.3] | 32.2 [29.8, 34.7] | 206 [164, 247] | 165.5 [133.4, 197.7] | 2.83 [2.04, 3.62] | 208 [165, 252] |
| 8 | 8 | 77.2 [67.1, 87.2] | 122 [110, 135] | 43.7 [33.8, 53.6] | 47.2 [37.1, 57.2] | 33.0 [29.7, 36.3] | 33.4 [31.1, 35.8] | 225 [198, 253] | 179.4 [158.9, 200.0] | 3.33 [2.79, 3.88] | 229 [199, 259] |
| 12 | 5 | 72.7 [65.6, 79.7] | 110 [89, 131] | 41.1 [33.3, 48.9] | 42.3 [34.9, 49.7] | 33.8 [31.2, 36.4] | 31.5 [30.3, 32.8] | 214 [186, 242] | 161.9 [141.9, 181.9] | 2.33 [1.79, 2.88] | 218 [191, 245] |
| 12 | 8 | 78.5 [74.4, 82.6] | 112 [99, 126] | 46.3 [43.3, 49.3] | 48.2 [44.0, 52.3] | 30.2 [26.2, 34.2] | 32.2 [29.3, 35.1] | 197 [141, 254] | 190.6 [161.3, 219.9] | 3.00 [3.00, 3.00] | 198 [140, 255] |
| 20 | 5 | 75.5 [68.4, 82.6] | 104 [97, 112] | 42.8 [35.0, 50.6] | 45.2 [37.9, 52.4] | 30.5 [26.4, 34.6] | 32.7 [31.3, 34.1] | 226 [204, 249] | 160.9 [147.2, 174.6] | 2.83 [2.04, 3.62] | 228 [207, 248] |
| 20 | 8 | 83.3 [78.8, 87.9] | 129 [114, 143] | 50.2 [43.6, 56.8] | 53.0 [49.0, 57.0] | 33.3 [29.8, 36.9] | 33.1 [30.4, 35.8] | 193 [145, 241] | 200.0 [158.5, 241.5] | 3.33 [2.79, 3.88] | 197 [146, 248] |

## H0 — slope of late-window mean on overshoot

Pooled across founder levels within each offspring level. Ordinary least squares treats each (founder, seed) row as a point. Seed agreement is the sign of that seed's own slope across founder levels.

| Offspring | n | Slope | 95% CI | Covers zero | Seeds agreeing | Directional claim |
| ---: | ---: | ---: | ---: | --- | ---: | --- |
| 5 | 24 | -0.008 | [-0.105, 0.090] | yes | 5/6 | none |
| 8 | 24 | -0.084 | [-0.224, 0.056] | yes | 5/6 | none |

## H1 — offspring 8 minus offspring 5

Paired by seed at each founder level.

| Founder | Metric | Mean difference | 95% CI | Excludes zero | Sign counts (+/−) | Directional claim |
| ---: | --- | ---: | ---: | --- | ---: | --- |
| 5 | `late_window_mean` | 1.64 | [-0.08, 3.35] | no | 6/0 | none |
| 8 | `late_window_mean` | 1.20 | [-1.57, 3.97] | no | 4/2 | none |
| 12 | `late_window_mean` | 0.64 | [-3.22, 4.50] | no | 3/3 | none |
| 20 | `late_window_mean` | 0.42 | [-2.90, 3.74] | no | 3/3 | none |
| 5 | `mean_lifespan_at_death` | 21.95 | [-5.85, 49.75] | no | 4/2 | none |
| 8 | `mean_lifespan_at_death` | 13.90 | [-10.05, 37.84] | no | 5/1 | none |
| 12 | `mean_lifespan_at_death` | 28.71 | [-12.81, 70.24] | no | 4/2 | none |
| 20 | `mean_lifespan_at_death` | 39.15 | [3.58, 74.71] | yes | 6/0 | positive |
| 5 | `max_generation` | 0.33 | [-0.21, 0.88] | no | 2/0 | none |
| 8 | `max_generation` | 0.50 | [-0.38, 1.38] | no | 2/0 | none |
| 12 | `max_generation` | 0.67 | [0.12, 1.21] | yes | 4/0 | none |
| 20 | `max_generation` | 0.50 | [-0.38, 1.38] | no | 2/0 | none |

## Buffer contrast

The resource-buffer comparison moved founder resources, node count, and regeneration together. On that axis the six-seed mean populations were 84 / 96 / 108 (conservative / balanced / buffered) and scaled with the ongoing food budget, while the single-seed peaks (137 / 120 / 171 on seed 42) did not line up with the endings. This sweep holds the food budget fixed, so a slope of late-window population on overshoot is the boom's own association, not the food supply's.
