# Cooperative Units: Results

**Design:** [Design.md](Design.md) ·
**Raw record:** [`experiments/cooperation/results/REPORT.md`](../../../../experiments/cooperation/results/REPORT.md)
(per-cell runs, founders, units and time series alongside) ·
**Run:** 780 cooperative runs (39 cells × seeds 101–120) and 20 solo twins, registration commit `7649080`.

## Verdicts

| | Hypothesis | Mean seed-paired difference [95% CI] | Seeds in direction | p (Holm) | Verdict |
| --- | --- | --- | --- | --- | --- |
| H1 | Enforced exit cost lengthens unit life | +80.0 ticks [77.5, 82.6] | 20/20 | <0.001 | Supported |
| H2 | Broad scope needs a high exit cost | +8.7 ticks [5.7, 11.8] | 19/20 | <0.001 | Supported |
| H3 | Recognition lengthens unit life at zero exit cost | +4.0 ticks [3.5, 4.6] | 20/20 | <0.001 | Supported |
| H4a | Welfare higher at exit cost 0.3 than 0 | +7.1 ticks [5.1, 9.2] | 19/20 | <0.001 | Significant |
| H4b | Welfare higher at exit cost 0.3 than 0.6 | −3.5 ticks [−5.8, −1.2] | 3/20 | 1.000 | Not significant; the difference runs the other way |
| H4 | Welfare peaks at a moderate exit cost | | | | Not supported |
| H5 | Trapped time rises with exit cost (0.6 vs 0.3) | −0.008 [−0.011, −0.004] | 4/20 | 1.000 | Not supported; slightly lower at 0.6 |

Founder welfare rose monotonically with enforced exit cost (21.8, 28.9 and
32.5 ticks of extra life at exit cost 0, 0.3 and 0.6 with full recognition).
Trapping is real and frequent where exit costs are enforced — trap is the
most common primary diagnosis there — but it did not outweigh the value of
units that last, and the share of member time spent trapped did not grow
from exit cost 0.3 to 0.6.

## Manipulation checks

**M2** passed: with no exit cost nobody is ever trapped.

**M1** failed in the strict sense. With zero recognition an exit cost is
never collected, yet raising the nominal exit cost from 0 to 0.6 shortened
unit life by 1.2 ticks (95% CI −1.7 to −0.7; 18 of 20 seeds) and raised
coverage and welfare. The exploratory follow-up
[`scripts/run_cooperation_m1_followup.py`](../../../../scripts/run_cooperation_m1_followup.py)
re-ran those cells with the agents' settings-fit term switched off: the
contrast became exactly zero on every seed, because the runs became
identical. The channel is therefore the fit term. Agents value an
institution's *stated* settings, so an unenforced exit cost still changes who
accepts a match (economic and religious agents prefer a high exit cost,
hedonists a low one) and how members value staying. This is a property of
the model, not a bug, but it means "unrecognised exit cost is inert" holds
only for enforcement, not for selection.

## The settings grid

Scope-averaged means (20 seeds per cell):

| Recognition | Exit cost | Unit life (RMST) | Welfare gain (ticks) | Coverage | Trap rate | Primary trap | Primary fragility | Healthy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 18.2 | 13.9 | 0.29 | 0 | 0 | 0.85 | 0.03 |
| 0 | 0.3 | 17.7 | 14.8 | 0.31 | 0 | 0 | 0.86 | 0.02 |
| 0 | 0.6 | 17.0 | 18.4 | 0.35 | 0 | 0 | 0.87 | 0.02 |
| 0.5 | 0 | 17.7 | 20.0 | 0.36 | 0 | 0 | 0.86 | 0.02 |
| 0.5 | 0.3 | 56.5 | 23.6 | 0.50 | 0.19 | 0.33 | 0.42 | 0.04 |
| 0.5 | 0.6 | 83.8 | 30.3 | 0.57 | 0.21 | 0.50 | 0.21 | 0.05 |
| 1 | 0 | 22.2 | 21.8 | 0.45 | 0 | 0 | 0.76 | 0.04 |
| 1 | 0.3 | 85.6 | 28.9 | 0.59 | 0.17 | 0.49 | 0.17 | 0.07 |
| 1 | 0.6 | 102.2 | 32.5 | 0.62 | 0.16 | 0.57 | 0.05 | 0.08 |

Scope is the strongest lever on welfare: 10.7, 25.8 and 31.5 extra ticks at
scope 0.2, 0.5 and 0.8, against falling effort (0.90, 0.87, 0.83) as members
keep less of each extra unit they earn. Without enforcement units rarely
outlive their first review: 85–87% of classified units fail by fragility.

## Institutions, structure and motives

| Condition | Unit life | Coverage | Welfare gain | Founders starved (solo 0.60) | Trap rate | Main diagnosis |
| --- | --- | --- | --- | --- | --- | --- |
| Marriage preset | 98.9 | 0.67 | 37.5 | 0.44 | 0.16 | Trap 0.55 |
| Cohabitation preset | 19.4 | 0.37 | 21.5 | 0.55 | 0.02 | Fragility 0.85 |
| Friendship preset | 31.7 | 0.16 | 3.3 | 0.60 | 0 | Fragility 0.63 |
| Business partnership preset | 108.9 | 0.62 | 44.6 | 0.43 | 0.30 | Trap 0.57 |
| Marriage, ranked payouts | 81.6 | 0.65 | 30.9 | 0.52 | 0.14 | Trap 0.34, capture 0.28 |
| Marriage with families | 97.6 | 0.69 | 32.5 | 0.49 | 0.14 | Trap 0.42 |
| Marriage, families arrange | 74.8 | 0.75 | 21.6 | 0.58 | 0.18 | Override 0.62 |
| Marriage, all emotional | 105.4 | 0.79 | 49.8 | 0.40 | 0.05 | Unclassified 0.52, healthy 0.20, trap 0.20 |
| Marriage, all religious | 101.8 | 0.77 | 49.9 | 0.39 | 0.18 | Trap 0.62 |
| Marriage, all political | 98.7 | 0.73 | 35.2 | 0.47 | 0.16 | Trap 0.63 |
| Marriage, all economic | 98.7 | 0.63 | 21.6 | 0.51 | 0.20 | Trap 0.61 |
| Marriage, all hedonist | 123.1 | 0.30 | 5.9 | 0.56 | 0.43 | Trap 0.83 |

## Who benefits

Every institution moved survival from high-skill to low-skill founders.
Mean lifespan change against the solo twin, by founder skill tercile:

| Condition | Low skill | Middle | High skill |
| --- | --- | --- | --- |
| No enforcement (scope 0.8) | +52.8 | +23.5 | −7.2 |
| Enforced exit (scope 0.8, exit 0.6, recognition 1) | +79.1 | +44.6 | −1.9 |
| Marriage preset | +73.6 | +41.3 | −2.4 |
| Business partnership preset | +106.9 | +30.6 | −3.7 |
| Marriage, ranked payouts | +82.5 | +22.1 | −11.9 |
| Marriage, families arrange | +65.5 | +18.6 | −19.3 |

Arranged matches were the costliest institution for high-skill founders:
25% of them starved, against 3% alone. Families pick the most productive
outsider as a partner for their own adult child, and that partner's income
then flows into the match and, as dues, into the arranging family.

The motive rows are exploratory (single-motive populations). The
all-hedonist population is the clearest contrast: hedonists dislike high exit
costs, so few of them marry (coverage 0.30), and those who do mostly end up
trapped (83% of units).

## Deviations

None from the registered protocol. M1's failure and its follow-up are
reported above; the follow-up is exploratory.

## Correction (post-publication)

Code review found that a member's appraisal of its *current* unit credited
dues paid to a parent unit as belonging even when that parent was the other
spouse's family, while the appraisal of a *prospective* unit did not. Only
the outsider in an arranged match was affected. The fix (commit `ba5b6361`)
counts belonging only for a member's own family. The full matrix was re-run
on the registered seeds with that code: 38 of 39 cells reproduce the
original record exactly, and the arranged-match cell changes slightly (unit
life 76.3 → 74.8, trap rate 0.15 → 0.18, high-skill founders starved 24% →
25%; tables above updated). No confirmatory test involves that cell, so no
verdict changes. The raw record under `experiments/cooperation/results/` is
the re-run. Its manifest records commit `fd02b642`; the code that ran was
`ba5b6361`, whose changes were not yet committed when the re-run started.
