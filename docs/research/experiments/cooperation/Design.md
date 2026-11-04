# Cooperative Units: Which Institutional Settings Make Cooperation Persist and Pay?

**Status:** Registered and run (`farm.experiments.cooperation`, model in `farm.core.cooperation`)
**Results:** [RESULTS.md](RESULTS.md)
**CLI:** [`scripts/run_cooperation.py`](../../../../scripts/run_cooperation.py)
**Model:** [Cooperative units](../../../concepts/cooperation.md)
**Package README:** [`farm/experiments/cooperation/README.md`](../../../../farm/experiments/cooperation/README.md)

This page is the registration for the confirmatory run. It fixes the
question, conditions, measures, hypotheses and analysis before any data from
the confirmatory seeds (101–120) exist. Calibration used seeds 1–2 only; §8
lists what calibration changed. Appendix A lists every parameter value.

## Abstract

Cooperative institutions differ along a small number of settings: how many
members, how much of their income is pooled, what leaving costs, whether
outsiders enforce the arrangement, whether membership is exclusive, how
members enter, and whether shares are equal. The model represents every
institution as one object, a *cooperative unit*, configured by those seven
settings, and the experiment sweeps them in a population of agents with
heterogeneous skill, private motives and idiosyncratic bad luck. It asks
which settings make units persist, which leave members better off than they
would have been alone, and how units fail when they do. Each cooperative run
is paired with a *solo twin* in which the same founders face the same shocks
with cooperation switched off, so welfare effects are measured founder by
founder.

## 1. Question

For a pair institution, how do scope, exit cost and recognition change
(a) how long units last, (b) how much better off founders are than their
solo selves, and (c) how often members are kept in a unit they would leave
if leaving were free? Secondary: how do named institutions (preset settings),
rank-ordered payouts and nesting inside families compare?

## 2. Model

The model is documented in [docs/concepts/cooperation.md](../../../concepts/cooperation.md).
In brief: members pool `scope` of their income; the pool creates a surplus
`synergy × inflow × (1 − 1/n)`; the unit keeps a reserve of
`reserve_target` per participant, pays emergency aid to members below the aid
threshold, and pays out the rest by rank weights. An exit forfeits
`exit_cost × holdings` to the unit, but only when an enforcement draw with
probability `recognition` succeeds; recognition also blocks outside offers
to members of exclusive units and makes a parent's veto stick.

Agents value a membership with one number,

```
V = H · (w_wealth · net + w_security · insurance + w_belonging · parent_gain)
    + k_fit · fit(settings) + k_aff · w_affinity · compatibility
```

where the weights come from each agent's motive (political, economic,
emotional, hedonist, religious). They join when V clears a threshold, leave
when V is below minus the expected exit payment, and are *trapped* when V is
negative but leaving would cost more.

## 3. World and agents

A non-spatial world of 60 founders running 600 ticks. Each tick an agent earns
`skill × effort`, earns nothing with probability 0.15, earns 40% during a
shared 20-tick drought every 120 ticks, and loses income to crowding above
70 living agents (`× 70/N`). It pays metabolism plus a quadratic effort cost,
dies at zero energy or age 300, and reproduces above 22 energy (hard cap
120). Skill is log-normal (median 1.0, σ 0.35); diligence is Beta(2, 2);
each agent has one motive (uniform over the five) and an affinity vector.
Offspring draw fresh traits, so the trait distribution is the same in every
condition and over time.

Effort is the privately optimal level given the agent's *marginal retention*
(the fraction of an extra unit of own income that comes back to it, which
falls with scope), blended toward full effort by diligence. Pooling therefore
carries a free-rider cost.

Each tick: work and pool → settle (surplus, dues, aid, payouts) → deaths →
membership reviews (every 10 ticks after a 10-tick tenure) → arranged
matches (nesting condition only) → one random encounter per adult →
births.

**Solo twin.** For every seed, one run with cooperation off. Each agent's
shocks come from its own random stream keyed by seed and slot, and founders'
traits and the drought phase are drawn from seed-keyed streams, so every
founder exists, with the same traits and the same shock sequence, in every
condition and in the twin.

## 4. Conditions

| Family | Cells | Definition |
| --- | --- | --- |
| Grid (primary) | 27 | Exclusive, chosen, symmetric pair; scope ∈ {0.2, 0.5, 0.8} × exit cost ∈ {0, 0.3, 0.6} × recognition ∈ {0, 0.5, 1} |
| Presets | 4 | `marriage`, `cohabitation`, `friendship` (2–6 members, non-exclusive), `business_partnership` (2–4, non-exclusive) |
| Structure | 3 | `marriage_ranked` (symmetry 0); `marriage_lineage` (couples that reproduce anchor a family unit, children born into it); `marriage_arranged` (families arrange matches for adult children; vetoes active) |
| Motive (exploratory) | 5 | The marriage preset with an all-one-motive population |

20 seeds per cell (101–120) plus 20 solo twins.

## 5. Measures

* **Bond life:** restricted mean survival time (RMST) of bond units up to
  τ = 200 ticks, from a Kaplan–Meier estimate (units alive at the end are
  censored). Also survival at 100 ticks.
* **Welfare:** per founder, lifespan in the cooperative run minus lifespan in
  its solo twin; per run, the mean over founders. Also the shares better and
  worse off, starvation with and without cooperation, and the gain by skill
  tercile.
* **Trap rate:** member-steps spent wanting out but priced in, over all
  member-steps in bonds.
* **Failure modes:** the primary diagnosis of each bond (fragility, override,
  capture, trap, drift, hollow, healthy, unclassified) with the thresholds in
  Appendix A.

## 6. Hypotheses

All contrasts are within seed, averaged over the three scope levels unless
stated, and tested one-sided with a Wilcoxon signed-rank test across the 20
seeds.

| | Hypothesis | Test |
| --- | --- | --- |
| H1 | With full recognition, a higher exit cost lengthens bond life. | RMST(E 0.6, R 1) − RMST(E 0, R 1) > 0 |
| H2 | Broad scope needs a high exit cost: the scope penalty on bond life at exit cost 0 shrinks at exit cost 0.6. | [RMST(S .8, E .6) − RMST(S .2, E .6)] − [RMST(S .8, E 0) − RMST(S .2, E 0)] > 0, at R 1 |
| H3 | With no exit cost, recognition still lengthens bond life by blocking outside offers. | RMST(E 0, R 1) − RMST(E 0, R 0) > 0 |
| H4 | Founder welfare peaks at a moderate exit cost, because high exit costs trap. | Both W(E .3) − W(E 0) > 0 (H4a) and W(E .3) − W(E .6) > 0 (H4b), at R 1 |
| H5 | Trapped time rises with exit cost. | Trap(E .6, R 1) − Trap(E .3, R 1) > 0 |

The six tests (H1, H2, H3, H4a, H4b, H5) form one family, Holm-corrected at
α = 0.05. H4 is supported only if H4a and H4b both survive correction.
Effect sizes are reported as the mean seed-paired difference with a
percentile bootstrap 95% CI (10,000 resamples).

**Manipulation checks** (not in the family): with zero recognition, exit cost
should not change bond life (two-sided, M1); with zero exit cost, the trap
rate should be zero at every recognition level (M2).

## 7. Exploratory analyses

Named presets, rank-ordered payouts (capture), nesting (override, drift,
family exits), all-one-motive populations, who benefits by skill tercile,
coverage and effort. These are descriptive and carry no inferential claims.

## 8. Calibration (seeds 1–2)

The hypotheses were drafted after a single-seed smoke test of the
uncalibrated model and before calibration, and were not changed afterwards.
Calibration pilots on seeds 1–2 were seen in full, including a final pass
through the analysis pipeline whose point estimates for H1–H5 were visible
(with two seeds no test can reach significance). Calibration changed the
world, not the hypotheses:

1. Costs recalibrated (metabolism 0.2, quadratic effort cost 1.0) so that
   effort responds to scope; in the first version effort stayed near 1.
2. A storage limit on energy (40) and a carrying capacity (70) were added.
   Without them the population sat at the hard cap with births blocked tens
   of thousands of times, and hoarded energy made holdings-proportional exit
   payments enormous.
3. Offspring traits are drawn fresh instead of inherited, so the trait
   distribution stays fixed; with inheritance, high-skill lineages took over
   within a few hundred ticks and conditions drifted into different worlds.
4. Members review memberships every 10 ticks rather than every tick (with
   per-tick reviews, decision noise alone dissolved near-zero-value units),
   and a member's running benefit estimate starts from what it expected
   rather than from the first, reserve-building step.
5. Fragility counts only exit-driven dissolutions; units ended by a member's
   death are not institutional failures.

## 9. Known limitations

* Non-spatial matching; agents meet uniformly at random.
* Decision rules are fixed heuristics, not learned policies.
* Crowding couples runs: an institution that keeps more agents alive lowers
  everyone's income, so welfare is a total effect, crowding included.
* The valuation scales (horizon, fit and affinity weights) are not swept in
  the registered run.

## Appendix A. Parameters

| Group | Parameter | Value |
| --- | --- | --- |
| World | founders / hard cap / carrying capacity | 60 / 120 / 70 |
| World | ticks | 600 |
| World | skill | log-normal, median 1.0, σ 0.35 |
| World | shock probability | 0.15 |
| World | drought | every 120 ticks for 20, income × 0.4 |
| World | metabolism / effort cost (½·c·e²) / minimum effort | 0.2 / 1.0 / 0.2 |
| World | initial / maximum energy | 8 / 40 |
| World | reproduction threshold / offspring cost / offspring energy | 22 / 12 / 8 |
| World | maturity / maximum age | 20 / 300 |
| World | trait and motive heritability / affinity noise | 0 / 0 / 0.15 |
| Decisions | encounters per tick | 1 |
| Decisions | horizon H / fit weight / affinity weight | 20 / 1.0 / 2.0 |
| Decisions | accept threshold / switch premium / logistic noise scale | 0.5 / 1.0 / 0.5 |
| Decisions | minimum tenure / review interval | 10 / 10 |
| Decisions | income-signal noise (log-normal σ) / income EMA α | 0.2 / 0.05 |
| Decisions | diligence | Beta(2, 2) |
| Decisions | motive mix | uniform over five motives |
| Pools | synergy / payout rate | 0.2 / 0.8 |
| Pools | aid threshold / aid amount | 2 / 1 |
| Pools | reserve target per participant | 2 |
| Pools | benefit EMA α | 0.1 |
| Diagnostics | minimum steps / window | 20 / 50 |
| Diagnostics | capture share / gap | 0.6 / 0.25 |
| Diagnostics | hollow throughput | 0.05 |
| Diagnostics | fragility lifetime (exit-driven only) | 30 |
| Diagnostics | trap steps / loss tolerance | 15 / 0.02 |
| Analysis | RMST τ / survival horizon | 200 / 100 |
| Analysis | α (Holm) / bootstrap resamples | 0.05 / 10,000 |
| Analysis | seeds | 101–120 |
