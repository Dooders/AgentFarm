# Cooperation in the Spatial World: Does an Institution Evolve When the Brain Chooses It?

**Status:** Registered, not yet run (`farm.runners.intrinsic_cooperation_experiment`, service in `farm.core.cooperation.service`)
**CLI:** [`scripts/run_intrinsic_cooperation_experiment.py`](../../../../scripts/run_intrinsic_cooperation_experiment.py)
**Model:** [Cooperative units, "In the spatial world"](../../../concepts/cooperation.md#in-the-spatial-world)
**Builds on:** [Intrinsic evolution](../intrinsic_evolution/intrinsic_evolution.md)
**Precursor:** [the non-spatial registered run](Design.md) and its [results](RESULTS.md)

This page fixes the question, arms, measures, hypotheses and analysis before
the confirmatory seeds (101–120) are run. The non-spatial run established what
the *institution* does when agents follow fixed decision rules: enforced exit
costs lengthen unit life and raise welfare, trapping is common but not
decisive. This run removes the fixed rules. Agents in the intrinsic-evolution
world decide whether to bond and leave with the same learned policy that
decides whether to gather or attack, and both the terms they offer and their
taste for cooperation are heritable loci under selection.

## 1. Question

In a spatial world where agents gather, reproduce with genetic inheritance and
learn with their own brains, and where cooperation is available only as two
learned actions whose terms are set by heritable genes:

(a) does cooperation take hold at all — do brains learn to bond, and do units
persist; (b) does it change the population's fate; (c) does selection act on
the institution — which settings loci move, and in which direction?

## 2. Arms

Every seed runs three arms with identical world, policy and starting
population; only the cooperation configuration differs.

| Arm | Action space | Cooperation loci | Units |
| --- | --- | --- | --- |
| `baseline` | 8 actions (unchanged world) | absent | none |
| `placebo` | 10 actions (`bond`, `leave` added) | present, randomised at founding, heritable | never form (`bond` is a no-op) |
| `cooperation` | 10 actions | present, randomised at founding, heritable | form, pool, pay out, insure, enforce |

The placebo is the primary control. It carries everything the cooperation arm
carries except a working institution: the larger policy output, the extra
mutation load, actions that can be wasted. Its loci drift neutrally, which
gives the null for the selection hypotheses.

## 3. World and inheritance

The `development` environment (50×50, 30 founders across three agent types,
`max_population` 50 unless overridden) with the intrinsic-goals startup knobs
(initial agent resources 12, 60 resource nodes). Inheritance follows the
intrinsic-evolution policy: Gaussian mutation at rate 0.1 and scale 0.1,
reflecting boundaries, uniform crossover with a co-parent within 10 units
(a unit partner when the agent has one), selection pressure `low`. Initial
diversity: independent mutation at scale 0.2 on every founder, applied after
the cooperation loci are drawn uniformly within their bounds.

Cooperation defaults: `bond_range` 6, `offer_window` 10, terms = mean of the
founders' settings loci, switching allowed, synergy 0.2, payout rate 0.8,
reserve 2 per participant, aid 1 below 2, membership reward scale 1 with
horizon 20, fit scale 2, affinity scale 4; partners as allies; partner
co-parenting; no family units. 1,000 steps per arm.

## 4. Measures

Per arm and seed, from the per-step record:

* **Coverage:** share of alive agents in a unit (mean over the run; final).
* **Bond share (late):** share of agents whose last action was `bond`, averaged over the last quarter of the run.
* **Population:** mean, peak and final; births and deaths.
* **Units:** formed, mean lifetime, failure-mode prevalence (the non-spatial diagnostics applied to the units' ledgers).
* **Loci:** population mean of each cooperation locus at the start and end; **settings drift** = mean absolute change of the six settings loci.

## 5. Hypotheses

Each is a seed-paired, one-sided Wilcoxon signed-rank test of the
`cooperation − placebo` difference across the 20 seeds. The five form one
family, Holm-corrected at α = 0.05. Effect sizes are the mean paired
difference with a percentile bootstrap 95% CI (10,000 resamples).

| | Hypothesis | Metric |
| --- | --- | --- |
| S1 | Cooperation takes hold: coverage exceeds the placebo's (which is zero by construction). | mean coverage |
| S2 | The brain learns to bond: late in the run `bond` is chosen more often than in the placebo, where it never pays. | late bond share |
| S3 | Cooperation sustains a larger population. | mean population |
| S4 | Selection acts on the institution: the settings loci move further from their founding means than under neutral drift. | settings drift |
| S5 | Selection favours binding terms: the mean exit-cost locus ends higher than under neutral drift. | mean `coop_exit_cost` at end |

S1 is close to a manipulation check (any formed unit makes it true in that
seed); it is kept in the family because the policy can in principle learn
never to bond. S5 is the one directional claim about *which* institution
evolves, taken from the non-spatial result that enforced exit costs raised
founder welfare; it may fail if trapping costs dominate once agents can be
kept in a unit against a learned preference.

**Manipulation checks** (not in the family): placebo coverage is exactly zero
(M1); `placebo − baseline` population difference is reported two-sided as the
cost of carrying the extra actions and loci (M2).

## 6. Exploratory analyses

Which failure modes the evolved units show; the joint distribution of the
settings loci at the end (do scope and exit cost co-evolve); motive-locus
drift; the `cooperation − baseline` contrast; runs with `family_on_birth`,
`terms="initiator"`, `fixed_settings` presets and frozen loci
(`--no-evolve-genes`) as follow-ups. Descriptive only.

## 7. Known limitations

* The membership reward is an intrinsic term the designer added; it is what
  lets motives bear on a learned decision, but it also means part of the
  "taste for cooperation" is hand-placed rather than evolved. The scale is a
  parameter (`membership_reward_scale`, 0 switches it off) and is not swept
  in the registered run.
* Checkpoint resume does not restore live units; the registered run does not
  resume.
* Bond formation is local (`bond_range`) and the policy sees partners only as
  allies in its observation; it has no channel showing who is *open*, so the
  bond action is partly a gamble on nearby intent.
* The credit-assignment fix in `LearningAgentBehavior` (see commit message)
  changes how every learning run trains; results are not comparable with
  runs made before it.

## Appendix A. Parameters

| Group | Parameter | Value |
| --- | --- | --- |
| World | environment / steps | `development` / 1,000 |
| World | initial agent resources / resource nodes | 12 / 60 |
| Inheritance | mutation rate / scale / mode / boundary | 0.1 / 0.1 / gaussian / reflect |
| Inheritance | crossover / co-parent radius | uniform / 10 |
| Inheritance | selection pressure | low |
| Inheritance | initial diversity scale | 0.2 |
| Cooperation | bond range / offer window / max candidates | 6 / 10 / 5 |
| Cooperation | terms / switching | mean / allowed |
| Cooperation | synergy / payout rate / reserve per participant | 0.2 / 0.8 / 2 |
| Cooperation | aid threshold / aid amount | 2 / 1 |
| Cooperation | membership reward scale / horizon / fit scale / affinity scale | 1 / 20 / 2 / 4 |
| Cooperation | partners as allies / membership marker / partner co-parent / family on birth | yes / yes / yes / no |
| Genes | settings loci defaults (scope, exit cost, recognition, symmetry, exclusivity, max members) | 0.5, 0.3, 0.5, 1.0, 1.0, 2 (founders drawn uniformly) |
| Genes | motive loci / propensities (bond, leave) | 0.5 each / 0.1, 0.05 (founders drawn uniformly) |
| Analysis | α (Holm) / bootstrap resamples / seeds | 0.05 / 10,000 / 101–120 |
