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

Every seed runs four arms with identical world, policy and starting
population; only the cooperation configuration differs.

| Arm | Action space | Cooperation loci | Units | Membership reward |
| --- | --- | --- | --- | --- |
| `baseline` | 8 actions (unchanged world) | absent | none | — |
| `placebo` | 10 actions (`bond`, `leave` added) | present, randomised at founding, heritable | never form (`bond` is a no-op) | — |
| `cooperation` | 10 actions | present, randomised at founding, heritable | form, pool, pay out, insure, enforce | on (scale 1) |
| `cooperation_no_motive` | 10 actions | as above | as above | off (scale 0) |

The placebo is the primary control. It carries everything the cooperation arm
carries except a working institution: the larger policy output, the extra
mutation load, actions that can be wasted. Its loci drift neutrally, which
gives the null for the selection hypotheses.

The fourth arm removes the one reward term the designer adds. The
membership reward can be as large as a full gather per step for a well-fit
member, so without this arm a positive result could be the reward term
rather than the institution. The registered hypotheses are tested on
`cooperation − placebo`; `cooperation_no_motive − placebo` is reported for
every metric as the institution's effect without that term, and
`cooperation − cooperation_no_motive` as the term's own contribution.

## 3. World and inheritance

The `development` environment (50×50, 30 founders across three agent types)
with the intrinsic-goals startup knobs (initial agent resources 12, 60
resource nodes) and the population cap raised from 50 to 300. At 50 every
arm sits on the cap and selection has nothing to act on (the intrinsic-goals
experiment documents the same problem); density-dependent reproduction cost
regulates the population instead. Inheritance follows the
intrinsic-evolution policy: Gaussian mutation at rate 0.1 and scale 0.1,
reflecting boundaries, uniform crossover with a co-parent within 10 units
(a unit partner when the agent has one), selection pressure `low`. Initial
diversity: independent mutation at scale 0.2 on every founder, applied after
the cooperation loci are drawn uniformly within their bounds.

Cooperation defaults: `bond_range` 6, `offer_window` 10, terms = mean of the
founders' settings loci, switching allowed, synergy 0.2, payout rate 0.8,
reserve 2 per participant, aid 1 below 2, membership reward scale 1 (0 in
the fourth arm) with horizon 20, fit scale 2, affinity scale 4; partners
added to the allies channel (same-type agents remain allies, so perception
is unchanged for anyone not in a unit); partner co-parenting; no family
units. The propensity loci (`bond_weight`, `leave_weight`) are bounded at
0.3 against the other action weights' sum of 1, so an innate prior can put
at most a fifth of an agent's steps on `bond` before learning. 3,000 steps
per arm, so that several generations of offspring, each learning from
scratch, pass under selection.

## 4. Measures

Per arm and seed, from the per-step record:

* **Coverage:** share of alive agents in a unit (mean over the run; final).
* **Policy bond probability (late):** the mean over agents of their policy's own probability of choosing `bond`, before innate priors and masks, averaged over the last quarter of the run. This separates what the network learned from what the propensity gene supplied. The sampled bond share is also recorded.
* **Population:** mean, peak and final; births and deaths; mean lifespan of agents that died; maximum generation reached.
* **Units:** formed, mean lifetime, failure-mode prevalence (the non-spatial diagnostics applied to the units' ledgers).
* **Loci:** population mean of each cooperation locus at the start and end, and the **change** over the run; the population mean of `coop_recognition × coop_exit_cost` (the enforced exit payment per unit of holdings) and its change.

## 5. Hypotheses

Each is a seed-paired Wilcoxon signed-rank test of the `cooperation −
placebo` difference across the 20 seeds. The nine tests form one family,
Holm-corrected at α = 0.05. Effect sizes are the mean paired difference with
a percentile bootstrap 95% CI (10,000 resamples).

| | Hypothesis | Metric | Test |
| --- | --- | --- | --- |
| S2 | The brain learns to bond: late in the run the policies' own probability of `bond` is higher than in the placebo, where it never pays. | late policy bond probability | one-sided |
| S3 | Cooperation sustains a larger population. | mean population | one-sided |
| S4 (six tests) | Selection acts on the institution: the change over the run in the population mean of each settings locus differs from its neutral drift in the placebo. | change in `coop_scope`, `coop_exit_cost`, `coop_recognition`, `coop_symmetry`, `coop_exclusivity`, `coop_max_members` | two-sided |
| S5 | Selection favours binding terms: the enforced exit payment (`recognition × exit_cost`) rises more than under neutral drift. | change in expected exit payment | one-sided |

S4 tests direction, not distance: selection moves a locus the same way in
every seed, drift moves it a random way, so a nonzero mean paired change is
the signature of selection, while a larger absolute change could come from
nothing more than a smaller or faster-turning population. S5 is the one
directional claim about *which* institution evolves, taken from the
non-spatial result that enforced exit costs raised founder welfare; it may
fail if trapping costs dominate once agents can be kept in a unit against a
learned preference. It uses the product of the two loci because an exit
cost binds only when recognition enforces it.

**Manipulation checks** (reported, not corrected with the family): units
form at all — coverage in the cooperation arm exceeds the placebo's, which
is zero by construction (M1); and the `placebo − baseline` population
difference, two-sided, as the cost of carrying the extra actions and loci
without an institution (M2).

## 6. Exploratory analyses

The `cooperation_no_motive` contrasts (does the institution pay for itself
without the motive reward); which failure modes the evolved units show; the
joint distribution of the settings loci at the end (do scope and exit cost
co-evolve); motive-locus drift; the `cooperation − baseline` contrast; the
sampled bond share against the policy's probability; runs with
`family_on_birth`, `terms="initiator"`, `fixed_settings` presets, frozen
loci (`--no-evolve-genes`) and zero synergy as follow-ups. Descriptive only.

## 7. Known limitations

* The membership reward is an intrinsic term the designer added; it is what
  lets motives bear on a learned decision, but it also means part of the
  "taste for cooperation" is hand-placed rather than evolved. The fourth arm
  removes it; the registered hypotheses are still tested on the arm that
  has it.
* The synergy surplus creates resources from nothing (20% of a pair's
  inflow), so S3 tests the institution *including* its productive surplus.
  A zero-synergy follow-up separates insurance from production.
* A unit's terms are the mean of its founders' loci, so an agent's own gene
  is half the institution it joins; this halves the selection gradient on
  every settings locus.
* Seed pairing removes initial-conditions variance only: the arms' policies
  have different output sizes, so their random streams diverge at the first
  decision.
* Checkpoint resume does not restore live units; the registered run does not
  resume.
* Bond formation is local (`bond_range`) and the policy sees partners as
  allies in its observation; it has no channel showing who is *open*, so the
  bond action is partly a gamble on nearby intent. This biases the test
  toward not finding cooperation.
* The credit-assignment fix in `LearningAgentBehavior` (see commit message)
  changes how every learning run trains; results are not comparable with
  runs made before it.

## Appendix A. Parameters

| Group | Parameter | Value |
| --- | --- | --- |
| World | environment / steps | `development` / 3,000 |
| World | initial agent resources / resource nodes / population cap | 12 / 60 / 300 |
| Inheritance | mutation rate / scale / mode / boundary | 0.1 / 0.1 / gaussian / reflect |
| Inheritance | crossover / co-parent radius | uniform / 10 |
| Inheritance | selection pressure | low |
| Inheritance | initial diversity scale | 0.2 |
| Cooperation | bond range / offer window / max candidates | 6 / 10 / 5 |
| Cooperation | terms / switching | mean / allowed |
| Cooperation | synergy / payout rate / reserve per participant | 0.2 / 0.8 / 2 |
| Cooperation | aid threshold / aid amount | 2 / 1 |
| Cooperation | membership reward scale / horizon / fit scale / affinity scale | 1 (0 in `cooperation_no_motive`) / 20 / 2 / 4 |
| Cooperation | partners as allies / membership marker / partner co-parent / family on birth | yes / yes / yes / no |
| Genes | settings loci defaults (scope, exit cost, recognition, symmetry, exclusivity, max members) | 0.5, 0.3, 0.5, 1.0, 1.0, 2 (founders drawn uniformly) |
| Genes | motive loci / propensities (bond, leave) | 0.5 each / 0.05, 0.03 in [0, 0.3] (founders drawn uniformly) |
| Analysis | α (Holm) / bootstrap resamples / seeds | 0.05 / 10,000 / 101–120 |
