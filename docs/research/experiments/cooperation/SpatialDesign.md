# Cooperation in the Spatial World: Does an Institution Evolve When the Brain Chooses It?

**Status:** Registered and calibrated on a reduced local pilot (§8), not yet run (`farm.runners.intrinsic_cooperation_experiment`, service in `farm.core.cooperation.service`)
**CLI:** [`scripts/run_intrinsic_cooperation_experiment.py`](../../../../scripts/run_intrinsic_cooperation_experiment.py); calibration gates: [`scripts/calibrate_intrinsic_cooperation.py`](../../../../scripts/calibrate_intrinsic_cooperation.py)
**Running the seeds:** [GCP runbook](GcpRunbook.md) (one VM per seed, merge with `scripts/merge_intrinsic_cooperation_runs.py`)
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
population; only the cooperation configuration differs. A fifth arm is
registered as optional: it is run when budget allows and enters only the
exploratory contrasts.

| Arm | Action space | Cooperation loci | Units | Membership reward |
| --- | --- | --- | --- | --- |
| `baseline` | 8 actions (unchanged world) | absent | none | — |
| `placebo` | 10 actions (`bond`, `leave` added) | present, randomised at founding, heritable | never form (`bond` is a no-op) | — |
| `cooperation` | 10 actions | present, randomised at founding, heritable | form, pool, pay out, insure, enforce | on (scale 1) |
| `cooperation_no_motive` | 10 actions | as above | as above | off (scale 0) |
| `cooperation_zero_synergy` (optional) | 10 actions | as above | as above, `synergy = 0` | on (scale 1) |

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

The zero-synergy arm removes the other designer-added term: pooling creates
a 20% surplus from nothing, so without this arm a population effect could
be production rather than insurance. `cooperation_zero_synergy − placebo`
is the institution's effect with no surplus and `cooperation −
cooperation_zero_synergy` the surplus's own share; both are exploratory.

## 3. World and inheritance

The `development` environment (50×50, 30 founders across three agent types)
with the intrinsic-goals startup knobs (initial agent resources 12, 60
resource nodes) and the population cap raised from 50 to 300. At 50 every
arm sits on the cap and selection has nothing to act on (the intrinsic-goals
experiment documents the same problem); density-dependent reproduction cost
regulates the population instead, at selection pressure `medium` so the
regulation bites below the cap. Inheritance follows the intrinsic-evolution
policy: Gaussian mutation at rate 0.1 and scale 0.1, reflecting boundaries,
uniform crossover with a co-parent within 10 units (a unit partner when the
agent has one). Crossover requires a co-parent of the same agent type, so a
partner of another type is never the co-parent; the propensity for
same-type bonds that this creates is a known constraint (§7).

**What varies.** Only Chromosome B (the innate action priors) and the
cooperation loci (Chromosome D) evolve. Chromosome A (the learning
hyperparameters, including the network and replay-buffer sizes) and
Chromosome C (the reward weights) are pinned at their defaults on every
founder, and the pins are inherited. The platform's initial-diversity layer
is off. Both choices follow from the intrinsic-goals pilot and the
calibration in §8: with the diversity layer on, scrambled learning rates and
exploration schedules leave most brains unable to learn, the policies'
action distributions stay near uniform, and the action mix is set by the
priors alone — which would make S2 a statement about `bond_weight`, not
about learning. Pinning C also keeps the membership reward worth the same
to every agent, so the fourth arm isolates the term cleanly. The
cooperation loci are still drawn uniformly per founder by the service and
mutate at every birth.

**Expressing the learned preference.** The policy's softmax over Q-values
is sharpened with a temperature of 0.1 before the innate priors are applied
(`learning.policy_temperature`). At temperature 1 the per-step Q gaps are
too small to move the action mix off the priors even when the ordering has
been learned; §8 reports the positive control that fixes the value.

Cooperation defaults: `bond_range` 6, `offer_window` 10, terms = mean of the
founders' settings loci, switching allowed, synergy 0.2 (0 in the optional
fifth arm), payout rate 0.8, reserve 2 per participant, aid 1 below 2,
membership reward scale 1 (0 in the fourth arm) with horizon 20, fit scale
2, affinity scale 4; partners added to the allies channel (same-type agents
remain allies, so perception is unchanged for anyone not in a unit); partner
co-parenting; no family units. Three mechanics amendments from the design
review:

* **Exit cost base.** An enforced exit forfeits `exit_cost` of what the
  unit has paid the leaver (payouts plus aid, capped by current holdings),
  not of the leaver's whole wealth. A random `leave` can therefore never
  cost more than the unit was ever worth to the agent, which removes a
  large, learning-unrelated penalty from the placebo-free arms.
* **Masking.** `leave` is offered only to current members and `bond` only
  when another agent is within `bond_range`; otherwise the action is masked
  for that step and the policy chooses among the rest. The same rule holds
  in the placebo, where nobody is ever a member, so the placebo's agents
  never see `leave`. This keeps the policy's probability mass meaningful
  (an unavailable action cannot absorb it) and matches the information a
  real agent would have.
* **Partners are not targets.** `attack` never selects a current unit
  partner. Without this, a unit's own members were the nearest and most
  frequent targets of its `attack` action.

The propensity loci (`bond_weight`, `leave_weight`) are bounded at 0.3
against the other action weights' sum of 1, so an innate prior can put at
most a fifth of an agent's steps on `bond` before learning. 3,000 steps per
arm, so that several generations of offspring, each learning from scratch,
pass under selection.

**Persistence and provenance.** Every arm writes `arm_result.json` (its
summary and time series) as it finishes and every replicate writes
`replicate_summary.json`; a restart loads finished arms instead of re-running
them. The summary records the git commit the run was made from.

## 4. Measures

Per arm and seed, from the per-step record:

* **Coverage:** share of alive agents in a unit (mean over the run; final).
* **Policy bond probability (late):** the mean over agents of their policy's own probability of choosing `bond`, before innate priors and masks, averaged over the last quarter of the run. This separates what the network learned from what the propensity gene supplied. Beside it: the probability of `bond` in the distribution the agent actually sampled from (policy × priors × mask), the rank of `bond` in the policy's own ordering, and the sampled bond and leave shares.
* **Learning diagnostics:** the policies' own probability of `gather` minus uniform (`policy_gather_lift_late`), the sampled gather share minus the innate prior's share (`gather_share_over_prior_late`), and the mean peak of the policy distribution. These are the positive controls (§5).
* **Population:** mean, peak and final; births and deaths; deaths by cause (starvation, combat, other); mean lifespan of agents that died; maximum generation reached. Lifespan and births are uncapped, so they still move when the population sits at the cap.
* **Units:** formed, mean lifetime, failure-mode prevalence (the non-spatial diagnostics applied to the units' ledgers); total forfeits and forfeits per enforced exit.
* **Loci:** population mean of each cooperation locus at the start and end, and the **change** over the run; the population mean of `coop_recognition × coop_exit_cost` (the enforced exit payment per unit of the exit-cost base) and its change.
* **Action mix:** the share of every action, including `communicate`, `bond` and `leave`.

## 5. Hypotheses

Each is a seed-paired Wilcoxon signed-rank test of the `cooperation −
placebo` difference across the 20 seeds. The seven tests form one family,
Holm-corrected at α = 0.05. Effect sizes are the mean paired difference with
a percentile bootstrap 95% CI (10,000 resamples).

| | Hypothesis | Metric | Test |
| --- | --- | --- | --- |
| S2 | The brain learns to bond: late in the run the policies' own probability of `bond` is higher than in the placebo, where it never pays. | late policy bond probability | one-sided |
| S3 | Cooperation sustains a larger population. | mean population | one-sided |
| S3b | Cooperation lengthens lives: agents that die lived longer than in the placebo. | mean lifespan at death | one-sided |
| S4 (three tests) | Selection acts on the institution: the change over the run in the population mean of each settings locus differs from its neutral drift in the placebo. | change in `coop_scope`, `coop_exit_cost`, `coop_recognition` | two-sided |
| S5 | Selection favours binding terms: the enforced exit payment (`recognition × exit_cost`) rises more than under neutral drift. | change in expected exit payment | one-sided |

S3b is in the family because S3's metric has a ceiling: when an arm sits
at the population cap, population cannot show an advantage but lifespan
still can. S4 is restricted to the three loci with a mechanism that gives
selection something to act on — how much is pooled, what leaving costs, and
whether that cost is enforced. The other three (`symmetry`, `exclusivity`,
`max_members`) have no directional prediction and are reported as
exploratory so they do not dilute the family.

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

**Learning positive controls** (reported, not corrected with the family;
one-sample tests across seeds in the `baseline` arm): the policies' own
late probability of `gather` exceeds uniform (L1), and the late gather
share exceeds what the innate priors alone would give (L2). If either
fails, the brains did not learn in this world and S2 cannot be read as a
statement about learning. Both also gate the confirmatory run (§8).

## 6. Exploratory analyses

Reported uncorrected and flagged `exploratory` in the summary: the change
in `coop_symmetry`, `coop_exclusivity` and `coop_max_members`
(`cooperation − placebo`, two-sided); starvation deaths (fewer in the
cooperation arm, if insurance works); births (more, if cooperation raises
turnover); and, when the fifth arm is run, `cooperation_zero_synergy −
placebo` and `cooperation − cooperation_zero_synergy` on population.

Descriptive only: the `cooperation_no_motive` contrasts (does the
institution pay for itself without the motive reward); which failure modes
the evolved units show; the joint distribution of the settings loci at the
end (do scope and exit cost co-evolve); motive-locus drift; the
`cooperation − baseline` contrast; the sampled bond share and the acting
bond probability against the policy's probability; runs with
`family_on_birth`, `terms="initiator"`, `fixed_settings` presets and frozen
loci (`--no-evolve-genes`) as follow-ups.

## 7. Known limitations

* The membership reward is an intrinsic term the designer added; it is what
  lets motives bear on a learned decision, but it also means part of the
  "taste for cooperation" is hand-placed rather than evolved. The fourth arm
  removes it; the registered hypotheses are still tested on the arm that
  has it.
* The synergy surplus creates resources from nothing (20% of a pair's
  inflow), so S3 tests the institution *including* its productive surplus.
  The optional `cooperation_zero_synergy` arm separates insurance from
  production, but only in the exploratory contrasts.
* A unit's terms are the mean of its founders' loci, so an agent's own gene
  is half the institution it joins; this halves the selection gradient on
  every settings locus.
* Crossover requires a co-parent of the same agent type. A partner of
  another type is therefore never a co-parent, and partner co-parenting
  (the route by which a unit's two lineages mix) only operates in same-type
  units.
* Learning is expressed through a sharpened softmax (temperature 0.1), and
  Chromosomes A and C are pinned. This is a choice about what the
  experiment is about — selection on priors and on the institution, with
  brains that can learn — not a claim that it is the only regime in which
  cooperation could evolve.
* Seed pairing removes initial-conditions variance only: the arms' policies
  have different output sizes, so their random streams diverge at the first
  decision.
* Resume is at the granularity of a finished arm: an interrupted arm is
  re-run from its start, because checkpoint resume does not restore live
  units. A finished arm is reused only when its stored provenance (config,
  commit, dirty flag, hash seed) matches the run.
* S3b's lifespan at death is right-censored: agents alive at the last step
  do not enter it, so an arm whose agents live longer contributes fewer,
  earlier-born deaths. At 3,000 steps against lifespans of ~130 the censored
  share is small, but the metric is a mean over deaths, not a restricted
  mean survival time. It is NaN for an arm in which nobody died.
* Bond formation is local (`bond_range`) and the policy sees partners as
  allies in its observation; it has no channel showing who is *open*, so the
  bond action is partly a gamble on nearby intent. This biases the test
  toward not finding cooperation.
* The credit-assignment fix in `LearningAgentBehavior` (see commit message)
  changes how every learning run trains; results are not comparable with
  runs made before it.

## 8. Calibration

A pilot on seeds 1–3 (outside the confirmatory 101–120) fixes the free
parameters before the registered run and gates it. The report is produced
by `scripts/calibrate_intrinsic_cooperation.py` from the merged summary;
the gates it evaluates are the pre-flight gate in the [GCP runbook](GcpRunbook.md).

**Gates** (all must pass on the pilot before the confirmatory seeds are
launched):

| Gate | Criterion |
| --- | --- |
| L1 | `policy_gather_lift_late > 0` in every baseline replicate |
| L2 | `gather_share_over_prior_late > 0` in every baseline replicate |
| Units form | mean coverage > 0.05 and units formed > 0 in every cooperation replicate |
| Generations | `max_generation ≥ 3` in every cooperation replicate |
| Not pinned | mean population < 0.95 × cap in every arm |
| Exit cost bounded | 0 < forfeits per enforced exit < 10 |

**Reduced local pilot.** Seeds 1–3, 400 steps, population cap 60, the four
registered arms, everything else at the registered values (one process per
seed, merged with `scripts/merge_intrinsic_cooperation_runs.py`; 12 min on
4 vCPUs). Its purpose was to fix the learning regime and to measure the
noise the family is tested against, not to anticipate the result. The
full-length pilot on the GCP configuration is the runbook's pre-flight step
and must reproduce the gate results before the confirmatory launch.

*Learning.* The positive controls pass in every seed: L1 = 0.18, 0.17, 0.19
(policy probability of `gather` minus uniform, baseline arm, last quarter)
and L2 = 0.24, 0.23, 0.25 (sampled gather share minus the innate prior's
share). The policy's mean peak probability is 0.40 against a uniform 0.125,
and the preference is visible by step 50. The same world at temperature 1.0
(seed 1, baseline arm) gives L1 = 0.014, L2 = 0.087 and a peak of 0.15: the
ordering is learned but not expressed, and the action mix tracks the
priors. Temperature 0.1 is therefore fixed. Learning is weaker but present
in the cooperation arms (gather lift 0.12 ± 0.02), which have ten actions
and the pooled income.

*Regime.* Mean population as a fraction of the cap: baseline 0.77, placebo
0.75, cooperation 0.90, `cooperation_no_motive` 0.89 — regulated below the
cap in every arm, but the cooperation arms sit close to it, so at the
registered cap of 300 the `not_pinned` gate must be re-read on the full
pilot. Every death was starvation (no combat deaths once partners are
protected); mean lifespan at death 123–150 steps; 2–3 generations in 400
steps, which fails the `generations` gate as expected at this length
(lifespans of ~130 steps give ~20 generations at 3,000). Units form
readily: coverage 0.42 ± 0.03, ~200 units per run, mean unit lifetime 19
steps, forfeits per enforced exit 0.38 resource on the received base (the
holdings base scales with the leaver's wealth instead). `leave` is 1.7%
of actions in the cooperation arms and 0% in the placebo, where it is
always masked.

*Noise for the selection tests.* Run-long change in the population mean of
a locus in the placebo (neutral drift, SD over three seeds): `coop_scope`
0.047, `coop_exit_cost` 0.023, `coop_recognition` 0.023, expected exit
payment 0.016; the paired `cooperation − placebo` SDs were 0.02–0.03. At
n = 20 under the family's strictest Holm threshold (α/7) the minimum
detectable standardised effect at 80% power is dz ≈ 0.90 (two-sided) and
0.82 (one-sided), i.e. a mean paired change of about 0.02–0.03 on a locus
in [0, 1]. Drift at 400 steps understates drift at 3,000; the full pilot
re-measures it.

*Pilot effects (three seeds, 400 steps; descriptive only, and not a basis
for changing any registered direction).* Population and lifespan
differences were large (S3 +8.8 agents, S3b +24 steps, all three seeds
positive); the policy bond probability did not differ between cooperation
and placebo (S2 +0.002, dz 0.18: the brains had not learned to bond more
where it pays, in 400 steps); the expected exit payment *fell* more in the
cooperation arm than under drift (S5 −0.017, all three seeds), the opposite
of the registered direction. S5 stays registered as written; if the
confirmatory run confirms the pilot's sign, that is the finding.

**Frozen after calibration:** `policy_temperature = 0.1`,
`initial_diversity_scale = None`, `freeze_platform_genes = True`,
`selection_pressure = medium`, `exit_cost_base = received`,
`mask_pointless_actions = True`, `protect_partners = True`. These are the
defaults of `IntrinsicCooperationExperimentConfig` and of the CLI; the
confirmatory run passes none of them explicitly.

## Appendix A. Parameters

| Group | Parameter | Value |
| --- | --- | --- |
| World | environment / steps | `development` / 3,000 |
| World | initial agent resources / resource nodes / population cap | 12 / 60 / 300 |
| Inheritance | mutation rate / scale / mode / boundary | 0.1 / 0.1 / gaussian / reflect |
| Inheritance | crossover / co-parent radius | uniform / 10 |
| Inheritance | selection pressure | medium |
| Inheritance | initial diversity layer / frozen chromosomes | off / A (learning hyperparameters) and C (reward weights) |
| Learning | policy temperature (softmax over Q, before priors) | 0.1 |
| Cooperation | bond range / offer window / max candidates | 6 / 10 / 5 |
| Cooperation | terms / switching | mean / allowed |
| Cooperation | exit cost base / pointless-action masking / partners protected from attack | received / on / on |
| Cooperation | synergy / payout rate / reserve per participant | 0.2 (0 in `cooperation_zero_synergy`) / 0.8 / 2 |
| Cooperation | aid threshold / aid amount | 2 / 1 |
| Cooperation | membership reward scale / horizon / fit scale / affinity scale | 1 (0 in `cooperation_no_motive`) / 20 / 2 / 4 |
| Cooperation | partners as allies / membership marker / partner co-parent / family on birth | yes / yes / yes / no |
| Genes | settings loci defaults (scope, exit cost, recognition, symmetry, exclusivity, max members) | 0.5, 0.3, 0.5, 1.0, 1.0, 2 (founders drawn uniformly) |
| Genes | motive loci / propensities (bond, leave) | 0.5 each / 0.05, 0.03 in [0, 0.3] (founders drawn uniformly) |
| Analysis | α (Holm) / bootstrap resamples / seeds | 0.05 / 10,000 / 101–120 (pilot 1–3) |
| Analysis | family | S2, S3, S3b, S4 × 3, S5 (seven tests) |
