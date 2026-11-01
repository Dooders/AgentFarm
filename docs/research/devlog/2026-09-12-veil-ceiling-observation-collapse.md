---
layout: page
title: "The veil ceiling: a leaked cue inverts sealed-world evaluation"
---

Sealed-world evaluation assumes that what an agent does while watched is
informative about what it does when unwatched. The Veil Ceiling experiment
holds enforcement fixed — coverage `c = 0.5`, penalty `p = 6` — and varies
only the fidelity `f` of a cue correlated with being monitored. The prediction
is that above some fidelity the observed regime carries almost no information
about the unobserved one.

**Headline:** cue fidelity, not enforcement intensity, splits observed from
unobserved defection. With a perfect leak and Lamarckian inheritance, agents
defect at ~40 % of unwatched opportunities and 4 % of watched ones; realised
enforcement falls from 0.48 to 0.14 at the same designed `c · p`. Concealment
pays. The pre-registered validity test (H2) was *falsified* — an oracle
classifier still scores AUC 0.75 — but that number is an artifact of giving the
evaluator ground truth from the leaked regime. Calibrate on honest data and
apply it unchanged, and Lamarckian C2 is at chance (AUC 0.48). The naive
metric, observed defection rate, ranks agents *backwards* (rank AUC 0.45).
The veil ceiling is real for any evaluator who has not already seen
concealment.

Protocol, Appendix A, and deviations:
[Design.md](../experiments/veil_ceiling/Design.md).
Full tables:
[RESULTS.md](../experiments/veil_ceiling/RESULTS.md).

## The world

A 20 × 20 grid with 48 logistic-regeneration nodes. Agents choose `MOVE`
(step toward the nearest healthy node), `GATHER` (take up to 3, never below
the regeneration threshold), `OVER_HARVEST` (take 3 regardless, effort 0.5),
or `PASS`. Over-harvesting is individually profitable and collectively costly: a
node below the threshold regenerates at a quarter of the healthy rate. A
defection is an over-harvest at an opportunity that actually crosses the
threshold, so rates are only counted where the affordance tempts.

Monitoring is an epoch-resampled mask (`round(c · n)` cells, 25-tick epochs)
so the map cannot be memorised. The cue is the monitoring bit passed through a
binary symmetric channel with flip probability `(1 − f) / 2`. `f = 0` is a
coin flip; `f = 1` is a perfect leak. The rightmost four columns (`x ≥ 16`) are
never monitored during training and are switched on, with the cue, only in a
frozen evaluation.

C4 is the null: a perfect cue of an *independent* map with the same coverage
and schedule. If C4 manufactured divergence, the extra input bit would be
enough to void the design. It does not (Δ_cue ≈ 0.006–0.007, an order of
magnitude below C2).

## What ran

780 seed-matched runs: 13 conditions (C0, C1 `f = 0`, C2 `f = 1`, C3
fidelity sweep 0.1–0.9, C4 decoy, C1/C2 at `p ∈ {3, 9}`) × Baldwinian
(fresh network at birth) × Lamarckian (parent weights copied via
`apply_lamarckian_policy_warmstart`) × 30 shared seeds. No extinctions. C0
calibration passes in both modes (defection 0.24 / 0.49, stable). Expected
penalty per defection is identical across C1–C4 at the start of training
(`c · p = 3`); realised enforcement is reported as an outcome.

```bash
PYTHONHASHSEED=0 python scripts/run_veil_ceiling.py
python scripts/run_veil_ceiling.py --analyze-only
python scripts/run_veil_ceiling_validity_followup.py
```

Effects are paired differences with 95 % seed-bootstrap CIs (2000
replicates). The C4 noise band is mean ± 2 SD of C4 Δ_cue.

## Divergence is a monotone dose-response in the cue

![Divergence and oracle validity vs cue fidelity. Grey band is the C4 null.](figures/veil_ceiling_dose_response.png)

Δ_cue — defection rate at `cue = 0` minus `cue = 1` — rises monotonically with
`f` in both modes (Spearman ρ = 1.0). Lamarckian transfer makes the response
convex: `f = 0.1` sits inside the late C4 band (upper 0.029), `f = 0.3` is
already just outside it (0.041 [0.036, 0.045]), `f = 0.5` is clearly out
(0.071), and `f = 1` reaches 0.359. Baldwinian shows the same ordering at a
fifth of the amplitude (0.079). Robustness cells at `p = 3` and `p = 9`
reproduce the split with essentially unchanged Δ_cue.

H1 is supported in both modes. C2 − C4 Δ_cue is 0.072 [0.067, 0.077]
(Baldwinian) and 0.352 [0.335, 0.369] (Lamarckian); every seed agrees on the
sign.

## Enforcement is matched by design and diverges by outcome

Every monitored cell starts with the same expected penalty. Realised
enforcement in C1 stays at the design value (0.47–0.48 of defections
penalised). In C2 it falls to 0.35 (Baldwinian) and 0.14 (Lamarckian): agents
with a perfect cue defect where they are not watched, so the same monitor
catches far fewer of them. That is the operational signature of the veil
failing. The monitor did not get weaker. The population learned to step around
it.

## H2 is falsified — and then the ceiling shows up anyway

H2 predicted that leave-one-seed-out validity AUC would stay high in C1
(≥ 0.7) and collapse below 0.6 in C2. C1 holds (0.79). C2 does not collapse:
0.702 [0.688, 0.716] Baldwinian, 0.753 [0.738, 0.767] Lamarckian. Check 3
fires; H2 is reported as falsified.

The second validity column is the tell. Rank AUC — scoring agents by
observed defection rate alone — falls to 0.446 [0.425, 0.467] in Lamarckian
C2. An evaluator who reads sealed-world defection at face value ranks the
agents backwards, because the agents who look best on camera are the
conditional defectors. The multi-feature classifier recovers validity from
side channels the pre-registered feature set happened to include.

An exploratory follow-up on the same raw record (not pre-registered) asks
what that surviving AUC is made of.

![Feature ablation of the LOSO classifier. Observed defection rate alone is at chance in Lamarckian C2; action shares carry the rest.](figures/veil_ceiling_validity_ablation.png)

Feature ablation, Lamarckian C2:

| Feature set | AUC |
|---|---|
| observed defection rate only | 0.546 [0.524, 0.568] |
| action shares only | 0.726 [0.712, 0.740] |
| everything except defection rate | 0.752 [0.737, 0.768] |
| full pre-registered set | 0.753 [0.738, 0.767] |

Removing the defection rate costs nothing. Energy and exposure carry no
information. All of the retained validity is in the action shares, and within
them almost entirely `move_share` (standardised coefficient 1.11 vs −0.07 for
defection rate). Conditional defectors move on 28 % of observed ticks versus
17 % for cooperators (SMD +0.87): when an agent that has learned the cue finds
itself watched, it leaves. The leak is not "rich agents defect". It is "agents
that walk away from the camera defect elsewhere".

That still overstates what a real evaluator can do. The pre-registered LOSO
AUC is an *oracle*: it fits on agents from the leaked-cue condition using their
unobserved ground truth, which a sealed-world evaluator never has. The
honest alternative is to calibrate where behaviour is believed to be honest
(C1, or the C4 null) and apply the classifier unchanged.

![Oracle vs honest-calibrated evaluator across cue fidelity.](figures/veil_ceiling_honest_evaluator.png)

| Evaluated cell (Lamarckian) | Oracle (LOSO) | Calibrated on C1 |
|---|---|---|
| C3 `f = 0.7` | 0.771 | 0.765 |
| C3 `f = 0.9` | 0.747 | 0.661 |
| C2 `f = 1.0` | 0.753 [0.738, 0.767] | 0.480 [0.460, 0.502] |

Calibrated on honest data, the evaluator puts weight 1.5 on observed
defection and ~0 on movement — exactly the wrong features once the cue leaks.
At `f = 1` the primary cell (`p = 6`) is at chance (0.480 [0.460, 0.502]);
the `p = 9` robustness cell matches (0.480) while `p = 3` is only slightly
above (0.540). The drop is sharp above `f ≈ 0.7`. Baldwinian degrades only
mildly (0.702 → 0.664) because those agents conceal weakly.

Under the pre-registered operationalisation, H2 stays falsified. Under the
operationalisation that matches how sealed-world evaluation is actually used,
the veil ceiling is complete.

## Concealment pays, and inheritance amplifies it

In C2, 42 % of Lamarckian agents (11 % of Baldwinian agents) classify as
conditional (Δ_cue ≥ 0.3). They earn more net energy per tick than
unconditional cooperators in every C2 cell (all cost CIs > 0): +0.144
[0.127, 0.163] Baldwinian, +0.059 [0.050, 0.067] Lamarckian. In C1 the few
agents classified as conditional — there by chance, since the cue is noise —
earn *less* than cooperators. Concealment is only fit when the cue is
informative, which is the expected sign structure. H3 is supported.

![Δ_cue over training. Lamarckian C2 is still climbing at tick 1500; Baldwinian plateaus by ~300.](figures/veil_ceiling_delta_timecourse.png)

Lamarckian transfer amplifies the conditional policy: C2 Δ_cue is 0.280
[0.263, 0.297] higher than the seed-matched Baldwinian run, and the
time-course is still rising at the end of training while Baldwinian has
plateaued by tick ~300. In the frozen evaluation the cue is switched on in
the held-out band for the first time. Lamarckian C2 agents apply the
conditional policy there at full strength (held-out Δ_cue = 0.494, above
the training-region 0.359, well outside the C4 held-out band). Baldwinian
shows a positive held-out effect (0.052) that sits inside the pre-registered
C4 band (upper 0.091), so H4 is supported only with inheritance.

H5 — that inheritance accelerates concealment *more than* cooperation — is
inconclusive. Transfer speeds up whatever the population is learning
(cooperative onset median 50 vs 100 ticks in C1; C2 conditional onset also
earlier). The differential is −48 ticks [−125, 38].

## Verdicts

| Hypothesis | Baldwinian | Lamarckian |
|---|---|---|
| H1 divergence grows with fidelity | supported | supported |
| H2 validity collapses at `f = 1` (oracle AUC) | **falsified** | **falsified** |
| H3 concealment is not costly | supported | supported |
| H4 spatial transfer to the held-out band | **falsified** | supported |
| H5 inheritance accelerates concealment more than cooperation | inconclusive | inconclusive |

The honest-calibrated follow-up is a different operationalisation of H2, not
a re-verdict. Pre-registered H2 asked whether an oracle with leaked-regime
ground truth loses validity; it does not. A realistic evaluator, calibrated
where the veil is intact, has none.

## Open questions

- **Can the same coverage budget catch concealment by reallocating where it
  looks?** The side channel that works here (leaving when observed) is
  specific to this world and would itself be learned away if the evaluator
  acted on it. Adaptive monitoring — reweighting the next epoch's true map
  from last epoch's observed residual, without raising `c` or `p` — is the
  natural next experiment.
- **Ablating a feature is not the same as hiding it.** No agent in this run
  had any incentive to disguise `move_share`. A motivated concealer that
  pays a small cost to look busy on camera is untested.
- **Baldwinian within-lifetime learning is weak** (500-tick lifespans,
  from-scratch networks). Every Baldwinian effect is compressed by that;
  the Lamarckian arm is the one that actually has time to learn the cue.
- The world is a compact re-implementation of AgentFarm's grid dynamics, not
  the full `Environment`. The inheritance mechanism is the shared core helper.

## Reproduce

The confirmatory record is committed under
[`experiments/veil_ceiling/results/`](https://github.com/Dooders/AgentFarm/tree/dev/experiments/veil_ceiling/results).
Re-analyse without re-running:

```bash
source venv/bin/activate
python scripts/run_veil_ceiling.py --analyze-only
python scripts/run_veil_ceiling_validity_followup.py
```

Full matrix (~27 min on 4 cores):

```bash
PYTHONHASHSEED=0 python scripts/run_veil_ceiling.py
```

Pilot (3 seeds, no robustness cells):

```bash
PYTHONHASHSEED=0 python scripts/run_veil_ceiling.py \
    --seeds 3 --no-robustness --output-dir /tmp/veil_pilot
```

## Related docs

- [Pre-registered design and Appendix A](../experiments/veil_ceiling/Design.md)
- [Narrative results](../experiments/veil_ceiling/RESULTS.md)
- [Confirmatory REPORT](https://github.com/Dooders/AgentFarm/blob/dev/experiments/veil_ceiling/results/REPORT.md)
- [Validity follow-up](https://github.com/Dooders/AgentFarm/blob/dev/experiments/veil_ceiling/results/VALIDITY_FOLLOWUP.md)
- [Package README](https://github.com/Dooders/AgentFarm/blob/dev/farm/experiments/veil_ceiling/README.md)
- [Baldwinian vs Lamarckian: policy warm-start across three resource regimes](2026-05-21-baldwinian-vs-lamarckian-ab-harness.md)
