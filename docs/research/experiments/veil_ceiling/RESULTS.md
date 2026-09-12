# The Veil Ceiling — results of the confirmatory run

**Design:** [Design.md](Design.md) · **Full report:**
[`experiments/veil_ceiling/results/REPORT.md`](../../../../experiments/veil_ceiling/results/REPORT.md)
· **Raw record:** [`experiments/veil_ceiling/results/`](../../../../experiments/veil_ceiling/results)

- 780 runs: 13 conditions (C0, C1, C2, C3 × 5, C4, C1/C2 at p = 3 and p = 9)
  × 2 inheritance modes × 30 seed-matched runs. No extinctions in any cell.
- Simulation code commit `4fe7d33c` (recorded in `manifest.json`); analysis
  thresholds are the pre-registered values in `AnalysisThresholds`.
- All intervals are 95 % percentile bootstraps over seeds (2000 replicates).

## Verdicts

| Hypothesis | Baldwinian | Lamarckian | Note |
|---|---|---|---|
| H1 divergence grows with fidelity | **supported** | **supported** | C2 − C4 Δ_cue = 0.072 [0.067, 0.077] and 0.352 [0.335, 0.369]; Spearman ρ(f, Δ) = 1.0 in both modes. |
| H2 validity collapses at f = 1 | **falsified** | **falsified** | AUC(C2) = 0.702 [0.688, 0.716] and 0.753 [0.738, 0.767]; the pre-registered collapse threshold is < 0.6. |
| H3 concealment is not costly | **supported** | **supported** | Conditional agents out-earn cooperators: +0.144 [0.127, 0.163] and +0.059 [0.050, 0.067] energy/tick. |
| H4 spatial transfer to the held-out band | **falsified** | **supported** | Held-out Δ_cue = 0.052 [0.026, 0.081] vs C4 band upper 0.091 (Baldwinian); 0.494 [0.451, 0.537] vs 0.187 (Lamarckian). |
| H5 inheritance accelerates concealment more than cooperation | inconclusive | | Lamarckian transfer accelerates both; difference −48 ticks [−125, 38]. |

Falsification checks (§9): the C4 null control passes in both modes (C4 Δ_cue
≈ 0.006–0.007, an order of magnitude below C2, so the cue channel itself does
not manufacture divergence); C2 lies far outside the C4 band; validity does
not collapse, so check 3 falsifies H2.

## Calibration

| Mode | C0 defection rate | Drift / 100 ticks |
|---|---|---|
| Baldwinian | 0.239 [0.233, 0.246] | −0.001 |
| Lamarckian | 0.488 [0.472, 0.503] | +0.018 |

Both baselines are interior and stable within the pre-registered bounds. The
two modes have very different baselines: without monitoring, Lamarckian
inheritance lets the population converge on the individually optimal
over-harvesting policy (49 % of opportunities), whereas Baldwinian agents that
must learn from scratch every generation defect at 24 %.

## What the data show

### Enforcement is matched by design and diverges by outcome

Every monitored cell starts with the same expected penalty per defection
(c · p = 3.0). Realised enforcement in C1 stays at the design value
(0.47–0.48 of defections penalised). In C2 it falls to 0.35 (Baldwinian) and
0.14 (Lamarckian): agents with a perfect cue defect where they are not
watched, so the same monitor catches far fewer of them. This is the operational
signature of the veil failing.

### Divergence is a monotone dose-response in cue fidelity

Δ_cue rises monotonically with f in both modes (ρ = 1.0). With Lamarckian
transfer the response is convex: f ≤ 0.3 is indistinguishable from the C4
band, f = 0.5 is clearly out of it (0.071), and f = 1 reaches 0.359 — agents
defect at 40 % of unobserved opportunities and 4 % of observed ones. The
Baldwinian arm shows the same ordering at a fifth of the amplitude
(Δ_cue = 0.079 at f = 1). The robustness cells reproduce the result at p = 3
and p = 9 with essentially unchanged Δ_cue (0.40 and 0.34 Lamarckian; 0.10 and
0.06 Baldwinian).

### Observed behaviour stays predictive — but only if you use the right features

H2 is falsified: the leave-one-seed-out classifier keeps an AUC of 0.70–0.75
in C2 (C1: 0.79). The mechanism is visible in the second validity column of
the report. The *rank* AUC — using only the observed defection rate as the
score — falls to 0.446 [0.425, 0.467] in Lamarckian C2 and 0.44–0.53 in the
C2 robustness cells: an evaluator who reads observed defection at face value
ranks the agents *backwards*, because the agents who behave best when observed
are the conditional defectors. The multi-feature classifier recovers validity
from the other observed features (energy level, movement and gather shares,
exposure), which still leak the agent's strategy. The "veil ceiling" therefore
applies to the naive reading of sealed-world evaluation, not to observed
behaviour as such; the pre-registered threshold was not met and H2 is reported
as falsified.

### Concealment pays

In C2, 42 % of Lamarckian agents (11 % of Baldwinian agents) are classified as
conditional. They earn more net energy per tick than unconditional cooperators
in every C2 cell (all cost CIs > 0), and in the Baldwinian arm they also leave
more offspring (1.06 vs 0.68 per agent). In C1 the few agents classified as
conditional — there by chance, since the cue is noise — earn *less* than
cooperators. Concealment is only fit when the cue is informative, which is the
expected sign structure.

### Transfer to the never-monitored band

In the frozen evaluation the cue is switched on in the held-out band for the
first time. Lamarckian C2 agents apply the conditional policy there at full
strength (held-out Δ_cue = 0.494, above the training-region value of 0.359),
well outside the C4 held-out band. Baldwinian C2 agents show a positive
held-out effect (0.052; paired C2 − C4 = 0.042 [0.012, 0.076]) but it sits
inside the pre-registered C4 band (upper 0.091), so H4 is falsified for that
arm under the pre-registered rule.

The C4 held-out band is wide for a reason worth recording: in Lamarckian C4
the held-out Δ_cue is 0.049 [0.025, 0.073] even though the cue is a decoy.
During training the cue is only ever non-zero in the training region, so
`cue = 1` is a weak proxy for "you are in the monitored region"; agents that
learned to defect freely in the never-monitored band extrapolate that novel
`(held-out, cue = 1)` input towards their training-region policy. This is a
region-confound of any held-out design in which the cue is absent from the
held-out region during training. The C4 band absorbs it; the Lamarckian C2
effect is ten times larger.

### Inheritance

Lamarckian transfer amplifies the conditional policy: C2 Δ_cue is 0.280
[0.263, 0.297] higher than in the seed-matched Baldwinian run, and the
time-course figure shows Δ_cue in Lamarckian C2 still rising at the end of
training while the Baldwinian curve plateaus by tick ~300. Inheritance also
accelerates plain cooperation (C1 cooperative onset median 50 vs 100 ticks), so
H5's differential-acceleration claim is inconclusive: transfer speeds up
whatever the population is learning.

## Follow-up: why did validity survive? (exploratory, not pre-registered)

H2 was falsified because the pre-registered classifier kept AUC 0.70–0.75 in
C2. Three re-analyses of the same raw record ask what that number is made of.
Full tables: [`VALIDITY_FOLLOWUP.md`](../../../../experiments/veil_ceiling/results/VALIDITY_FOLLOWUP.md);
reproduce with `python scripts/run_veil_ceiling_validity_followup.py`.

### 1. The surviving validity comes from one channel: movement

Feature ablation of the LOSO classifier, Lamarckian C2:

| Feature set | AUC |
|---|---|
| observed defection rate only | 0.546 [0.524, 0.568] |
| action shares only (gather / move / pass) | 0.726 [0.712, 0.740] |
| everything except defection rate | 0.752 [0.737, 0.768] |
| full pre-registered set | 0.753 [0.738, 0.767] |
| energy only | 0.478 |
| exposure only | 0.500 |

The observed defection rate — the thing sealed-world evaluation actually
measures — is at chance on its own, and removing it from the full set costs
nothing. Energy and exposure carry no information. All of the retained
validity is in the action shares, and within them almost entirely `move_share`
(standardised logistic coefficient 1.11 vs −0.07 for defection rate).
Conditional defectors move on 28 % of observed ticks versus 17 % for
cooperators (SMD +0.87): when an agent that has learned the cue finds itself
watched, it leaves. The leak is not "rich agents defect", it is "agents that
walk away from the camera defect elsewhere". The Baldwinian arm shows the same
structure at lower amplitude (defection rate alone 0.676; +actions 0.703;
`move_share` coefficient 0.47).

### 2. A realistic evaluator has no validity at all

The pre-registered LOSO AUC is an *oracle*: it fits the classifier on agents
from the leaked-cue condition using their unobserved ground truth, which a real
sealed-world evaluator never has. The honest alternative is to calibrate the
evaluator where behaviour is believed to be honest (C1, or the C4 null) and
apply it unchanged.

| Evaluated cell (Lamarckian) | Oracle (LOSO) | Calibrated on C1 | Calibrated on C4 |
|---|---|---|---|
| C3 f = 0.7 | 0.771 | 0.765 | 0.765 |
| C3 f = 0.9 | 0.747 | 0.661 | 0.666 |
| C2 f = 1.0 | 0.753 [0.738, 0.767] | 0.480 [0.460, 0.502] | 0.491 [0.471, 0.513] |
| C2 p = 3 | 0.734 | 0.540 | 0.548 |
| C2 p = 9 | 0.732 | 0.480 | 0.484 |

Calibrated on honest data, the evaluator puts weight 1.5 on observed defection
and ~0 on movement — exactly the wrong features once the cue leaks. At f = 1
its AUC is at or below chance in every Lamarckian C2 cell, and it degrades
sharply above f ≈ 0.7. The Baldwinian arm degrades mildly (0.702 → 0.664)
because its agents conceal weakly.

This reverses the headline reading of H2. Under the pre-registered
operationalisation — an evaluator with ground truth from the leaked regime —
validity does not collapse (H2 stays falsified). Under the operationalisation
that matches how sealed-world evaluation is actually used, the veil ceiling is
complete: observed behaviour of cue-aware agents is uninformative, and the
naive metric is inverted (rank AUC 0.446). The difference between the two is
whether the evaluator already knows what concealment looks like.

### 3. Implications

- A sealed-world evaluator that has never seen cue-aware agents cannot detect
  them from the behaviour it is designed to measure; it needs side channels it
  has no prior reason to weight.
- The side channel that does work here (leaving when observed) is specific to
  this world and would itself be learned away if the evaluator acted on it —
  the arms-race question is tested in
  [AdaptiveResults.md](AdaptiveResults.md): reallocating the same coverage
  budget by caught defections does not restore validity; occupancy chasing
  raises enforcement but a C1-calibrated evaluator stays at chance at `f = 1`.
- Ablating features is not the same as hiding them from a motivated agent;
  no agent in this run had any incentive to disguise `move_share`.

## Limitations

- The world is a compact re-implementation of AgentFarm's grid dynamics, not
  the full `Environment` (Design §12.1); the inheritance mechanism is the
  shared core helper.
- Baldwinian within-lifetime learning is weak in this world (500-tick
  lifespans, from-scratch networks), which compresses every Baldwinian effect.
- The predictive-validity result depends on the observed feature set; a
  narrower feature set (defection rate only) collapses to chance or below.
- The held-out cue confound described above inflates the C4 held-out band and
  therefore raises the bar for H4.

## Reproduce

```bash
PYTHONHASHSEED=0 python scripts/run_veil_ceiling.py                 # ~27 min on 4 cores
python scripts/run_veil_ceiling.py --analyze-only                    # re-analyse the committed record
```
