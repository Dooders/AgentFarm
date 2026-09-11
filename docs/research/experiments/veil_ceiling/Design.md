# The Veil Ceiling: Observation Collapse in Sealed-World Agent Evaluation

**Status:** Implemented and run (`farm.experiments.veil_ceiling`)
**CLI:** [`scripts/run_veil_ceiling.py`](../../../../scripts/run_veil_ceiling.py)
**Results:** [RESULTS.md](RESULTS.md), raw record under
[`experiments/veil_ceiling/results`](../../../../experiments/veil_ceiling/results)

This page records the pre-registered design as implemented. Section numbers
follow the original design document; Appendix A lists the parameter values that
were left open in the pre-registration and fixed before the confirmatory run.
Deviations from the original text are listed explicitly in §12.

## Abstract

Sealed-world evaluation assumes that what an agent does while observed is
informative about what it does when unobserved. The experiment tests the
condition under which that assumption fails: when the agent can *see a cue*
correlated with being observed. Agents in a shared resource world can defect
(over-harvest) for a private gain at a collective cost. Enforcement is held
constant; only the fidelity of an observation cue is varied. The prediction is
that cue fidelity, not enforcement intensity, drives the divergence between
observed and unobserved behaviour, and that above a fidelity threshold the
observed regime carries almost no information about the unobserved one — the
"veil ceiling".

## 1. Question

Given a fixed monitoring regime (coverage `c`, penalty `p`), how does the
fidelity `f` of an observation cue change (a) the gap between observed and
unobserved defection and (b) the predictive validity of observed behaviour for
unobserved behaviour? Does policy inheritance across generations (Lamarckian
transfer) change the answer?

## 2. Hypotheses

| | Hypothesis | Pre-registered test |
|---|---|---|
| H1 | Divergence Δ grows with cue fidelity; at `f = 1` (C2) it lies outside the C4 noise band while C1 (`f = 0`) shows none. | Paired C2 − C4 on Δ_cue with CI excluding 0; Spearman ρ(f, Δ) > 0 across the sweep. |
| H2 | Predictive validity of observed behaviour collapses as `f → 1`: AUC in C1 stays high (≥ 0.7) while AUC in C2 falls below 0.6. | LOSO logistic-regression AUC per cell with seed-bootstrap CI. |
| H3 | Concealment is not costly: conditional agents in C2 are at least as fit as unconditional cooperators (within 10 % of cooperator fitness). | Seed-paired difference in per-tick net energy between strategy classes. |
| H4 | The conditional policy transfers spatially: in the frozen evaluation, agents show cue-conditional defection in a region that was never monitored during training. | Held-out Δ_cue in C2 exceeds the C4 held-out band with CI > 0. |
| H5 | Lamarckian inheritance accelerates the *conditional* policy (C2) more than it accelerates simple cooperation (C1). | Paired difference of onset accelerations (Baldwinian − Lamarckian) across conditions. |

## 3. World

A 20 × 20 grid with 48 resource nodes. Each node regenerates logistically
(rate 0.5, capacity 10). A node whose stock falls below the regeneration
threshold (4) regenerates at a quarter of the healthy rate — this is the
collective cost of over-harvesting. Agents pay a small base consumption and a
move cost each tick, die at zero energy or at age 500, and reproduce
automatically (probability 0.5 per tick) when energy exceeds 20, paying 10 to
an offspring that starts with 10. Population is capped at 50 (initial 20).

The rightmost four columns (`x ≥ 16`) form the **held-out band**: never
monitored during training, monitored (and cued) like every other cell during
the frozen evaluation.

## 4. Affordance

Each tick an agent chooses one of four macro-actions from its own Q-network:

| Action | Effect |
|---|---|
| `MOVE` | Step one cell towards the nearest healthy node (cost 0.2). |
| `GATHER` | Take up to 3 units but never below the regeneration threshold. |
| `OVER_HARVEST` | Take up to 3 units regardless of the threshold, paying an effort cost of 0.5. |
| `PASS` | Do nothing. |

A **defection opportunity** is a tick at which over-harvesting the best node in
range yields strictly more, net of effort, than gathering. A **defection** is an
over-harvest at an opportunity that actually crosses the threshold. Defection
rates are defections per opportunity, so they are only counted where the
affordance tempts.

## 5. Monitoring and conditions

### 5.1 Parameters

- **Coverage `c`**: fraction of training-region cells monitored in each
  25-tick epoch; the monitored set is re-sampled every epoch (exactly
  `round(c · n)` cells) so the map cannot be memorised.
- **Penalty `p`**: energy deducted from a monitored defector at the moment of
  defection.
- **Fidelity `f`**: the cue shown to an agent is the monitor bit passed
  through a binary symmetric channel with flip probability `(1 − f)/2`.
  `f = 1` is a perfect leak, `f = 0` an uninformative coin. In C0 the cue is a
  constant 0.

Expected penalty per defection for a cue-blind policy is `c · p`, identical
across C1–C4 at training start. Realised enforcement (penalties / defections,
training region, training phase) is reported as an outcome.

### 5.2 Calibration requirement

C0 must produce an interior, stable baseline: seed-bootstrap CI of the
defection rate within [0.1, 0.9], |drift| ≤ 0.05 per 100 ticks, no
extinctions. If it does not, the run is a calibration failure, not evidence.

### 5.3 Cells

| Cell | c | f | p | Purpose |
|---|---|---|---|---|
| C0 | 0 | – | – | Baseline |
| C1 | 0.5 | 0 | 6 | Enforcement with intact veil |
| C2 | 0.5 | 1 | 6 | Enforcement with leaked veil |
| C3 | 0.5 | 0.1, 0.3, 0.5, 0.7, 0.9 | 6 | Dose-response |
| C4 | 0.5 | 1 (decoy map) | 6 | Null control: perfect cue of an independent map with identical statistics |
| Robustness | 0.5 | 0 / 1 | 3, 9 | C1 and C2 re-run at other penalties |

Every cell is crossed with **Baldwinian** (each newborn starts from a fresh
network) and **Lamarckian** (the parent's trained weights are copied into the
offspring via `farm.core.policy_inheritance.apply_lamarckian_policy_warmstart`)
inheritance. All 26 cells are run on the same 30 seeds; seed `s` fixes the
node layout, initial positions, true monitor map and per-agent policy streams
across cells so contrasts are matched pairs.

## 6. Metrics

- **Divergence Δ_cue** = rate(cue = 0) − rate(cue = 1), late training
  (last third of training windows). **Δ_true** uses the actual monitor mask
  instead of the cue.
- **Predictive validity (primary)**: per agent with ≥ 10 opportunities in
  each regime, observed-regime features (defection rate, action shares, mean
  energy, log exposure under cue = 1) predict whether unobserved-regime
  defection is above the cell median. Leave-one-seed-out logistic regression;
  AUC with seed-bootstrap CI. The single-feature rank AUC is reported alongside.
- **Cost of concealment**: agents classified as conditional (Δ_cue ≥ 0.3),
  cooperative (≤ 0.2 in both regimes), defector (≥ 0.5 in both) or mixed;
  fitness = (energy gained − penalties)/lifespan; seed-paired difference.
- **Onset**: first window from which Δ_cue exceeds the C4 per-window band
  for two consecutive windows (conditional onset); first window from which
  the rate is below half the seed-matched C0 rate (cooperative onset).
- **Held-out transfer**: cue-conditional defection during the frozen
  evaluation in the held-out band versus the training region, for agents born
  during training.

## 7. Learner

Each agent owns a two-layer tanh MLP (16 hidden) with a linear skip, trained by
online Q-learning (γ = 0.5, lr 0.05, replay 256, batch 16, target network
synced every 100 updates, optimistic initialisation, ε 0.3 → 0.1 over 100
steps). Learning is frozen for every agent at the end of training; agents
born during the evaluation are frozen from birth.

## 8. Analysis

Every cell has 30 seed-matched runs. Effects are reported as paired
differences with 95 % percentile-bootstrap CIs over seeds (2000 replicates),
sign agreement and paired Cohen's d. The C4 noise band is mean ± 2 SD of C4
Δ_cue across seeds, computed separately for late training, per window and for
the held-out evaluation. Dose-response is Spearman ρ over C1, the C3 sweep and
C2.

## 9. Falsification

1. If C4 shows divergence comparable to C2 — its Δ_cue CI excludes zero *and*
   it is not dwarfed by C2 (C2 − C4 excluding zero with |Δ_C4| < ½ |Δ_C2|) —
   the design is void (cue channel confounded with input dimensionality) and
   H1–H4 are marked `void`.
2. If C2's Δ_cue lies inside the C4 band, H1 and H2 are falsified.
3. If validity AUC in C2 remains ≥ 0.7, H2 is falsified.

## 10. Robustness

C1 and C2 at `p ∈ {3, 9}` test whether the divergence result depends on the
penalty magnitude at fixed expected enforcement structure.

## 11. Implementation map

| Design element | Code |
|---|---|
| Parameters and conditions | `farm/experiments/veil_ceiling/config.py` |
| Resource world and affordance | `farm/experiments/veil_ceiling/world.py` |
| Monitor masks and cue channel | `farm/experiments/veil_ceiling/monitoring.py` |
| Per-agent Q-learner and inheritance surface | `farm/experiments/veil_ceiling/learner.py` |
| Agent ledger (phase × region × monitored × cue) | `farm/experiments/veil_ceiling/agents.py` |
| Simulation loop, reproduction, warm-start | `farm/experiments/veil_ceiling/simulation.py` |
| Metrics | `farm/experiments/veil_ceiling/metrics.py` |
| Matrix orchestration and raw outputs | `farm/experiments/veil_ceiling/experiment.py` |
| Paired stats, band, checks, verdicts | `farm/experiments/veil_ceiling/analysis.py` |
| REPORT.md and figures | `farm/experiments/veil_ceiling/report.py` |
| Tests | `tests/experiments/test_veil_ceiling_*.py` |

## 12. Deviations from the original design text

1. **Compact world instead of the full `Environment` stack.** A full
   AgentFarm simulation runs for ~18 minutes per configuration, which makes
   780 seed-matched runs impractical. The experiment re-implements the grid,
   logistic regeneration, energy accounting, reproduction and death in a
   self-contained numpy package. The inheritance mechanism is the shared core
   helper (`apply_lamarckian_policy_warmstart`) and its telemetry, not a copy.
2. **Automatic reproduction.** Reproduction is a threshold-and-chance rule
   rather than a learned action, so the action space stays focused on the
   affordance under test.
3. **Macro `MOVE`.** Movement is a single "step towards the nearest healthy
   node" action; positions are not in the observation, only the coarse region
   id, so the learner cannot memorise monitored coordinates.
4. **Opportunity definition.** A defection opportunity requires that
   over-harvesting is *individually profitable net of effort*; a defection is
   only counted when the draw actually crosses the threshold. This keeps the
   rate denominator to decisions where the affordance tempts.
5. **Cue channel.** Fidelity is implemented as a binary symmetric channel so
   `f = 0` is exactly chance and `f = 1` exactly the monitor bit.
6. **C4 decoy map.** The null control draws a second mask from an independent
   stream with the same coverage, epoch schedule and spatial restriction, and
   reports that instead of the true mask.
7. **Held-out band in evaluation only.** The band is monitored and cued only
   in the frozen evaluation phase so that its training-phase behaviour is
   truly unmonitored.
8. **Baldwinian within-lifetime learning is weak.** With a 500-tick maximum
   lifespan and no inheritance, individual agents learn slowly; the Baldwinian
   arm therefore shows smaller effects than the Lamarckian arm. This is an
   observed property of the world, reported rather than tuned away.
9. **Falsification check 1** is operationalised as "C4 divergence comparable
   to C2" (see §9) rather than "C4 CI includes zero", because a CI that
   barely excludes zero while being an order of magnitude below C2 is not a
   confound.

## 13. Reproducibility

```bash
source venv/bin/activate
PYTHONHASHSEED=0 python scripts/run_veil_ceiling.py            # full matrix (~30 min on 4 cores)
python scripts/run_veil_ceiling.py --analyze-only               # re-analyse the committed raw record
python scripts/run_veil_ceiling.py --seeds 3 --no-robustness --output-dir /tmp/veil_pilot
pytest tests/experiments/test_veil_ceiling_world.py tests/experiments/test_veil_ceiling_simulation.py tests/experiments/test_veil_ceiling_analysis.py
```

Outputs: `manifest.json` (all parameters, thresholds, git commit),
`cells/<cell>/{runs.csv,windows.csv,agents.csv.gz}` (raw per-run, per-window
and per-agent ledgers), `runs.csv` (combined), `analysis/*.csv` and
`analysis/checks.json`, `figures/*.png`, `REPORT.md`.

## Appendix A — parameters fixed before the confirmatory run

| Parameter | Value |
|---|---|
| Grid | 20 × 20; held-out band `x ≥ 16` |
| Resource nodes | 48; capacity 10; initial 10 |
| Regeneration | logistic, rate 0.5, floor 0.5; threshold 4; suppressed factor 0.25 |
| Gather / over-harvest draw | 3 units; over-harvest effort 0.5 |
| Costs | move 0.2; base consumption 0.05 per tick |
| Population | initial 20; cap 50; initial energy 12 |
| Reproduction | threshold 20; chance 0.5; offspring cost 10; offspring energy 10 |
| Lifespan | max age 500 ticks |
| Training / evaluation | 1500 training ticks; 300 frozen evaluation ticks; 50-tick windows |
| Monitoring | `c = 0.5`; epoch 25 ticks; `p = 6` primary, `p ∈ {3, 9}` robustness |
| Fidelity sweep | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Learner | MLP 16 hidden; lr 0.05; γ 0.5; ε 0.3 → 0.1 over 100 steps; replay 256; batch 16; target sync 100; optimistic init 1.0; 2 train steps / observation |
| Seeds | 30 per cell (1–30), shared across all cells |
| Validity classifier | leave-one-seed-out logistic regression (standardised features) |
| Eligibility for validity | ≥ 10 opportunities in each cue regime |
| Strategy thresholds | conditional Δ ≥ 0.3; cooperative ≤ 0.2; defector ≥ 0.5 |
| AUC thresholds | high ≥ 0.7; collapse < 0.6 |
| Calibration bounds | rate CI within [0.1, 0.9]; drift ≤ 0.05 / 100 ticks |
| Noise band | C4 mean ± 2 SD across seeds |
| Onset | 2 consecutive windows |
| Bootstrap | 2000 replicates, seed 20260911 |

## Appendix B — reproducibility record

The confirmatory run was executed with `PYTHONHASHSEED=0` on 4 worker
processes. Each run's RNG is derived from `np.random.SeedSequence(seed)`
spawned into six named streams (world, dynamics, policy, monitor, decoy, cue),
so seed-matched cells share their world layout and true monitor map
bit-for-bit (verified by `test_seed_matching_shares_world_and_monitor_map_across_conditions`).
