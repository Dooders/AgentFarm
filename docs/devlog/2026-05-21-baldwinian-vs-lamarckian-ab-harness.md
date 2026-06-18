---
layout: page
title: "Baldwinian vs Lamarckian: policy warm-start across three resource regimes"
---

Issue [#849](https://github.com/Dooders/AgentFarm/issues/849) asked for a matched
A/B path to quantify when offspring policy warm-starting helps and when it
destabilizes intrinsic-evolution runs. This post reports the full matrix:
36 paired simulations, one aggregate comparison, and a clear answer on whether
Lamarckian inheritance is worth turning on under the stable-profile regimes we
already use elsewhere.

## What we tested

Two inheritance modes on the same intrinsic-evolution stack:

- **Baldwinian** (baseline): offspring inherit the hyperparameter chromosome
  and start with a fresh decision policy.
- **Lamarckian** (treatment): same chromosome inheritance, plus a compatible
  policy warm-start copied from the parent at reproduction.

Everything else is held fixed across arms — see the
[protocol doc](../experiments/intrinsic_evolution/inheritance_mode_ab.md)
for the full parameter list. Highlights:

| Knob | Value |
| --- | --- |
| Profiles | `conservative`, `balanced`, `buffered` |
| Seeds | `42`, `7`, `19`, `101`, `137`, `256` |
| Logged steps | 1000 (200-step warmup) |
| Crossover | off (isolates inheritance mode) |
| Selection pressure | low |
| Speciation | GMM, `max_k=4` |

Matrix size: **2 arms × 3 profiles × 6 seeds = 36 runs**.

Runner: `scripts/run_inheritance_mode_ab.py`.
Comparator: `scripts/compare_inheritance_arms.py`.
Outputs: `experiments/inheritance_ab/` (manifest, per-arm sweeps, aggregate
summary and plots).

## Headline result

**No robust effect in any profile.** Under the protocol's acceptance gate
(paired 95% CI excludes zero *and* sign agreement ≥ 75%), Lamarckian
warm-starting does not earn a regime-wide recommendation — not as a win, not
as a stability loss, not as a speciation-collapse risk.

| Profile | Verdict |
| --- | --- |
| conservative | no robust effect |
| balanced | no robust effect |
| buffered | no robust effect |

That is not the same as "nothing happened." Warm-start executed at scale and
paired runs diverged on population counts. The treatment just did not clear a
strict small-sample bar on the ecological readouts we score for recommendations.

## Mechanism coverage

Lamarckian warm-start was active throughout the Lamarckian arm:

| Profile | Mean success rate | 95% CI | Applied | Skipped |
| --- | --- | --- | --- | --- |
| conservative | 0.852 | [0.838, 0.867] | 3118 | 538 |
| balanced | 0.849 | [0.836, 0.862] | 3837 | 681 |
| buffered | 0.849 | [0.835, 0.863] | 3773 | 671 |

Every skip was `incompatible_state` — parent and child policy shapes did not
match at reproduction time, so those offspring fell back to a cold start.
`decide_action_failures` were zero in both arms across all 36 runs.

Wall-clock: ~5.5 h per arm (~19.7k s Baldwinian, ~20.2k s Lamarckian). All
36 runs completed without error.

## Paired deltas (Lamarckian − Baldwinian)

### Performance

Population is the primary performance readout. Effects are seed-noisy and
profile-dependent:

**Conservative** — mildly negative on average, mixed by seed:

| Metric | Mean Δ | 95% CI | Sign agreement |
| --- | --- | --- | --- |
| population mean | −8.2 | [−19.5, 3.1] | 67% |
| population final | −6.2 | [−21.5, 9.2] | 67% |

Per-seed final population (Baldwinian → Lamarckian): 74→76 (+2), 64→76 (+12),
63→31 (−32), 67→62 (−5), 75→67 (−8), 67→61 (−6). Seed 42 alone accounts for
most of the negative mean.

**Balanced** — the strongest directional signal, still not robust:

| Metric | Mean Δ | 95% CI | Sign agreement |
| --- | --- | --- | --- |
| population mean | +6.2 | [−0.4, 12.8] | 83% |
| population final | +11.8 | [−2.1, 25.8] | 83% |

Five of six seeds gained population (+6 to +27 agents). Seed 7 lost nine
agents, widening the CI enough to include zero.

Per-seed final population: 97→88 (−9), 72→88 (+16), 67→94 (+27), 60→67 (+7),
67→91 (+24), 74→80 (+6).

**Buffered** — flat:

| Metric | Mean Δ | 95% CI | Sign agreement |
| --- | --- | --- | --- |
| population mean | −3.8 | [−16.8, 9.2] | 50% |
| population final | 0.0 | [−9.0, 9.0] | 67% |

Per-seed final population: 99→98, 88→86, 105→104, 93→102, 74→83, 95→81 —
small swings in both directions.

### Stability

Startup death rate was **0.0** in every run for both arms, so the stability-loss
path never fired. Oscillation amplitude deltas were small and CI-wide in all
profiles (conservative +3.3, balanced −2.2, buffered −0.3; none robust).

### Diversity

Speciation slope moved slightly positive under conservative (+0.007/100 steps,
83% sign agreement, CI barely excludes zero) but that metric alone does not
trigger a recommendation — and the classifier treats *negative* slope as
collapse risk, not positive. Buffered and balanced speciation deltas were
essentially flat.

## How to read this

Three layers stack on top of each other:

1. **The mechanism works.** ~85% of reproduction events in the Lamarckian arm
   successfully copied parent policy weights. Arms are not equivalent at the
   action-selection layer.

2. **Ecological outcomes are a second-order perturbation.** Both arms share
   the same chromosome inheritance path; decisions combine policy probabilities
   with chromosome action weights multiplicatively. Inherited weights nudge
   behavior, but population and speciation are emergent, high-variance
   summaries — especially with only six paired seeds.

3. **The verdict gate is conservative.** Balanced came closest to a Lamarckian
   performance win (83% sign agreement, mean +12 final population) but one
   dissenting seed kept the 95% CI straddling zero. Conservative and buffered
   never approached a clean call.

Practical takeaway for now: **keep Baldwinian as the default.** Lamarckian
warm-start adds ~2% wall-clock overhead and ~15% cold-start fallbacks without
a demonstrated regime-wide payoff on the metrics we care about at this scale.

## What shipped (harness)

The experiment path that produced these numbers:

- `IntrinsicEvolutionPolicy.inheritance_mode`: `baldwinian` | `lamarckian`
- Reproduction applies warm-start only in Lamarckian mode via
  `apply_lamarckian_policy_warmstart`
- Telemetry in run metadata:
  `policy_inheritance_metrics.lamarckian_warmstart_applied/skipped`
  (renamed to mode-neutral `warmstart_applied/skipped` on 2026-06-17 when the
  P2–P4 variants landed)
- `scripts/run_inheritance_mode_ab.py` — orchestrates both arms
- `scripts/compare_inheritance_arms.py` — paired-seed deltas and verdicts
- `scripts/run_stable_profile_seed_sweep.py` accepts `--inheritance-mode`

To reproduce:

```bash
PYTHONHASHSEED=0 python scripts/run_inheritance_mode_ab.py \
  --output-dir experiments/inheritance_ab \
  --disk-database \
  --resume

python scripts/compare_inheritance_arms.py \
  --baseline-dir experiments/inheritance_ab/baldwinian \
  --baseline-label baldwinian \
  --treatment-dir experiments/inheritance_ab/lamarckian \
  --arm-labels lamarckian \
  --output-dir experiments/inheritance_ab/aggregate
```

## Open questions

- **Balanced near-miss:** with more seeds or a longer horizon, does the +12
  mean final-population delta on balanced consolidate into a robust win?
- **Conservative seed sensitivity:** is the seed-42 collapse (−32 agents) a
  genuine Lamarckian failure mode or run noise?
- **Mechanism-proximal metrics:** offspring fitness in the first *N* steps after
  birth may show a Lamarckian advantage even when whole-population summaries
  do not.

Those are follow-ups, not revisions to this aggregate. At n=6 per profile, the
honest read is: warm-start runs, sometimes helps individual seeds, and does not
yet justify flipping the default inheritance mode.
