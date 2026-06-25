---
layout: page
title: "Gene flow and the buffer: when crossover compresses (and when it doesn't)"
---

The [05-12 seed sweep](2026-05-12-seed-sweep-reality-check.md) gave a clean baseline:
with crossover off, speciation trajectories diverged across conservative,
balanced, and buffered resource profiles. The obvious mechanism question was
whether that divergence was mostly an artifact of mutation-only inheritance.
If we add gene flow, do clusters collapse?

I re-ran the same 6-seed matrix with crossover enabled and compared each run
against its no-crossover counterpart by profile and seed.

Short answer: **it depends on the profile**. Gene flow robustly *compresses*
endpoint separation in conservative runs (it doesn't merge clusters — all
arms still end clearly above the merging regime), leaves buffered runs
essentially unchanged, and stays noisy in balanced runs.

## Setup

Two crossover arms, each paired against the existing no-crossover baseline:

- `uniform`: per-gene coin-flip recombination
- `blend`: BLX-alpha interpolation (`alpha=0.5`)

Shared setup:

- profiles: `conservative`, `balanced`, `buffered`
- seeds: `[42, 7, 19, 101, 137, 256]`
- logged steps: 1000 (after 200-step warmup)
- selection pressure: `low`
- speciation tracking: GMM, max k = 4
- database mode: **disk-backed SQLite** (`--disk-database`)

Commands and full artifact paths live in
[crossover_rerun.md](../research/experiments/intrinsic_evolution/crossover_rerun.md).

## The headline result

| Profile | uniform | blend |
| --- | --- | --- |
| conservative | **robustly compresses** | **robustly compresses** |
| balanced | no robust effect | no robust effect |
| buffered | no robust effect | no robust effect |

Verdict rule is the same one used in analysis scripts: paired delta on
`speciation_final` or `speciation_slope` with 95% CI excluding zero and at
least 75% within-profile sign agreement. The analyzer labels this verdict
"robustly collapses"; I'm calling it "compresses" here because the magnitude
is small (Δ ≈ -0.06 to -0.09) and the trajectories still end well above the
merging regime — see the conservative panel below.

![Speciation trajectories with crossover arms and no-crossover baseline](figures/speciation_trajectories_with_arms.png)

## What changes by profile

### Conservative: crossover compresses (it does not collapse)

This is the biggest surprise in the rerun. Against no-crossover baseline:

- baseline final speciation: `0.748`
- `uniform` final: `0.685` (Δ `-0.063`, 95% CI excludes zero, 6/6 negative)
- `blend` final: `0.663` (Δ `-0.086`, 95% CI excludes zero, 6/6 negative)

Both arms still classify as `diverging` (6/6 seeds, positive slope), and all
three arms end well above the merging regime. The chart shows this clearly:
the orange baseline line sits ~0.07-0.09 above the uniform/blend lines, but
all three are rising and stay in the 0.65-0.78 band. So crossover does not
invert trajectory direction and does not pull clusters together; it shifts
the rising trace down by a small, robust amount.

### Buffered (#845): trajectory survives gene flow

Issue [#845](https://github.com/Dooders/AgentFarm/issues/845) asked whether
gene flow collapses the buffered "rising speciation" pattern.

Across all six seeds, buffered trajectories are still `diverging` under both
arms. Paired deltas on final speciation are not robustly negative; slope
deltas are slightly positive (uniform +0.007, blend +0.006) but also not
robust.

| Metric | Baseline | uniform | blend |
| --- | --- | --- | --- |
| Mean final speciation | 0.689 | 0.698 | 0.641 |
| Mean slope (/100 steps) | 0.020 | 0.027 | 0.026 |
| Direction agreement | diverging (6/6) | diverging (6/6) | diverging (6/6) |

Blend does lower mean speciation over the run, but not enough to qualify as a
robust collapse on the final-index or slope criteria.

### Balanced: still the high-variance middle

Balanced remains the least stable profile in this line of experiments.
Neither arm produces a robust paired shift in final speciation or slope. This
matches the earlier "balanced is unusually variable" pattern from 05-12.

## What this says about the mechanism

The buffer seems to control how much recombination can pull lineages
together:

- under tighter-resource conservative conditions, crossover compresses
  endpoint separation by a small but robust amount (without merging
  clusters);
- under buffered conditions, crossover mixes genes but does not erase the
  diverging trajectory pattern;
- balanced remains near a regime boundary where variance dominates.

So the updated claim is more measured than a "gene flow erases speciation"
story: **resource profile shapes the strength of crossover's compressive
effect on speciation, but in this regime it modulates rather than reverses
the diverging trajectory.**

## Caveats

- This remains a low-selection-pressure regime (`selection_pressure="low"`).
- Metrics answer trajectory and endpoint questions, not causality at the
  behavioral-policy level.
- Several per-gene shifts (including `learning_rate`) remain seed-sensitive.

## What's next

- Long-horizon conservative sweeps (Issue
  [#867](https://github.com/Dooders/AgentFarm/issues/867)):
  does crossover compression persist or saturate over 3k-5k steps?
- Wider profile axis with `stress` and `legacy` (Issue
  [#846](https://github.com/Dooders/AgentFarm/issues/846)).
- Follow-up on why `balanced` is the variance peak.

## Related docs

- [When one seed disagrees with six](2026-05-12-seed-sweep-reality-check.md)
- [Does the resource buffer pick the genes?](2026-05-04-resource-buffer-shapes-intrinsic-evolution.md)
- [Crossover rerun experiment doc](../research/experiments/intrinsic_evolution/crossover_rerun.md)
- [Intrinsic evolution docs](../research/experiments/intrinsic_evolution/intrinsic_evolution.md)
