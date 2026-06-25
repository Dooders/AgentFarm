# Crossover Rerun — Intrinsic Evolution

This experiment re-runs the May-12 [stable-profile seed sweep](../../devlog/2026-05-12-seed-sweep-reality-check.md)
with intrinsic-evolution **crossover enabled**, then compares the result
against the no-crossover baseline paired by seed.

Related issues / context:

- Issue [#845](https://github.com/Dooders/AgentFarm/issues/845) — buffered + crossover follow-up
- Issue [#867](https://github.com/Dooders/AgentFarm/issues/867) — long-horizon conservative follow-up (out of scope here)
- Devlog: [When one seed disagrees with six](../../devlog/2026-05-12-seed-sweep-reality-check.md)
- Devlog: [Does the resource buffer pick the genes?](../../devlog/2026-05-04-resource-buffer-shapes-intrinsic-evolution.md)

## Hypothesis

The 6-seed sweep established that all three stable profiles
(`conservative`, `balanced`, `buffered`) show diverging speciation
trajectories under the current setup, with mean final speciation index
~0.59–0.75. That divergence was generated **without** gene flow: the
intrinsic-evolution policy ran with `crossover_enabled=False`, so children
inherited the parent chromosome modulo mutation.

If gene flow is the missing homogenizing force, enabling crossover should
**collapse** the rising speciation index. If the divergence is structural
(carried by speciation/spatial selection rather than weak mixing), gene
flow may shift gene-level magnitudes but should leave the speciation
direction alone.

## Design

Two crossover arms × three profiles × six seeds = **36 runs**, paired
by seed against the existing baseline at
`experiments/stable_profile_sweep/stable_{profile}/seed_{seed}/`.

### Arms

| Arm | `crossover_mode` | `blend_alpha` | Notes |
| --- | --- | --- | --- |
| `uniform` | `uniform` | n/a | per-gene coin flip from either parent; standard "recombination" interpretation |
| `blend`   | `blend`   | 0.5 | BLX-α continuous interpolation; strongest continuous-gene mixer |

### Profiles

`conservative` / `balanced` / `buffered` from
[`STABLE_SUB_PROFILES`](../../../farm/runners/intrinsic_evolution_experiment.py).
Resource-buffer settings are unchanged from the baseline sweep.

### What was held fixed (matches baseline)

| Setting | Value |
| --- | --- |
| Seeds | 42, 7, 19, 101, 137, 256 |
| Steps logged | 1000 |
| Warmup steps | 200 |
| Snapshot interval | 50 |
| Mutation | gaussian, rate 0.15, scale 0.10, reflect boundary |
| Selection pressure | low |
| Co-parent strategy | nearest_alive_same_type |
| Co-parent max radius | unbounded |
| Cross-type pollination | off |
| Speciation tracking | GMM, max_k=4 |
| Initial diversity | INDEPENDENT_MUTATION, rate 1.0, scale 0.25 |

## Commands

### 1. Run both arms across all profiles and seeds

```bash
python scripts/run_crossover_rerun.py \
    --output-dir experiments/crossover_rerun
```

This invokes [`scripts/run_stable_profile_seed_sweep.py`](../../../scripts/run_stable_profile_seed_sweep.py)
in-process for each arm. Each arm writes to
`experiments/crossover_rerun/{arm}/stable_{profile}/seed_{seed}/` and a
top-level `rerun_manifest.json`. By default, `analyze_stable_profile_seed_sweep.py`
is then invoked on each arm directory to produce per-arm aggregates.

For dry-run inspection only:

```bash
python scripts/run_crossover_rerun.py --output-dir experiments/crossover_rerun --dry-run
```

### 2. Paired-seed comparison vs baseline

```bash
python scripts/compare_crossover_arms.py \
    --baseline-dir experiments/stable_profile_sweep \
    --treatment-dir experiments/crossover_rerun/uniform \
    --treatment-dir experiments/crossover_rerun/blend \
    --arm-labels uniform blend \
    --output-dir experiments/crossover_rerun/aggregate
```

## Outputs

The comparison analyzer writes the following to `--output-dir`:

- `crossover_rerun_summary.md` — verdict-per-profile narrative ("Does
  gene flow collapse the rising speciation pattern?") plus paired-delta
  and gene-shift tables. Verdict thresholds: paired-delta 95% CI excludes
  zero **and** ≥75% within-profile sign agreement (matches
  `SIGN_AGREEMENT_THRESHOLD` in
  [`scripts/analyze_stable_profile_seed_sweep.py`](../../../scripts/analyze_stable_profile_seed_sweep.py)).
- `crossover_rerun_summary.json` — machine-readable form of the above.
- `speciation_trajectories_with_arms.png` — speciation traces per
  profile, baseline + each arm (mean ribbon + per-seed lines).
- `paired_delta_heatmap.png` — rows = (profile × arm), columns = key
  metrics (`speciation_final`, `speciation_slope`, lineage cluster count
  and churn, plus convergent and direction-flip genes). Asterisk
  annotations mark cells where the paired delta is robust.
- `lineage_cluster_count.png` — mean cluster count per step, one curve
  per arm, per profile.

## Out of scope

- Long-horizon (3k / 5k step) runs — see Issue [#867](https://github.com/Dooders/AgentFarm/issues/867).
- The `stress` / `legacy` initial-conditions profiles — see Issue [#846](https://github.com/Dooders/AgentFarm/issues/846).
- Tuning `coparent_max_radius` and `allow_cross_type_pollination`. Both
  stay at the policy defaults so the only varied axes are
  `crossover_mode` and the existing `stable` sub-profile.

## Results

Completed 2026-05-18: 36 runs (2 arms × 3 profiles × 6 seeds), all
successful, `--disk-database` enabled (~4.6 h per arm wall time). Paired
comparison against `experiments/stable_profile_sweep/` is in
`experiments/crossover_rerun/aggregate/` (gitignored).

This sweep used the updated chromosome crossover implementation (not the
2026-05-14 in-memory run). Outcomes differ materially from that earlier
pass, especially for **conservative**.

### Headline verdict (all profiles)

| Profile | uniform | blend |
| --- | --- | --- |
| conservative | **robustly collapses** | **robustly collapses** |
| balanced | no robust effect | no robust effect |
| buffered | no robust effect | no robust effect |

*Verdict = paired delta on `speciation_final` or `speciation_slope` with 95%
CI excluding zero and ≥75% within-profile sign agreement (see
`scripts/compare_crossover_arms.py`). "Collapses" = lower final speciation
and/or less-positive slope vs. no-crossover baseline.*

### Cross-profile summary

- **Conservative:** Gene flow **does** homogenize. Both arms lower final
  speciation index (uniform −0.063, blend −0.086; 6/6 seeds negative) and
  mean speciation over the run, without flipping the modal trajectory
  direction (still diverging 6/6 seeds in each arm — clusters separate, but
  less far apart).
- **Balanced:** No robust paired shift in final index or slope; high
  seed-to-seed variance persists.
- **Buffered:** No robust collapse of final index or slope (Issue #845). All
  six seeds remain **diverging** under both crossover operators. Blend
  lowers mean speciation over the run (−0.065, CI excludes zero, 6/6
  negative) similar to the conservative pattern but without a robust final-index
  drop.

### Buffered profile — Issue [#845](https://github.com/Dooders/AgentFarm/issues/845)

| Metric | Baseline (no crossover) | uniform crossover | blend crossover | Paired Δ (uniform) | Paired Δ (blend) |
| --- | --- | --- | --- | --- | --- |
| Mean final speciation index | 0.689 | 0.698 | 0.641 | +0.010 | −0.048 |
| Mean speciation slope (/100 steps) | 0.020 | 0.027 | 0.026 | +0.007 | +0.006 |
| Trajectory direction (6 seeds) | diverging (6/6) | diverging (6/6) | diverging (6/6) | — | — |
| Mean cluster count (lineage) | — | — | — | −0.37 | −0.09 |
| Cluster churn rate | — | — | — | ~0 | +0.004 |

**Conclusion (#845):** Under buffered resource conditions, crossover does
**not** collapse the rising speciation *trajectory* (no robust paired shift
in final index or slope; 6/6 seeds still diverging). Gene flow may trim
average cluster separation (blend arm, mean index over the run) but does
not reverse buffered runs into a conservative-style merge. The May-4
single-seed rising trace (0.653 → 0.753) is consistent with the multi-seed
baseline; crossover leaves that direction intact.

`learning_rate` shifts under crossover remain seed-sensitive for buffered
(see per-seed logs in the sweep output; e.g. uniform seed 42 at +4.9%,
seed 256 at +23.5%).

### Artifacts

| Artifact | Path |
| --- | --- |
| Paired comparison (MD) | `experiments/crossover_rerun/aggregate/crossover_rerun_summary.md` |
| Paired comparison (JSON) | `experiments/crossover_rerun/aggregate/crossover_rerun_summary.json` |
| Speciation traces (all profiles) | `experiments/crossover_rerun/aggregate/speciation_trajectories_with_arms.png` |
| Paired-delta heatmap | `experiments/crossover_rerun/aggregate/paired_delta_heatmap.png` |
| Lineage cluster count | `experiments/crossover_rerun/aggregate/lineage_cluster_count.png` |
| Per-arm aggregates | `experiments/crossover_rerun/{uniform,blend}/aggregate/seed_sweep_summary.md` |
| Orchestrator manifest | `experiments/crossover_rerun/rerun_manifest.json` |
