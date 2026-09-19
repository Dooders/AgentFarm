# Union emergence — full experiment scope

This is the protocol for
[PR 1014](https://github.com/Dooders/AgentFarm/pull/1014): exclusive,
costly-to-exit pair-bonds as a selected *unit* versus promiscuous `share`.
The through-line is **sharing is a transaction; a union is an agent**.

The PR landed the pre-register (devlog + catalog) without the standalone
runner. This directory is that runner plus the first-glance artifacts.

## Claim

Two non-committed agents optimize locally. A locked pair is a new unit with
its own persistence (accumulated `bond_strength`, shared horizon, costly
exit, complementary roles, joint reproductive budget). Selection keeps the
unit only when leaving costs more than staying, the neighborhood is small
enough to stop shopping, and the surplus split is not silent extraction.

This is **not** the existing `share_weight` / `reward_share_bonus` machinery.
Those are one-shot transfers. The hole in Nowak / tag / LLM-IPD work is the
same: they vary a share-or-defect knob and never grow a `partner_id`.

## Questions

| Version | Question |
|---|---|
| v1 | Does an exclusive, costly-to-exit union produce superadditive returns that promiscuous `share` does not? |
| v2 | Under what bonding cost, divorce cost, courtship delay, and neighborhood size does that exclusive unit appear — and is it symbiotic or extractive? |
| Port | Do the same cells hold after the genes and `bond` / `leave` actions live on the AgentFarm chromosome? |

## Layers

| Layer | Status | What it is |
|---|---|---|
| **A. Standalone v1 pilot** | Specified; runnable here | Instant synergy, no bonding cost, binary lock. 4 arms × 8 seeds × 900 steps. |
| **B. Standalone v2 grid** | Specified; default runner | Courtship, bonding cost, accumulated strength, accidental divorce, extraction. 4 worlds × 4 arms × 4 seeds × 550 steps. |
| **C. AgentFarm port** | Outlined, not wired | Three genes on the chromosome, `bond` / `leave` next to `share`, `UnionEmergenceExperiment` or a `unique+union` arm on `IntrinsicGoalsExperiment`. |
| **D. Port follow-ups** | After C | Seed-matched A/B on the baseline cell; ablate synergy vs split-cost vs exit tax; heterogeneous goals × union. |

First-glance evidence is layer B (or a seed/step-reduced B that still
crosses every arm and every world). Layer C is scoped here and not run.

## Literature constraints (do not drop)

| Paper | Design constraint |
|---|---|
| Song, Feldman & Gavrilets 2013 | Do not treat rising `pair_commitment` as success. Sweep bonding cost and accident rate. |
| Reynolds 2018 | Keep a gift/share-off control (`solo_only`). Keep `max_pop`. |
| Akçay et al. 2023 | Courtship gate before synergy / split-cost. Sweep `exit_tax`. |
| Leimar & McNamara 2024 | `bond_strength` is accumulated, not a binary lock. Keep `social_range` small; wide range is an ablation. |
| Ogbo, Elragig & Han 2022 | Log extraction = energy gap × commitment gap. Forced ≠ chosen. |

Out of scope for now: institutional monogamy norms (Bauch), LLM courtship
talk, lifting `max_pop`, tit-for-tat share.

## Genes

| Gene | Range | Role |
|---|---|---|
| `pair_commitment` | [0, 1] | willingness to *start* a lock |
| `fidelity` | [0, 1] | leave resistance; scales exit tax and grief |
| `specialize` | [0, 1] | gather/guard complementarity once bonded |
| `share_weight` | [0, 2] | promiscuous transfer (reused) |
| `attack_weight` | [0, 2] | steal pressure (reused; v1 logged drift) |

Commitment starts the bond. Strength is the unit. Fidelity is what
selection is allowed to keep.

## Mechanics (v2, frozen as the port baseline)

1. **`bond` / `leave`.** Assortative on commitment. Optional pairing forms
   with \(p = c_i c_j\); forced pairing uses a high fixed rate. Pay
   `bonding_cost` split; refuse if either cannot cover it.
2. **Courtship.** No gather synergy and no split offspring cost until
   `pair_age ≥ courtship_steps`.
3. **Strength.** Each co-located step:
   `strength += bond_gain * min(c) − bond_decay`, clipped to [0, 1].
   Synergy = \(1 + k \cdot \mathrm{strength} \cdot \mathrm{mean}(s)\).
4. **While bonded.** Co-location pull; loose equalization toward the poorer
   partner (Leimar, not TFT). Promiscuous share suppressed.
5. **Exit.** Accidental divorce at `accident_divorce`, then leave with
   \((1-f)\cdot(0.04+\mathrm{stress})\cdot(1-0.5\cdot\mathrm{strength})\).
   Exit tax \(\propto f\). Death applies grief \(\propto f\).
6. **No external fitness.** Survival and reproduction only.

Default world: `bonding_cost=0.8`, `courtship_steps=12`, `exit_tax=1.1`,
`social_range=3.2`, `accident_divorce=0.01`, `max_pop=80`.

v1 differs only by turning the literature knobs off: `bonding_cost=0`,
`courtship_steps=0`, binary lock (`bond_strength` snaps to 1).

## Arms

| Arm | Pairing | Share |
|---|---|---|
| `solo_only` | clamped off | residual only |
| `promiscuous` | off | open `share_weight` |
| `optional_union` | evolvable commitment | pair-internal after courtship |
| `forced_union` | high pairing rate | same pair-internal rules |
| *(port only)* `unique+union` | fourth arm on `IntrinsicGoalsExperiment` | goals heterogeneous |

## v2 worlds (compact threshold grid)

| Cell | bonding_cost | courtship_steps | exit_tax | social_range |
|---|---|---|---|---|
| `baseline` | 0.8 | 12 | 1.1 | 3.2 |
| `cheap_exit_cheap_bond` | 0.2 | 12 | 0.2 | 3.2 |
| `no_courtship` | 0.8 | 0 | 1.1 | 3.2 |
| `wide_neighborhood` | 0.8 | 12 | 1.1 | 8.0 |

First three AgentFarm port ablations (layer C), if the baseline cell holds:
`courtship ∈ {0, 12}`, `exit_tax ∈ {0.2, 1.1, 2.2}`,
`social_range ∈ {2.0, 3.2, 8.0}`.

## Metrics (pre-registered)

Primary:

- `synergy_index = E[energy | paired] / E[energy | solo]` in the **same**
  world, counting currently paired vs unpaired agents. Courtship still
  matters because only mature pairs receive the gather multiplier.
- `mean_pair_duration`, `mean_bond_strength`
- `fidelity` Δ (not commitment Δ)
- `mean_extraction = E[ |E_i−E_j| · |c_i−c_j| ]` among live pairs
- `lineages_alive`

Secondary: `pair_commitment` Δ (expected ≤ 0 under optional pairing),
action mix, mean energy, paired fraction.

## Win conditions / falsifiers

Win:

1. Synergy CI excludes 1 on baseline optional/forced; promiscuous cannot
   match it (share is a transfer, not a multiplier).
2. Optional commitment does **not** climb; fidelity does when `exit_tax`
   is above threshold.
3. Ablating courtship inflates synergy (part of the effect was free).
4. Cheap exit and wide `social_range` shrink duration and flatten fidelity Δ.
5. Forced extraction > optional extraction.

Falsifiers: synergy CI includes 1; `reward_share_bonus` / open share
matches union synergy; fidelity falls in the costly-exit forced arm;
optional `paired_frac → 0` at medium/high selection pressure.

## Run commands

```bash
# Default = v2 compact threshold grid (4 worlds × 4 arms × 4 seeds × 550)
python experiments/union_emergence/union_intrinsic_evolution.py

# First glance: every cell, fewer seeds/steps
python experiments/union_emergence/union_intrinsic_evolution.py --mode first_glance

# v1 pilot (instant synergy, binary lock)
python experiments/union_emergence/union_intrinsic_evolution.py --mode v1
```

Writes `experiments/union_emergence/compact_threshold_summary.json` (v2 /
first-glance) or `v1_pilot_summary.json`. The README also mentions a
sandbox `union_experiment/` path; the runner copies the summary there
when that directory exists or `--sandbox-dir` is set.

## Layer C — AgentFarm port (not in this runner)

1. Add `pair_commitment`, `fidelity`, `specialize` to
   `farm/core/hyperparameter_chromosome.py` (linear 8-bit, same table as
   `share_weight`).
2. `partner_id`, `pair_age`, `bond_strength`, `role` on agent state;
   `bond` / `leave` next to `share`.
3. Gather path: synergy only if partner alive, co-located, and
   `pair_age ≥ courtship_steps`.
4. Reproduction path: split cost + co-parent crossover only for mature bonds.
5. `social_dynamics.py`: `bond` as cooperation; log dissolution separately.
6. Runner: `UnionEmergenceExperiment`, or a `unique+union` arm on
   `intrinsic_goals_experiment.py`. Freeze learning genes; evolve union +
   share + goal loci.
7. CLI: `scripts/run_union_emergence.py` (`--num-steps`,
   `--selection-pressure`, `--exit-tax`, `--courtship-steps`,
   `--social-range`, `--bonding-cost`).
8. Artifacts: `union_emergence_summary.json`, gene-drift panel, synergy
   bar, paired-frac trajectory, extraction vs leave-rate.
9. Tests: bond is symmetric and charges `bonding_cost`; leave clears both
   sides and zeros strength; synergy is 1.0 during courtship and when
   either partner is dead; solo arm never sets `partner_id`.

## Invariants the standalone must already satisfy

These are the port tests, enforced on the arena first:

- Bond is symmetric and charges `bonding_cost` (split).
- Leave clears both sides and zeros `bond_strength`.
- Synergy is 1.0 during courtship and when either partner is dead.
- `solo_only` never sets `partner_id`.

## First-glance bar

A first-glance run is enough to say whether the *direction* of the
pre-register appears:

- optional/forced synergy > 1 and above promiscuous (no pairing → no synergy)
- forced extraction > optional extraction
- no-courtship synergy ≥ baseline optional synergy
- cheap-exit optional synergy ≤ baseline
- optional Δ commitment ≲ 0 on baseline

It is **not** enough to claim the published 1.30 / 1.36 / 1.20 point
estimates. Those need the full 4×550 v2 grid (and later the seed-matched
port).
