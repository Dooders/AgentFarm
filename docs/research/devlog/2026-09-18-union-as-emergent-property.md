---
layout: page
title: "The union as an emergent property"
subtitle: "Exclusive pair-bonds vs two non-committed agents — pilot, literature constraints, and a compact threshold grid."
date: 2026-09-18
related:
  - experiments/intrinsic_evolution/intrinsic_goals.md
  - experiments/intrinsic_evolution/intrinsic_evolution.md
  - devlog/2026-07-29-selection-pressure-and-intrinsic-goals.md
  - farm/core/social_dynamics.py
---

# The union as an emergent property

**Status:** standalone pilot + compact threshold grid complete. AgentFarm port wired and first-glanced — see [the 19 September Layer C post](2026-09-19-union-emergence-layer-c.md).

**Question (v1).** Does an exclusive, costly-to-exit union produce superadditive returns that promiscuous `share` does not?

**Question (v2, after Song / Reynolds / PNAS 2023 / Leimar / Han).** Under what bonding cost, divorce cost, courtship delay, and neighborhood size does an exclusive unit appear that promiscuous share cannot fake — and is that unit symbiotic or extractive?

This is the marriage claim in AgentFarm language: two non-committed agents optimize locally; a locked pair is a new unit with its own persistence. The union is the object of selection, not just the two genomes.

## Why this is not the existing share machinery

AgentFarm already evolves cooperation as a *transaction*:

| Existing | Missing |
|---|---|
| `share_weight`, `share_mult_poor` / `share_mult_wealthy` | exclusive `partner_id` + accumulated `bond_strength` |
| `reward_share_bonus` | stay cost / exit tax / bonding cost |
| `cooperation_threshold` | complementary roles after courtship |
| one-shot `share` / `assist` / `defend` in `social_dynamics.py` | split `offspring_cost` + co-parent crossover *conditional on a mature bond* |

Promiscuous share is the two-agent market. A union changes the payoff matrix: shared horizon, costly exit, specialization, and a joint reproductive budget.

## Related experiments (what not to reinvent)

Closest published cousins — cite these in the catalog entry:

| Paper | What they already showed | Constraint on this design |
|---|---|---|
| Song, Feldman & Gavrilets 2013, *J. Evol. Biol.* | Pair-bond preference coevolves with parental-care cooperation only below a bonding-cost threshold; accidental divorce sets the ESS; balancing selection, not fixation | Do not treat rising `pair_commitment` as success. Sweep bonding cost and accident rate. |
| Reynolds 2018, *J. Hum. Evol.* | Semipermanent breeding bonds emerge from gene-based forage gifting in constrained hominin-like ABMs; disappear when the gift gene is off or *N* is unconstrained | Keep a gift/share-off control. Keep `max_pop`. |
| Akçay et al. 2023, *PNAS* | Emotional bookkeeping + courtship stabilizes intra-pair IPD when divorce is costly and deceit is expensive | Courtship gate before synergy / split-cost. Sweep `exit_tax`. |
| Leimar & McNamara 2024, *PNAS* | Helping evolves via accumulating bond strength; reciprocity is loose; small neighborhoods required | `bond_strength` not a binary lock. Keep `social_range` small; wide range is an ablation. |
| Ogbo, Elragig & Han 2022, *Adaptive Behavior* | Prior commitment coordinates only when the deal specifies the split of asymmetric gains | Log extraction = energy gap × commitment gap. Forced ≠ chosen. |

Nowak’s five rules, tag models, and LLM-IPD papers vary a share-or-defect knob. They do not grow a `partner_id`. That is the hole.

## Design (current standalone + AgentFarm port)

### New genes

| Gene | Range | Role |
|---|---|---|
| `pair_commitment` | [0, 1] | willingness to *start* a lock |
| `fidelity` | [0, 1] | leave resistance; scales exit tax and grief |
| `specialize` | [0, 1] | gather/guard complementarity once bonded |

Reuse: `share_weight`, `reward_share_bonus`, `reward_reproduce_bonus`, existing co-parent crossover.

Commitment starts the bond. Strength is the unit. Fidelity is what selection is allowed to keep.

### Mechanics (v2 — literature knobs)

1. **`bond` / `leave`.** Assortative on commitment. Form with $p = c_i c_j$ (optional) or a high fixed rate (forced). Pay `bonding_cost` split; refuse if either cannot cover it.
2. **Courtship.** No gather synergy and no split offspring cost until `pair_age ≥ courtship_steps`.
3. **Strength.** Each co-located step: `strength += bond_gain * min(c) − bond_decay`, clipped to [0, 1]. Synergy = $1 + k \cdot \mathrm{strength} \cdot \mathrm{mean}(s)$.
4. **While bonded.** Co-location pull; loose equalization toward the poorer partner (Leimar, not TFT). Promiscuous share suppressed.
5. **Exit.** Accidental divorce at `accident_divorce` (Song noise), then leave with $(1-f)\cdot(0.04+\mathrm{stress})\cdot(1-0.5\cdot\mathrm{strength})$. Exit tax $\propto f$. Death applies grief $\propto f$.
6. **No external fitness.** Survival and reproduction only.

Default world (frozen as the port baseline): `bonding_cost=0.8`, `courtship_steps=12`, `exit_tax=1.1`, `social_range=3.2`, `accident_divorce=0.01`, `max_pop=80`.

### Arms

| Arm | Pairing | Share |
|---|---|---|
| `solo_only` | clamped off | residual only |
| `promiscuous` | off | open `share_weight` |
| `optional_union` | evolvable commitment | pair-internal after courtship |
| `forced_union` | high pairing rate | same |
| *(port)* `unique+union` | fourth arm on `IntrinsicGoalsExperiment` | goals heterogeneous |

### Metrics and pre-register (v2)

Primary:

- `synergy_index = E[energy | paired] / E[energy | solo]` in the **same** world, **after** courtship
- `mean_pair_duration`, `mean_bond_strength`
- `fidelity` Δ (not commitment Δ)
- `mean_extraction = E[ |E_i−E_j| · |c_i−c_j| ]` among live pairs
- `lineages_alive`

Secondary: `pair_commitment` Δ (expected ≤ 0 under optional pairing), action mix.

Win conditions:

1. Synergy CI excludes 1 on baseline optional/forced; promiscuous cannot match it.
2. Optional commitment does **not** climb; fidelity does when `exit_tax` is above threshold.
3. Ablating courtship inflates synergy (part of the effect was free).
4. Cheap exit and wide `social_range` shrink duration and flatten fidelity Δ.
5. Forced extraction > optional extraction.

Falsifiers: synergy CI includes 1 after the AgentFarm port; `reward_share_bonus` alone matches union synergy; fidelity falls in the costly-exit forced arm; optional `paired_frac → 0` at medium/high selection pressure.

## Pilot results (v1 arena, 8 seeds × 900 steps)

Instant synergy, no bonding cost, binary lock. Population cap bound every arm (~65–68). Signal is node quality, not headcount.

| Arm | synergy | paired frac | pair duration | lineages alive | notable gene Δ |
|---|---|---|---|---|---|
| solo_only | — | 0 | — | 5.3 | attack +0.15, share −0.03 |
| promiscuous | — | 0 | — | 5.5 | share −0.10; lowest mean energy (3.19) |
| optional_union | **1.43 ± 0.19** | 0.55 | 15 steps | 6.9 | commitment −0.12, fidelity ~0 |
| forced_union | **1.48 ± 0.04** | 0.90 | 38 steps | 8.1 | fidelity **+0.05**, commitment −0.07 |

Read: union is superadditive in-world; open share is not a substitute; cheap pairing + cheap exit → churn; costly stay is what selection buys; lineage diversity is higher under unions.

## Historical threshold notes from the unavailable sandbox runner

The table below records the earlier sandbox notes that motivated the reconstruction. It is **not** generated by the committed standalone runner and does **not** match the reproducible `compact_threshold_summary.json`; for the checked-in v2 results, use `experiments/union_emergence/FIRST_GLANCE.md` and the JSON artifact instead.

| Cell | optional synergy | optional duration | Δ commitment | Δ fidelity | extraction |
|---|---|---|---|---|---|
| **baseline** (cost 0.8, courtship 12, exit 1.1, range 3.2) | **1.30** | 16 | ~0 | **+0.07** | 0.08 |
| cheap exit + cheap bond | 1.20 | 14 | ~0 | +0.06 | 0.08 |
| no courtship | **1.36** | 15 | **−0.08** | +0.09 | 0.07 |
| wide neighborhood (range 8) | 1.37 | 14 | **−0.12** | ~0 | 0.06 |

Forced arm: duration ~21–23, extraction 0.12–0.17 (about 2× optional).

How the papers showed up:

- Song — optional commitment stays flat on baseline; falls when pairing is too easy.
- PNAS courtship — delay off inflates synergy 1.30 → 1.36. Part of the old 1.43 was the multiplier firing on day one.
- PNAS costly divorce — cheap exit drops optional synergy 1.30 → 1.20 and does not lengthen bonds.
- Leimar — wide range collapses commitment (−0.12) and stops fidelity from rising.
- Han — imposed pairs are more extractive than chosen ones.

Default port cell is **baseline**. First three AgentFarm ablations: courtship ∈ {0, 12}, exit_tax ∈ {0.2, 1.1, 2.2}, social_range ∈ {2.0, 3.2, 8.0}.

## Implementation sketch on AgentFarm

1. Add the three genes to `farm/core/hyperparameter_chromosome.py` (linear 8-bit, same table as `share_weight`).
2. `partner_id`, `pair_age`, `bond_strength`, `role` on agent state; `bond` / `leave` next to `share`.
3. Gather path: synergy only if partner alive, co-located, and `pair_age ≥ courtship_steps`.
4. Reproduction path: split cost + co-parent crossover only for mature bonds.
5. `social_dynamics.py`: `bond` as cooperation; log dissolution rate separately.
6. Runner: `UnionEmergenceExperiment`, or a `unique+union` arm on `intrinsic_goals_experiment.py`. Freeze learning genes; evolve union + share + goal loci.
7. CLI: `scripts/run_union_emergence.py` (`--num-steps`, `--selection-pressure`, `--exit-tax`, `--courtship-steps`, `--social-range`, `--bonding-cost`).
8. Artifacts: `union_emergence_summary.json`, gene-drift panel, synergy bar, paired-frac trajectory, extraction vs leave-rate.
9. Tests: bond is symmetric and charges `bonding_cost`; leave clears both sides and zeros strength; synergy is 1.0 during courtship and when either partner is dead; solo arm never sets `partner_id`.

Do not add yet: institutional monogamy norms (Bauch), LLM courtship talk, lifting `max_pop`, tit-for-tat share.

## Placement in the catalog

Under **Emergent behavior & dominance** in `docs/research/experiments-catalog.md`, after One of a Kind / Cooperation, with a pointer from Intrinsic Goals: goals can be social; a union is a heritable *structure*, not just a reward weight.

Drop-in path: `docs/research/devlog/2026-09-18-union-as-emergent-property.md` plus an index bullet.

Layer C first-glance (100 steps × 2 seeds × 6 cells × 4 arms) is written up in
[The union on the chromosome](2026-09-19-union-emergence-layer-c.md).
Optional synergy 1.12; forced 0.90; 5 of 8 direction checks. Layer D:

- Seed-matched AgentFarm A/B on the baseline cell (this pre-register).
- Ablate synergy multiplier vs split-cost vs exit tax (which term carries the 1.12×).
- Heterogeneous goals × union: does `reward_*` distance predict `leave`?

## Files

- `experiments/union_emergence/union_intrinsic_evolution.py` — v2 arena
- `experiments/union_emergence/compact_threshold_summary.json` — v2 grid
- `experiments/union_emergence/layer_c/` — port first-glance
- [Layer C outcome](2026-09-19-union-emergence-layer-c.md)

Through-line: **sharing is a transaction; a union is an agent.** Selection keeps the second only when leaving costs more than staying, the neighborhood is small enough to stop shopping, and the surplus split is not silent extraction.
