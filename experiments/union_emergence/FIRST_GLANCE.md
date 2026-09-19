# Union emergence — first-glance evidence

Standalone arena reconstructed from the
[PR 1014](https://github.com/Dooders/AgentFarm/pull/1014) pre-register
(`docs/research/devlog/2026-09-18-union-as-emergent-property.md`) and
`SCOPE.md`. The original `union_intrinsic_evolution.py` was not in the
PR; this is a from-spec v2 implementation, so the point estimates will
not match the post. The question is whether the **direction** of the
claim appears.

Command:

```bash
PYTHONHASHSEED=0 python experiments/union_emergence/union_intrinsic_evolution.py --mode first_glance
PYTHONHASHSEED=0 python experiments/union_emergence/union_intrinsic_evolution.py --mode v2
```

- Scout: 4 worlds × 4 arms × 2 seeds × 220 steps (`first_glance_summary.json`)
- Full compact grid: 4 worlds × 4 arms × 4 seeds × 550 steps
  (`compact_threshold_summary.json`) — this is the first-glance *of the
  experiment as specified*, not a substitute cell.

Population sits on the cap (`max_pop=80`) in every arm, matching the
post's "signal is node quality, not headcount."

## Headline (baseline world, v2 4×550)

| Arm | mean energy | synergy | paired frac | duration | Δ commit | Δ fidelity | extraction | lineages |
|---|---|---|---|---|---|---|---|---|
| solo_only | 15.0 | — | 0 | — | −0.022 | +0.007 | — | 29.8 |
| promiscuous | 17.3 | — | 0 | — | −0.009 | +0.011 | — | 29.5 |
| optional_union | **27.9** | **1.15 ± 0.06** | 0.88 | 23.5 | +0.007 | **+0.012** | 0.081 | **45.8** |
| forced_union | 18.7 | 2.58 ± 0.48 | 0.91 | 16.3 | +0.021 | +0.078 | 0.065 | 31.2 |

Approximate 95% CI on optional synergy (4 seeds, SEM from seed pstdev):
**1.09–1.21, excludes 1**.

Read:

1. **Promiscuous share is not a substitute.** Open `share_weight` lifts
   mean energy 15.0 → 17.3. Optional union lifts it to 27.9 — a
   population-level surplus share cannot fake.
2. **In-world synergy is superadditive.** Paired agents in the optional
   arm have ~15% more energy than unpaired agents in the *same* world.
3. **Commitment does not climb; fidelity does.** Song's constraint
   holds on baseline. Forced fidelity rises harder (+0.078): costly stay
   is what selection buys when pairing is imposed.
4. **Lineage diversity is higher under optional unions** (45.8 vs ~30).
5. **Forced pairing is a worse unit than optional pairing.** Forced
   synergy looks huge because the unpaired remainder is poor; population
   energy (18.7) is only a little above promiscuous and well below
   optional. Duration is shorter (16 vs 24).

The 2×220 scout already showed the same directions (optional synergy
1.17, energy 12.0 vs promiscuous 8.4). The 4×550 grid is the one to
cite.

## Threshold-grid directions (optional arm)

| Cell | synergy | duration | Δ commit | Δ fidelity | energy |
|---|---|---|---|---|---|
| baseline | **1.15** | 23.5 | +0.007 | **+0.012** | 27.9 |
| cheap exit + cheap bond | **1.03** | 24.9 | +0.007 | −0.001 | 36.4 |
| no courtship | 1.14 | 23.7 | −0.002 | 0.000 | 28.2 |
| wide neighborhood | 1.36 | 23.5 | 0.000 | +0.014 | 27.1 |

How the pre-register scored:

| Win condition | v2 outcome |
|---|---|
| Synergy CI excludes 1 on baseline optional/forced; promiscuous cannot match | **Holds** (energy 27.9 vs 17.3; synergy CI excludes 1) |
| Optional commitment does not climb; fidelity does when exit is costly | **Holds** |
| Ablating courtship inflates synergy | **Does not hold** (1.14 vs 1.15) |
| Cheap exit drops synergy and does not lengthen bonds | **Synergy drops** (1.15 → 1.03); duration is slightly *longer* (24.9 vs 23.5) |
| Forced extraction > optional extraction | **Does not hold** on the logged index (0.065 vs 0.081). Forced *is* worse on population energy and duration. |
| Wide range flattens fidelity / collapses commitment | Fidelity still rises; commitment stays flat. Synergy *increases*. Forced wide-range is pathological (synergy 5.45, energy 9.5, 19.5 lineages). |

6 / 9 automated direction checks pass. The three misses are the
courtship inflation, the Han extraction index, and cheap-exit duration.

## What this is and is not

This is first-glance evidence that **an exclusive unit can produce a
surplus promiscuous share cannot**, and that **selection keeps fidelity
rather than commitment** on the baseline cell.

It is **not** a replication of the posted 1.30 / 1.36 / 1.20 point
estimates. The original sandbox runner was never committed; this arena
implements the documented knobs (`bonding_cost`, `courtship_steps`,
`exit_tax`, `social_range`, leave formula, extraction index) on an
energy economy that had to be reconstructed. Courtship and extraction
need another pass before the literature-ablation claims are trusted.

Layer C (AgentFarm chromosome + `bond` / `leave` actions) is in
`scripts/run_union_emergence.py`. See `SCOPE.md` and
`experiments/union_emergence/layer_c/`.
