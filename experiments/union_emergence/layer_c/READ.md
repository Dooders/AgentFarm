# Layer C first-glance read

100 steps × 2 seeds × 6 cells × 4 arms on the AgentFarm chromosome
(`scripts/run_union_emergence.py --mode first_glance`). Table and figures
sit next to this file.

## Direction

Optional synergy is **1.12** on baseline (standalone first-glance was 1.15).
Courtship off inflates it to **1.36**; cheap exit drops it to **0.91**.
Commitment stays flat (Δ ≈ +0.003). Fidelity Δ is +0.001 — the `> 0`
check passes but is noise at this horizon.

Solo never sets `partner_id`. Optional paired-frac settles near 0.77;
forced near 0.90.

## Misses (same two as the standalone arena)

- Forced baseline synergy is 0.90, not > 1.
- Forced extraction (0.26) is below optional (0.39). Han's index did not
  replicate.
- Promiscuous energy (36.5) slightly beats optional (35.6) in this rich
  24×24 world. Share is still a transfer: it has no synergy index because
  there is no pair.

Wide-range forced synergy 7.39 is leftover-solo inflation, not a 7×
gather multiplier. Read it next to mean energy 34.4.

Not enough horizon or seeds to claim the published 1.30 / 1.36 / 1.20
point estimates. Layer D is seed-matched A/B plus ablating synergy vs
split-cost vs exit tax.
