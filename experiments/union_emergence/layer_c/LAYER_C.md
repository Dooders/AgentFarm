# Union emergence Layer C

Outcome post: [`docs/research/devlog/2026-09-19-union-emergence-layer-c.md`](../../../docs/research/devlog/2026-09-19-union-emergence-layer-c.md).

| cell | arm | synergy | energy | paired | Δ commit | Δ fidelity | extraction | pop |
|---|---|---|---|---|---|---|---|---|
| baseline | solo_only | — | 35.4 | 0.000 | 0.001 | -0.000 | — | 80.0 |
| baseline | promiscuous | — | 36.5 | 0.000 | 0.001 | -0.001 | — | 80.0 |
| baseline | optional_union | 1.123 | 35.6 | 0.774 | 0.003 | 0.001 | 0.386 | 80.0 |
| baseline | forced_union | 0.897 | 34.2 | 0.905 | 0.000 | -0.001 | 0.263 | 80.0 |
| no_courtship | solo_only | — | 35.4 | 0.000 | 0.001 | -0.000 | — | 80.0 |
| no_courtship | promiscuous | — | 36.5 | 0.000 | 0.001 | -0.001 | — | 80.0 |
| no_courtship | optional_union | 1.358 | 33.7 | 0.775 | 0.000 | 0.002 | 0.278 | 79.5 |
| no_courtship | forced_union | 1.044 | 33.8 | 0.882 | 0.003 | 0.001 | 0.247 | 80.0 |
| cheap_exit | solo_only | — | 35.4 | 0.000 | 0.001 | -0.000 | — | 80.0 |
| cheap_exit | promiscuous | — | 36.5 | 0.000 | 0.001 | -0.001 | — | 80.0 |
| cheap_exit | optional_union | 0.905 | 36.2 | 0.785 | 0.002 | 0.001 | 0.214 | 80.0 |
| cheap_exit | forced_union | 1.138 | 34.1 | 0.897 | 0.006 | -0.003 | 0.448 | 80.0 |
| costly_exit | solo_only | — | 35.4 | 0.000 | 0.001 | -0.000 | — | 80.0 |
| costly_exit | promiscuous | — | 36.5 | 0.000 | 0.001 | -0.001 | — | 80.0 |
| costly_exit | optional_union | 1.316 | 33.7 | 0.766 | 0.006 | -0.001 | 0.230 | 80.0 |
| costly_exit | forced_union | 1.511 | 34.8 | 0.862 | 0.006 | -0.002 | 0.301 | 79.0 |
| tight_range | solo_only | — | 35.4 | 0.000 | 0.001 | -0.000 | — | 80.0 |
| tight_range | promiscuous | — | 36.5 | 0.000 | 0.001 | -0.001 | — | 80.0 |
| tight_range | optional_union | 1.159 | 35.7 | 0.786 | 0.003 | -0.002 | 0.438 | 80.0 |
| tight_range | forced_union | 1.651 | 33.8 | 0.869 | 0.003 | 0.002 | 0.097 | 80.0 |
| wide_range | solo_only | — | 35.4 | 0.000 | 0.001 | -0.000 | — | 80.0 |
| wide_range | promiscuous | — | 36.5 | 0.000 | 0.001 | -0.001 | — | 80.0 |
| wide_range | optional_union | 1.382 | 32.8 | 0.840 | 0.009 | -0.001 | 0.350 | 79.5 |
| wide_range | forced_union | 7.394 | 34.4 | 0.909 | 0.002 | -0.000 | 0.118 | 80.0 |

Win-condition checks: 5/8.
- PASS: baseline_optional_synergy_gt_1
- FAIL: baseline_forced_synergy_gt_1
- FAIL: optional_energy_beats_promiscuous
- PASS: optional_commitment_does_not_climb
- PASS: optional_fidelity_rises_on_baseline
- PASS: cheap_exit_drops_optional_synergy
- PASS: no_courtship_inflates_optional_synergy
- FAIL: forced_extraction_gt_optional

## Takeaway

First-glance Layer C (100 steps × 2 seeds × 6 cells × 4 arms) on the
AgentFarm chromosome. Direction matches the standalone arena on the
claims that actually replicated there:

- Optional synergy is 1.12 on baseline (standalone first-glance was 1.15).
  Courtship off inflates it to 1.36; cheap exit drops it to 0.91.
- Commitment stays flat (Δ ≈ +0.003). Fidelity Δ is +0.001 — the check
  passes the `> 0` gate but is noise at this horizon.
- Solo never pairs. Optional paired-frac settles near 0.77; forced near 0.90.
- Promiscuous energy (36.5) slightly beats optional (35.6) in this rich
  24×24 economy. Share remains a transfer; it does not produce a synergy
  index because there is no `partner_id`.
- Forced baseline synergy is 0.90 and forced extraction (0.26) is below
  optional (0.39). Same two misses as the standalone first-glance: Han
  extraction and forced superadditivity did not replicate.
- Wide-range forced synergy 7.39 is leftover-solo inflation (a few poor
  unpaired agents), not a 7× gather multiplier. Read it next to mean energy
  34.4.

Not enough horizon or seeds to claim the published 1.30 / 1.36 / 1.20
point estimates. Layer D is seed-matched A/B on the baseline cell plus
ablating synergy vs split-cost vs exit tax.
