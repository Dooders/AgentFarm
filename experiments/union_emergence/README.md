# Union emergence (standalone arena)

Pilot + compact threshold grid for exclusive pair-bonds under implicit
selection. Not yet wired into `farm.runners`.

Full protocol: [`SCOPE.md`](SCOPE.md). Pre-register:
[`docs/research/devlog/2026-09-18-union-as-emergent-property.md`](../../docs/research/devlog/2026-09-18-union-as-emergent-property.md).

```bash
# Default = v2 compact threshold grid (4 worlds × 4 arms × 4 seeds × 550)
python experiments/union_emergence/union_intrinsic_evolution.py

# First glance: every cell, 2 seeds × 220 steps
python experiments/union_emergence/union_intrinsic_evolution.py --mode first_glance

# v1 binary-lock pilot (8 seeds × 900)
python experiments/union_emergence/union_intrinsic_evolution.py --mode v1
```

Writes JSON/markdown summaries next to this README
(`compact_threshold_summary.json`, `first_glance_summary.json`, or
`v1_pilot_summary.json`). Pass `--sandbox-dir union_experiment` to also
copy the summary to the original sandbox path.

First-glance read of the v2 grid: [`FIRST_GLANCE.md`](FIRST_GLANCE.md).

The three new genes (`pair_commitment`, `fidelity`, `specialize`) and the
`bond` / `leave` actions are **not** on the AgentFarm chromosome yet.
That is layer C in `SCOPE.md`.
