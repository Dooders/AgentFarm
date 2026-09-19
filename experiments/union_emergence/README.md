# Union emergence

Standalone arena (layers A/B) plus the AgentFarm chromosome port (layer C).

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

Layer C (AgentFarm chromosome + `bond` / `leave`) lives in
`farm/runners/union_emergence_experiment.py`:

```bash
PYTHONHASHSEED=0 python scripts/run_union_emergence.py --mode first_glance
```

Writes `experiments/union_emergence/layer_c/`. First-glance read:
[`layer_c/READ.md`](layer_c/READ.md). Outcome post:
[`docs/research/devlog/2026-09-19-union-emergence-layer-c.md`](../../docs/research/devlog/2026-09-19-union-emergence-layer-c.md).
Full protocol: [`SCOPE.md`](SCOPE.md).
