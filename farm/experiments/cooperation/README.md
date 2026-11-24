# `farm.experiments.cooperation`

Self-contained implementation of the registered experiment *Cooperative
Units: Which Institutional Settings Make Cooperation Persist and Pay?*, built
on [`farm.core.cooperation`](../../core/cooperation/__init__.py).
Design and parameters: [docs/research/experiments/cooperation/Design.md](../../../docs/research/experiments/cooperation/Design.md).
Model: [docs/concepts/cooperation.md](../../../docs/concepts/cooperation.md).

```bash
PYTHONHASHSEED=0 python scripts/run_cooperation.py                  # registered matrix, seeds 101-120
python scripts/run_cooperation.py --exploratory                      # plus all-one-motive populations
python scripts/run_cooperation.py --seeds 1 2 --output-dir /tmp/coop_pilot
python scripts/run_cooperation.py --analyze-only                     # re-analyse existing raw outputs
```

| Module | Responsibility |
|---|---|
| `config.py` | Registered constants, `WorldConfig`, `DecisionConfig`, `Condition`, `CONDITIONS`, `RunConfig`, diagnostic thresholds |
| `world.py` | `IncomeWorld`: per-agent shocks, droughts, crowding, costs |
| `agents.py` | `CoopAgent` and per-agent income streams keyed by seed and slot |
| `decisions.py` | Appraising prospective and current units, effort under pooling, noisy acceptance |
| `simulation.py` | `CooperationSimulation`: tick loop, exits, matching, arranged matches, births and families; solo twins |
| `metrics.py` | Kaplan-Meier, RMST, run summaries, founder pairs against the solo twin |
| `experiment.py` | `MatrixConfig`, `run_matrix`, `load_outputs` with resumable per-cell outputs |
| `analysis.py` | Registered seed-paired tests with Holm correction, manipulation checks, descriptive tables |
| `report.py` | `REPORT.md` and figures |

Outputs under the output directory: `manifest.json`, `runs.csv` (all cells),
`solo/` (twins), `cells/<condition>/` (`runs.csv`, `founders.csv.gz`,
`units.csv.gz`, `timeseries.csv.gz`), `analysis.json`, summary CSVs,
`REPORT.md` and `figures/`.
