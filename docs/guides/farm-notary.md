# Notarizing an AgentFarm run

Stamp a finished experiment directory so config + artifact hashes can be anchored. Simulation stays local.

## Prerequisites

```bash
pip install -e ".[notary]"
```

For FarmNotary development, a sibling clone still works: `pip install -e ../FarmNotary`.

## After a run

```bash
python scripts/notarize_run.py --run-dir experiments/<name>/results --runner experiment_runner
```

Writes `manifest.json` into that directory (via FarmNotary). `anchor` is dry-run until a chain backend is configured in FarmNotary.

```bash
python -m farm_notary.cli verify --run-dir experiments/<name>/results
python -m farm_notary.cli anchor --run-dir experiments/<name>/results
```

## From Python

```python
from farm.provenance.notary import notarize_run_dir

receipt = notarize_run_dir("experiments/demo/results", runner="demo")
```

The adapter stamps only the official allowlist (`trials.csv`, `summary.csv`,
`allocation_means.csv`, `contrasts.csv`, `run_config.json`, `REPORT.md`,
`figures/*.png`). FarmNotary also refuses name fragments `ballot`, `vote`,
`voter`, `individual_choice`, and `private`.

## What must not be in the run dir you stamp

Do not leave voter-level or agent-level choice files next to the official outputs.
Prefer writing those under a `private/` subfolder; they are never hashed.

## Related

- [Design RFC](../design/onchain_run_provenance.md)
- [FarmNotary](https://github.com/Dooders/FarmNotary)
- [Deterministic simulations](deterministic-simulations.md)
