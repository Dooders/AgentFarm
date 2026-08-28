# Notarizing an AgentFarm run

Stamp a finished experiment directory so config + artifact hashes can be anchored. Simulation stays local.

## Prerequisites

```bash
# FarmNotary as a sibling clone (current workflow)
git clone https://github.com/Dooders/FarmNotary.git ../FarmNotary
pip install -e ../FarmNotary
```

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

## What must not be in the run dir you stamp

Do not leave voter-level or agent-level choice files next to the official outputs unless they use a skipped name fragment (`ballot`, `vote`, …). Prefer writing those under a `private/` subfolder that is not hashed.

## Related

- [Design RFC](../design/onchain_run_provenance.md)
- [FarmNotary](https://github.com/Dooders/FarmNotary)
- [Deterministic simulations](deterministic-simulations.md)
