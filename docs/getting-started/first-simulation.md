# First simulation

## Command line

```bash
python run_simulation.py --environment development --steps 1000
```

Results are written under `simulations/` (database files, logs, and analysis output).

## Experiments and analysis tools

```bash
# Parameter sweeps
python farm/core/cli.py --mode experiment --environment development --experiment-name test --iterations 3

# Visualize existing results
python farm/core/cli.py --mode visualize --db-path simulations/simulation.db

# Generate analysis reports
python farm/core/cli.py --mode analyze --db-path simulations/simulation.db
```

See [Experiment quickstart](experiments-quickstart.md) for parameter studies.

## API server

```bash
uvicorn farm.api.server:app --host 0.0.0.0 --port 5000
```

Use the `uvicorn` command directly. Running `python -m farm.api.server` enables `reload=True`, which requires an import string and exits with a warning.

Defaults:

- Port 5000
- Structured logs in `logs/application.json.log` and `logs/application.log`

Key endpoints:

- `POST /api/simulation/new` — create and run a simulation
- `GET /api/simulation/<sim_id>/step/<step>` — fetch a step
- `GET /api/simulation/<sim_id>/analysis` — run analysis
- `GET /api/simulation/<sim_id>/export` — export data
- `WS /ws/<client_id>` — WebSocket client channel

See [Deployment](../guides/deployment.md) for production notes.

## Benchmarks

```bash
python -m benchmarks.run_benchmarks --list
python -m benchmarks.run_benchmarks --spec benchmarks/specs/memory_db_baseline.yaml
```

See [benchmarks/README.md](../../benchmarks/README.md).

## Testing

```bash
pytest -q
# or
python run_tests.py
```
