## Deployment guide

### Python simulation / API

Deploy the Python package and API according to your environment (venv, container, or process manager). Typical steps:

- Install with `pip install -r requirements.txt` and `pip install -e .` (or install a built wheel in production).
- Run simulations via `run_simulation.py`, `farm.core.cli`, or your own entrypoints.
- Run the HTTP API with `python -m farm.api.server` (see [README](../README.md) for defaults and endpoints).

There is no single Dockerfile or cloud manifest maintained in this repository; treat deployment as environment-specific.

### Configuration Explorer (web / Electron)

**Not applicable in-tree today:** the full Vite/Electron Configuration Explorer app is not present under `farm/editor/` (only Jest tests exist). When a deployable UI is added, expect something like:

- Static hosting of a `dist/` web build, and/or
- Packaged Electron artifacts from `electron-builder` (or similar),

documented in that application’s own README. Design targets are described in [Electron Config Explorer architecture](electron/config_explorer_architecture.md).

### Environment variables (target UI)

When a Config Explorer frontend exists, it may use variables such as `IS_ELECTRON`, `PERF_LOG`, and optional RUM/error endpoints. See the app’s documentation when available.

### CI

GitHub Actions workflows live under [`.github/workflows/`](../.github/workflows/). [`tests.yml`](../.github/workflows/tests.yml) runs Python tests (`pytest` with coverage) and `farm/editor` Jest tests (Node 20). There is no `ci.yml` workflow in this repository.
