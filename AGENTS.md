# Agent instructions (AgentFarm)

This file is for **automated coding agents** working in this repository. Human contributors may prefer [CONTRIBUTING.md](CONTRIBUTING.md) and the [README](README.md).

## Project

**AgentFarm** is a Python-first simulation and analysis platform for agent-based modeling, reinforcement learning experiments, and related research (see README for feature overview). The installable package lives under `farm/`. Tests live under `tests/`. There is a small JavaScript test suite for the editor under `farm/editor/`.

**Note:** [docs/agents.md](docs/agents.md) describes a *research experiment* about system vs. individual agents in simulations. It is **not** this file and is unrelated to agent tooling.

## Environment setup

Use a **virtual environment** for development and tests.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

- **Python:** 3.8+ (3.9+ recommended; CI uses 3.10).
- **Redis:** Optional; used for some memory features (see README).
- **Editor UI:** From `farm/editor/`, use Node 20 and `npm ci` (or `npm install`) before running Jest.

## Commands

Run these from the repository root unless noted.

| Task | Command |
|------|---------|
| Default test suite | `pytest` |
| Tests with coverage (matches CI style) | `pytest -q --cov=farm --cov-report=term-missing` |
| Include slow or integration tests | `pytest -m ""` or `pytest -m integration` / `-m slow` as needed |
| Editor Jest tests | `cd farm/editor && npm test -- --runInBand` |

`pytest.ini` excludes `slow` and `integration` markers by default. Use `-m` to opt in when you need those suites.

**Linting:** Ruff and Pylint are configured in `pyproject.toml` (`[tool.ruff]`, `[tool.pylint.*]`). Run `ruff check .` and/or `pylint farm` if you change Python code.

## Code conventions

- **Style:** PEP 8; line length 120 for Ruff/Pylint where configured.
- **Design:** Prefer SOLID, DRY, KISS, and composition over inheritance when refactoring or adding features (see `.cursor/rules/design-principles.mdc` in this repo).
- **Tests:** Add or update tests under `tests/` for behavior changes. Use markers defined in `pytest.ini` (`unit`, `integration`, `slow`, `db`, `ml`, etc.) where appropriate.

## Layout (quick reference)

| Path | Role |
|------|------|
| `farm/` | Main Python package (simulation core, API, analysis, config, etc.) |
| `tests/` | Pytest suite |
| `farm/editor/` | Editor-related assets and Jest tests |
| `docs/` | Documentation and design notes |
| `benchmarks/` | Performance benchmarks |

When changing behavior, locate the closest existing patterns in `farm/` and mirror their structure, typing, and logging (`structlog` via `farm.utils` logging helpers where applicable).

## Cursor Cloud specific instructions

### Services overview

| Service | How to run | Notes |
|---------|-----------|-------|
| Simulation (CLI) | `source venv/bin/activate && python run_simulation.py --environment development --steps N` | Produces a `.db` file in `simulations/` |
| API server | `source venv/bin/activate && uvicorn farm.api.server:app --host 0.0.0.0 --port 5000` | Do **not** use `python -m farm.api.server` — the `reload=True` flag in `__main__` requires import-string mode and may crash. Use the `uvicorn` command directly. |
| Pytest | `source venv/bin/activate && pytest` | See `AGENTS.md > Commands` table for variants. |
| Jest (editor) | `cd farm/editor && npm test -- --runInBand` | Requires Node 20 (`nvm use 20`). |
| Lint | `source venv/bin/activate && ruff check .` | Ruff/Pylint may need to be installed separately if not in your env. |

### Gotchas

- **API server startup**: Always use `uvicorn farm.api.server:app` instead of `python -m farm.api.server`. The latter's `reload=True` flag causes `WARNING: You must pass the application as an import string to enable 'reload' or 'workers'` and silently exits.
- **Node version**: The editor tests require Node 20. Use `nvm use 20` (or your Node 20 install) before `npm test`.
- **Simulation determinism**: `run_simulation.py` automatically restarts itself with `PYTHONHASHSEED=0` if not set. This is normal behavior, not an error.
- **Pytest logging noise**: Some tests trigger `structlog` memory-monitor warnings on stderr (`CRITICAL: Memory usage at …`). This is cosmetic and does not indicate test failure — check the actual pass/fail summary line.
