# Development setup

Guide for contributors working on the Python simulation core and the editor JavaScript tests.

## Python simulation (`farm/`)

Most development happens in the Python package under `farm/`.

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

See [CONTRIBUTING](../../CONTRIBUTING.md) and [AGENTS](../../AGENTS.md) for branching, linting, and layout.

- **Getting started:** [Installation](../getting-started/installation.md) · [First simulation](../getting-started/first-simulation.md)
- **CI:** [`.github/workflows/tests.yml`](../../.github/workflows/tests.yml) (job `python`, Python 3.10)

## Editor / JavaScript (`farm/editor/`)

[`farm/editor/`](../../farm/editor/) currently contains a **Jest** suite only (`npm test`). There is no Vite dev server or Electron app checked in yet.

```bash
cd farm/editor
npm ci   # or npm install
npm test -- --runInBand
```

CI runs the same under the `js-ui` job (Node 20).

## Planned Configuration Explorer

The sections below describe the **intended** Configuration Explorer product. These are design specs, not runnable steps in this repository today.

- [Electron Config Explorer architecture](../reference/electron/config_explorer_architecture.md)
- [IPC API reference](../reference/ipc-api.md) — target main/preload contract

When a full web or Electron app lands in-tree, its `package.json` will own scripts such as `dev`, `build`, and `electron:dev`.

### Intended workflow (future UI)

- Configuration Explorer: sections, validation, editing
- Comparison mode: load another configuration, diff, selective apply
- Templates: save, list, apply, delete presets
- History: undo/redo and history save/load

## Contributing

- **Python:** follow [CONTRIBUTING](../../CONTRIBUTING.md); run `pytest` and `ruff check .`.
- **Editor JS:** add or update Jest tests under `farm/editor/` and keep them passing under Node 20.
