## Developer guide

### Python simulation (`farm/`)

Most development happens in the Python package. Use a virtual environment, `pip install -r requirements.txt`, `pip install -e .`, and `pytest` from the repository root. See [CONTRIBUTING](../CONTRIBUTING.md) and [AGENTS](../AGENTS.md) for commands, layout, and linting.

CI for Python is defined in [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) (job `python`, Python 3.10).

### Editor / JavaScript (`farm/editor/`)

**Current state:** [`farm/editor/package.json`](../farm/editor/package.json) defines a **Jest** suite only (`npm test`). There is no `dev`, `build`, or Electron tooling in that package today.

From `farm/editor/`:

```bash
npm ci   # or npm install
npm test -- --runInBand
```

CI runs the same under the `js-ui` job in `.github/workflows/tests.yml` (Node 20).

### Planned Configuration Explorer (reference)

Design notes for a future Vite + React (+ optional Electron) Config Explorer live in [Electron Config Explorer architecture](electron/config_explorer_architecture.md). The [IPC API reference](ipc-api.md) describes the target preload/main channel contract. Use those documents when implementing or reviewing UI work; they are not guaranteed to match any checked-in `src/` tree until that application is added.

### Contributing

- Python: follow [CONTRIBUTING](../CONTRIBUTING.md); run `pytest` and Ruff/Pylint as appropriate.
- Editor JS: add or update Jest tests under `farm/editor/` and keep them passing under Node 20.
