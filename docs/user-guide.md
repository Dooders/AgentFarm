## User guide

### Simulation platform (Python)

AgentFarm’s core product is the Python package under `farm/`. For install, CLI, API server, and benchmarks, use the [repository README](../README.md) and [CONTRIBUTING](../CONTRIBUTING.md). Automated tests run with `pytest` from the repo root after `pip install -r requirements.txt` and `pip install -e .`.

### Configuration Explorer (UI)

**In this repository today**, [`farm/editor/`](../farm/editor/) contains a **Jest** test suite for editor-related JavaScript (see `farm/editor/package.json`). There is no Vite dev server or Electron app checked in here yet.

The sections below describe the **intended** Configuration Explorer product (navigation, compare/diff, templates, IPC). That design is spelled out in:

- [Electron Config Explorer architecture](electron/config_explorer_architecture.md)
- [IPC API reference](ipc-api.md) (target main/preload contract)

When a full web or Electron app lands in-tree, its `package.json` will own scripts such as `dev`, `build`, and `electron:dev`; until then, treat the following as **product/spec** rather than runnable steps in this clone.

#### Intended install and launch (future)

- Development (web): install deps, run the dev server, open the local URL (see future app README).
- Development (Electron): Electron + renderer dev command (see future app README).
- Production: static build and/or `electron-builder` packaging as defined by that app.

#### Intended concepts and workflow

- Configuration Explorer: sections, validation, editing.
- Comparison mode: load another configuration, diff, selective apply.
- Templates: save, list, apply, delete presets.
- History: undo/redo and history save/load.

#### Intended keyboard shortcuts

- Open / Save / Save As, grayscale toggle, export shortcuts as implemented by the future UI.

#### Accessibility and troubleshooting

- Keyboard navigation and live regions as implemented by the future UI.
- IPC vs browser-only mode and `PERF_LOG=1` behavior as documented when the app exists.
