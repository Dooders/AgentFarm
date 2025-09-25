## Electron Config Explorer Architecture

This document defines a modular architecture for the Electron-based Configuration Explorer that will replace and extend the current sidebar in `farm/editor`. It establishes clear module boundaries, IPC contracts between the main and renderer processes, and a phased migration plan that minimizes disruption.

### Goals
- Provide a structured, SRP-aligned architecture for configuration exploration and editing
- Enable safe, validated editing of simulation configuration files and presets
- Decouple UI concerns from file system and process orchestration via IPC
- Support future extension (OCP) without modifying core modules

### Design Principles
- Single Responsibility: each module has one reason to change (view, state, service, bridge)
- Open-Closed: extend via new views, commands, and IPC routes rather than modifying existing logic
- Interface Segregation: small, purpose-built interfaces for services and IPC contracts
- Dependency Inversion: UI depends on abstractions (services) not concrete Node/Electron APIs
- Composition over inheritance for UI components and services

---

## High-Level Architecture

The Config Explorer spans both Electron processes:

- Main Process (trusted):
  - File system, process management, validation/preview orchestration
  - Persistent config store and preset registry
  - IPC handlers (request/response; event publishers)

- Renderer (UI):
  - Views: tree, details, diff, validation panel, search
  - State: normalized config tree, selection, dirty state, validation diagnostics
  - Services: typed IPC client, schema helpers, diff/merge helpers

### Proposed Directory Structure

Renderer (under `farm/editor`):

```
farm/editor/
  components/
    config-explorer/
      TreeView.js
      DetailsPanel.js
      DiffView.js
      ValidationPanel.js
      SearchBar.js
    shared/
      SplitPane.js
      Toolbar.js
  state/
    explorerStore.js
    selectors.js
    actions.js
  services/
    ipcClient.js
    schemaService.js
    diffService.js
```

Main process (under `farm/editor` or `farm/editor/main`):

```
farm/editor/
  main.js                       # app bootstrap
  main/
    ipcRoutes.js                # registers all routes
    configStore.js              # loads/saves configs, presets
    fileSystemService.js        # file IO, dialogs
    validationService.js        # JSON schema validation (AJV)
    previewService.js           # dry-run/preview integration to Python
```

Note: Files listed are architectural targets; implement incrementally during migration.

---

## Module Responsibilities (Renderer)

- TreeView: presents hierarchical configuration (files, sections, keys); supports search/filter
- DetailsPanel: form editor for selected node with schema-driven widgets and validation hints
- DiffView: side-by-side diff between working changes and on-disk or between presets
- ValidationPanel: lists diagnostics with jump-to capabilities
- explorerStore: single source of truth for selection, data, dirty state, diagnostics
- ipcClient: typed wrapper around `ipcRenderer` providing request/response and event streams
- schemaService: resolves JSON schemas, defaults, and field metadata
- diffService: computes diffs, applies patches, resolves conflicts

## Module Responsibilities (Main)

- ipcRoutes: binds channel names to handlers, validates payloads, and returns typed responses
- configStore: manages in-memory cache of configs; snapshot/rollback; preset registry
- fileSystemService: reads/writes YAML/JSON/TOML; shows open/save dialogs; watches files
- validationService: runs schema validation; returns structured diagnostics
- previewService: coordinates with Python backend to dry-run or simulate validation-only steps

---

## IPC Contracts

All requests are request/response with `id`, `ok`, `error?`, and `data?`. Renderer subscribes to event channels for async updates (file watch, preview progress).

Channel names are namespaced as `config:*` and `explorer:*`.

- config:listRoots
  - req: `{}`
  - res: `{ roots: Array<{id, label, path, type}> }`

- config:listTree
  - req: `{ rootId: string, expand?: string[] }`
  - res: `{ nodes: Array<{id, parentId, key, path, kind, hasChildren}> }`

- config:get
  - req: `{ nodePath: string }`
  - res: `{ value: unknown, schema?: object, source: 'file'|'preset' }`

- config:update
  - req: `{ nodePath: string, patch: { op: 'replace'|'add'|'remove', path: string, value?: unknown }[] }`
  - res: `{ value: unknown, diagnostics: Diagnostic[] }`

- config:validate
  - req: `{ rootId: string }`
  - res: `{ diagnostics: Diagnostic[] }`

- config:save
  - req: `{ rootId: string }`
  - res: `{ saved: boolean, path: string }`

- config:presets:list
  - req: `{ scope?: 'global'|'project' }`
  - res: `{ presets: Array<{id, name, description}> }`

- config:presets:apply
  - req: `{ presetId: string }`
  - res: `{ success: boolean, appliedPaths: string[] }`

- explorer:search
  - req: `{ query: string, scope: 'keys'|'values'|'both' }`
  - res: `{ matches: Array<{nodePath, excerpt}> }`

- explorer:watch (event)
  - payload: `{ path: string, type: 'changed'|'deleted'|'created' }`

- preview:run
  - req: `{ rootId: string, steps?: number }`
  - res: `{ runId: string }`

- preview:progress (event)
  - payload: `{ runId: string, percent: number, message?: string }`

### Native File Dialogs (Renderer-triggered)

- dialog:openConfig
  - req: none
  - res: `{ canceled: boolean, filePath?: string }`

- dialog:saveConfig
  - req: `{ suggestedPath?: string }`
  - res: `{ canceled: boolean, filePath?: string }`

Diagnostic shape:

Diagnostic shape:
  - `nodePath`: string
  - `level`: 'error' | 'warning' | 'info'
  - `code?`: string (optional)
  - `message`: string

Validation rules are enforced on the main side (AJV or equivalent) so the renderer remains untrusted. Renderer performs optimistic UI validation only for UX hints.

---

## State Management (Renderer)

- Store shape:
  - `tree`: normalized nodes by id
  - `selection`: current nodePath
  - `workingValues`: edited values keyed by nodePath
  - `dirty`: boolean or set of dirty nodePaths
  - `diagnostics`: Diagnostic[] keyed by nodePath
  - `presets`: available presets and metadata
  - `search`: query and results

Selectors compute derived views (e.g., visible nodes) and keep components simple (KISS, SRP).

---

## Electron Process Concerns

- Context Isolation: enable `contextIsolation: true` and expose a minimal, typed preload API
- No direct FS access in renderer; all IO via IPC
- Validate every IPC request; never trust renderer input
- Long-running tasks (preview) are cancellable and emit progress events
- Watchers are debounced and coalesced to avoid floods

Note: Until the preload boundary is introduced, a temporary renderer service (`window.dialogService`) wraps `ipcRenderer.invoke` calls for `dialog:openConfig` and `dialog:saveConfig`. This will be migrated behind a typed preload API in a future phase.

---

## Migration Plan (Phased)

Phase 1: Architecture & Skeleton
- Add `ipcRoutes`, `configStore`, `fileSystemService` in main; `ipcClient` in renderer
- Implement read-only features: `config:listRoots`, `config:listTree`, `config:get`
- Introduce `ConfigExplorer` panel behind a feature flag while keeping existing sidebar
 - Add native file dialogs for open/save and basic toolbar controls (Open, Save, Save As)

Phase 2: Editing & Validation
- Implement `config:update`, `config:validate`, `config:save`
- Add `DetailsPanel` with schema-driven forms and `ValidationPanel`
- Add `DiffView` and dirty state tracking

Phase 3: Presets, Search, and Preview
- Implement presets list/apply and global/project scopes
- Add `explorer:search` and file watchers for external edits
- Implement `preview:run` with progress events to the Python backend
- Remove legacy sidebar configuration controls once feature is stable

Rollback Strategy
- Keep legacy sidebar behind a toggle until Phase 3 completes
- Maintain snapshot/rollback in `configStore` to discard unsafe changes

---

## Risks & Mitigations
- Risk: Renderer accessing Node APIs directly
  - Mitigation: Context isolation + preload, no `nodeIntegration`
- Risk: Schema drift with Python backend
  - Mitigation: Single schema source of truth; versioned schemas; contract tests
- Risk: Large configs cause UI lag
  - Mitigation: Virtualized tree, incremental loading, memoized selectors

---

## Acceptance Criteria Mapping
- Architecture document (this file) checked into repository
- Covers module responsibilities, SRP, Electron process concerns
- Defines IPC message types and contracts
- Provides a phased migration plan from current sidebar to config explorer
- Renderer implements multi-config workflows: open primary and compare configurations simultaneously
- Visual diffing available:
  - Form-based: field-level diff highlighting and one-click "Copy from compare"
  - YAML-based: side-by-side grid listing key paths and values (current vs compare)
- Preset bundles supported: apply preset (deep merge) and undo last applied preset
- Validation and unsaved state clearly indicated in UI during edits/merges

---

## Notes on Current Codebase

Today, the Electron editor (`farm/editor`) directly loads UI with Node integration and talks to a Python backend via HTTP and Socket.IO. This proposal introduces typed IPC boundaries, moves file IO and validation to the main process, and gradually replaces the sidebar controls with a dedicated configuration explorer.

