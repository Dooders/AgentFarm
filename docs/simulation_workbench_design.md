## Simulation Workbench Design (Configure • Run • Analyze)

### Purpose
Design a full‑fledged desktop workbench that lets users configure simulations, run them with live feedback, and analyze/compare results. The workbench builds directly on the existing Config Explorer stack and patterns.

### Scope at a Glance
- Configure: Schema‑driven editor, presets, validation, diff/merge, YAML/JSON export
- Run: Launch, pause/resume/stop, progress streaming, logs, step playback, result persistence
- Analyze: On‑demand analysis modules, charts/dashboards, cross‑run comparison, export
- Works in Electron first, degrades to browser where possible

---

## Technology Stack (Same as Config Explorer)
- React 18 + TypeScript + Vite
- Zustand for state management
- Zod for client‑side validation (mirrors Python schema)
- Leva for compact controls and section folders
- styled‑components for theming; existing greyscale palette and a11y patterns
- Electron (main + preload + renderer IPC)
- Python backend: Flask + Flask‑SocketIO + the existing `farm` modules

Additional libraries
- socket.io‑client (renderer) for progress/events
- recharts or visx (charts; lightweight, typed)
- idb‑keyval (optional) for local result caching in browser mode

Environment
- `VITE_API_BASE_URL` is respected (already proxied in `vite.config.ts`)
- Electron preload exposes a minimal, typed API surface

---

## Product Goals & Primary Use Cases
- Create a simulation configuration from presets or files, edit with live validation
- Start a run and watch progress in real time; pause/resume/stop as needed
- Inspect step snapshots; scrub a timeline; view selected metrics as they stream
- Persist run artifacts and reopen them later from a Results Gallery
- Execute analysis modules on any completed run or a set of runs; view charts/tables
- Compare configurations and results across multiple runs side‑by‑side

---

## High‑Level Architecture

### Processes
- Electron Main (trusted): file dialogs, process orchestration, spawn Python, path management, safe persistence
- Renderer (UI): views, domain stores, IPC/HTTP clients, charts
- Python Server: REST + WebSocket API, backed by `SimulationController`, database, and analysis services

### Data Flow
1) Renderer edits a config (Zod) -> optional preview validation -> save to disk via IPC
2) Renderer requests "start run" -> Main forwards to Python API (or spawns controller locally)
3) Python emits progress via SocketIO; renderer updates `runStore`
4) Completed run persists DB and metadata; analysis modules operate against DB
5) Renderer requests analysis -> Python computes aggregates -> returns series/tables -> charts

---

## Directory Structure (Renderer)

```
src/
  app/
    routes.tsx               # top-level routing (Workspace, Configure, Run, Analyze)
    AppShell.tsx             # left-nav + status bar + toasts
  pages/
    Configure/
      ConfigurePage.tsx      # wraps existing Config Explorer + YAML preview
    Run/
      RunPage.tsx            # launch form + active run dashboard
    Analyze/
      AnalyzePage.tsx        # module picker + dashboards
    Results/
      ResultsGallery.tsx     # list of prior runs with search/filter
  components/
    run/RunToolbar.tsx
    run/ProgressMeter.tsx
    run/TimelineScrubber.tsx
    run/LiveMetricsPanel.tsx
    analyze/AnalysisCard.tsx
    analyze/ComparisonMatrix.tsx
  stores/
    configStore.ts           # existing + file ops
    runStore.ts              # active jobs + status
    resultsStore.ts          # persisted runs & metadata
    analysisStore.ts         # module selection & outputs
    uiStore.ts               # theme, layout, toasts
  services/
    ipcService.ts            # existing
    api/httpClient.ts        # fetch/axios; respects proxy
    api/socketClient.ts      # socket.io-client wrapper
    resultsService.ts        # load/save run metadata, caches
    chartService.ts          # transform series → chart data
  types/
    config.ts                # shared SimulationConfigType
    run.ts                   # SimulationRun, JobStatus, Progress
    analysis.ts              # AnalysisRequest/Response types
```

Electron (`electron/`)
- `main.js` (existing) extended with: run process lifecycle, safe paths, single‑instance lock, protocol handling
- `preload.js/ts`: typed API: dialogs, paths, app info, safe fs helpers
- `ipcHandlers.js`: config, presets, history, results (extend existing handlers referenced in docs)

Python (`farm/api/server.py` and controllers)
- Extend REST + SocketIO to expose job controls and analysis registry

---

## Domain Models (Renderer‑side Types)

```ts
export type JobStatus = 'queued' | 'starting' | 'running' | 'paused' | 'completed' | 'stopped' | 'error'

export interface SimulationRun {
  id: string
  name: string
  createdAt: string
  status: JobStatus
  totalSteps: number
  currentStep: number
  configPath?: string
  dbPath?: string
  tags?: string[]
  metrics?: Record<string, number>
}

export interface ProgressEvent {
  runId: string
  step: number
  totalSteps: number
  percent: number
  message?: string
}

export interface AnalysisModuleMeta {
  name: string
  title: string
  description?: string
  inputs: Record<string, unknown>
}
```

---

## API Contracts

### REST (HTTP)
- POST `/api/simulation/new` → `{ sim_id }` (accepts partial config overrides)
- GET `/api/simulation/:sim_id/step/:step` → step snapshot
- GET `/api/simulation/:sim_id/analysis` → default analysis summary
- GET `/api/simulations` → active run map
- GET `/api/simulation/:sim_id/export` → CSV path
- GET `/api/config/schema` → combined schema from `farm.core.config_schema.generate_combined_config_schema()`
- GET `/api/analysis/modules` → `AnalysisModuleMeta[]`
- POST `/api/analysis/:module` → run module; returns output path, counts, or series

Browser mode uses these HTTP endpoints directly. Electron mode may alternatively execute local controllers for improved performance; both expose the same shapes to the renderer.

### WebSocket (SocketIO)
- `simulation_progress` → `ProgressEvent`
- `subscription_success` / `subscription_error`

### Electron Preload (Typed)
- `openConfigDialog(): Promise<string | undefined>`
- `saveConfigDialog(suggested?: string): Promise<string | undefined>`
- `readFile(path): Promise<string` + `writeFile(path, data): Promise<void>`
- `paths(): { appData: string; userData: string; temp: string }`

---

## Key UI Flows

### Configure
- Leverage existing `ConfigExplorer` for sectioned editing
- YAML/JSON preview and export; import from file
- Presets: list/apply/undo; diff to current; store in `electron-store`
- Validation: Zod on change; server‑side validation endpoint for final checks

### Run
- "New Run" form: select config (path or in‑memory), steps, run name/tags
- Start → creates run card with live progress, ETA, logs
- Controls: pause, resume, stop; on error, show diagnostics with action to open logs
- Timeline: scrub steps (lazy load snapshots via `/step/:step`), auto‑play preview
- Live metrics: agent count, rewards, resources, FPS; streamed in aggregate buckets

### Analyze
- Pick a completed run (or multi‑select)
- Module picker (from `/api/analysis/modules`)
- Run module → results cached and rendered as charts/tables
- Comparison Matrix: select up to N runs, pick metrics → side‑by‑side cards
- Export: CSV/PNG for charts; link to output directory

### Results Gallery
- Persisted list of runs with search/filter by tag/date/status
- Clicking a run opens a details drawer: config hash, parameters, quick charts, actions (analyze, export, open folder)

## Components to Build

### MVP components (build first)
- Configure
  - ConfigExplorerPage: wraps existing `ConfigExplorer` with file open/save and YAML/JSON preview
  - PresetManagerModal: list/apply/undo presets
  - SaveBar: dirty-state banner with Save/Discard
- Run
  - RunPage: layout shell for an active run
  - RunToolbar: New, Start, Pause/Resume, Stop, step count, run name/tags
  - ProgressMeter: percent, ETA, status chip
  - TimelineScrubber: step slider with play/pause; requests `/step/:step`
  - LiveMetricsPanel: small charts for agents/resources/reward (socket stream)
  - LogPanel: recent messages/errors
- Analyze
  - AnalyzePage: module picker + results area
  - AnalysisModulePicker: fetch list, configure params
  - AnalysisCard: renders a module result (table/chart) with Export
- Results
  - ResultsGallery: grid/list of past runs (search, filter)
  - RunCard: status, tags, quick metrics; open details
  - RunDetailsDrawer: config summary and actions (Analyze, Export, Open folder)

### Phase 2 (nice-to-haves)
- ComparisonMatrix (runs × metrics)
- MetricSelector and ChartLegend controls
- ResourceUsagePanel (CPU/RAM, optional)
- DiffInspector (current vs preset/file)
- EmptyState, ErrorBoundary, Toast/Notifications, ConfirmDialog

### Notes
- Reuse existing components: `ConfigExplorer`, `LeftPanel`, `ComparisonPanel`, validation components, `ipcService`
- Provide a single reusable `TimeSeriesChart` used by `LiveMetricsPanel` and `AnalysisCard`

---

## State Management (Zustand Stores)

### `configStore`
- Source of truth for current config, compare config, diff, dirty state
- File operations via Electron IPC with browser fallbacks

### `runStore`
- Holds active `SimulationRun` map and selected run id
- Actions: `startRun`, `pauseRun`, `resumeRun`, `stopRun`, `subscribeToProgress`
- Persists minimal run metadata to `resultsStore` upon completion

### `resultsStore`
- Indexed by run id; contains metadata, derived metrics, thumbnail sparkline data
- Backed by disk (Electron) or IndexedDB (browser)

### `analysisStore`
- Module catalog and cached outputs keyed by `(module, runId, paramsHash)`

Selectors keep components lean; expensive derivations are memoized.

---

## Backend Extensions (Python)

### REST Additions
- `/api/config/schema` → return JSON produced by `generate_combined_config_schema()`
- `/api/analysis/modules` → inspect `farm/analysis` registry and expose metadata

### Controller Integration
- `SimulationController` already supports initialize/start/pause/stop/step/get_state/cleanup
- Add a small manager that maps `sim_id` → controller instance; exposes pause/resume/stop endpoints and emits `simulation_progress`

### Streaming & Aggregation
- For live charts, compute rolling aggregates server‑side:
  - downsample step metrics to 10–20 Hz; transmit deltas only
  - endpoints for windowed series: `/api/run/:id/metrics?from=t0&to=t1&every=100ms`

---

## Theming, Accessibility, and UX
- Use the existing greyscale system and typography (Albertus for labels, JetBrains Mono for numbers)
- All controls maintain the 28px height and monochrome focus rings
- Keyboard navigation: roving tabindex in lists, TimelineScrubber responds to Arrow/Home/End
- Live regions for run status updates and errors

---

## Testing Strategy
- Unit: stores, services, validation utils (Vitest)
- Component: key views (RunPage, AnalyzePage) with MSW mocks
- E2E: critical flows with Playwright (start run, receive progress, complete, analyze)
- Contract tests: validate that Python API responses conform to TS types (schema tests)

---

## Performance & Reliability
- Budget: primary interactions under 100ms; charts render under 200ms for 5k points
- Use windowed rendering and virtualization for long lists and galleries
- Socket reconnect with exponential backoff; offline cues with retry actions
- Large series: pre‑aggregate in Python; lazy request further detail on zoom

---

## Security
- Electron: `contextIsolation: true`, no `nodeIntegration`, strict preload surface
- Validate all IPC inputs; never trust renderer
- Sanitize any file paths; write only to app‑scoped directories by default

---

## Rollout Plan

### Phase 1 — Run Skeleton (2 weeks)
- Add `runStore`, `RunPage`, socket client, basic progress card
- Expose `/api/config/schema`, wire config → start run
- Persist completed runs to `resultsStore`

### Phase 2 — Live Inspection (2–3 weeks)
- Timeline scrubbing + step snapshots
- Live metrics panel with rolling aggregates
- Pause/resume/stop controls wired to controller manager

### Phase 3 — Analysis & Gallery (3 weeks)
- Results Gallery with search/filter
- Analysis modules UI; charts and exports; comparison matrix

### Phase 4 — Polish & Electron Harden (2 weeks)
- A11y sweep, dark/contrast modes, error boundary UX
- Preload hardening, typed IPC, packaging and auto‑updates

---

## Acceptance Criteria
- Start a run from an edited config; see progress within 1s and control execution
- Scrub steps for an active or completed run; load snapshots on demand
- View at least three live metrics while running and a summary on completion
- Run at least three analysis modules and render charts; export CSV/PNG
- Persist completed runs and reopen them from a gallery; compare at least two runs
- All flows usable via keyboard with clear focus indications and ARIA labels
- Browser mode functional for configure/analyze with remote API (no OS dialogs)

---

## Open/Closed Extension Points
- New analysis modules appear automatically via `/api/analysis/modules`
- New metrics can be added server‑side; the renderer reads advertised series and renders cards declaratively
- New run types (e.g., batch/experiment sweeps) plug into `runStore` via a `Runner` interface without changing existing callers

---

## Risks & Mitigations
- Schema drift between Python and Zod → single source of truth endpoint `/api/config/schema`; nightly contract test
- Large DBs/series stall UI → server‑side aggregation + windowing + virtualization
- Electron security regressions → context isolation, typed preload, IPC validation

---

## Implementation Notes (Mapping to Current Codebase)
- Reuse `src/components/ConfigExplorer/*` for Configure page and diff UX
- Use `farm/api/server.py` for start/step/analysis; extend with schema and module listing
- Build on `farm/controllers/simulation_controller.py` for fine‑grained controls
- Generate schemas using `farm/core/config_schema.py` and hydrate Zod client‑side
- Respect existing `vite.config.ts` proxy and `VITE_API_BASE_URL`

This document defines the target product and technical shape for the unified Simulation Workbench using the established stack and patterns already present in the repository.