## Developer Guide

### Prerequisites
- Node 18+
- npm 9+

### Setup
```
npm install
npm run dev
```

Electron development:
```
npm run electron:dev
```

### Codebase Overview
- `src/` React UI, Zustand stores, services
- `electron/` main, preload, IPC handlers
- `docs/` documentation

Key modules:
- `src/services/ipcService.ts`: typed IPC client with metrics and error handling
- `electron/ipcHandlers.js`: main-process handlers with security checks
- `src/stores/configStore.ts`: domain state and actions, diff/merge utilities

### Scripts
- `npm run dev` web dev server
- `npm run electron:dev` Electron + Vite dev
- `npm run build` web build
- `npm run electron:pack` package Electron app
- `npm run typecheck`, `npm run lint`, `npm run test:run`, `npm run test:e2e`
- `npm run analyze` bundle visualization

### Testing
- Unit: Vitest (`vitest.config.ts`), jsdom environment, MSW for network
- E2E: Playwright (`playwright.config.ts`), spins Vite server automatically
- Electron IPC tests: Node env; CJS guarded to skip under pure ESM

### Electron Security
- `contextIsolation: true`, preload exposes minimal `window.electronAPI`
- Channel allowlist in preload for invoke/send and event listeners
- File-system access restricted to safe roots in main handlers

### Performance
- Vite code-splitting for vendor/ui/utils
- Perf logs via `PERF_LOG=1`
- RUM hooks (`rumMark`, `rumNav`) in `src/utils/perf.ts`

### Contributing
- Follow SRP, OCP, ISP, DIP, DRY, KISS, Composition Over Inheritance
- Add tests for new features and IPC endpoints
- Run `npm run typecheck && npm run test:run` before PR
