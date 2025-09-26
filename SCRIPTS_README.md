# Available Scripts

## `dev`
```bash
vite
```

## `dev:electron`
```bash
cross-env IS_ELECTRON=true vite
```

## `build`
```bash
tsc && vite build
```

## `build:dev`
```bash
tsc && cross-env NODE_ENV=development vite build
```

## `build:prod`
```bash
tsc && cross-env NODE_ENV=production vite build
```

## `build:analyze`
```bash
tsc && vite build --mode analyze
```

## `lint`
```bash
eslint src --ext ts,tsx --report-unused-disable-directives --max-warnings 0
```

## `lint:fix`
```bash
eslint src --ext ts,tsx --fix
```

## `lint:electron`
```bash
eslint electron --ext js,ts
```

## `format`
```bash
prettier --write src
```

## `format:check`
```bash
prettier --check src
```

## `format:electron`
```bash
prettier --write electron
```

## `preview`
```bash
vite preview
```

## `preview:electron`
```bash
wait-on tcp:4173 && cross-env IS_ELECTRON=true electron electron/main.js
```

## `electron`
```bash
wait-on tcp:3000 && cross-env IS_ELECTRON=true electron electron/main.js
```

## `electron:dev`
```bash
concurrently "npm run dev:electron" "npm run electron"
```

## `electron:pack`
```bash
npm run build:prod && electron-builder
```

## `electron:pack:dev`
```bash
npm run build:dev && electron-builder
```

## `electron:dist`
```bash
electron-builder --publish=never
```

## `electron:dist:all`
```bash
electron-builder --publish=never --mac --win --linux
```

## `typecheck`
```bash
tsc --noEmit
```

## `typecheck:watch`
```bash
tsc --noEmit --watch
```

## `test`
```bash
vitest
```

## `test:ui`
```bash
vitest --ui
```

## `test:run`
```bash
vitest run
```

## `test:coverage`
```bash
vitest run --coverage
```

## `test:jest`
```bash
jest
```

## `test:jest:watch`
```bash
jest --watch
```

## `test:jest:coverage`
```bash
jest --coverage
```

## `test:all`
```bash
npm run test:run && npm run test:jest
```

## `test:coverage:all`
```bash
npm run test:coverage && npm run test:jest:coverage
```

## `docs`
```bash
npm run build && npm run typecheck && npm run test:run
```

## `docs:serve`
```bash
npm run docs && npx http-server dist -p 3000
```

## `clean`
```bash
rm -rf dist dist-electron node_modules/.vite
```

## `clean:all`
```bash
rm -rf dist dist-electron node_modules/.vite node_modules
```

## `install:clean`
```bash
rm -rf node_modules package-lock.json && npm install
```

## `prebuild`
```bash
npm run clean && npm run typecheck
```

## `version`
```bash
npm run docs && git add .
```

## `analyze`
```bash
npm run build:analyze && npx vite-bundle-analyzer dist/stats.html
```

## `docker:build`
```bash
docker build -t live-simulation-config-explorer .
```

## `docker:run`
```bash
docker run -p 3000:3000 live-simulation-config-explorer
```

