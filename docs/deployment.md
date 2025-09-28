## Deployment Guide

### Web (Vite)
- Build: `npm run build`
- Serve `dist/` with any static server

### Electron
- Dev: `npm run electron:dev`
- Package (dev build): `npm run electron:pack:dev`
- Package (prod build): `npm run electron:pack`
- Distribute artifacts from `dist-electron/` or builder output; use `electron-builder` config in `electron-builder.json`.

### Environment Variables
- `IS_ELECTRON=true` to enable Electron-specific behavior at runtime
- `PERF_LOG=1` to enable performance logs in console
- `VITE_RUM_ENDPOINT` POST endpoint for RUM
- `VITE_ERROR_ENDPOINT` POST endpoint for error reporting

### CI
- GitHub Actions workflow `.github/workflows/ci.yml` runs typecheck, lints, unit, e2e, build

### Code Signing/Notarization (macOS)
- Configure Apple ID and certificates for `electron-builder` if publishing for macOS