## User Guide

This guide walks you through installing, launching, and using the Config Explorer.

### Install & Launch
- Development (web): `npm install` then `npm run dev` and open `http://localhost:3000`
- Development (Electron): `npm run electron:dev`
- Production build: `npm run build` (web) or `npm run electron:pack` (Electron)

### Main Concepts
- Configuration Explorer: Navigate sections, edit values, and validate.
- Comparison Mode: Load another configuration to diff and selectively apply.
- Templates: Save, list, apply, and delete reusable presets.
- History: Edits are tracked; undo/redo and history save/load supported.

### Workflow
1. Open a configuration (File > Open or toolbar)
2. Edit fields in the left panel; validation hints appear inline
3. Save (Ctrl/Cmd+S) or Save As to a new file
4. Use Compare to load another config and apply differences
5. Export to JSON/YAML/TOML

### Keyboard Shortcuts
- Open: Ctrl/Cmd+O
- Save: Ctrl/Cmd+S
- Save As: Ctrl/Cmd+Shift+S
- Toggle Grayscale UI: Ctrl/Cmd+G
- Export YAML: Ctrl/Cmd+Y

### Accessibility
- Keyboard navigable controls with visible focus
- Live regions announce key actions

### Troubleshooting
- If IPC is unavailable, the app runs in browser mode with limited features
- Check developer console for `[perf]` logs when `PERF_LOG=1`
