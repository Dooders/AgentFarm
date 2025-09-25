# GitHub Issues: Settings UI/UX and Platform Improvements

## 1) Implement schema-driven settings UI

**Problem**
Our settings UI is hand-built and inconsistent. Validation and defaults aren’t centralized, making changes risky and error-prone.

**Proposal**
Adopt a schema-driven form approach (e.g., JSON Schema via react-jsonschema-form or JSON Forms) so settings structure, validation, and UI are generated from a single source of truth.

**Tasks**
- Choose schema approach (JSON Schema + AJV or Zod + adapter)
- Define schema for all current settings with titles, descriptions, defaults
- Implement form renderer and custom widgets where needed
- Add schema-driven validation and error messages
- Migrate existing settings to schema definitions

**Acceptance Criteria**
- All current settings render from a schema
- Invalid inputs show inline, accessible errors
- Defaults appear consistently and survive resets
- Adding a new setting requires only schema changes plus optional widget

**Labels**
area:settings, type:enhancement, priority:high

---

## 2) Integrate robust settings persistence with electron-store

**Problem**
Settings persistence lacks schema enforcement, migrations, and change events.

**Proposal**
Use electron-store with schema, defaults, migrations, and atomic writes. Broadcast changes to all windows.

**Tasks**
- Install and initialize electron-store with schema + defaults
- Implement safe migrations for existing users
- Wire change events to update UI and relevant modules
- Add debounced writes for high-frequency changes

**Acceptance Criteria**
- Store validates and rejects invalid writes
- Existing installs migrate safely without data loss
- UI reflects external changes instantly
- Writes are atomic and debounced where appropriate

**Labels**
area:settings, type:enhancement

---

## 3) Improve settings navigation and layout

**Problem**
Settings are hard to scan and navigate; related options aren’t grouped predictably.

**Proposal**
Use a left sidebar for categories, a top search field, and a right content panel with clear grouping and minimal nesting.

**Tasks**
- Define categories and grouping for all settings
- Implement left-nav category list
- Add right-pane section headers and minimal sub-tabs if needed
- Show “Requires restart” badges inline where applicable

**Acceptance Criteria**
- All settings belong to a clear category
- Navigation is keyboard-accessible and screen reader friendly
- “Requires restart” uses a consistent badge style

**Labels**
area:ux, area:settings, type:enhancement

---

## 4) Refine interactions and microcopy in settings

**Problem**
Interactions (save/reset/validation) are inconsistent; microcopy is sparse or unclear.

**Proposal**
Adopt auto-save with visible confirmation, per-field reset, section reset, and concise helper text under labels.

**Tasks**
- Auto-save on change with small inline “Saved” confirmation
- Add per-field “Reset to default” and section-level reset
- Add helper text for non-obvious controls; avoid tooltip-only explanations
- Standardize boolean/enums/numeric controls for consistency

**Acceptance Criteria**
- No blocking modals for typical edits
- All fields have reset behaviors if not default
- Helper text clarifies purpose and constraints
- Controls are consistent across the app

**Labels**
area:ux, area:settings, type:enhancement

---

## 5) Add power features: search, import/export, reset

**Problem**
Users can’t quickly find or migrate settings. No single place to reset safely.

**Proposal**
Add fuzzy search across labels/descriptions, JSON import/export with validation, and global reset with a preview.

**Tasks**
- Implement Fuse.js search across settings registry (labels, descriptions, keys)
- Highlight matches in results
- Import/export JSON with schema validation and safe merge preview
- Global reset flow with confirmation and changed-keys preview

**Acceptance Criteria**
- Search returns relevant results as you type
- Invalid imports are rejected with actionable errors
- Export produces valid, schema-compliant JSON
- Global reset shows exactly what will change

**Labels**
area:settings, type:feature

---

## 6) Performance and stability improvements for settings

**Problem**
Large panels and rapid changes can cause jank and unnecessary IPC traffic.

**Proposal**
Lazy-load category panels, virtualize heavy lists, debounce frequent writes, and batch IPC messages.

**Tasks**
- Code-split settings categories and load on demand
- Virtualize any long lists (e.g., advanced options) if present
- Debounce slider/typing updates before persisting
- Batch IPC updates to reduce message chatter

**Acceptance Criteria**
- Initial settings load is noticeably faster
- Scrolling and editing remain smooth under load
- IPC traffic is reduced during rapid changes

**Labels**
performance, area:settings

---

## 7) Harden security and Electron hygiene

**Problem**
Renderer may have excess privileges and IPC may be loosely validated.

**Proposal**
Enforce contextIsolation, disable nodeIntegration, expose a minimal typed preload API, validate all IPC, and apply a strict CSP.

**Tasks**
- Ensure BrowserWindow uses contextIsolation: true and nodeIntegration: false
- Implement typed, minimal preload bridge with capability-based API
- Validate and sanitize all IPC payloads with AJV/Zod
- Add strict Content Security Policy and safe shell.openExternal usage

**Acceptance Criteria**
- No direct Node.js access in renderer
- Only whitelisted APIs are exposed via preload
- Invalid IPC messages are rejected with errors
- CSP blocks inline/eval and unsafe sources

**Labels**
security, area:platform

---

## 8) Add automated testing for settings UI and behavior

**Problem**
No automated coverage for critical settings flows; regressions are likely.

**Proposal**
Use Playwright for end-to-end settings flows, Storybook for component isolation with visual regression, and unit tests for schema, migrations, and IPC.

**Tasks**
- Playwright: test navigation, edit, reset, import/export, restart-required labels
- Storybook: add stories for settings components and widgets
- Visual regression: baseline key stories
- Unit tests: schema validation, migrations, and IPC validators

**Acceptance Criteria**
- CI runs E2E and unit tests reliably
- Visual diffs catch unintended UI changes
- Migrations and validators have robust unit coverage

**Labels**
testing, area:settings, type:enhancement

