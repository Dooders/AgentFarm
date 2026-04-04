## IPC API reference

**Status:** This document specifies the **intended** Electron IPC surface for the planned Configuration Explorer (see [Electron Config Explorer architecture](electron/config_explorer_architecture.md)). A matching preload/main implementation is **not** checked into this repository yet; [`farm/editor/`](../farm/editor/) currently hosts Jest tests only.

When implemented, requests are expected to use `ipcRenderer.invoke(channel, payload)` with structured responses, and preload should expose `window.electronAPI` with methods that wrap allowed channels.

### General Envelope
- Success: `{ success: true, payload }`
- Error: `{ success: false, error: string }`

### Channels

Config
- `config:load` req `{ filePath?: string, format?: 'json'|'yaml'|'toml' }` → payload `{ config, metadata, filePath?, timestamp }`
- `config:save` req `{ config, filePath?, format?, backup?, ifMatchMtime? }` → payload `{ filePath, size, mtime, format, timestamp }`
- `config:export` req `{ config, format, filePath?, includeMetadata?, subsetPath?, paths? }` → payload `{ filePath, content, size, format, timestamp }`
- `config:import` req `{ filePath?, content?, format?, validate?, merge? }` → payload `{ config, metadata, source, timestamp }`
- `config:validate` req `{ config, partial?, rules? }` → payload `{ isValid, errors, warnings, config, validationTime, timestamp }`

Templates
- `config:template:load` req `{ templateName, category? }`
- `config:template:save` req `{ template, config, overwrite? }`
- `config:template:delete` req `{ templateName, category? }`
- `config:template:list` req `{ category?, includeSystem?, includeUser? }`

History
- `config:history:save` req `{ history, currentIndex, maxEntries? }`
- `config:history:load` req `{ maxEntries?, since?, filter? }`
- `config:history:clear` req `{}` → payload `{ success: true, timestamp }`

File System
- `fs:file:exists` req `{ filePath }` → payload `{ exists, isFile, isDirectory, size?, modified? }`
- `fs:file:read` req `{ filePath, encoding?, options? }` → payload `{ content, encoding, size, modified, mimeType }`
- `fs:file:write` req `{ filePath, content, encoding?, options? }` → payload `{ filePath, written, size, backupCreated?, backupPath? }`
- `fs:file:delete` req `{ filePath, backup? }` → payload `{ filePath, deleted, backupCreated?, backupPath? }`
- `fs:directory:read` req `{ dirPath, recursive?, filter? }` → payload `{ entries[], totalCount, fileCount, directoryCount }`
- `fs:directory:create` req `{ dirPath, recursive?, mode? }` → payload `{ dirPath, created, existing }`
- `fs:directory:delete` req `{ dirPath, recursive?, backup? }` → payload `{ dirPath, deleted, backupCreated?, backupPath? }`

Dialogs
- `dialog:open` req `{ ...options }` → Native file open dialog result
- `dialog:save` req `{ ...options }` → Native file save dialog result

App/System
- `app:ping` req `{}` → payload `{ success, timestamp }`
- `app:settings:get` req `{ keys?, category? }` → payload `{ settings, timestamp }`
- `app:settings:set` req `{ settings, category?, persist? }` → payload `{ success, updatedKeys[], timestamp }`
- `app:version:get` req `{}` → payload `{ version, build, platform, arch, timestamp }`
- `app:path:get` req `{ type }` → payload `{ path, type, timestamp }`
- `system:info:get` req `{}` → payload hardware and OS info

### Events (preload on/once allowlist)
- `config:loaded`, `config:saved`, `config:validation:complete`, `validation:error`
- `config:template:created`, `config:template:deleted`, `config:history:updated`
- `fs:operation:complete`

### Security
- Preload allowlists channels for `invoke`/`send` and for events
- Main restricts file access to: home, appData, userData, documents, desktop, downloads

