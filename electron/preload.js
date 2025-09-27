const { contextBridge, ipcRenderer } = require('electron')

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Config file operations
  loadConfig: (requestOrPath) => {
    const payload = typeof requestOrPath === 'string' || requestOrPath === undefined
      ? { filePath: requestOrPath }
      : requestOrPath
    return ipcRenderer.invoke('config:load', payload)
  },
  saveConfig: (requestOrConfig, maybeFilePath) => {
    const looksLikeRequest = requestOrConfig && typeof requestOrConfig === 'object' && !Array.isArray(requestOrConfig) && (
      Object.prototype.hasOwnProperty.call(requestOrConfig, 'filePath') ||
      Object.prototype.hasOwnProperty.call(requestOrConfig, 'format') ||
      (requestOrConfig.config && typeof requestOrConfig.config === 'object')
    )
    const payload = looksLikeRequest
      ? requestOrConfig
      : { config: requestOrConfig, filePath: maybeFilePath }
    return ipcRenderer.invoke('config:save', payload)
  },
  exportConfig: (requestOrConfig, maybeFormat) => {
    const looksLikeRequest = requestOrConfig && typeof requestOrConfig === 'object' && !Array.isArray(requestOrConfig) && (
      Object.prototype.hasOwnProperty.call(requestOrConfig, 'filePath') ||
      Object.prototype.hasOwnProperty.call(requestOrConfig, 'format') ||
      (requestOrConfig.config && typeof requestOrConfig.config === 'object')
    )
    const payload = looksLikeRequest
      ? requestOrConfig
      : { config: requestOrConfig, format: maybeFormat }
    return ipcRenderer.invoke('config:export', payload)
  },
  importConfig: (request) => ipcRenderer.invoke('config:import', request),
  validateConfig: (request) => ipcRenderer.invoke('config:validate', request),

  // Template operations
  loadTemplate: (request) => ipcRenderer.invoke('config:template:load', request),
  saveTemplate: (request) => ipcRenderer.invoke('config:template:save', request),
  deleteTemplate: (request) => ipcRenderer.invoke('config:template:delete', request),
  listTemplates: (request) => ipcRenderer.invoke('config:template:list', request),

  // History operations
  saveHistory: (request) => ipcRenderer.invoke('config:history:save', request),
  loadHistory: (request) => ipcRenderer.invoke('config:history:load', request),
  clearHistory: () => ipcRenderer.invoke('config:history:clear'),

  // File system operations
  fileExists: (request) => ipcRenderer.invoke('fs:file:exists', request),
  readFile: (request) => ipcRenderer.invoke('fs:file:read', request),
  writeFile: (request) => ipcRenderer.invoke('fs:file:write', request),
  deleteFile: (request) => ipcRenderer.invoke('fs:file:delete', request),
  readDirectory: (request) => ipcRenderer.invoke('fs:directory:read', request),
  createDirectory: (request) => ipcRenderer.invoke('fs:directory:create', request),
  deleteDirectory: (request) => ipcRenderer.invoke('fs:directory:delete', request),

  // Dialog operations
  showOpenDialog: (options) => ipcRenderer.invoke('dialog:open', options),
  showSaveDialog: (options) => ipcRenderer.invoke('dialog:save', options),

  // Application operations
  getSettings: (request) => ipcRenderer.invoke('app:settings:get', request),
  setSettings: (request) => ipcRenderer.invoke('app:settings:set', request),
  getAppVersion: () => ipcRenderer.invoke('app:version:get'),
  getAppPath: (request) => ipcRenderer.invoke('app:path:get', request),
  getSystemInfo: () => ipcRenderer.invoke('system:info:get', {}),

  // Platform information
  platform: process.platform,

  // IPC event listeners
  on: (channel, callback) => {
    // Whitelist of allowed channels
    const validChannels = [
      'config:loaded',
      'config:saved',
      'validation:error',
      'app:version',
      'config:validation:complete',
      'config:template:created',
      'config:template:deleted',
      'config:history:updated',
      'fs:operation:complete'
    ]

    if (validChannels.includes(channel)) {
      ipcRenderer.on(channel, callback)
      return () => ipcRenderer.removeListener(channel, callback)
    }
  },

  // IPC one-time listeners
  once: (channel, callback) => {
    const validChannels = ['app:ready', 'app:ping']

    if (validChannels.includes(channel)) {
      ipcRenderer.once(channel, callback)
    }
  },

  // Generic IPC methods
  invoke: (channel, payload) => ipcRenderer.invoke(channel, payload),
  send: (channel, payload) => ipcRenderer.send(channel, payload),
  removeListener: (channel, listener) => ipcRenderer.removeListener(channel, listener),
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
})