const { contextBridge, ipcRenderer } = require('electron')

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Config file operations
  loadConfig: (filePath) => ipcRenderer.invoke('config:load', filePath),
  saveConfig: (config, filePath) => ipcRenderer.invoke('config:save', config, filePath),
  exportConfig: (config, format) => ipcRenderer.invoke('config:export', config, format),

  // Dialog operations
  showOpenDialog: (options) => ipcRenderer.invoke('dialog:open', options),
  showSaveDialog: (options) => ipcRenderer.invoke('dialog:save', options),

  // Platform information
  platform: process.platform,

  // IPC event listeners
  on: (channel, callback) => {
    // Whitelist of allowed channels
    const validChannels = [
      'config:loaded',
      'config:saved',
      'validation:error',
      'app:version'
    ]

    if (validChannels.includes(channel)) {
      ipcRenderer.on(channel, callback)
      return () => ipcRenderer.removeListener(channel, callback)
    }
  },

  // IPC one-time listeners
  once: (channel, callback) => {
    const validChannels = ['app:ready']

    if (validChannels.includes(channel)) {
      ipcRenderer.once(channel, callback)
    }
  }
})