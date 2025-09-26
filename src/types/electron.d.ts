// Global window augmentations for Electron APIs
// Keep global declarations in a single ambient types file for organization

export {}

declare global {
  interface Window {
    electronAPI?: {
      loadConfig: (filePath?: string) => Promise<any>
      saveConfig: (config: any, filePath?: string) => Promise<any>
      exportConfig: (config: any, format: string) => Promise<any>
      showOpenDialog: (options?: any) => Promise<any>
      showSaveDialog: (options?: any) => Promise<any>
      platform: string
      on: (channel: string, callback: Function) => void
      once: (channel: string, callback: Function) => void
      invoke: (channel: string, payload?: any) => Promise<any>
      send: (channel: string, payload?: any) => Promise<any>
      removeListener: (channel: string, listener: Function) => void
      removeAllListeners: (channel?: string) => void
    }

    electron?: {
      openFileDialog: (options: {
        mode: 'file' | 'directory' | 'save'
        filters?: { name: string; extensions: string[] }[]
      }) => Promise<{ filePaths: string[] }>
      fileExists: (path: string, isDirectory?: boolean) => Promise<boolean>
    }
  }
}

