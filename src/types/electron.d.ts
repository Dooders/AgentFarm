// Global window augmentations for Electron APIs
// Keep global declarations in a single ambient types file for organization

export {}

declare global {
  interface Window {
    electronAPI?: {
      // Config operations
      loadConfig: (requestOrPath?: string | { filePath?: string; format?: string }) => Promise<any>
      saveConfig: (
        requestOrConfig:
          | { config: any; filePath?: string; format?: string; backup?: boolean; ifMatchMtime?: number }
          | any,
        maybeFilePath?: string
      ) => Promise<any>
      exportConfig: (
        requestOrConfig:
          | { config: any; format: 'json' | 'yaml' | 'xml'; filePath?: string; includeMetadata?: boolean; subsetPath?: string; paths?: string[] }
          | any,
        maybeFormat?: 'json' | 'yaml' | 'xml'
      ) => Promise<any>
      importConfig: (request: { filePath?: string; content?: string; format?: string; validate?: boolean; merge?: boolean }) => Promise<any>
      validateConfig: (request: { config: any; partial?: boolean; rules?: string[]; context?: Record<string, any> }) => Promise<any>

      // Template operations
      loadTemplate: (request: { templateName: string; category?: string }) => Promise<any>
      saveTemplate: (request: { template: any; config: any; overwrite?: boolean }) => Promise<any>
      deleteTemplate: (request: { templateName: string; category?: string }) => Promise<any>
      listTemplates: (request: { category?: string; includeSystem?: boolean; includeUser?: boolean }) => Promise<any>

      // History operations
      saveHistory: (request: any) => Promise<any>
      loadHistory: (request: any) => Promise<any>
      clearHistory: () => Promise<any>

      // File system operations
      fileExists: (request: { filePath: string }) => Promise<any>
      readFile: (request: { filePath: string; encoding?: string; options?: any }) => Promise<any>
      writeFile: (request: { filePath: string; content: string | any; encoding?: string; options?: any }) => Promise<any>
      deleteFile: (request: { filePath: string; backup?: boolean }) => Promise<any>
      readDirectory: (request: { dirPath: string; recursive?: boolean; filter?: any }) => Promise<any>
      createDirectory: (request: { dirPath: string; recursive?: boolean; mode?: number }) => Promise<any>
      deleteDirectory: (request: { dirPath: string; recursive?: boolean; backup?: boolean }) => Promise<any>

      // Dialog operations
      showOpenDialog: (options?: any) => Promise<any>
      showSaveDialog: (options?: any) => Promise<any>

      // Application operations
      getSettings: (request: { keys?: string[]; category?: string }) => Promise<any>
      setSettings: (request: { settings: Record<string, any>; category?: string; persist?: boolean }) => Promise<any>
      getAppVersion: () => Promise<any>
      getAppPath: (request: { type: string }) => Promise<any>
      getSystemInfo: () => Promise<any>

      // Platform information
      platform: string

      // Event listeners
      on: (channel: string, callback: (payload: any) => void) => void
      once: (channel: string, callback: (payload: any) => void) => void

      // Generic IPC
      invoke: (channel: string, payload?: any) => Promise<any>
      send: (channel: string, payload?: any) => void
      removeListener: (channel: string, listener: (payload: any) => void) => void
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

