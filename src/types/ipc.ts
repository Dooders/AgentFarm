import { SimulationConfigType, ConfigImportResult, ConfigExport, ConfigurationMetadata, ConfigTemplate, ConfigPath } from './config'
import { ValidationResult } from './validation'

// =====================================================
// IPC Channel Types
// =====================================================

// Base IPC channel interface
export interface IPCChannel {
  name: string
  request: any
  response: any
}

// IPC event types
export type IPCEventType =
  | 'config:load'
  | 'config:save'
  | 'config:export'
  | 'config:import'
  | 'config:validate'
  | 'config:template:load'
  | 'config:template:save'
  | 'config:template:delete'
  | 'config:template:list'
  | 'config:history:load'
  | 'config:history:save'
  | 'config:history:clear'
  | 'fs:file:exists'
  | 'fs:file:read'
  | 'fs:file:write'
  | 'fs:file:delete'
  | 'fs:directory:read'
  | 'fs:directory:create'
  | 'fs:directory:delete'
  | 'app:settings:get'
  | 'app:settings:set'
  | 'app:version:get'
  | 'app:path:get'
  | 'system:info:get'

// IPC request/response types
export interface IPCRequest {
  id: string
  type: IPCEventType
  payload: any
  timestamp: number
  timeout?: number
}

export interface IPCResponse {
  id: string
  type: IPCEventType
  success: boolean
  payload: any
  error?: string
  timestamp: number
  duration: number
}

// =====================================================
// Configuration IPC Types
// =====================================================

// Configuration load request/response
export interface ConfigLoadRequest {
  filePath?: string
  templateName?: string
}

export interface ConfigLoadResponse {
  config: SimulationConfigType
  metadata: ConfigurationMetadata
  filePath?: string
  templateName?: string
  timestamp: number
}

// Configuration save request/response
export interface ConfigSaveRequest {
  config: SimulationConfigType
  metadata?: ConfigurationMetadata
  filePath?: string
  format?: 'json' | 'yaml' | 'xml'
  backup?: boolean
}

export interface ConfigSaveResponse {
  filePath: string
  size: number
  timestamp: number
  backupCreated?: boolean
  backupPath?: string
}

// Configuration export request/response
export interface ConfigExportRequest {
  config: SimulationConfigType
  format: 'json' | 'yaml' | 'xml'
  filePath?: string
  includeMetadata?: boolean
  compress?: boolean
}

export interface ConfigExportResponse {
  filePath?: string
  content: string
  size: number
  format: 'json' | 'yaml' | 'xml'
  compressed?: boolean
  timestamp: number
}

// Configuration import request/response
export interface ConfigImportRequest {
  filePath?: string
  content?: string
  format?: 'json' | 'yaml' | 'xml'
  validate?: boolean
  merge?: boolean
}

export interface ConfigImportResponse extends ConfigImportResult {
  config: SimulationConfigType
  metadata?: ConfigurationMetadata
  source: 'file' | 'content'
  timestamp: number
}

// Configuration validation request/response
export interface ConfigValidateRequest {
  config: SimulationConfigType
  partial?: boolean
  rules?: string[]
  context?: Record<string, any>
}

export interface ConfigValidateResponse extends ValidationResult {
  config: SimulationConfigType
  validationTime: number
  timestamp: number
}

// =====================================================
// Template IPC Types
// =====================================================

// Template load request/response
export interface TemplateLoadRequest {
  templateName: string
  category?: string
}

export interface TemplateLoadResponse {
  template: ConfigTemplate
  config: SimulationConfigType
  timestamp: number
}

// Template save request/response
export interface TemplateSaveRequest {
  template: ConfigTemplate
  config: SimulationConfigType
  overwrite?: boolean
}

export interface TemplateSaveResponse {
  templateName: string
  category: string
  size: number
  timestamp: number
  created: boolean
  overwritten: boolean
}

// Template delete request/response
export interface TemplateDeleteRequest {
  templateName: string
  category?: string
}

export interface TemplateDeleteResponse {
  templateName: string
  category: string
  timestamp: number
  deleted: boolean
}

// Template list request/response
export interface TemplateListRequest {
  category?: string
  includeSystem?: boolean
  includeUser?: boolean
}

export interface TemplateListResponse {
  templates: Array<{
    name: string
    description: string
    category: string
    author?: string
    version?: string
    lastModified: number
    size: number
    type: 'system' | 'user'
  }>
  totalCount: number
  categoryCount: Record<string, number>
  timestamp: number
}

// =====================================================
// History IPC Types
// =====================================================

// History save request/response
export interface HistorySaveRequest {
  history: Array<{
    id: string
    config: SimulationConfigType
    timestamp: number
    action: string
    description?: string
    metadata?: Record<string, any>
  }>
  currentIndex: number
  maxEntries?: number
}

export interface HistorySaveResponse {
  entriesSaved: number
  currentIndex: number
  timestamp: number
}

// History load request/response
export interface HistoryLoadRequest {
  maxEntries?: number
  since?: number
  filter?: {
    actions?: string[]
    dateRange?: { start: number; end: number }
  }
}

export interface HistoryLoadResponse {
  history: Array<{
    id: string
    config: SimulationConfigType
    timestamp: number
    action: string
    description?: string
    metadata?: Record<string, any>
  }>
  currentIndex: number
  totalEntries: number
  timestamp: number
}

// =====================================================
// File System IPC Types
// =====================================================

// File operations
export interface FileExistsRequest {
  filePath: string
}

export interface FileExistsResponse {
  exists: boolean
  isFile: boolean
  isDirectory: boolean
  size?: number
  modified?: number
}

export interface FileReadRequest {
  filePath: string
  encoding?: 'utf8' | 'binary' | 'base64'
  options?: {
    flag?: string
    start?: number
    end?: number
  }
}

export interface FileReadResponse {
  content: string | Buffer
  encoding: 'utf8' | 'binary' | 'base64'
  size: number
  modified: number
  mimeType?: string
}

export interface FileWriteRequest {
  filePath: string
  content: string | Buffer
  encoding?: 'utf8' | 'binary' | 'base64'
  options?: {
    flag?: string
    mode?: number
    backup?: boolean
  }
}

export interface FileWriteResponse {
  filePath: string
  written: number
  size: number
  created: boolean
  backupCreated?: boolean
  backupPath?: string
}

export interface FileDeleteRequest {
  filePath: string
  backup?: boolean
}

export interface FileDeleteResponse {
  filePath: string
  deleted: boolean
  backupCreated?: boolean
  backupPath?: string
}

// Directory operations
export interface DirectoryReadRequest {
  dirPath: string
  recursive?: boolean
  filter?: {
    extensions?: string[]
    exclude?: string[]
    includeHidden?: boolean
  }
}

export interface DirectoryReadResponse {
  entries: Array<{
    name: string
    path: string
    type: 'file' | 'directory' | 'symlink'
    size?: number
    modified?: number
    extension?: string
  }>
  totalCount: number
  fileCount: number
  directoryCount: number
}

export interface DirectoryCreateRequest {
  dirPath: string
  recursive?: boolean
  mode?: number
}

export interface DirectoryCreateResponse {
  dirPath: string
  created: boolean
  existing: boolean
}

export interface DirectoryDeleteRequest {
  dirPath: string
  recursive?: boolean
  backup?: boolean
}

export interface DirectoryDeleteResponse {
  dirPath: string
  deleted: boolean
  backupCreated?: boolean
  backupPath?: string
}

// =====================================================
// Application IPC Types
// =====================================================

// Settings get/set
export interface SettingsGetRequest {
  keys?: string[]
  category?: string
}

export interface SettingsGetResponse {
  settings: Record<string, any>
  timestamp: number
}

export interface SettingsSetRequest {
  settings: Record<string, any>
  category?: string
  persist?: boolean
}

export interface SettingsSetResponse {
  success: boolean
  updatedKeys: string[]
  timestamp: number
  errors?: Record<string, string>
}

// Application info
export interface AppVersionResponse {
  version: string
  build: string
  platform: string
  arch: string
  timestamp: number
}

export interface AppPathRequest {
  type: 'home' | 'appData' | 'userData' | 'cache' | 'temp' | 'exe' | 'module' | 'desktop' | 'documents' | 'downloads' | 'music' | 'pictures' | 'videos' | 'logs' | 'crashDumps'
}

export interface AppPathResponse {
  path: string
  type: string
  timestamp: number
}

// System info
export interface SystemInfoResponse {
  platform: string
  arch: string
  release: string
  hostname: string
  userInfo: {
    uid?: number
    gid?: number
    username: string
    homedir: string
    shell?: string
  }
  memory: {
    total: number
    free: number
    used: number
  }
  cpus: Array<{
    model: string
    speed: number
    times: {
      user: number
      nice: number
      sys: number
      idle: number
      irq: number
    }
  }>
  loadavg: number[]
  uptime: number
  timestamp: number
}

// =====================================================
// IPC Service Interface
// =====================================================

// IPC service methods
export interface IPCService {
  // Configuration operations
  loadConfig(request: ConfigLoadRequest): Promise<ConfigLoadResponse>
  saveConfig(request: ConfigSaveRequest): Promise<ConfigSaveResponse>
  exportConfig(request: ConfigExportRequest): Promise<ConfigExportResponse>
  importConfig(request: ConfigImportRequest): Promise<ConfigImportResponse>
  validateConfig(request: ConfigValidateRequest): Promise<ConfigValidateResponse>

  // Template operations
  loadTemplate(request: TemplateLoadRequest): Promise<TemplateLoadResponse>
  saveTemplate(request: TemplateSaveRequest): Promise<TemplateSaveResponse>
  deleteTemplate(request: TemplateDeleteRequest): Promise<TemplateDeleteResponse>
  listTemplates(request: TemplateListRequest): Promise<TemplateListResponse>

  // History operations
  saveHistory(request: HistorySaveRequest): Promise<HistorySaveResponse>
  loadHistory(request: HistoryLoadRequest): Promise<HistoryLoadResponse>

  // File system operations
  fileExists(request: FileExistsRequest): Promise<FileExistsResponse>
  readFile(request: FileReadRequest): Promise<FileReadResponse>
  writeFile(request: FileWriteRequest): Promise<FileWriteResponse>
  deleteFile(request: FileDeleteRequest): Promise<FileDeleteResponse>
  readDirectory(request: DirectoryReadRequest): Promise<DirectoryReadResponse>
  createDirectory(request: DirectoryCreateRequest): Promise<DirectoryCreateResponse>
  deleteDirectory(request: DirectoryDeleteRequest): Promise<DirectoryDeleteResponse>

  // Application operations
  getSettings(request: SettingsGetRequest): Promise<SettingsGetResponse>
  setSettings(request: SettingsSetRequest): Promise<SettingsSetResponse>
  getAppVersion(): Promise<AppVersionResponse>
  getAppPath(request: AppPathRequest): Promise<AppPathResponse>
  getSystemInfo(): Promise<SystemInfoResponse>

  // Generic operations
  send<T = any>(channel: string, payload?: any): Promise<T>
  invoke<T = any>(channel: string, payload?: any): Promise<T>
  on<T = any>(channel: string, listener: (payload: T) => void): () => void
  once<T = any>(channel: string, listener: (payload: T) => void): Promise<T>
  removeListener(channel: string, listener: Function): void
  removeAllListeners(channel?: string): void
}

// IPC error types
export interface IPCError {
  code: string
  message: string
  details?: any
  stack?: string
  timestamp: number
}

// IPC connection status
export type IPCConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error'

// IPC connection info
export interface IPCConnectionInfo {
  status: IPCConnectionStatus
  lastPing: number
  lastError?: IPCError
  retryCount: number
  maxRetries: number
  timeout: number
}

// IPC performance metrics
export interface IPCPerformanceMetrics {
  totalRequests: number
  totalResponses: number
  averageResponseTime: number
  errorRate: number
  requestsByChannel: Record<string, number>
  responseTimeByChannel: Record<string, number>
  peakResponseTime: number
  lastRequestTime: number
  lastResponseTime: number
}