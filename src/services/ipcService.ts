import { IPCService, IPCEventType, IPCError, IPCConnectionStatus, IPCConnectionInfo, IPCPerformanceMetrics } from '../types/ipc'
import { SimulationConfigType } from '../types/config'
import { ValidationResult } from '../types/validation'

// =====================================================
// IPC Service Implementation
// =====================================================

export class IPCServiceImpl implements IPCService {
  private connectionStatus: IPCConnectionStatus = 'disconnected'
  private connectionInfo: IPCConnectionInfo = {
    status: 'disconnected',
    lastPing: 0,
    retryCount: 0,
    maxRetries: 3,
    timeout: 5000
  }

  private performanceMetrics: IPCPerformanceMetrics = {
    totalRequests: 0,
    totalResponses: 0,
    averageResponseTime: 0,
    errorRate: 0,
    requestsByChannel: {},
    responseTimeByChannel: {},
    peakResponseTime: 0,
    lastRequestTime: 0,
    lastResponseTime: 0
  }

  private listeners: Map<string, Array<(payload: any) => void>> = new Map()
  // Reserved for future once-only listener support
  // private onceListeners: Map<string, Array<(payload: any) => Promise<any>>> = new Map()

  constructor() {
    this.initializeConnection()
  }

  // =====================================================
  // Connection Management
  // =====================================================

  private async initializeConnection(): Promise<void> {
    if (typeof window === 'undefined' || !window.electronAPI) {
      this.connectionStatus = 'disconnected'
      return
    }

    this.connectionStatus = 'connecting'

    try {
      // Test connection with a simple ping
      await this.ping()
      this.connectionStatus = 'connected'
      this.connectionInfo.lastPing = Date.now()
      this.connectionInfo.retryCount = 0
    } catch (error) {
      this.connectionStatus = 'error'
      this.connectionInfo.lastError = this.createIPCError('CONNECTION_ERROR', error)
    }
  }

  private async ping(): Promise<boolean> {
    try {
      if (!window.electronAPI || typeof window.electronAPI.invoke !== 'function') {
        throw new Error('Electron API not available')
      }
      const response = await window.electronAPI.invoke('app:ping', { timestamp: Date.now() })
      return response.success
    } catch (error) {
      throw new Error('IPC connection test failed')
    }
  }

  private createIPCError(code: string, error: any): IPCError {
    return {
      code,
      message: error.message || 'Unknown IPC error',
      details: error,
      stack: error.stack,
      timestamp: Date.now()
    }
  }

  private async executeRequest<T = any>(type: IPCEventType, payload: any): Promise<T> {

    this.performanceMetrics.totalRequests++
    this.performanceMetrics.requestsByChannel[type] = (this.performanceMetrics.requestsByChannel[type] || 0) + 1
    this.performanceMetrics.lastRequestTime = Date.now()

    try {
      if (!window.electronAPI || typeof window.electronAPI.invoke !== 'function') {
        throw new Error('Electron API not available')
      }

      const startTime = Date.now()
      type Enveloped<R> = { success: boolean; payload?: R; error?: string }
      const response = (await window.electronAPI.invoke(type, payload)) as T | Enveloped<T>
      const responseTime = Date.now() - startTime

      this.performanceMetrics.totalResponses++
      this.performanceMetrics.responseTimeByChannel[type] = (this.performanceMetrics.responseTimeByChannel[type] || 0) + responseTime
      this.performanceMetrics.averageResponseTime = this.performanceMetrics.totalResponses > 0
        ? (this.performanceMetrics.averageResponseTime * (this.performanceMetrics.totalResponses - 1) + responseTime) / this.performanceMetrics.totalResponses
        : responseTime
      this.performanceMetrics.peakResponseTime = Math.max(this.performanceMetrics.peakResponseTime, responseTime)
      this.performanceMetrics.lastResponseTime = Date.now()

      // Accept both enveloped and bare responses from main
      const maybe = response as Enveloped<T>
      if (maybe && typeof maybe === 'object' && 'success' in maybe) {
        if (!maybe.success) {
          throw new Error(maybe.error || 'IPC operation failed')
        }
        return (maybe.payload as T)
      }
      return response as T
    } catch (error) {
      this.performanceMetrics.errorRate = this.performanceMetrics.totalRequests > 0
        ? (this.performanceMetrics.totalRequests - this.performanceMetrics.totalResponses) / this.performanceMetrics.totalRequests
        : 0

      throw this.createIPCError('REQUEST_FAILED', error)
    }
  }

  // Keeping ID generator for future tracing (not currently used)
  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`
  }

  // =====================================================
  // Configuration Operations
  // =====================================================

  async loadConfig(request: { filePath?: string; templateName?: string }): Promise<{
    config: SimulationConfigType
    metadata: any
    filePath?: string
    templateName?: string
    timestamp: number
  }> {
    const result = await this.executeRequest('config:load', request)
    return result
  }

  async saveConfig(request: {
    config: SimulationConfigType
    metadata?: any
    filePath?: string
    format?: 'json' | 'yaml' | 'xml'
    backup?: boolean
  }): Promise<{
    filePath: string
    size: number
    timestamp: number
    backupCreated?: boolean
    backupPath?: string
  }> {
    const result = await this.executeRequest('config:save', request)
    return result
  }

  async exportConfig(request: {
    config: SimulationConfigType
    format: 'json' | 'yaml' | 'xml'
    filePath?: string
    includeMetadata?: boolean
    compress?: boolean
  }): Promise<{
    filePath?: string
    content: string
    size: number
    format: 'json' | 'yaml' | 'xml'
    compressed?: boolean
    timestamp: number
  }> {
    const result = await this.executeRequest('config:export', request)
    return result
  }

  async importConfig(request: {
    filePath?: string
    content?: string
    format?: 'json' | 'yaml' | 'xml'
    validate?: boolean
    merge?: boolean
  }): Promise<any> {
    const result = await this.executeRequest('config:import', request)
    return result
  }

  async validateConfig(request: {
    config: SimulationConfigType
    partial?: boolean
    rules?: string[]
    context?: Record<string, any>
  }): Promise<ValidationResult & { config: SimulationConfigType; validationTime: number; timestamp: number }> {
    const result = await this.executeRequest('config:validate', request)
    return result
  }

  // =====================================================
  // Template Operations
  // =====================================================

  async loadTemplate(request: { templateName: string; category?: string }): Promise<any> {
    const result = await this.executeRequest('config:template:load', request)
    return result
  }

  async saveTemplate(request: {
    template: any
    config: SimulationConfigType
    overwrite?: boolean
  }): Promise<any> {
    const result = await this.executeRequest('config:template:save', request)
    return result
  }

  async deleteTemplate(request: { templateName: string; category?: string }): Promise<any> {
    const result = await this.executeRequest('config:template:delete', request)
    return result
  }

  async listTemplates(request: { category?: string; includeSystem?: boolean; includeUser?: boolean }): Promise<any> {
    const result = await this.executeRequest('config:template:list', request)
    return result
  }

  // =====================================================
  // History Operations
  // =====================================================

  async saveHistory(request: any): Promise<any> {
    const result = await this.executeRequest('config:history:save', request)
    return result
  }

  async loadHistory(request: any): Promise<any> {
    const result = await this.executeRequest('config:history:load', request)
    return result
  }

  // =====================================================
  // File System Operations
  // =====================================================

  async fileExists(request: { filePath: string }): Promise<any> {
    const result = await this.executeRequest('fs:file:exists', request)
    return result
  }

  async readFile(request: { filePath: string; encoding?: string; options?: any }): Promise<any> {
    const result = await this.executeRequest('fs:file:read', request)
    return result
  }

  async writeFile(request: { filePath: string; content: string | any; encoding?: string; options?: any }): Promise<any> {
    const result = await this.executeRequest('fs:file:write', request)
    return result
  }

  async deleteFile(request: { filePath: string; backup?: boolean }): Promise<any> {
    const result = await this.executeRequest('fs:file:delete', request)
    return result
  }

  async readDirectory(request: { dirPath: string; recursive?: boolean; filter?: any }): Promise<any> {
    const result = await this.executeRequest('fs:directory:read', request)
    return result
  }

  async createDirectory(request: { dirPath: string; recursive?: boolean; mode?: number }): Promise<any> {
    const result = await this.executeRequest('fs:directory:create', request)
    return result
  }

  async deleteDirectory(request: { dirPath: string; recursive?: boolean; backup?: boolean }): Promise<any> {
    const result = await this.executeRequest('fs:directory:delete', request)
    return result
  }

  // =====================================================
  // Application Operations
  // =====================================================

  async getSettings(request: { keys?: string[]; category?: string }): Promise<any> {
    const result = await this.executeRequest('app:settings:get', request)
    return result
  }

  async setSettings(request: { settings: Record<string, any>; category?: string; persist?: boolean }): Promise<any> {
    const result = await this.executeRequest('app:settings:set', request)
    return result
  }

  async getAppVersion(): Promise<any> {
    const result = await this.executeRequest('app:version:get', {})
    return result
  }

  async getAppPath(request: { type: string }): Promise<any> {
    const result = await this.executeRequest('app:path:get', request)
    return result
  }

  async getSystemInfo(): Promise<any> {
    const result = await this.executeRequest('system:info:get', {})
    return result
  }

  // =====================================================
  // Generic Operations
  // =====================================================

  async send<T = any>(channel: string, payload?: any): Promise<T> {
    return this.executeRequest(channel as IPCEventType, payload)
  }

  async invoke<T = any>(channel: string, payload?: any): Promise<T> {
    return this.executeRequest(channel as IPCEventType, payload)
  }

  on<T = any>(channel: string, listener: (payload: T) => void): () => void {
    if (!this.listeners.has(channel)) {
      this.listeners.set(channel, [])
    }

    this.listeners.get(channel)!.push(listener)

    // Set up the listener with electronAPI if available
    if (window.electronAPI && window.electronAPI.on) {
      const electronListener = (payload: T) => {
        this.listeners.get(channel)?.forEach(l => l(payload))
      }

      window.electronAPI.on(channel, electronListener)

      // Return unsubscribe function
      return () => {
        if (window.electronAPI && window.electronAPI.removeListener) {
          window.electronAPI.removeListener(channel, electronListener)
        }
        const listeners = this.listeners.get(channel) || []
        const index = listeners.indexOf(listener)
        if (index > -1) {
          listeners.splice(index, 1)
        }
      }
    }

    // Return no-op function if electronAPI not available
    return () => {
      const listeners = this.listeners.get(channel) || []
      const index = listeners.indexOf(listener)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  once<T = any>(channel: string, listener: (payload: T) => void): Promise<T> {
    return new Promise((resolve) => {
      const onceListener = (payload: T) => {
        listener(payload)
        resolve(payload)
      }

      this.on(channel, onceListener)
    })
  }

  removeListener(channel: string, listener: Function): void {
    if (window.electronAPI && window.electronAPI.removeListener) {
      window.electronAPI.removeListener(channel, listener)
    }

    const listeners = this.listeners.get(channel) || []
    const index = listeners.indexOf(listener as any)
    if (index > -1) {
      listeners.splice(index, 1)
    }
  }

  removeAllListeners(channel?: string): void {
    if (channel) {
      if (window.electronAPI && window.electronAPI.removeAllListeners) {
        window.electronAPI.removeAllListeners(channel)
      }
      this.listeners.delete(channel)
    } else {
      if (window.electronAPI && window.electronAPI.removeAllListeners) {
        window.electronAPI.removeAllListeners()
      }
      this.listeners.clear()
    }
  }

  // =====================================================
  // Status and Metrics
  // =====================================================

  getConnectionStatus(): IPCConnectionStatus {
    return this.connectionStatus
  }

  getConnectionInfo(): IPCConnectionInfo {
    return { ...this.connectionInfo }
  }

  getPerformanceMetrics(): IPCPerformanceMetrics {
    return { ...this.performanceMetrics }
  }

  async reconnect(): Promise<void> {
    this.connectionInfo.retryCount++
    await this.initializeConnection()
  }
}

// =====================================================
// Global IPC Service Instance
// =====================================================

export const ipcService = new IPCServiceImpl()

// Global Window type augmentation moved to src/types/electron.d.ts