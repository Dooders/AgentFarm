import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { IPCServiceImpl } from './ipcService'
import { IPCConnectionStatus } from '../types/ipc'

// Mock electronAPI
const mockElectronAPI = {
  loadConfig: vi.fn(),
  saveConfig: vi.fn(),
  exportConfig: vi.fn(),
  importConfig: vi.fn(),
  validateConfig: vi.fn(),
  loadTemplate: vi.fn(),
  saveTemplate: vi.fn(),
  deleteTemplate: vi.fn(),
  listTemplates: vi.fn(),
  saveHistory: vi.fn(),
  loadHistory: vi.fn(),
  clearHistory: vi.fn(),
  fileExists: vi.fn(),
  readFile: vi.fn(),
  writeFile: vi.fn(),
  deleteFile: vi.fn(),
  readDirectory: vi.fn(),
  createDirectory: vi.fn(),
  deleteDirectory: vi.fn(),
  getSettings: vi.fn(),
  setSettings: vi.fn(),
  getAppVersion: vi.fn(),
  getAppPath: vi.fn(),
  getSystemInfo: vi.fn(),
  platform: 'darwin',
  on: vi.fn(),
  once: vi.fn(),
  invoke: vi.fn(),
  send: vi.fn(),
  removeListener: vi.fn(),
  removeAllListeners: vi.fn()
}

describe('IPCServiceImpl', () => {
  let ipcService: IPCServiceImpl

  beforeEach(() => {
    // Reset the singleton instance
    (IPCServiceImpl as any).instance = null

    // Mock window.electronAPI BEFORE creating the service
    Object.defineProperty(window, 'electronAPI', {
      value: mockElectronAPI,
      writable: true
    })

    // Mock successful invoke responses
    mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: {} })

    // Create the service instance
    ipcService = new IPCServiceImpl()

    // Reset all mocks after service creation
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Connection Management', () => {
    it('initializes with disconnected status when electronAPI is unavailable', async () => {
      Object.defineProperty(window, 'electronAPI', {
        value: undefined,
        writable: true
      })

      const service = new IPCServiceImpl()
      expect(service.getConnectionStatus()).toBe('disconnected')
    })

    it('attempts to connect when electronAPI is available', async () => {
      mockElectronAPI.invoke.mockResolvedValue({ success: true })

      const service = new IPCServiceImpl()

      // Wait for initialization
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(service.getConnectionStatus()).toBe('connected')
    })

    it('handles connection failure gracefully', async () => {
      mockElectronAPI.invoke.mockRejectedValue(new Error('Connection failed'))

      const service = new IPCServiceImpl()

      // Wait for initialization
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(service.getConnectionStatus()).toBe('error')
    })
  })

  describe('Configuration Operations', () => {
    beforeEach(async () => {
      mockElectronAPI.invoke.mockResolvedValue({ success: true })
      await new Promise(resolve => setTimeout(resolve, 100))
    })

    it('loads configuration successfully', async () => {
      const mockConfig = { width: 100, height: 100 }
      const mockResponse = {
        config: mockConfig,
        metadata: { version: '1.0' },
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.loadConfig({ filePath: 'test.json' })

      expect(mockElectronAPI.invoke).toHaveBeenCalledWith('config:load', { filePath: 'test.json' })
      expect(result).toEqual(mockResponse)
    })

    it('saves configuration successfully', async () => {
      const mockConfig = { width: 100, height: 100 }
      const mockResponse = {
        filePath: 'test.json',
        size: 1024,
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.saveConfig({
        config: mockConfig,
        format: 'json',
        backup: true
      })

      expect(mockElectronAPI.invoke).toHaveBeenCalledWith('config:save', {
        config: mockConfig,
        format: 'json',
        backup: true
      })
      expect(result).toEqual(mockResponse)
    })

    it('exports configuration successfully', async () => {
      const mockConfig = { width: 100, height: 100 }
      const mockResponse = {
        filePath: 'export.json',
        content: JSON.stringify(mockConfig),
        size: 1024,
        format: 'json' as const,
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.exportConfig({
        config: mockConfig,
        format: 'json'
      })

      expect(result).toEqual(mockResponse)
    })

    it('imports configuration successfully', async () => {
      const configContent = '{"width": 200, "height": 200}'
      const mockResponse = {
        config: { width: 200, height: 200 },
        metadata: { importedAt: expect.any(String) },
        source: 'content',
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.importConfig({
        content: configContent,
        format: 'json'
      })

      expect(result).toEqual(mockResponse)
    })

    it('validates configuration successfully', async () => {
      const mockConfig = { width: 100, height: 100 }
      const mockResponse = {
        isValid: true,
        errors: [],
        warnings: [],
        config: mockConfig,
        validationTime: 10,
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.validateConfig({
        config: mockConfig,
        rules: ['population_positive']
      })

      expect(result).toEqual(mockResponse)
    })
  })

  describe('Template Operations', () => {
    beforeEach(async () => {
      mockElectronAPI.invoke.mockResolvedValue({ success: true })
      await new Promise(resolve => setTimeout(resolve, 100))
    })

    it('loads template successfully', async () => {
      const mockResponse = {
        template: { name: 'test-template', description: 'Test template' },
        config: { width: 100, height: 100 },
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.loadTemplate({
        templateName: 'test-template'
      })

      expect(result).toEqual(mockResponse)
    })

    it('saves template successfully', async () => {
      const template = { name: 'test-template', category: 'user' }
      const config = { width: 100, height: 100 }
      const mockResponse = {
        templateName: 'test-template',
        category: 'user',
        size: 1024,
        timestamp: Date.now(),
        created: true
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.saveTemplate({
        template,
        config
      })

      expect(result).toEqual(mockResponse)
    })

    it('lists templates successfully', async () => {
      const mockResponse = {
        templates: [
          {
            name: 'template1',
            description: 'Template 1',
            category: 'user',
            type: 'user' as const
          }
        ],
        totalCount: 1,
        categoryCount: { user: 1 },
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.listTemplates({})

      expect(result).toEqual(mockResponse)
    })
  })

  describe('File System Operations', () => {
    beforeEach(async () => {
      mockElectronAPI.invoke.mockResolvedValue({ success: true })
      await new Promise(resolve => setTimeout(resolve, 100))
    })

    it('checks file existence', async () => {
      const mockResponse = {
        exists: true,
        isFile: true,
        isDirectory: false,
        size: 1024,
        modified: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.fileExists({ filePath: 'test.json' })

      expect(result).toEqual(mockResponse)
    })

    it('reads file successfully', async () => {
      const mockResponse = {
        content: '{"width": 100}',
        encoding: 'utf8',
        size: 1024,
        modified: Date.now(),
        mimeType: 'application/json'
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.readFile({
        filePath: 'test.json',
        encoding: 'utf8'
      })

      expect(result).toEqual(mockResponse)
    })

    it('writes file successfully', async () => {
      const mockResponse = {
        filePath: 'test.json',
        written: 1024,
        size: 1024,
        created: true
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.writeFile({
        filePath: 'test.json',
        content: '{"width": 100}'
      })

      expect(result).toEqual(mockResponse)
    })
  })

  describe('Application Operations', () => {
    beforeEach(async () => {
      mockElectronAPI.invoke.mockResolvedValue({ success: true })
      await new Promise(resolve => setTimeout(resolve, 100))
    })

    it('gets application version', async () => {
      const mockResponse = {
        version: '1.0.0',
        build: 'dev',
        platform: 'darwin',
        arch: 'x64',
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.getAppVersion()

      expect(result).toEqual(mockResponse)
    })

    it('gets application path', async () => {
      const mockResponse = {
        path: '/test/path',
        type: 'userData',
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.getAppPath({ type: 'userData' })

      expect(result).toEqual(mockResponse)
    })

    it('gets system information', async () => {
      const mockResponse = {
        platform: 'darwin',
        arch: 'x64',
        release: '21.0.0',
        hostname: 'test-host',
        userInfo: {
          username: 'testuser',
          homedir: '/home/testuser'
        },
        memory: {
          total: 17179869184,
          free: 8589934592,
          used: 8589934592
        },
        cpus: [
          {
            model: 'Intel Core i7',
            speed: 2800,
            times: {
              user: 1000,
              nice: 0,
              sys: 2000,
              idle: 3000,
              irq: 0
            }
          }
        ],
        loadavg: [1.5, 1.2, 1.0],
        uptime: 3600,
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.getSystemInfo()

      expect(result.platform).toBe('darwin')
      expect(result.arch).toBe('x64')
    })
  })

  describe('Settings Management', () => {
    beforeEach(async () => {
      mockElectronAPI.invoke.mockResolvedValue({ success: true })
      await new Promise(resolve => setTimeout(resolve, 100))
    })

    it('gets settings successfully', async () => {
      const mockResponse = {
        settings: { theme: 'dark', language: 'en' },
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.getSettings({})

      expect(result).toEqual(mockResponse)
    })

    it('sets settings successfully', async () => {
      const mockResponse = {
        success: true,
        updatedKeys: ['theme'],
        timestamp: Date.now()
      }

      mockElectronAPI.invoke.mockResolvedValue({ success: true, payload: mockResponse })

      const result = await ipcService.setSettings({
        settings: { theme: 'light' }
      })

      expect(result).toEqual(mockResponse)
    })
  })

  describe('Event Handling', () => {
    beforeEach(async () => {
      mockElectronAPI.invoke.mockResolvedValue({ success: true })
      await new Promise(resolve => setTimeout(resolve, 100))
    })

    it('registers event listeners', () => {
      const callback = vi.fn()

      const unsubscribe = ipcService.on('test-event', callback)

      expect(mockElectronAPI.on).toHaveBeenCalledWith('test-event', expect.any(Function))
      expect(typeof unsubscribe).toBe('function')
    })

    it('registers one-time listeners', async () => {
      const callback = vi.fn()
      mockElectronAPI.on.mockImplementation((channel, cb) => {
        // Simulate immediate event emission
        setTimeout(() => cb({ data: 'test' }), 0)
      })

      const result = await ipcService.once('test-event', callback)

      expect(callback).toHaveBeenCalledWith({ data: 'test' })
      expect(result).toEqual({ data: 'test' })
    })

    it('removes listeners correctly', () => {
      const callback = vi.fn()

      ipcService.on('test-event', callback)
      ipcService.removeListener('test-event', callback)

      expect(mockElectronAPI.removeListener).toHaveBeenCalled()
    })

    it('removes all listeners for a channel', () => {
      ipcService.removeAllListeners('test-event')

      expect(mockElectronAPI.removeAllListeners).toHaveBeenCalledWith('test-event')
    })
  })

  describe('Error Handling', () => {
    it('handles IPC errors gracefully', async () => {
      const service = new IPCServiceImpl()
      mockElectronAPI.invoke.mockRejectedValue(new Error('IPC Error'))

      await expect(service.loadConfig({ filePath: 'test.json' })).rejects.toThrow('IPC Error')
    })

    it('provides fallback for connection failures', async () => {
      // Create service with electronAPI undefined
      Object.defineProperty(window, 'electronAPI', {
        value: undefined,
        writable: true
      })

      const service = new IPCServiceImpl()

      // Should throw error when electronAPI is not available
      await expect(service.loadConfig({ filePath: 'test.json' })).rejects.toThrow('Electron API not available')
    })
  })

  describe('Performance Metrics', () => {
    beforeEach(async () => {
      mockElectronAPI.invoke.mockResolvedValue({ success: true })
      await new Promise(resolve => setTimeout(resolve, 100))
    })

    it('tracks request metrics', async () => {
      await ipcService.loadConfig({ filePath: 'test.json' })

      const metrics = ipcService.getPerformanceMetrics()
      expect(metrics.totalRequests).toBe(1)
      expect(metrics.requestsByChannel['config:load']).toBe(1)
    })

    it('tracks response metrics', async () => {
      // Mock a successful response with timing
      mockElectronAPI.invoke.mockImplementation(() => {
        return new Promise(resolve => {
          setTimeout(() => {
            resolve({ success: true, payload: { config: {} } })
          }, 10)
        })
      })

      await ipcService.loadConfig({ filePath: 'test.json' })

      const metrics = ipcService.getPerformanceMetrics()
      expect(metrics.totalResponses).toBe(1)
      expect(metrics.averageResponseTime).toBeGreaterThanOrEqual(10)
    })
  })
})