import React from 'react'
import '@testing-library/jest-dom'
import '@testing-library/jest-dom/vitest'
import { beforeAll, afterAll, afterEach, vi } from 'vitest'
import { startMSW, stopMSW, resetMSW } from './mocks/server'

// Start MSW before all tests
beforeAll(async () => {
  await startMSW()
})

// Stop MSW after all tests
afterAll(async () => {
  await stopMSW()
})

// Reset handlers after each test
afterEach(async () => {
  await resetMSW()
  vi.clearAllMocks()
})

// Mock ResizeObserver which is not available in test environment
globalThis.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock matchMedia which is not available in test environment
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock IntersectionObserver
globalThis.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock WebSocket for testing
globalThis.WebSocket = vi.fn().mockImplementation(() => ({
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  dispatchEvent: vi.fn(),
  send: vi.fn(),
  close: vi.fn(),
  readyState: 1, // OPEN
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
})) as any

// Mock AccessibilityProvider for testing
vi.mock('@/components/UI/AccessibilityProvider', () => ({
  AccessibilityProvider: ({ children }: { children: React.ReactNode }) => children,
  useAccessibility: () => ({
    announce: vi.fn(),
    announceToScreenReader: vi.fn(),
    setFocus: vi.fn(),
    getFocus: vi.fn(),
    isKeyboardNavigation: false,
    isScreenReader: false,
    isHighContrast: false,
    isReducedMotion: false,
    setKeyboardNavigation: vi.fn(),
    setScreenReader: vi.fn(),
    setHighContrast: vi.fn(),
    setReducedMotion: vi.fn(),
  }),
}))

// Mock IPC service for testing
const mockIpcService = {
  getConnectionStatus: vi.fn(() => 'connected'),
  initializeConnection: vi.fn(() => Promise.resolve()),
  loadConfig: vi.fn(() => Promise.resolve({
    success: true,
    config: {
      width: 100,
      height: 100,
      position_discretization_method: 'floor',
      use_bilinear_interpolation: true,
      system_agents: 20,
      independent_agents: 20,
      control_agents: 10,
      agent_type_ratios: {
        SystemAgent: 0.4,
        IndependentAgent: 0.4,
        ControlAgent: 0.2,
      },
      learning_rate: 0.001,
      epsilon_start: 1.0,
      epsilon_min: 0.1,
      epsilon_decay: 0.995,
      agent_parameters: {
        SystemAgent: {
          target_update_freq: 100,
          memory_size: 1000,
          learning_rate: 0.001,
          gamma: 0.99,
          epsilon_start: 1.0,
          epsilon_min: 0.1,
          epsilon_decay: 0.995,
          dqn_hidden_size: 64,
          batch_size: 32,
          tau: 0.01,
          success_reward: 1.0,
          failure_penalty: -0.1,
          base_cost: 0.01,
        },
        IndependentAgent: {
          target_update_freq: 100,
          memory_size: 1000,
          learning_rate: 0.001,
          gamma: 0.99,
          epsilon_start: 1.0,
          epsilon_min: 0.1,
          epsilon_decay: 0.995,
          dqn_hidden_size: 64,
          batch_size: 32,
          tau: 0.01,
          success_reward: 1.0,
          failure_penalty: -0.1,
          base_cost: 0.01,
        },
        ControlAgent: {
          target_update_freq: 100,
          memory_size: 1000,
          learning_rate: 0.001,
          gamma: 0.99,
          epsilon_start: 1.0,
          epsilon_min: 0.1,
          epsilon_decay: 0.995,
          dqn_hidden_size: 64,
          batch_size: 32,
          tau: 0.01,
          success_reward: 1.0,
          failure_penalty: -0.1,
          base_cost: 0.01,
        },
      },
      visualization: {
        canvas_width: 800,
        canvas_height: 600,
        background_color: '#000000',
        agent_colors: {
          SystemAgent: '#ff6b6b',
          IndependentAgent: '#4ecdc4',
          ControlAgent: '#45b7d1',
        },
        show_metrics: true,
        font_size: 12,
        line_width: 1,
      },
      gather_parameters: {
        target_update_freq: 100,
        memory_size: 1000,
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_min: 0.1,
        epsilon_decay: 0.995,
        dqn_hidden_size: 64,
        batch_size: 32,
        tau: 0.01,
        success_reward: 1.0,
        failure_penalty: -0.1,
        base_cost: 0.01,
      },
      share_parameters: {
        target_update_freq: 100,
        memory_size: 1000,
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_min: 0.1,
        epsilon_decay: 0.995,
        dqn_hidden_size: 64,
        batch_size: 32,
        tau: 0.01,
        success_reward: 1.0,
        failure_penalty: -0.1,
        base_cost: 0.01,
      },
      move_parameters: {
        target_update_freq: 100,
        memory_size: 1000,
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_min: 0.1,
        epsilon_decay: 0.995,
        dqn_hidden_size: 64,
        batch_size: 32,
        tau: 0.01,
        success_reward: 1.0,
        failure_penalty: -0.1,
        base_cost: 0.01,
      },
      attack_parameters: {
        target_update_freq: 100,
        memory_size: 1000,
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_min: 0.1,
        epsilon_decay: 0.995,
        dqn_hidden_size: 64,
        batch_size: 32,
        tau: 0.01,
        success_reward: 1.0,
        failure_penalty: -0.1,
        base_cost: 0.01,
      },
    }
  })),
  saveConfig: vi.fn(() => Promise.resolve({ success: true })),
  executeRequest: vi.fn(() => Promise.resolve({ success: true })),
  validateConfig: vi.fn(() => Promise.reject(new Error('IPC validation not available in tests'))),
  // Add missing methods for tests
  readFile: vi.fn(() => Promise.resolve({ success: true, content: '{}' })),
  writeFile: vi.fn(() => Promise.resolve({ success: true })),
  fileExists: vi.fn(() => Promise.resolve({ exists: true })),
  exportConfig: vi.fn(() => Promise.resolve({ success: true })),
  importConfig: vi.fn(() => Promise.resolve({ success: true })),
  loadTemplate: vi.fn(() => Promise.resolve({ success: true })),
  saveTemplate: vi.fn(() => Promise.resolve({ success: true })),
  listTemplates: vi.fn(() => Promise.resolve({ success: true, templates: [] })),
  getAppVersion: vi.fn(() => Promise.resolve('1.0.0')),
  getAppPath: vi.fn(() => Promise.resolve('/app/path')),
  getSystemInfo: vi.fn(() => Promise.resolve({ platform: 'darwin', arch: 'x64' })),
  getSettings: vi.fn(() => Promise.resolve({})),
  setSettings: vi.fn(() => Promise.resolve({ success: true })),
  on: vi.fn(() => vi.fn()), // Return unsubscribe function
  once: vi.fn(() => Promise.resolve()),
  removeListener: vi.fn(),
  removeAllListeners: vi.fn(),
  send: vi.fn(() => Promise.resolve()),
  invoke: vi.fn(() => Promise.resolve()),
}

// Mock IPCServiceImpl class
class MockIPCServiceImpl {
  connectionStatus = 'connected'
  connectionInfo = {
    status: 'connected' as const,
    lastPing: Date.now(),
    retryCount: 0,
    maxRetries: 3,
    timeout: 5000
  }

  performanceMetrics = {
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

  listeners = new Map()
  onceListeners = new Map()

  constructor() {
    // Initialize connection
  }

  async initializeConnection(): Promise<void> {
    this.connectionStatus = 'connected'
  }

  getConnectionStatus() {
    return this.connectionStatus
  }

  getConnectionInfo() {
    return { ...this.connectionInfo }
  }

  getPerformanceMetrics() {
    return { ...this.performanceMetrics }
  }

  async loadConfig(request: any) {
    return mockIpcService.loadConfig(request)
  }

  async saveConfig(request: any) {
    return mockIpcService.saveConfig(request)
  }

  async executeRequest(type: string, payload: any) {
    return mockIpcService.executeRequest(payload)
  }

  async validateConfig(request: any) {
    return mockIpcService.validateConfig(request)
  }

  async reconnect(): Promise<void> {
    await this.initializeConnection()
  }
}

vi.mock('@/services/ipcService', () => ({
  IPCServiceImpl: MockIPCServiceImpl,
  ipcService: mockIpcService,
}))

// Mock window.electronAPI for testing
Object.defineProperty(window, 'electronAPI', {
  value: {
    invoke: vi.fn(() => Promise.resolve({ success: true }))
  },
  writable: true
})

// Mock Leva hooks and components for testing
vi.mock('leva', () => ({
  Leva: ({ children }: { children: React.ReactNode }) => React.createElement('div', { 'data-testid': 'leva-root' }, children),
  useControls: vi.fn(() => ({})),
  folder: vi.fn((config) => config),
  button: vi.fn((config) => config),
  monitor: vi.fn((config) => config),
  LevaPanel: ({ children }: { children: React.ReactNode }) => React.createElement('div', { 'data-testid': 'leva-panel' }, children)
}))

// Mock MetadataSystem for testing
vi.mock('@/components/LevaControls/MetadataSystem', () => ({
  MetadataProvider: ({ children }: { children: React.ReactNode }) => children,
  useMetadata: () => ({
    metadata: {},
    isLoading: false,
    error: null,
    getFieldMetadata: vi.fn(() => ({})),
    getValidationRules: vi.fn(() => []),
    getFieldDescription: vi.fn(() => ''),
    getFieldRange: vi.fn(() => ({ min: 0, max: 100 })),
    getFieldType: vi.fn(() => 'number'),
    hasMetadata: false,
    validateControl: vi.fn(() => [])
  }),
  MetadataTemplates: {
    percentage: vi.fn(() => ({})),
    ratio: vi.fn(() => ({})),
    coordinates: vi.fn(() => ({})),
    color: vi.fn(() => ({})),
    filePath: vi.fn(() => ({})),
    number: vi.fn(() => ({}))
  }
}))

// Mock styled-components for testing
const createStyledComponent = (tagName: string) => {
  const StyledComponent = vi.fn(({ children, ...props }: any) => React.createElement(tagName, props, children))
  StyledComponent.attrs = vi.fn(() => StyledComponent)
  StyledComponent.withConfig = vi.fn(() => StyledComponent)
  StyledComponent.displayName = `styled.${tagName}`
  return StyledComponent
}

vi.mock('styled-components', () => ({
  default: new Proxy(createStyledComponent('div'), {
    get(target, prop) {
      if (typeof prop === 'string') {
        return createStyledComponent(prop)
      }
      return target
    }
  })
}))