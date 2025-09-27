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

// Mock matchMedia which is not available in test environment (guard for node env)
if (typeof window !== 'undefined') {
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
}

// Mock IntersectionObserver
globalThis.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock WebSocket for testing
(globalThis as any).WebSocket = vi.fn().mockImplementation(() => ({
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
  AccessibilityProvider: ({ children }: { children: React.ReactNode }) => (
    React.createElement('div', null,
      React.createElement('div', { 'aria-live': 'polite' }),
      React.createElement('div', { 'aria-live': 'assertive' }),
      children
    )
  ),
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
}

// Partially mock ipcService to preserve class exports
vi.mock('@/services/ipcService', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/ipcService')>()
  return {
    ...actual,
    ipcService: mockIpcService,
  }
})

// Mock window.electronAPI for testing (guard for node env)
if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'electronAPI', {
    value: {
      invoke: vi.fn(() => Promise.resolve({ success: true }))
    },
    writable: true
  })
}

// Mock Leva hooks and components for testing
vi.mock('leva', () => ({
  Leva: ({ children }: { children: React.ReactNode }) => React.createElement('div', { 'data-testid': 'leva-root' }, children),
  useControls: vi.fn(() => ({})),
  folder: vi.fn((config) => config),
  button: vi.fn((config) => config),
  monitor: vi.fn((config) => config),
  LevaPanel: ({ children }: { children: React.ReactNode }) => React.createElement('div', { 'data-testid': 'leva-panel' }, children)
}))

// Mock MetadataSystem for testing with required exports
vi.mock('@/components/LevaControls/MetadataSystem', () => {
  const ValidationRules = {
    range: (min: number, max: number) => ({
      name: `range_${min}_${max}`,
      description: `Valid range is ${min} to ${max}`,
      validator: (val: any) => typeof val === 'number' && val >= min && val <= max,
      errorMessage: `Value must be between ${min} and ${max}`,
      severity: 'error'
    })
  }

  const MetadataTemplates = {
    percentage: () => ({ category: 'display', inputType: 'number', format: 'percentage', validationRules: [ValidationRules.range(0, 1)] }),
    ratio: () => ({ category: 'parameters', inputType: 'number', format: 'number', validationRules: [ValidationRules.range(0, 1)] }),
    coordinates: () => ({ category: 'position', inputType: 'vector2', format: 'number' }),
    color: () => ({ category: 'visualization', inputType: 'color', format: 'hex' }),
    filePath: () => ({ category: 'input', inputType: 'file' }),
    number: (min?: number, max?: number) => ({ category: 'parameters', inputType: 'number', format: 'number', validationRules: min !== undefined ? [ValidationRules.range(min, max ?? Number.POSITIVE_INFINITY)] : [] })
  }

  const MetadataProvider = ({ children }: { children: React.ReactNode }) => children

  const useMetadata = () => ({
    metadata: {},
    groups: {},
    categories: {},
    registerControl: vi.fn(),
    updateControl: vi.fn(),
    getControlMetadata: vi.fn(() => ({} as any)),
    getGroupControls: vi.fn(() => []),
    getCategoryGroups: vi.fn(() => []),
    validateControl: vi.fn((path: string, value: any) => {
      // very basic example validation
      if (path && value === 'invalid-value') {
        return [{ message: 'Value must be valid', severity: 'error', rule: 'test-rule' }]
      }
      return []
    })
  })

  return {
    MetadataProvider,
    useMetadata,
    ValidationRules,
    MetadataTemplates
  }
})

// Use real styled-components to avoid invalid element issues