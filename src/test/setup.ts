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
vi.mock('@/services/ipcService', () => ({
  ipcService: {
    getConnectionStatus: vi.fn(() => 'connected'),
    initializeConnection: vi.fn(() => Promise.resolve()),
    loadConfig: vi.fn(() => Promise.resolve({ success: true, config: {} })),
    saveConfig: vi.fn(() => Promise.resolve({ success: true })),
    executeRequest: vi.fn(() => Promise.resolve({ success: true })),
  },
}))

// Mock window.electronAPI for testing
Object.defineProperty(window, 'electronAPI', {
  value: {
    invoke: vi.fn(() => Promise.resolve({ success: true })),
    on: vi.fn(),
    removeAllListeners: vi.fn(),
  },
  writable: true,
})