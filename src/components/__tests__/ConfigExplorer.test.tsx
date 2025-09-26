import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { ConfigExplorer } from '../ConfigExplorer/ConfigExplorer'

// Mock the IPC service
vi.mock('@/services/ipcService', () => ({
  ipcService: {
    getConnectionStatus: vi.fn(),
    initializeConnection: vi.fn()
  }
}))

// Mock the DualPanelLayout component
vi.mock('@/components/Layout/DualPanelLayout', () => ({
  DualPanelLayout: () => <div data-testid="dual-panel-layout">Dual Panel Layout</div>
}))

describe('ConfigExplorer', () => {
  beforeEach(() => {
    // Reset mocks
    vi.clearAllMocks()

    // Mock window.electronAPI as undefined (browser mode)
    Object.defineProperty(window, 'electronAPI', {
      value: undefined,
      writable: true
    })
  })

  it('renders without crashing', () => {
    const { container } = render(<ConfigExplorer />)
    expect(container.firstChild).toHaveClass('config-explorer')
  })

  it('displays loading state initially', () => {
    render(<ConfigExplorer />)
    expect(screen.getByText('Initializing application...')).toBeInTheDocument()
  })

  it('shows connection status when disconnected', async () => {
    // Mock IPC service to return disconnected status
    const { ipcService } = require('@/services/ipcService')
    ipcService.getConnectionStatus.mockReturnValue('disconnected')

    render(<ConfigExplorer />)

    await waitFor(() => {
      expect(screen.getByText('Running in browser mode - some features may be limited')).toBeInTheDocument()
    })
  })

  it('renders dual panel layout after initialization', async () => {
    // Mock IPC service to return connected status
    const { ipcService } = require('@/services/ipcService')
    ipcService.getConnectionStatus.mockReturnValue('connected')

    render(<ConfigExplorer />)

    await waitFor(() => {
      expect(screen.getByTestId('dual-panel-layout')).toBeInTheDocument()
    })
  })

  it('handles error state gracefully', async () => {
    // Mock IPC service to return error status
    const { ipcService } = require('@/services/ipcService')
    ipcService.getConnectionStatus.mockReturnValue('error')

    render(<ConfigExplorer />)

    await waitFor(() => {
      expect(screen.getByText('Connection Error')).toBeInTheDocument()
      expect(screen.getByText('Failed to establish connection to backend services.')).toBeInTheDocument()
    })
  })

  it('renders retry button in error state', async () => {
    const { ipcService } = require('@/services/ipcService')
    ipcService.getConnectionStatus.mockReturnValue('error')

    render(<ConfigExplorer />)

    await waitFor(() => {
      const retryButton = screen.getByRole('button', { name: 'Retry' })
      expect(retryButton).toBeInTheDocument()
    })
  })
})