import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ThemeProvider } from '@/components/UI/ThemeProvider'
import { ConfigExplorer } from '../ConfigExplorer/ConfigExplorer'

// Mock the modules BEFORE importing the component
vi.mock('@/services/ipcService', () => ({
  ipcService: {
    getConnectionStatus: vi.fn(() => 'disconnected'),
    initializeConnection: vi.fn()
  }
}))

vi.mock('@/components/Layout/DualPanelLayout', () => ({
  DualPanelLayout: () => <div data-testid="dual-panel-layout">Dual Panel Layout</div>
}))

describe('ConfigExplorer', () => {
  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks()

    // Mock window.electronAPI as undefined (browser mode)
    Object.defineProperty(window, 'electronAPI', {
      value: undefined,
      writable: true
    })
  })

  it('renders without crashing', () => {
    const { container } = render(
      <ThemeProvider>
        <ConfigExplorer />
      </ThemeProvider>
    )
    expect(container.firstChild).toHaveClass('config-explorer')
  })

  it('renders dual panel layout', () => {
    render(
      <ThemeProvider>
        <ConfigExplorer />
      </ThemeProvider>
    )
    expect(screen.getByTestId('dual-panel-layout')).toBeInTheDocument()
  })

  it('shows browser mode warning', () => {
    render(
      <ThemeProvider>
        <ConfigExplorer />
      </ThemeProvider>
    )
    expect(screen.getByText(/Running in browser mode/)).toBeInTheDocument()
  })
})