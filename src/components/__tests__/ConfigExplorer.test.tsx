import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ThemeProvider } from '@/components/UI/ThemeProvider'
import { ConfigExplorer } from '../ConfigExplorer/ConfigExplorer'
import { useConfigStore } from '@/stores/configStore'

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

  it('toggles comparison panel visibility', async () => {
    render(
      <ThemeProvider>
        <ConfigExplorer />
      </ThemeProvider>
    )

    // Initially hidden
    expect(screen.queryByText(/Comparison Panel/)).not.toBeInTheDocument()

    // Find and click the toggle button
    const toggleBtn = await screen.findByRole('button', { name: /Show Comparison Panel/i })
    fireEvent.click(toggleBtn)

    // Panel now visible
    await waitFor(() => expect(screen.getByText(/Comparison Panel/)).toBeInTheDocument())

    // Click again to hide
    fireEvent.click(screen.getByRole('button', { name: /Hide Comparison Panel/i }))
    await waitFor(() => expect(screen.queryByText(/Comparison Panel/)).not.toBeInTheDocument())
  })

  it('handles invalid comparison file gracefully', async () => {
    render(
      <ThemeProvider>
        <ConfigExplorer />
      </ThemeProvider>
    )

    const toggleBtn = await screen.findByRole('button', { name: /Show Comparison Panel/i })
    fireEvent.click(toggleBtn)

    const loadBtn = await screen.findByRole('button', { name: /Load Comparison Config/i })
    // Simulate a click on hidden input by dispatching change with invalid JSON
    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(["{ invalid json"], 'bad.json', { type: 'application/json' })
    Object.defineProperty(input, 'files', { value: [file] })
    fireEvent.change(input)

    await waitFor(() => {
      expect(screen.getByText(/Invalid comparison file/)).toBeInTheDocument()
    })
  })
})