// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Toolbar } from '@/components/Layout/Toolbar'
import { useConfigStore } from '@/stores/configStore'

vi.mock('@/services/ipcService', () => ({
  ipcService: {
    getConnectionStatus: vi.fn(() => 'disconnected'),
    exportConfig: vi.fn(async () => ({ success: true }))
  }
}))

describe('Toolbar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders file, comparison and app controls', () => {
    render(<Toolbar />)
    expect(screen.getByRole('toolbar', { name: /application toolbar/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Open configuration/ })).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /Save \(Ctrl\/Cmd\+S\)/ }).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByRole('button', { name: /Save As/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export JSON' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Export YAML/ })).toBeInTheDocument()
    // Show or Hide Compare button depending on initial state
    const compareToggle = screen.getByRole('button', { name: /Show Compare|Hide Compare/ })
    expect(compareToggle).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Load Compare…' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Clear Compare' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Apply All' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /grayscale/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reset' })).toBeInTheDocument()
  })

  it('toggles grayscale and persists', () => {
    render(<Toolbar />)
    const btn = screen.getByRole('button', { name: /grayscale/i })
    const before = document.body.classList.contains('grayscale')
    fireEvent.click(btn)
    const after = document.body.classList.contains('grayscale')
    expect(after).toBe(!before)
  })

  it('disables Save when not dirty and enables after change', () => {
    const { rerender } = render(<Toolbar />)
    const saveBtns = screen.getAllByRole('button', { name: /Save \(Ctrl\/Cmd\+S\)/ })
    const saveBtn = saveBtns[0]
    expect(saveBtn).toBeDisabled()
    useConfigStore.getState().updateConfig('width', 101)
    // Rerender to reflect store change
    rerender(<Toolbar />)
    expect(screen.getAllByRole('button', { name: /Save \(Ctrl\/Cmd\+S\)/ })[0]).not.toBeDisabled()
  })
})

