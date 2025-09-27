// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { StatusBar } from '@/components/Layout/StatusBar'
import { useConfigStore } from '@/stores/configStore'
import { useValidationStore } from '@/stores/validationStore'

vi.mock('@/services/ipcService', () => ({
  ipcService: {
    getConnectionStatus: vi.fn(() => 'disconnected')
  }
}))

describe('StatusBar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // reset stores
    useValidationStore.setState({ errors: [], warnings: [], isValidating: false, lastValidationTime: 0 })
    useConfigStore.setState({ isDirty: false, currentFilePath: undefined, lastSaveTime: undefined, lastLoadTime: undefined })
    // setup validation section in DOM
    const existing = document.getElementById('validation-content')
    if (!existing) {
      const el = document.createElement('div')
      el.id = 'validation-content'
      // @ts-ignore
      el.scrollIntoView = vi.fn()
      // @ts-ignore
      el.focus = vi.fn()
      document.body.appendChild(el)
    }
  })

  it('renders validation counts and controls', () => {
    render(<StatusBar />)
    expect(screen.getByRole('status', { name: /application status bar/i })).toBeInTheDocument()
    expect(screen.getByText(/Errors:/)).toBeInTheDocument()
    expect(screen.getByText(/Warnings:/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Validate configuration now/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Toggle auto validation/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /View validation issues/i })).toBeInTheDocument()
  })

  it('shows unsaved indicator when dirty', () => {
    useConfigStore.setState({ isDirty: true })
    render(<StatusBar />)
    expect(screen.getByText(/Unsaved/)).toBeInTheDocument()
  })

  it('toggles auto validation state text', () => {
    render(<StatusBar />)
    const toggle = screen.getByRole('button', { name: /Toggle auto validation/i })
    const before = toggle.textContent
    fireEvent.click(toggle)
    const after = toggle.textContent
    expect(after).not.toBe(before)
  })

  it('navigates to validation section on "View Issues"', () => {
    render(<StatusBar />)
    const btn = screen.getByRole('button', { name: /View validation issues/i })
    const target = document.getElementById('validation-content') as any
    const spyScroll = vi.spyOn(target, 'scrollIntoView')
    const spyFocus = vi.spyOn(target, 'focus')
    fireEvent.click(btn)
    expect(spyScroll).toHaveBeenCalled()
    expect(spyFocus).toHaveBeenCalled()
  })

  it('updates error and warning counts when store changes', () => {
    render(<StatusBar />)
    expect(screen.getByText(/Errors: 0/)).toBeInTheDocument()
    expect(screen.getByText(/Warnings: 0/)).toBeInTheDocument()
    useValidationStore.setState({
      errors: [{ path: 'width', message: 'too small', code: 'min' }],
      warnings: [{ path: 'memory', message: 'high usage', code: 'warn' }]
    }, true)
    // Re-render to reflect state change (testing-library auto re-renders reactive components)
    expect(screen.getByText(/Errors: 1/)).toBeInTheDocument()
    expect(screen.getByText(/Warnings: 1/)).toBeInTheDocument()
  })
})

