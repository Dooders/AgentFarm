import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useLevaStore } from '../levaStore'

describe('LevaStore', () => {
  beforeEach(() => {
    // Reset the store before each test
    useLevaStore.getState().resetToDefaults()
  })

  it('initializes with default state', () => {
    const { result } = renderHook(() => useLevaStore())

    expect(result.current.isVisible).toBe(true)
    expect(result.current.isCollapsed).toBe(false)
    expect(result.current.panelPosition).toBe('right')
    expect(result.current.panelWidth).toBe(300)
    expect(result.current.theme).toBe('custom')
    expect(result.current.expandedFolders.size).toBeGreaterThan(0)
    expect(result.current.collapsedFolders.size).toBe(0)
  })

  it('manages panel state correctly', () => {
    const { result } = renderHook(() => useLevaStore())

    act(() => {
      result.current.setPanelVisible(false)
    })
    expect(result.current.isVisible).toBe(false)

    act(() => {
      result.current.setPanelCollapsed(true)
    })
    expect(result.current.isCollapsed).toBe(true)

    act(() => {
      result.current.setPanelPosition('left')
    })
    expect(result.current.panelPosition).toBe('left')

    act(() => {
      result.current.setPanelWidth(400)
    })
    expect(result.current.panelWidth).toBe(400)

    // Test width constraints
    act(() => {
      result.current.setPanelWidth(100)
    })
    expect(result.current.panelWidth).toBe(200) // Should be clamped to minimum

    act(() => {
      result.current.setPanelWidth(1000)
    })
    expect(result.current.panelWidth).toBe(800) // Should be clamped to maximum
  })

  it('manages control state correctly', () => {
    const { result } = renderHook(() => useLevaStore())

    expect(result.current.activeControls).toHaveLength(0)

    act(() => {
      result.current.toggleControl('width')
    })
    expect(result.current.activeControls).toContain('width')

    act(() => {
      result.current.setControlEnabled('width', false)
    })
    expect(result.current.disabledControls.has('width')).toBe(true)

    act(() => {
      result.current.setControlVisible('width', false)
    })
    expect(result.current.hiddenControls.has('width')).toBe(true)

    expect(result.current.isControlEnabled('width')).toBe(false)
    expect(result.current.isControlVisible('width')).toBe(false)
    expect(result.current.getActiveControls()).toHaveLength(0)
    expect(result.current.getVisibleControls()).toHaveLength(0)
  })

  it('manages folder state correctly', () => {
    const { result } = renderHook(() => useLevaStore())

    expect(result.current.expandedFolders.has('environment')).toBe(true)

    act(() => {
      result.current.toggleFolder('environment')
    })
    expect(result.current.collapsedFolders.has('environment')).toBe(true)
    expect(result.current.expandedFolders.has('environment')).toBe(false)

    act(() => {
      result.current.setFolderCollapsed('environment', false)
    })
    expect(result.current.collapsedFolders.has('environment')).toBe(false)
    expect(result.current.expandedFolders.has('environment')).toBe(true)

    expect(result.current.isFolderCollapsed('environment')).toBe(false)
    expect(result.current.isFolderExpanded('environment')).toBe(true)
  })

  it('manages bulk folder operations', () => {
    const { result } = renderHook(() => useLevaStore())

    act(() => {
      result.current.expandAllFolders()
    })
    expect(result.current.collapsedFolders.size).toBe(0)

    act(() => {
      result.current.collapseAllFolders()
    })
    expect(result.current.expandedFolders.size).toBe(0)
    expect(result.current.collapsedFolders.size).toBeGreaterThan(0)
  })

  it('manages theme state correctly', () => {
    const { result } = renderHook(() => useLevaStore())

    act(() => {
      result.current.setTheme('dark')
    })
    expect(result.current.theme).toBe('dark')

    act(() => {
      result.current.setCustomTheme({ colors: { accent1: '#ff0000' } })
    })
    expect(result.current.customTheme.colors.accent1).toBe('#ff0000')
  })

  it('exports and imports settings correctly', () => {
    const { result } = renderHook(() => useLevaStore())

    act(() => {
      result.current.setPanelVisible(false)
      result.current.setPanelWidth(400)
      result.current.setTheme('dark')
      result.current.toggleControl('width')
    })

    const settings = result.current.exportSettings()

    expect(settings.isVisible).toBe(false)
    expect(settings.panelWidth).toBe(400)
    expect(settings.theme).toBe('dark')
    expect(settings.activeControls).toContain('width')

    // Reset and import
    act(() => {
      result.current.resetToDefaults()
    })

    expect(result.current.isVisible).toBe(true)
    expect(result.current.theme).toBe('custom')

    act(() => {
      result.current.importSettings(settings)
    })

    expect(result.current.isVisible).toBe(false)
    expect(result.current.panelWidth).toBe(400)
    expect(result.current.theme).toBe('dark')
    expect(result.current.activeControls).toContain('width')
  })

  it('returns correct current theme', () => {
    const { result } = renderHook(() => useLevaStore())

    const customTheme = result.current.getCurrentTheme()
    expect(customTheme.colors).toBeDefined()
    expect(customTheme.fonts).toBeDefined()
    expect(customTheme.radii).toBeDefined()
    expect(customTheme.space).toBeDefined()

    // Test with different theme
    act(() => {
      result.current.setTheme('dark')
    })

    const darkTheme = result.current.getCurrentTheme()
    expect(darkTheme.colors.elevation1).toBe('#1a1a1a')
  })
})