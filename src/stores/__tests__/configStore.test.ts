import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useConfigStore } from '../configStore'
import { useValidationStore } from '../validationStore'

describe('ConfigStore', () => {
  beforeEach(() => {
    // Reset the store before each test
    useConfigStore.getState().config = useConfigStore.getState().originalConfig
    useConfigStore.getState().isDirty = false
    useConfigStore.getState().validationErrors = []
  })

  it('initializes with default config', () => {
    const { result } = renderHook(() => useConfigStore())

    expect(result.current.config.width).toBe(100)
    expect(result.current.config.height).toBe(100)
    expect(result.current.config.system_agents).toBe(20)
    expect(result.current.isDirty).toBe(false)
  })

  it('updates config and marks as dirty', () => {
    const { result } = renderHook(() => useConfigStore())

    act(() => {
      result.current.updateConfig('width', 200)
    })

    expect(result.current.config.width).toBe(200)
    expect(result.current.isDirty).toBe(true)
  })

  it('updates nested config properties', () => {
    const { result } = renderHook(() => useConfigStore())

    act(() => {
      result.current.updateConfig('agent_type_ratios.SystemAgent', 0.5)
    })

    expect(result.current.config.agent_type_ratios.SystemAgent).toBe(0.5)
    expect(result.current.isDirty).toBe(true)
  })

  it('toggles sections in expanded folders', () => {
    const { result } = renderHook(() => useConfigStore())

    expect(result.current.expandedFolders.has('environment')).toBe(true)

    act(() => {
      result.current.toggleSection('environment')
    })

    expect(result.current.expandedFolders.has('environment')).toBe(false)
  })

  it('adds new sections to expanded folders', () => {
    const { result } = renderHook(() => useConfigStore())

    expect(result.current.expandedFolders.has('new_section')).toBe(false)

    act(() => {
      result.current.toggleSection('new_section')
    })

    expect(result.current.expandedFolders.has('new_section')).toBe(true)
  })

  it('validates configuration and sets errors', () => {
    const { result } = renderHook(() => useConfigStore())
    const validationStore = renderHook(() => useValidationStore())

    // Set invalid width
    act(() => {
      result.current.updateConfig('width', 2000) // Too large
      result.current.validateConfig()
    })

    // Check that validation errors are in the validation store
    expect(validationStore.result.current.errors.length).toBeGreaterThan(0)
    expect(validationStore.result.current.errors[0].path).toBe('width')
  })

  it('loads config and resets state', async () => {
    const { result } = renderHook(() => useConfigStore())

    // Mark as dirty first
    act(() => {
      result.current.updateConfig('width', 150)
    })

    expect(result.current.isDirty).toBe(true)

    // Load config (should reset to defaults)
    await act(async () => {
      await result.current.loadConfig('test.json')
    })

    expect(result.current.config.width).toBe(100)
    expect(result.current.isDirty).toBe(false)
  })

  it('saves config and marks as clean', async () => {
    const { result } = renderHook(() => useConfigStore())

    act(() => {
      result.current.updateConfig('width', 150)
    })

    expect(result.current.isDirty).toBe(true)

    // Save config (should mark as clean)
    await act(async () => {
      await result.current.saveConfig('test.json')
    })

    expect(result.current.isDirty).toBe(false)
  })

  it('sets comparison config', () => {
    const { result } = renderHook(() => useConfigStore())

    const comparisonConfig = {
      ...result.current.config,
      width: 200,
      height: 200
    }

    act(() => {
      result.current.setComparison(comparisonConfig)
    })

    expect(result.current.compareConfig?.width).toBe(200)
    expect(result.current.compareConfig?.height).toBe(200)
  })
})