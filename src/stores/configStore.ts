import { create } from 'zustand'
import { SimulationConfigType, ConfigStore, ConfigSection, BatchConfigUpdate } from '@/types/config'
import { persistState, retrieveState } from './persistence'
import { useValidationStore } from './validationStore'
import { validationService } from '@/services/validationService'
import { ipcService } from '@/services/ipcService'

// Default configuration for initial state
const defaultConfig: SimulationConfigType = {
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
    ControlAgent: 0.2
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
      base_cost: 0.01
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
      base_cost: 0.01
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
      base_cost: 0.01
    }
  },
  visualization: {
    canvas_width: 800,
    canvas_height: 600,
    background_color: '#000000',
    agent_colors: {
      SystemAgent: '#ff6b6b',
      IndependentAgent: '#4ecdc4',
      ControlAgent: '#45b7d1'
    },
    show_metrics: true,
    font_size: 12,
    line_width: 1
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
    base_cost: 0.01
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
    base_cost: 0.01
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
    base_cost: 0.01
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
    base_cost: 0.01
  }
}

export const useConfigStore = create<ConfigStore>((set, get) => ({
  config: defaultConfig,
  originalConfig: defaultConfig,
  isDirty: false,
  compareConfig: null,
  showComparison: false,
  comparisonFilePath: undefined,
  selectedSection: 'environment',
  expandedFolders: new Set(['environment', 'agents', 'learning', 'visualization']),
  validationErrors: [],
  history: [{
    id: 'initial',
    config: defaultConfig,
    timestamp: Date.now(),
    action: 'create',
    description: 'Initial configuration'
  }],
  historyIndex: 0,

  // Layout state - default to 50/50 split for balanced desktop experience
  leftPanelWidth: 0.5,
  rightPanelWidth: 0.5,
  // Arbitrary layout sizes persistence map (percentages per layout key)
  layoutSizes: {},

  updateConfig: (path: string, value: any) => {
    const currentConfig = get().config
    const { history, historyIndex } = get()

    // Simple path-based update (in a real implementation, this would use a library like lodash.set)
    const keys = path.split('.')
    const newConfig = { ...currentConfig }

    let target = newConfig as any
    for (let i = 0; i < keys.length - 1; i++) {
      if (!target[keys[i]]) {
        target[keys[i]] = {}
      }
      target = target[keys[i]]
    }
    target[keys[keys.length - 1]] = value

    // Update history - truncate future history if we're not at the end
    const newHistory = history.slice(0, historyIndex + 1)
    newHistory.push({
      id: 'update',
      config: newConfig,
      timestamp: Date.now(),
      action: 'update',
      description: 'Configuration updated'
    })

    set({
      config: newConfig,
      isDirty: true,
      history: newHistory,
      historyIndex: historyIndex + 1
    })
  },

  loadConfig: async (filePath: string) => {
    try {
      console.log('Loading config from:', filePath)

      // Load configuration using IPC service
      const result = await ipcService.loadConfig({ filePath })

      set({
        config: result.config,
        originalConfig: result.config,
        isDirty: false,
        validationErrors: [],
        history: [{
          id: 'loaded',
          config: result.config,
          timestamp: Date.now(),
          action: 'load',
          description: 'Configuration loaded from file'
        }],
        historyIndex: 0
      })

      // Trigger validation
      await get().validateConfig()
    } catch (error) {
      console.error('Failed to load config:', error)
      throw error
    }
  },

  saveConfig: async (filePath?: string) => {
    try {
      console.log('Saving config to:', filePath || 'default location')

      // Save configuration using IPC service
      await ipcService.saveConfig({
        config: get().config,
        filePath,
        format: 'json',
        backup: true
      })

      set({
        originalConfig: get().config,
        isDirty: false,
        history: [{
          id: 'reset',
          config: get().config,
          timestamp: Date.now(),
          action: 'reset',
          description: 'Configuration reset to defaults'
        }],
        historyIndex: 0
      })

      return // Return void as expected by interface
    } catch (error) {
      console.error('Failed to save config:', error)
      throw error
    }
  },

  setComparison: (config: SimulationConfigType | null) => {
    set({ compareConfig: config })
  },

  toggleComparison: () => {
    const current = get().showComparison
    set({ showComparison: !current })
    const persistUI = get().persistUIState
    persistUI()
  },

  clearComparison: () => {
    set({ compareConfig: null, comparisonFilePath: undefined })
    const persistUI = get().persistUIState
    persistUI()
  },

  setComparisonPath: (path?: string) => {
    set({ comparisonFilePath: path })
    const persistUI = get().persistUIState
    persistUI()
  },

  toggleSection: (section: ConfigSection) => {
    const expandedFolders = new Set(get().expandedFolders)
    if (expandedFolders.has(section)) {
      expandedFolders.delete(section)
    } else {
      expandedFolders.add(section)
    }
    set({ expandedFolders })
  },

  validateConfig: async () => {
    try {
      const config = get().config
      const result = await ipcService.validateConfig({
        config,
        partial: false,
        rules: ['population_positive', 'world_size_valid']
      })

      // Update validation store with results
      useValidationStore.getState().setValidationResult(result)
    } catch (error) {
      console.error('Failed to validate config:', error)
      // Fallback to local validation
      const config = get().config
      const result = await validationService.validateConfig(config)
      useValidationStore.getState().setValidationResult(result)
    }
  },

  // Persistence methods for UI preferences
  persistUIState: () => {
    const state = get()
    const uiPreferences = {
      selectedSection: state.selectedSection,
      expandedFolders: Array.from(state.expandedFolders),
      showComparison: state.showComparison,
      comparisonFilePath: state.comparisonFilePath,
      leftPanelWidth: state.leftPanelWidth,
      rightPanelWidth: state.rightPanelWidth,
      layoutSizes: state.layoutSizes
    }

    persistState(uiPreferences, {
      name: 'config-ui-preferences',
      version: 1,
      onError: (error) => console.error('Failed to persist UI preferences:', error)
    })
  },

  restoreUIState: () => {
    const persistedState = retrieveState({
      name: 'config-ui-preferences',
      version: 1,
      onError: (error) => console.warn('Failed to restore UI preferences:', error)
    })

    if (persistedState && typeof persistedState === 'object' && persistedState !== null) {
      const state = persistedState as Record<string, unknown>
      const allowedSections: ConfigSection[] = [
        'environment',
        'agents',
        'learning',
        'agent_parameters',
        'modules',
        'visualization'
      ]
      const maybeSelected = state.selectedSection as string | undefined
      const selectedSectionValid = maybeSelected && (allowedSections as readonly string[]).includes(maybeSelected)
        ? (maybeSelected as ConfigSection)
        : 'environment'
      const maybeExpanded = (state.expandedFolders as string[] | undefined) || undefined
      const expandedFoldersValid = new Set<ConfigSection>(
        Array.isArray(maybeExpanded)
          ? (maybeExpanded.filter(s => (allowedSections as readonly string[]).includes(s)) as ConfigSection[])
          : ['environment', 'agents', 'learning', 'visualization']
      )
      set({
        selectedSection: selectedSectionValid,
        expandedFolders: expandedFoldersValid,
        showComparison: (state.showComparison as boolean) || false,
        comparisonFilePath: (state.comparisonFilePath as string | undefined) || undefined,
        leftPanelWidth: (state.leftPanelWidth as number) || 0.5,
        rightPanelWidth: (state.rightPanelWidth as number) || 0.5,
        layoutSizes: (state.layoutSizes as Record<string, number[]>) || {}
      })
    }
  },

  // Layout actions
  setPanelWidths: (leftWidth: number, rightWidth: number) => {
    const total = leftWidth + rightWidth
    const normalizedLeft = leftWidth / total
    const normalizedRight = rightWidth / total

    set({
      leftPanelWidth: normalizedLeft,
      rightPanelWidth: normalizedRight
    })

    // Persist the updated layout state
    get().persistUIState()
  },

  resetPanelWidths: () => {
    set({
      leftPanelWidth: 0.5,
      rightPanelWidth: 0.5
    })

    // Persist the reset layout state
    get().persistUIState()
  },

  // Clear persisted state
  clearPersistedState: () => {
    try {
      // Clear UI preferences
      const storage = typeof window !== 'undefined' && window.localStorage
        ? window.localStorage
        : null

      if (storage) {
        storage.removeItem('config-ui-preferences')
      }
    } catch (error) {
      console.warn('Failed to clear persisted state:', error)
    }
  },

  // Generic layout persistence helpers
  setLayoutSizes: (key: string, sizes: number[]) => {
    const current = get().layoutSizes
    const next = { ...current, [key]: sizes }
    set({ layoutSizes: next })
    get().persistUIState()
  },
  getLayoutSizes: (key: string) => {
    const current = get().layoutSizes
    return current ? current[key] : undefined
  },
  resetLayoutSizes: (key?: string) => {
    if (!key) {
      set({ layoutSizes: {} })
    } else {
      const current = { ...get().layoutSizes }
      delete current[key]
      set({ layoutSizes: current })
    }
    get().persistUIState()
  },

  // Advanced features
  // Batch update multiple config values
  batchUpdateConfig: (updates: BatchConfigUpdate) => {
    const currentConfig = get().config
    const { history, historyIndex } = get()
    const newConfig = { ...currentConfig }

    const updatePaths: string[] = []

    // Apply all updates
    for (const update of updates.updates) {
      const keys = update.path.split('.')
      let currentTarget = newConfig as any

      for (let i = 0; i < keys.length - 1; i++) {
        if (!currentTarget[keys[i]]) {
          currentTarget[keys[i]] = {}
        }
        currentTarget = currentTarget[keys[i]]
      }
      currentTarget[keys[keys.length - 1]] = update.value
      updatePaths.push(update.path)
    }

    // Update history - truncate future history if we're not at the end
    const newHistory = history.slice(0, historyIndex + 1)
    newHistory.push({
      id: 'update',
      config: newConfig,
      timestamp: Date.now(),
      action: 'update',
      description: 'Configuration updated'
    })

    set({
      config: newConfig,
      isDirty: true,
      history: newHistory,
      historyIndex: historyIndex + 1
    })

    // Trigger validation for updated fields
    updatePaths.forEach(path => {
      const result = validationService.validateField(path, getNestedValue(newConfig, path))
      if (result.errors.length > 0) {
        useValidationStore.getState().addErrors(result.errors)
      }
    })
  },

  // Undo/redo functionality
  undo: async () => {
    const { history, historyIndex } = get()
    if (historyIndex > 0) {
      const historyEntry = history[historyIndex - 1]
      set({
        config: historyEntry.config,
        historyIndex: historyIndex - 1,
        isDirty: true
      })

      // Trigger validation
      await get().validateConfig()
    }
  },

  redo: async () => {
    const { history, historyIndex } = get()
    if (historyIndex < history.length - 1) {
      const historyEntry = history[historyIndex + 1]
      set({
        config: historyEntry.config,
        historyIndex: historyIndex + 1,
        isDirty: true
      })

      // Trigger validation
      await get().validateConfig()
    }
  },

  // Reset to default configuration
  resetToDefaults: () => {
    set({
      config: defaultConfig,
      originalConfig: defaultConfig,
      isDirty: false,
      validationErrors: [],
      history: [{
        id: 'reset',
        config: defaultConfig,
        timestamp: Date.now(),
        action: 'reset',
        description: 'Configuration reset to defaults'
      }],
      historyIndex: 0
    })
    useValidationStore.getState().clearErrors()
  },

  // Export configuration metadata
  exportConfig: (format: 'json' | 'yaml' | 'xml' = 'json') => {
    const { config } = get()
    return {
      config,
      metadata: {
        version: '1.0.0',
        lastModified: Date.now(),
        created: Date.now()
      },
      exportFormat: format,
      exportVersion: '1.0'
    }
  },

  // Import configuration from JSON
  importConfig: async (configJson: string, format: 'json' | 'yaml' | 'xml' = 'json') => {
    try {
      const result = await ipcService.importConfig({
        content: configJson,
        format,
        validate: true,
        merge: false
      })

      set({
        config: result.config,
        isDirty: true
      })

      // Trigger validation
      await get().validateConfig()

      return { success: true }
    } catch (error) {
      console.error('Failed to import config:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Import failed'
      }
    }
  },

  // Get configuration diff (against original)
  getConfigDiff: () => {
    const { config, originalConfig } = get()
    const diff: Record<string, { original: any; current: any; changed: boolean }> = {}

    // Simple diff - in real implementation, use a proper diff library
    const allKeys = new Set([
      ...getAllKeys(originalConfig as unknown as Record<string, unknown>),
      ...getAllKeys(config as unknown as Record<string, unknown>)
    ])

    allKeys.forEach(key => {
      const originalValue = getNestedValue(originalConfig as unknown as Record<string, unknown>, key)
      const currentValue = getNestedValue(config as unknown as Record<string, unknown>, key)
      const changed = !deepEqual(originalValue, currentValue)

      if (changed) {
        diff[key] = { original: originalValue, current: currentValue, changed: true }
      }
    })

    return diff
  },

  // Get diff between current config and comparison config (added/removed/changed/unchanged)
  // Semantics:
  // - added: key exists in comparison but not in current (would be added to current)
  // - removed: key exists in current but not in comparison (would be removed from current)
  // - changed: key exists in both but values differ (current -> comparison)
  getComparisonDiff: () => {
    const { config, compareConfig } = get()
    const result = {
      added: {} as Record<string, unknown>,
      removed: {} as Record<string, unknown>,
      changed: {} as Record<string, { from: unknown; to: unknown }>,
      unchanged: {} as Record<string, unknown>
    }

    if (!compareConfig) return result

    const allKeys = new Set([
      ...getAllKeys(compareConfig as unknown as Record<string, unknown>),
      ...getAllKeys(config as unknown as Record<string, unknown>)
    ])

    allKeys.forEach(key => {
      const compareValue = getNestedValue(compareConfig as unknown as Record<string, unknown>, key)
      const currentValue = getNestedValue(config as unknown as Record<string, unknown>, key)

      const hasInCompare = compareValue !== undefined
      const hasInCurrent = currentValue !== undefined

      if (hasInCompare && !hasInCurrent) {
        ;(result.added as Record<string, unknown>)[key] = compareValue
      } else if (!hasInCompare && hasInCurrent) {
        ;(result.removed as Record<string, unknown>)[key] = currentValue
      } else if (hasInCompare && hasInCurrent) {
        const equal = deepEqual(compareValue, currentValue)
        if (!equal) {
          ;(result.changed as Record<string, { from: unknown; to: unknown }>)[key] = { from: currentValue, to: compareValue }
        } else {
          ;(result.unchanged as Record<string, unknown>)[key] = currentValue
        }
      }
    })

    return result
  },

  // Copy a field value from comparison into current config
  copyFromComparison: (path: string) => {
    const { compareConfig } = get()
    if (!compareConfig) return false
    const value = getNestedValue(compareConfig as unknown as Record<string, unknown>, path)
    if (value === undefined) return false
    get().updateConfig(path, value)
    return true
  },

  // Batch copy multiple paths from comparison
  batchCopyFromComparison: (paths: string[]) => {
    const { compareConfig } = get()
    if (!compareConfig || paths.length === 0) return false
    const updates = paths
      .map(path => ({ path, value: getNestedValue(compareConfig as unknown as Record<string, unknown>, path) }))
      .filter(update => update.value !== undefined)
    if (updates.length === 0) return false
    get().batchUpdateConfig({ updates })
    return true
  },

  // Remove a config path from current config (used for aligning deletions)
  removeConfigPath: (path: string) => {
    const currentConfig = get().config
    const { history, historyIndex } = get()
    const before = getNestedValue(currentConfig as unknown as Record<string, unknown>, path)
    if (before === undefined) {
      return
    }
    const newConfig = deepClone(currentConfig)
    deleteNestedPath(newConfig as unknown as Record<string, unknown>, path)

    // If nothing changed, bail
    const after = getNestedValue(newConfig as unknown as Record<string, unknown>, path)
    if (after !== undefined) {
      return
    }

    const newHistory = history.slice(0, historyIndex + 1)
    newHistory.push({
      id: 'update',
      config: newConfig,
      timestamp: Date.now(),
      action: 'update',
      description: `Removed ${path}`
    })

    set({
      config: newConfig,
      isDirty: true,
      history: newHistory,
      historyIndex: historyIndex + 1
    })
  },

  // Apply all differences from comparison into current config
  applyAllDifferencesFromComparison: () => {
    const { config, compareConfig, history, historyIndex } = get()
    if (!compareConfig) return
    const diff = get().getComparisonDiff()

    let newConfig = deepClone(config)

    // Apply additions and changes
    const toSet = [
      ...Object.keys(diff.added),
      ...Object.keys(diff.changed)
    ]
    toSet.forEach(path => {
      const value = getNestedValue(compareConfig as unknown as Record<string, unknown>, path)
      setNestedValue(newConfig as unknown as Record<string, unknown>, path, deepClone(value))
    })

    // Apply removals
    Object.keys(diff.removed).forEach(path => {
      deleteNestedPath(newConfig as unknown as Record<string, unknown>, path)
    })

    // If no actual change, bail
    if (deepEqual(config, newConfig)) {
      return
    }

    const newHistory = history.slice(0, historyIndex + 1)
    newHistory.push({
      id: 'update',
      config: newConfig,
      timestamp: Date.now(),
      action: 'update',
      description: 'Applied all differences from comparison'
    })

    set({
      config: newConfig,
      isDirty: true,
      history: newHistory,
      historyIndex: historyIndex + 1
    })
  },

  // Enhanced validation with field-specific feedback
  validateField: async (path: string, value: any) => {
    const result = validationService.validateField(path, value)

    // Update validation store with results
    if (result.errors.length > 0) {
      useValidationStore.getState().addErrors(result.errors)
    } else {
      useValidationStore.getState().clearFieldErrors(path)
    }

    return result.errors
  }
}) as unknown as ConfigStore)

// Helper functions for advanced features
function getNestedValue<T extends Record<string, unknown>>(obj: T, path: string): unknown {
  const keys = path.split('.')
  let current: unknown = obj

  for (const key of keys) {
    if (current === null || current === undefined || typeof current !== 'object') {
      return undefined
    }
    current = (current as Record<string, unknown>)[key]
  }

  return current
}

function getAllKeys(obj: Record<string, unknown>, prefix = ''): string[] {
  const keys: string[] = []

  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key

    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      keys.push(...getAllKeys(value as Record<string, unknown>, fullKey))
    } else {
      keys.push(fullKey)
    }
  }

  return keys
}

function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): void {
  const parts = path.split('.')
  let current: Record<string, unknown> = obj
  for (let i = 0; i < parts.length - 1; i++) {
    const key = parts[i]
    const next = current[key]
    if (typeof next !== 'object' || next === null) {
      current[key] = {}
    }
    current = current[key] as Record<string, unknown>
  }
  current[parts[parts.length - 1]] = value as unknown as never
}

function deleteNestedPath(obj: Record<string, unknown>, path: string): void {
  const parts = path.split('.')
  let current: Record<string, unknown> | undefined = obj
  for (let i = 0; i < parts.length - 1; i++) {
    const key = parts[i]
    const next = current[key]
    if (typeof next !== 'object' || next === null) return
    current = next as Record<string, unknown>
  }
  const lastKey = parts[parts.length - 1]
  if (current && Object.prototype.hasOwnProperty.call(current, lastKey)) {
    delete current[lastKey]
  }
}

function deepClone<T>(value: T): T {
  try {
    return JSON.parse(JSON.stringify(value)) as T
  } catch {
    // Fall back to a shallow clone for plain objects/arrays; otherwise throw
    if (Array.isArray(value)) {
      return (value.slice() as unknown) as T
    }
    if (value && typeof value === 'object') {
      return ({ ...(value as unknown as Record<string, unknown>) } as unknown) as T
    }
    return value
  }
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== typeof b) return false
  if (a === null || b === null) return a === b
  if (typeof a !== 'object' || typeof b !== 'object') return a === b

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false
    }
    return true
  }

  if (Array.isArray(a) || Array.isArray(b)) return false

  const aObj = a as Record<string, unknown>
  const bObj = b as Record<string, unknown>
  const aKeys = Object.keys(aObj)
  const bKeys = Object.keys(bObj)
  if (aKeys.length !== bKeys.length) return false
  for (const key of aKeys) {
    if (!Object.prototype.hasOwnProperty.call(bObj, key)) return false
    if (!deepEqual(aObj[key], bObj[key])) return false
  }
  return true
}
