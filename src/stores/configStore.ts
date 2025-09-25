import { create } from 'zustand'
import { SimulationConfig, ConfigStore, ValidationError } from '@/types/config'
import { persistState, retrieveState } from './persistence'
import { useValidationStore } from './validationStore'

// Default configuration for initial state
const defaultConfig: SimulationConfig = {
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
  selectedSection: 'environment',
  expandedFolders: new Set(['environment', 'agents', 'learning', 'visualization']),
  validationErrors: [],
  history: [defaultConfig],
  historyIndex: 0,

  updateConfig: (path: string, value: any) => {
    const currentConfig = get().config

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

    set({
      config: newConfig,
      isDirty: true
    })
  },

  loadConfig: async (filePath: string) => {
    try {
      // This would be implemented with IPC service in a real app
      console.log('Loading config from:', filePath)

      // For now, just reset to default - in a real implementation, this would be:
      // const config = await ipcRenderer.invoke('config:load', filePath)
      await new Promise(resolve => setTimeout(resolve, 100)) // Simulate async operation

      set({
        config: defaultConfig,
        originalConfig: defaultConfig,
        isDirty: false,
        validationErrors: []
      })
    } catch (error) {
      console.error('Failed to load config:', error)
      throw error
    }
  },

  saveConfig: async (filePath?: string) => {
    try {
      // This would be implemented with IPC service in a real app
      console.log('Saving config to:', filePath || 'default location')

      // In a real implementation, this would be:
      // await ipcRenderer.invoke('config:save', get().config, filePath)
      await new Promise(resolve => setTimeout(resolve, 100)) // Simulate async operation

      set({
        originalConfig: get().config,
        isDirty: false
      })
    } catch (error) {
      console.error('Failed to save config:', error)
      throw error
    }
  },

  setComparison: (config: SimulationConfig | null) => {
    set({ compareConfig: config })
  },

  toggleSection: (section: string) => {
    const expandedFolders = new Set(get().expandedFolders)
    if (expandedFolders.has(section)) {
      expandedFolders.delete(section)
    } else {
      expandedFolders.add(section)
    }
    set({ expandedFolders })
  },

  validateConfig: () => {
    // Basic validation - in real implementation, this would use Zod schemas
    const config = get().config
    const errors: ValidationError[] = []

    // Simple validation rules
    if (config.width < 10 || config.width > 1000) {
      errors.push({
        path: 'width',
        message: 'Width must be between 10 and 1000',
        code: 'invalid_range'
      })
    }

    if (config.height < 10 || config.height > 1000) {
      errors.push({
        path: 'height',
        message: 'Height must be between 10 and 1000',
        code: 'invalid_range'
      })
    }

    // Validate agent ratios sum to 1
    const ratioSum = Object.values(config.agent_type_ratios).reduce((sum, ratio) => sum + ratio, 0)
    if (Math.abs(ratioSum - 1.0) > 0.001) {
      errors.push({
        path: 'agent_type_ratios',
        message: 'Agent type ratios must sum to 1.0',
        code: 'invalid_sum'
      })
    }

    set({ validationErrors: errors })
  },

  // Persistence methods for UI preferences
  persistUIState: () => {
    const state = get()
    const uiPreferences = {
      selectedSection: state.selectedSection,
      expandedFolders: Array.from(state.expandedFolders),
      showComparison: state.showComparison
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
      set({
        selectedSection: (persistedState as any).selectedSection || 'environment',
        expandedFolders: new Set((persistedState as any).expandedFolders || [
          'environment',
          'agents',
          'learning',
          'visualization'
        ]),
        showComparison: (persistedState as any).showComparison || false
      })
    }
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

  // Advanced features
  // Batch update multiple config values
  batchUpdateConfig: (updates: Array<{ path: string; value: any }>) => {
    const currentConfig = get().config
    const newConfig = { ...currentConfig }

    const updatePaths: string[] = []

    // Apply all updates
    for (const update of updates) {
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

    set({
      config: newConfig,
      isDirty: true
    })

    // Trigger validation for updated fields
    updatePaths.forEach(path => {
      useValidationStore.getState().validateField(path, getNestedValue(newConfig, path))
    })
  },

  // Undo/redo functionality
  undo: () => {
    const { history, historyIndex } = get()
    if (historyIndex > 0) {
      const previousConfig = history[historyIndex - 1]
      set({
        config: previousConfig,
        historyIndex: historyIndex - 1,
        isDirty: true
      })

      // Trigger validation
      get().validateConfig()
    }
  },

  redo: () => {
    const { history, historyIndex } = get()
    if (historyIndex < history.length - 1) {
      const nextConfig = history[historyIndex + 1]
      set({
        config: nextConfig,
        historyIndex: historyIndex + 1,
        isDirty: true
      })

      // Trigger validation
      get().validateConfig()
    }
  },

  // Reset to default configuration
  resetToDefaults: () => {
    set({
      config: defaultConfig,
      originalConfig: defaultConfig,
      isDirty: false,
      validationErrors: []
    })
    useValidationStore.getState().clearErrors()
  },

  // Export configuration as JSON
  exportConfig: () => {
    const { config } = get()
    return JSON.stringify(config, null, 2)
  },

  // Import configuration from JSON
  importConfig: (configJson: string) => {
    try {
      const importedConfig = JSON.parse(configJson)

      // Basic validation
      if (typeof importedConfig !== 'object' || importedConfig === null) {
        throw new Error('Invalid configuration format')
      }

      set({
        config: { ...defaultConfig, ...importedConfig },
        isDirty: true
      })

      // Trigger validation
      get().validateConfig()

      return { success: true }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Import failed'
      }
    }
  },

  // Get configuration diff
  getConfigDiff: () => {
    const { config, originalConfig } = get()
    const diff: Record<string, { original: any; current: any; changed: boolean }> = {}

    // Simple diff - in real implementation, use a proper diff library
    const allKeys = new Set([
      ...getAllKeys(originalConfig),
      ...getAllKeys(config)
    ])

    allKeys.forEach(key => {
      const originalValue = getNestedValue(originalConfig, key)
      const currentValue = getNestedValue(config, key)
      const changed = JSON.stringify(originalValue) !== JSON.stringify(currentValue)

      if (changed) {
        diff[key] = { original: originalValue, current: currentValue, changed: true }
      }
    })

    return diff
  },

  // Enhanced validation with field-specific feedback
  validateField: async (path: string, value: any) => {
    const validationStore = useValidationStore.getState()

    // Clear existing errors for this field
    validationStore.clearFieldErrors(path)

    try {
      // Basic field validation
      const errors = validateSingleField(path, value)

      if (errors.length > 0) {
        validationStore.addErrors(errors)
      } else {
        // Clear any existing success state
        validationStore.clearFieldErrors(path)
      }

      return { success: errors.length === 0, errors }
    } catch (error) {
      const validationError: ValidationError = {
        path,
        message: error instanceof Error ? error.message : 'Validation failed',
        code: 'validation_error'
      }

      validationStore.addError(validationError)
      return { success: false, errors: [validationError] }
    }
  }
}))

// Helper functions for advanced features
function getNestedValue(obj: any, path: string): any {
  const keys = path.split('.')
  let current = obj

  for (const key of keys) {
    if (current === null || current === undefined) {
      return undefined
    }
    current = current[key]
  }

  return current
}

function getAllKeys(obj: any, prefix = ''): string[] {
  const keys: string[] = []

  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key

    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      keys.push(...getAllKeys(value, fullKey))
    } else {
      keys.push(fullKey)
    }
  }

  return keys
}

function validateSingleField(path: string, value: any): ValidationError[] {
  const errors: ValidationError[] = []

  // Field-specific validation rules
  switch (path) {
    case 'width':
    case 'height':
      if (typeof value !== 'number' || value < 10 || value > 1000) {
        errors.push({
          path,
          message: `${path} must be a number between 10 and 1000`,
          code: 'invalid_range'
        })
      }
      break

    case 'system_agents':
    case 'independent_agents':
    case 'control_agents':
      if (typeof value !== 'number' || value < 0 || value > 10000) {
        errors.push({
          path,
          message: `${path} must be a number between 0 and 10000`,
          code: 'invalid_range'
        })
      }
      break

    case 'learning_rate':
      if (typeof value !== 'number' || value <= 0 || value > 1) {
        errors.push({
          path,
          message: 'learning_rate must be a number between 0 and 1',
          code: 'invalid_range'
        })
      }
      break

    case 'epsilon_start':
    case 'epsilon_min':
      if (typeof value !== 'number' || value < 0 || value > 1) {
        errors.push({
          path,
          message: `${path} must be a number between 0 and 1`,
          code: 'invalid_range'
        })
      }
      break

    case 'epsilon_decay':
      if (typeof value !== 'number' || value <= 0 || value > 1) {
        errors.push({
          path,
          message: 'epsilon_decay must be a number between 0 and 1',
          code: 'invalid_range'
        })
      }
      break
  }

  return errors
}