import { create } from 'zustand'
import { SimulationConfig, ConfigStore, ValidationError } from '@/types/config'

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
  }
}))