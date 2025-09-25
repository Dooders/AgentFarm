// Base configuration interface (matching the Zod schema structure)
export interface SimulationConfig {
  // Environment settings
  width: number
  height: number
  position_discretization_method: 'floor' | 'round' | 'ceil'
  use_bilinear_interpolation: boolean

  // Agent settings
  system_agents: number
  independent_agents: number
  control_agents: number
  agent_type_ratios: {
    SystemAgent: number
    IndependentAgent: number
    ControlAgent: number
  }

  // Learning parameters
  learning_rate: number
  epsilon_start: number
  epsilon_min: number
  epsilon_decay: number

  // Agent parameters
  agent_parameters: {
    SystemAgent: AgentParameters
    IndependentAgent: AgentParameters
    ControlAgent: AgentParameters
  }

  // Visualization
  visualization: VisualizationConfig

  // Module parameters
  gather_parameters: ModuleParameters
  share_parameters: ModuleParameters
  move_parameters: ModuleParameters
  attack_parameters: ModuleParameters
}

// Agent parameters interface
export interface AgentParameters {
  target_update_freq: number
  memory_size: number
  learning_rate: number
  gamma: number
  epsilon_start: number
  epsilon_min: number
  epsilon_decay: number
  dqn_hidden_size: number
  batch_size: number
  tau: number
  success_reward: number
  failure_penalty: number
  base_cost: number
}

// Module parameters interface
export interface ModuleParameters {
  target_update_freq: number
  memory_size: number
  learning_rate: number
  gamma: number
  epsilon_start: number
  epsilon_min: number
  epsilon_decay: number
  dqn_hidden_size: number
  batch_size: number
  tau: number
  success_reward: number
  failure_penalty: number
  base_cost: number
}

// Visualization configuration interface
export interface VisualizationConfig {
  canvas_width: number
  canvas_height: number
  background_color: string
  agent_colors: {
    SystemAgent: string
    IndependentAgent: string
    ControlAgent: string
  }
  show_metrics: boolean
  font_size: number
  line_width: number
}

// Validation error interface
export interface ValidationError {
  path: string
  message: string
  code: string
}

// Store state interfaces
export interface ConfigStore {
  config: SimulationConfig
  originalConfig: SimulationConfig
  isDirty: boolean
  compareConfig: SimulationConfig | null
  showComparison: boolean
  selectedSection: string
  expandedFolders: Set<string>
  validationErrors: ValidationError[]
  history: SimulationConfig[]
  historyIndex: number

  // Basic actions
  updateConfig: (path: string, value: any) => void
  loadConfig: (filePath: string) => Promise<void>
  saveConfig: (filePath?: string) => Promise<void>
  setComparison: (config: SimulationConfig | null) => void
  toggleSection: (section: string) => void
  validateConfig: () => void

  // Persistence
  persistUIState: () => void
  restoreUIState: () => void
  clearPersistedState: () => void

  // Advanced features
  batchUpdateConfig: (updates: Array<{ path: string; value: any }>) => void
  undo: () => void
  redo: () => void
  resetToDefaults: () => void
  exportConfig: () => string
  importConfig: (configJson: string) => { success: boolean; error?: string }
  getConfigDiff: () => Record<string, { original: any; current: any; changed: boolean }>
  validateField: (path: string, value: any) => { success: boolean; errors: ValidationError[] }
}