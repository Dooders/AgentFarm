import { SimulationConfigType, AgentParameterType, VisualizationConfigType, AgentTypeRatiosType, ModuleParameterType, ValidationError } from './validation'
import { SimulationConfig } from './validation'

// Use Zod-inferred types for type safety
export type SimulationConfigType = SimulationConfigType

// Use Zod-inferred types for type safety
export type AgentParameters = AgentParameterType
export type ModuleParameters = ModuleParameterType
export type VisualizationConfig = VisualizationConfigType

// Re-export types for convenience
export type {
  SimulationConfigType,
  AgentParameterType,
  ModuleParameterType,
  VisualizationConfigType,
  AgentTypeRatiosType
} from './validation'

// ConfigStore interface definition
export interface ConfigStore {
  // Configuration state
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

  // Layout state
  leftPanelWidth: number
  rightPanelWidth: number

  // Actions
  updateConfig: (path: string, value: any) => void
  loadConfig: (filePath: string) => Promise<void>
  saveConfig: (filePath?: string) => Promise<void>
  setComparison: (config: SimulationConfig | null) => void
  toggleSection: (section: string) => void
  validateConfig: () => void

  // Persistence actions
  persistUIState: () => void
  restoreUIState: () => void
  clearPersistedState: () => void

  // Layout actions
  setPanelWidths: (leftWidth: number, rightWidth: number) => void
  resetPanelWidths: () => void

  // Advanced features
  batchUpdateConfig: (updates: Array<{ path: string; value: any }>) => void
  undo: () => void
  redo: () => void
  resetToDefaults: () => void
  exportConfig: () => string
  importConfig: (configJson: string) => { success: boolean; error?: string }
  getConfigDiff: () => Record<string, { original: any; current: any; changed: boolean }>
  validateField: (path: string, value: any) => any
}