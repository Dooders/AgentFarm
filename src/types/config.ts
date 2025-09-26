import { SimulationConfigType, AgentParameterType, ModuleParameterType, VisualizationConfigType, ValidationError } from './validation'

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
  config: SimulationConfigType
  originalConfig: SimulationConfigType
  isDirty: boolean
  compareConfig: SimulationConfigType | null
  showComparison: boolean
  selectedSection: string
  expandedFolders: Set<string>
  validationErrors: ValidationError[]
  history: SimulationConfigType[]
  historyIndex: number

  // Layout state
  leftPanelWidth: number
  rightPanelWidth: number

  // Actions
  updateConfig: (path: string, value: any) => void
  loadConfig: (filePath: string) => Promise<void>
  saveConfig: (filePath?: string) => Promise<void>
  setComparison: (config: SimulationConfigType | null) => void
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