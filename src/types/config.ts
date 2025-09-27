import { SimulationConfigType, AgentParameterType, ModuleParameterType, VisualizationConfigType, ValidationError } from './validation'

// Base types for configuration system
export type AgentType = 'SystemAgent' | 'IndependentAgent' | 'ControlAgent'
export type PositionDiscretizationMethod = 'floor' | 'round' | 'ceil'

// Configuration section identifiers
export type ConfigSection =
  | 'environment'
  | 'agents'
  | 'learning'
  | 'agent_parameters'
  | 'modules'
  | 'visualization'

// Configuration path type for nested updates
export type ConfigPath = string

// Configuration update operation
export interface ConfigUpdate {
  path: ConfigPath
  value: any
  timestamp?: number
}

// Batch configuration update
export interface BatchConfigUpdate {
  updates: ConfigUpdate[]
  description?: string
}

// Use Zod-inferred types for type safety
export type AgentParameters = AgentParameterType
export type ModuleParameters = ModuleParameterType
export type VisualizationConfig = VisualizationConfigType

// Re-export types for convenience
export type {
  SimulationConfigType,
  AgentParameterType,
  ModuleParameterType,
  VisualizationConfigType
} from './validation'

// Extended configuration interfaces with additional metadata
export interface ConfigurationMetadata {
  version: string
  lastModified: number
  created: number
  author?: string
  description?: string
  tags?: string[]
}

// Configuration with metadata
export interface ConfigWithMetadata extends SimulationConfigType {
  metadata: ConfigurationMetadata
}

// Configuration template for creating new configs
export interface ConfigTemplate {
  name: string
  description: string
  category: string
  baseConfig: Partial<SimulationConfigType>
  tags: string[]
}

// Configuration comparison result
export interface ConfigComparison {
  added: Record<string, any>
  removed: Record<string, any>
  changed: Record<string, { from: any; to: any }>
  unchanged: Record<string, any>
}

// Configuration validation context
export interface ConfigValidationContext {
  config: SimulationConfigType
  path?: ConfigPath
  parent?: any
  siblings?: Record<string, any>
}

// Configuration export/import types
export interface ConfigExport {
  config: SimulationConfigType
  metadata: ConfigurationMetadata
  exportFormat: 'json' | 'yaml' | 'xml'
  exportVersion: string
}

export interface ConfigImportResult {
  success: boolean
  config?: SimulationConfigType
  metadata?: ConfigurationMetadata
  errors?: string[]
  warnings?: string[]
}

// Configuration history entry
export interface ConfigHistoryEntry {
  id: string
  config: SimulationConfigType
  timestamp: number
  action: 'create' | 'update' | 'save' | 'load' | 'reset'
  description?: string
  metadata?: Record<string, any>
}

// Store state interfaces
export interface ConfigState {
  // Configuration state
  config: SimulationConfigType
  originalConfig: SimulationConfigType
  isDirty: boolean
  currentFilePath?: string
  lastSaveTime?: number
  lastLoadTime?: number
  compareConfig: SimulationConfigType | null
  showComparison: boolean
  comparisonFilePath?: string
  selectedSection: ConfigSection
  expandedFolders: Set<ConfigSection>
  validationErrors: ValidationError[]
  history: ConfigHistoryEntry[]
  historyIndex: number
  templates: ConfigTemplate[]

  // Layout state
  leftPanelWidth: number
  rightPanelWidth: number
  // Arbitrary persisted layout sizes by key (percentages per panel)
  layoutSizes: Record<string, number[]>

  // UI state
  isLoading: boolean
  isSaving: boolean
}

// Store action interfaces
export interface ConfigActions {
  // Basic configuration actions
  updateConfig: (path: ConfigPath, value: any) => void
  batchUpdateConfig: (updates: BatchConfigUpdate) => void
  loadConfig: (filePath: string) => Promise<void>
  saveConfig: (filePath?: string) => Promise<void>
  openConfigFromContent: (content: string, format?: 'json' | 'yaml') => Promise<void>
  importConfig: (configJson: string) => Promise<ConfigImportResult>
  exportConfig: (format?: 'json' | 'yaml' | 'xml') => ConfigExport
  resetToDefaults: () => void

  // Comparison actions
  setComparison: (config: SimulationConfigType | null) => void
  toggleComparison: () => void
  clearComparison: () => void
  setComparisonPath: (path?: string) => void

  // Diff and copy actions
  getComparisonDiff: () => ConfigComparison
  copyFromComparison: (path: ConfigPath) => boolean
  batchCopyFromComparison: (paths: ConfigPath[]) => boolean
  removeConfigPath: (path: ConfigPath) => void
  applyAllDifferencesFromComparison: () => void

  // Navigation actions
  setSelectedSection: (section: ConfigSection) => void
  toggleSection: (section: ConfigSection) => void
  expandSection: (section: ConfigSection) => void
  collapseSection: (section: ConfigSection) => void

  // Validation actions
  validateConfig: () => Promise<void>
  validateField: (path: ConfigPath, value: any) => Promise<ValidationError[]>
  clearValidationErrors: () => void

  // History actions
  undo: () => void
  redo: () => void
  clearHistory: () => void
  addToHistory: (action: ConfigHistoryEntry['action'], description?: string) => void

  // Layout actions
  setPanelWidths: (leftWidth: number, rightWidth: number) => void
  resetPanelWidths: () => void

  // Generic layout sizes persistence helpers
  setLayoutSizes: (key: string, sizes: number[]) => void
  getLayoutSizes: (key: string) => number[] | undefined
  resetLayoutSizes: (key?: string) => void

  // Persistence actions
  persistUIState: () => void
  restoreUIState: () => void
  clearPersistedState: () => void

  // Template actions
  loadTemplate: (templateName: string) => Promise<void>
  saveTemplate: (template: ConfigTemplate) => Promise<void>
  deleteTemplate: (templateName: string) => Promise<void>
}

// Computed properties interface
export interface ConfigComputed {
  readonly isValid: boolean
  readonly hasChanges: boolean
  readonly canUndo: boolean
  readonly canRedo: boolean
  readonly currentConfigDiff: ConfigComparison
  readonly configSummary: {
    totalAgents: number
    environmentSize: number
    memoryUsage: string
  }
  readonly affectedFields: ConfigPath[]
}

// Complete store interface combining all parts
export interface ConfigStore extends ConfigState, ConfigActions, ConfigComputed {}

// Store selector types
export type ConfigSelector<T> = (state: ConfigStore) => T
export type ConfigSelectorWithProps<T, P> = (state: ConfigStore, props: P) => T

// Store subscription callback
export type ConfigStoreListener = (state: ConfigStore) => void
export type PartialConfigStoreListener = (partial: Partial<ConfigStore>) => void

// Store middleware types
export type ConfigStoreMiddleware = (
  config: any,
  enhancer: any
) => (next: any) => (action: any) => any

// Store context for debugging
export interface ConfigStoreContext {
  name: string
  version: string
  devtools?: boolean
  persist?: boolean | string
}

// =====================================================
// Utility Types for Nested Configs and Manipulation
// =====================================================

// Note: ConfigPathValue was removed as it was not being used.
// For nested config access, use the Zod-inferred types directly:
// - SimulationConfigType for full config access
// - AgentParameterType for agent parameters
// - ModuleParameterType for module parameters
// - VisualizationConfigType for visualization settings

// Type for creating a new config type with optional properties at a specific path
export type OptionalConfigPath = Partial<SimulationConfigType>

// Simplified type for config updates
export type ConfigPathUpdate = Record<string, any>

// Simplified type for config paths
export type ConfigPaths = string

// Type for deep partial configuration
export type DeepPartialConfig<T> = Partial<T>

// Type for required configuration fields
export type RequiredConfigFields<T extends SimulationConfigType> = Required<T>

// Type for configuration field metadata
export interface ConfigFieldMetadata {
  path: ConfigPath
  type: 'number' | 'string' | 'boolean' | 'object' | 'array'
  required: boolean
  defaultValue: any
  min?: number
  max?: number
  step?: number
  options?: string[] | { [key: string]: any }
  description?: string
  category: ConfigSection
  validationRules?: string[]
  dependencies?: ConfigPath[]
}

// Configuration schema type for runtime introspection
export interface ConfigSchema {
  fields: Record<ConfigPath, ConfigFieldMetadata>
  sections: Record<ConfigSection, {
    name: string
    description: string
    fields: ConfigPath[]
    collapsed?: boolean
  }>
  dependencies: Record<ConfigPath, ConfigPath[]>
  validationRules: Record<string, {
    description: string
    validator: (value: any, context: ConfigValidationContext) => boolean
    errorMessage: string
  }>
}

// Configuration value transformer type
export type ConfigValueTransformer<T = any> = {
  serialize: (value: T) => any
  deserialize: (value: any) => T
  validate?: (value: any) => boolean
}

// Configuration value formatters
export type ConfigValueFormatter = {
  display: (value: any) => string
  edit: (value: any) => string
  tooltip?: (value: any) => string
}

// Configuration field change event
export interface ConfigFieldChangeEvent {
  path: ConfigPath
  previousValue: any
  newValue: any
  timestamp: number
  source: 'user' | 'system' | 'import' | 'reset'
  validationErrors?: ValidationError[]
}

// Configuration batch operation result
export interface ConfigBatchOperationResult {
  success: boolean
  operations: {
    path: ConfigPath
    success: boolean
    error?: string
    previousValue?: any
    newValue?: any
  }[]
  timestamp: number
}