import { ConfigPath, BatchConfigUpdate, ConfigSection, ConfigImportResult, ConfigExport } from './config'
import { ValidationError } from './validation'
import { ValidationResult } from './validation'

// =====================================================
// Event and Callback Type Definitions
// =====================================================

// Base event interface
export interface BaseEvent {
  type: string
  timestamp: number
  source: 'user' | 'system' | 'validation' | 'import' | 'export' | 'file' | 'template' | 'default' | 'clipboard' | 'drag_drop' | 'reset' | 'restore'
}

// Configuration change events
export interface ConfigChangeEvent extends BaseEvent {
  type: 'config:change'
  path: ConfigPath
  previousValue: any
  newValue: any
  validationErrors?: ValidationError[]
}

export interface ConfigBatchChangeEvent extends BaseEvent {
  type: 'config:batch_change'
  updates: BatchConfigUpdate
  results: Array<{
    path: ConfigPath
    success: boolean
    error?: string
    validationErrors?: ValidationError[]
  }>
}

export interface ConfigResetEvent extends BaseEvent {
  type: 'config:reset'
  previousConfig: any
  resetType: 'defaults' | 'template' | 'file'
  templateName?: string
  filePath?: string
}

export interface ConfigLoadEvent extends BaseEvent {
  type: 'config:load'
  filePath?: string
  source: 'file' | 'template' | 'default'
  success: boolean
  error?: string
}

export interface ConfigSaveEvent extends BaseEvent {
  type: 'config:save'
  filePath?: string
  success: boolean
  error?: string
}

export interface ConfigImportEvent extends BaseEvent {
  type: 'config:import'
  source: 'file' | 'clipboard' | 'drag_drop'
  format: 'json' | 'yaml' | 'xml'
  result: ConfigImportResult
}

export interface ConfigExportEvent extends BaseEvent {
  type: 'config:export'
  destination: 'file' | 'clipboard' | 'download'
  format: 'json' | 'yaml' | 'xml'
  exportData: ConfigExport
  success: boolean
  error?: string
}

// Validation events
export interface ValidationStartEvent extends BaseEvent {
  type: 'validation:start'
  paths?: ConfigPath[]
  context?: Record<string, any>
}

export interface ValidationCompleteEvent extends BaseEvent {
  type: 'validation:complete'
  result: ValidationResult
  duration: number
  paths?: ConfigPath[]
}

export interface ValidationErrorEvent extends BaseEvent {
  type: 'validation:error'
  errors: ValidationError[]
  path?: ConfigPath
}

export interface ValidationWarningEvent extends BaseEvent {
  type: 'validation:warning'
  warnings: ValidationError[]
  path?: ConfigPath
}

// UI interaction events
export interface SectionToggleEvent extends BaseEvent {
  type: 'ui:section_toggle'
  section: ConfigSection
  expanded: boolean
}

export interface PanelResizeEvent extends BaseEvent {
  type: 'ui:panel_resize'
  leftWidth: number
  rightWidth: number
  source: 'user' | 'reset' | 'restore'
}

export interface ComparisonToggleEvent extends BaseEvent {
  type: 'ui:comparison_toggle'
  enabled: boolean
  compareConfig?: any
}

export interface TemplateSelectEvent extends BaseEvent {
  type: 'ui:template_select'
  templateName: string
  category: string
}

// Performance and system events
export interface PerformanceEvent extends BaseEvent {
  type: 'performance:metric'
  metric: string
  value: number
  unit?: string
  context?: Record<string, any>
}

export interface ErrorEvent extends BaseEvent {
  type: 'error'
  error: Error
  context?: Record<string, any>
  severity: 'low' | 'medium' | 'high' | 'critical'
}

// Event union type
export type AppEvent =
  | ConfigChangeEvent
  | ConfigBatchChangeEvent
  | ConfigResetEvent
  | ConfigLoadEvent
  | ConfigSaveEvent
  | ConfigImportEvent
  | ConfigExportEvent
  | ValidationStartEvent
  | ValidationCompleteEvent
  | ValidationErrorEvent
  | ValidationWarningEvent
  | SectionToggleEvent
  | PanelResizeEvent
  | ComparisonToggleEvent
  | TemplateSelectEvent
  | PerformanceEvent
  | ErrorEvent

// =====================================================
// Callback Type Definitions
// =====================================================

// Base callback interface
export interface BaseCallback {
  (event: any): void
}

// Configuration change callbacks
export type ConfigChangeCallback = (event: ConfigChangeEvent) => void
export type ConfigBatchChangeCallback = (event: ConfigBatchChangeEvent) => void
export type ConfigResetCallback = (event: ConfigResetEvent) => void
export type ConfigLoadCallback = (event: ConfigLoadEvent) => void
export type ConfigSaveCallback = (event: ConfigSaveEvent) => void
export type ConfigImportCallback = (event: ConfigImportEvent) => void
export type ConfigExportCallback = (event: ConfigExportEvent) => void

// Validation callbacks
export type ValidationStartCallback = (event: ValidationStartEvent) => void
export type ValidationCompleteCallback = (event: ValidationCompleteEvent) => void
export type ValidationErrorCallback = (event: ValidationErrorEvent) => void
export type ValidationWarningCallback = (event: ValidationWarningEvent) => void

// UI interaction callbacks
export type SectionToggleCallback = (event: SectionToggleEvent) => void
export type PanelResizeCallback = (event: PanelResizeEvent) => void
export type ComparisonToggleCallback = (event: ComparisonToggleEvent) => void
export type TemplateSelectCallback = (event: TemplateSelectEvent) => void

// System callbacks
export type PerformanceCallback = (event: PerformanceEvent) => void
export type ErrorCallback = (event: ErrorEvent) => void

// Generic event callback
export type EventCallback<T extends AppEvent> = (event: T) => void | Promise<void>

// =====================================================
// Event Handler and Listener Types
// =====================================================

// Event handler interface
export interface EventHandler<T extends AppEvent = AppEvent> {
  handle: EventCallback<T>
  priority?: number
  once?: boolean
  filter?: (event: T) => boolean
}

// Event listener interface
export interface EventListener<T extends AppEvent = AppEvent> {
  id: string
  handler: EventHandler<T>
  enabled: boolean
}

// Event bus interface
export interface EventBus {
  emit<T extends AppEvent>(event: T): void | Promise<void>
  on<T extends AppEvent>(type: T['type'], handler: EventHandler<T>): EventListener<T>
  off(listenerId: string): void
  offType(type: string): void
  once<T extends AppEvent>(type: T['type'], handler: EventCallback<T>): EventListener<T>
  clear(): void
  getListeners(type?: string): EventListener[]
}

// =====================================================
// Hook Types for React Components
// =====================================================

// Configuration change hooks
export interface UseConfigChangeHook {
  onChange: (path: ConfigPath, callback: (value: any, previousValue: any) => void) => () => void
  onBatchChange: (callback: (updates: BatchConfigUpdate) => void) => () => void
  onReset: (callback: () => void) => () => void
}

// Validation hooks
export interface UseValidationHook {
  onValidationStart: (callback: () => void) => () => void
  onValidationComplete: (callback: (result: ValidationResult) => void) => () => void
  onError: (callback: (error: ValidationError) => void) => () => void
  onWarning: (callback: (warning: ValidationError) => void) => () => void
}

// UI interaction hooks
export interface UseUIHook {
  onSectionToggle: (callback: (section: ConfigSection, expanded: boolean) => void) => () => void
  onPanelResize: (callback: (leftWidth: number, rightWidth: number) => void) => () => void
  onComparisonToggle: (callback: (enabled: boolean) => void) => () => void
}

// =====================================================
// Event Filter and Matcher Types
// =====================================================

// Event filter function
export type EventFilter<T extends AppEvent> = (event: T) => boolean

// Event matcher for pattern matching
export interface EventMatcher<T extends AppEvent> {
  type: T['type']
  filter?: EventFilter<T>
  handler: EventCallback<T>
}

// Event pattern for complex matching
export interface EventPattern {
  types?: string[]
  sources?: Array<AppEvent['source']>
  timeRange?: { start: number; end: number }
  customFilter?: (event: AppEvent) => boolean
}

// Event subscription options
export interface EventSubscriptionOptions {
  priority?: number
  once?: boolean
  debounceMs?: number
  throttleMs?: number
  filter?: EventFilter<any>
  enabled?: boolean
}

// Event history entry
export interface EventHistoryEntry {
  id: string
  event: AppEvent
  timestamp: number
  handled: boolean
  handlerCount: number
}

// Event statistics
export interface EventStatistics {
  totalEvents: number
  eventsByType: Record<string, number>
  eventsBySource: Record<string, number>
  averageProcessingTime: number
  peakProcessingTime: number
  errorRate: number
}