import { z } from 'zod'
import {
  SimulationConfigSchema,
  AgentParameterSchema,
  ModuleParameterSchema,
  VisualizationConfigSchema,
  AgentTypeRatiosSchema
} from './zodSchemas'

// Validation error types
export interface ValidationError {
  path: string
  message: string
  code: string
}

// Validation result type
export interface ValidationResult {
  success: boolean
  errors: ValidationError[]
  warnings: ValidationError[]
}

// Validation state type
export interface ValidationState {
  isValidating: boolean
  errors: ValidationError[]
  warnings: ValidationError[]
  lastValidationTime: number

  // Computed properties - these are implemented as getters in the store
  readonly isValid: boolean
  readonly hasErrors: boolean
  readonly hasWarnings: boolean
  readonly errorCount: number
  readonly warningCount: number
}

// Zod schema exports
export interface ZodSchemas {
  SimulationConfigSchema: z.ZodSchema
  AgentParameterSchema: z.ZodSchema
  ModuleParameterSchema: z.ZodSchema
  VisualizationConfigSchema: z.ZodSchema
  AgentTypeRatiosSchema: z.ZodSchema
}

// Re-export schemas for convenience
export const schemas: ZodSchemas = {
  SimulationConfigSchema,
  AgentParameterSchema,
  ModuleParameterSchema,
  VisualizationConfigSchema,
  AgentTypeRatiosSchema
} as const

// Type exports for better TypeScript integration
export type SimulationConfigType = z.infer<typeof SimulationConfigSchema>
export type AgentParameterType = z.infer<typeof AgentParameterSchema>
export type ModuleParameterType = z.infer<typeof ModuleParameterSchema>
export type VisualizationConfigType = z.infer<typeof VisualizationConfigSchema>
export type AgentTypeRatiosType = z.infer<typeof AgentTypeRatiosSchema>

// Extended validation result with parsed data
export interface ValidationResultWithData<T = any> extends ValidationResult {
  data?: T
}

// Field validation options
export interface FieldValidationOptions {
  path: string
  value: unknown
  required?: boolean
  customValidators?: Array<(value: unknown) => ValidationError | null>
}


// =====================================================
// Store Action Types for Validation
// =====================================================

// Validation store state interface
export interface ValidationState {
  // Core validation state
  isValidating: boolean
  errors: ValidationError[]
  warnings: ValidationError[]
  lastValidationTime: number

  // Field-specific validation state
  fieldValidationCache: Record<string, {
    result: ValidationResult
    timestamp: number
    ttl: number
  }>

  // Validation configuration
  validationRules: Record<string, {
    enabled: boolean
    severity: 'error' | 'warning' | 'info'
    debounceMs?: number
  }>

  // Performance metrics
  validationMetrics: {
    totalValidations: number
    averageValidationTime: number
    lastValidationDuration: number
    validationErrorsByPath: Record<string, number>
  }
}

// Validation store actions interface
export interface ValidationActions {
  // Core validation actions
  setValidating: (validating: boolean) => void
  setValidationResult: (result: ValidationResult) => Promise<void>
  updateLastValidationTime: () => void

  // Error management actions
  addError: (error: ValidationError) => void
  addErrors: (errors: ValidationError[]) => void
  clearErrors: () => void
  removeError: (path: string) => void

  // Warning management actions
  addWarning: (warning: ValidationError) => void
  addWarnings: (warnings: ValidationError[]) => void
  clearWarnings: () => void
  removeWarning: (path: string) => void

  // Field validation actions
  validateField: (path: string, value: any, config?: Record<string, any>) => Promise<ValidationResult>
  validateFields: (fields: Array<{ path: string; value: any }>, config?: Record<string, any>) => Promise<ValidationResult>
  clearFieldValidation: (path: string) => void
  clearAllFieldValidation: () => void

  // Batch validation actions
  validateConfig: (config: Record<string, any>) => Promise<ValidationResult>
  validatePartialConfig: (partialConfig: Record<string, any>, fullConfig: Record<string, any>) => Promise<ValidationResult>

  // Validation rules management
  enableValidationRule: (ruleName: string) => void
  disableValidationRule: (ruleName: string) => void
  setValidationRuleSeverity: (ruleName: string, severity: 'error' | 'warning' | 'info') => void

  // Cache management
  clearValidationCache: () => void
  setCacheTTL: (ttl: number) => void

  // Utility actions
  exportValidationReport: () => ValidationReport
  importValidationRules: (rules: Record<string, any>) => Promise<void>
}

// Validation computed properties interface
export interface ValidationComputed {
  readonly isValid: boolean
  readonly hasErrors: boolean
  readonly hasWarnings: boolean
  readonly errorCount: number
  readonly warningCount: number
  readonly infoCount: number
  readonly validationProgress: number
  readonly mostCommonErrorPaths: Array<{ path: string; count: number }>
  readonly validationScore: number
  readonly lastValidationDuration: number
  readonly averageValidationTime: number
}

// Complete validation store interface
export interface ValidationStore extends ValidationState, ValidationActions, ValidationComputed {}

// Validation selector types
export type ValidationSelector<T> = (state: ValidationStore) => T
export type ValidationSelectorWithProps<T, P> = (state: ValidationStore, props: P) => T

// Validation store listeners
export type ValidationStoreListener = (state: ValidationStore) => void
export type ValidationStoreErrorListener = (errors: ValidationError[]) => void
export type ValidationStoreFieldListener = (path: string, result: ValidationResult) => void

// Validation report types
export interface ValidationReport {
  summary: {
    totalErrors: number
    totalWarnings: number
    totalInfo: number
    validationScore: number
    validationTime: number
    validatedAt: number
  }
  errors: ValidationError[]
  warnings: ValidationError[]
  info: ValidationError[]
  metrics: {
    errorsByPath: Record<string, number>
    errorsByRule: Record<string, number>
    validationHistory: Array<{
      timestamp: number
      duration: number
      errorCount: number
      warningCount: number
    }>
  }
  recommendations: string[]
}

// Validation rule types
export interface ValidationRule {
  name: string
  description: string
  enabled: boolean
  severity: 'error' | 'warning' | 'info'
  validator: (value: any, context: ValidationContext) => boolean | Promise<boolean>
  message: string | ((value: any, context: ValidationContext) => string)
  dependencies?: string[]
  category: string
}

// Validation batch request
export interface ValidationBatchRequest {
  config: Record<string, any>
  fields?: string[]
  rules?: string[]
  context?: ValidationContext
  options?: {
    skipCache?: boolean
    forceRevalidation?: boolean
    includeMetrics?: boolean
  }
}

// Validation performance metrics
export interface ValidationMetrics {
  totalValidations: number
  cacheHitRate: number
  averageValidationTime: number
  validationTimeByRule: Record<string, number>
  validationTimeByPath: Record<string, number>
  peakValidationTime: number
  validationQueueLength: number
}

// Validation middleware types - generic middleware for validation operations
export type ValidationMiddleware<T = any> = (
  config: T,
  enhancer: (next: (action: any) => any) => (action: any) => any
) => (next: (action: any) => any) => (action: any) => any

// Specific middleware types for different validation contexts
export type ValidationStoreMiddleware = ValidationMiddleware<ValidationStore>
export type ValidationStateMiddleware = ValidationMiddleware<ValidationState>
export type ValidationActionsMiddleware = ValidationMiddleware<ValidationActions>

// Validation context for cross-field validation
export interface ValidationContext {
  config: Record<string, unknown>
  errors: ValidationError[]
  warnings: ValidationError[]
  path?: string
  parent?: any
  siblings?: Record<string, any>
  globalContext?: Record<string, any>
}