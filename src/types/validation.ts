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

// Validation context for cross-field validation
export interface ValidationContext {
  config: Record<string, unknown>
  errors: ValidationError[]
  warnings: ValidationError[]
}