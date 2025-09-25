import { z } from 'zod'

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

// Zod schema exports (will be defined in services/validationService.ts)
export interface ZodSchemas {
  SimulationConfigSchema: z.ZodSchema
  AgentParameterSchema: z.ZodSchema
  ModuleParameterSchema: z.ZodSchema
  VisualizationConfigSchema: z.ZodSchema
}