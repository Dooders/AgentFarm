import { z } from 'zod'
import {
  SimulationConfigSchema,
  AgentParameterSchema,
  ModuleParameterSchema,
  VisualizationConfigSchema,
  AgentTypeRatiosSchema,
  type SimulationConfigType,
  type AgentParameterType,
  type ModuleParameterType,
  type VisualizationConfigType,
  type AgentTypeRatiosType
} from '@/types/zodSchemas'
import { ValidationError, ValidationResult } from '@/types/validation'

// Custom error formatter for Zod validation errors
export function formatZodError(zodError: z.ZodError): ValidationError[] {
  return zodError.errors.map((error) => {
    const path = error.path.length > 0 ? error.path.join('.') : 'root'
    const message = getCustomErrorMessage(error)

    return {
      path,
      message,
      code: error.code
    }
  })
}

// Get custom error messages based on error code and path
function getCustomErrorMessage(error: z.ZodIssue): string {
  const { message } = error

  // Handle specific error types with custom messages
  switch (error.code) {
    case 'invalid_type':
      if (error.expected === 'number' && error.received === 'string') {
        return `Expected a number, but received a string. Please enter a valid number.`
      }
      if (error.expected === 'boolean' && error.received === 'string') {
        return `Expected true or false, but received a string. Please enter true or false.`
      }
      return `Invalid type. Expected ${error.expected}, received ${error.received}.`

    case 'invalid_enum_value':
      // Use custom message if provided, otherwise use generic message
      if (message && message !== 'Invalid enum value') {
        return message
      }
      const allowedValues = error.options?.join(', ') || 'unknown'
      return `Invalid value. Allowed values are: ${allowedValues}`

    case 'too_small':
      if (error.type === 'number') {
        if (error.inclusive) {
          return `Value must be at least ${error.minimum} (inclusive)`
        }
        return `Value must be greater than ${error.minimum}`
      }
      if (error.type === 'string') {
        return `Text must be at least ${error.minimum} characters long`
      }
      return `Value is too small`

    case 'too_big':
      if (error.type === 'number') {
        if (error.inclusive) {
          return `Value must be at most ${error.maximum} (inclusive)`
        }
        return `Value must be less than ${error.maximum}`
      }
      if (error.type === 'string') {
        return `Text must be at most ${error.maximum} characters long`
      }
      return `Value is too large`

    case 'invalid_string':
      if (error.validation === 'regex') {
        // Use custom message if provided, otherwise use generic message
        if (message && message !== 'Invalid string') {
          return message
        }
        return `Invalid format. Please check the required format.`
      }
      return `Invalid text format`

    case 'custom':
      // Custom validation errors already have descriptive messages
      return message

    default:
      return message || 'Validation error occurred'
  }
}

// Validate configuration using Zod schemas
export async function validateSimulationConfig(config: unknown): ValidationResult {
  try {
    // Import and validate using Zod schema
    const { SimulationConfigSchema } = await import('@/types/zodSchemas')
    SimulationConfigSchema.parse(config)

    return {
      success: true,
      errors: [],
      warnings: []
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      const formattedErrors = formatZodError(error)

      return {
        success: false,
        errors: formattedErrors,
        warnings: []
      }
    }

    return {
      success: false,
      errors: [{
        path: 'root',
        message: 'Unexpected validation error occurred',
        code: 'unexpected_error'
      }],
      warnings: []
    }
  }
}

// Validate specific field using Zod schema
export function validateField(path: string, value: unknown): ValidationError[] {
  try {
    // Map field paths to appropriate schemas
    const fieldSchema = getFieldSchema(path)

    if (!fieldSchema) {
      return [{
        path,
        message: `No validation schema found for field: ${path}`,
        code: 'schema_not_found'
      }]
    }

    // Parse the value with the schema
    fieldSchema.parse(value)

    return []
  } catch (error) {
    if (error instanceof z.ZodError) {
      // For field validation, we want the path to be the field name, not 'root'
      const formattedErrors = formatZodError(error)
      return formattedErrors.map(error => ({
        ...error,
        path: error.path === 'root' ? path : error.path
      }))
    }

    return [{
      path,
      message: 'Validation failed',
      code: 'validation_error'
    }]
  }
}

// Get appropriate schema for a field path
function getFieldSchema(path: string): z.ZodSchema | null {
  const pathSegments = path.split('.')

  if (pathSegments.length === 1) {
    // Root level fields
    switch (path) {
      case 'width':
      case 'height':
        return z.number().min(10).max(1000).int()
      case 'system_agents':
      case 'independent_agents':
      case 'control_agents':
        return z.number().min(0).max(10000).int()
      case 'learning_rate':
        return z.number().min(0.0001).max(1.0).positive()
      case 'epsilon_start':
      case 'epsilon_min':
        return z.number().min(0.0).max(1.0)
      case 'epsilon_decay':
        return z.number().min(0.9).max(0.9999)
      case 'position_discretization_method':
        return z.enum(['floor', 'round', 'ceil'])
      case 'use_bilinear_interpolation':
        return z.boolean()

      case 'agent_type_ratios':
        return AgentTypeRatiosSchema

      case 'visualization':
        return VisualizationConfigSchema

      case 'agent_parameters':
      case 'gather_parameters':
      case 'share_parameters':
      case 'move_parameters':
      case 'attack_parameters':
        return z.object({})

      default:
        return null
    }
  }

  if (pathSegments.length === 2) {
    const [parent, child] = pathSegments

    // Agent parameters
    if (parent === 'agent_parameters' && ['SystemAgent', 'IndependentAgent', 'ControlAgent'].includes(child)) {
      return AgentParameterSchema
    }

    // Module parameters
    if (['gather_parameters', 'share_parameters', 'move_parameters', 'attack_parameters'].includes(parent)) {
      return ModuleParameterSchema
    }

    // Visualization fields
    if (parent === 'visualization') {
      switch (child) {
        case 'canvas_width':
        case 'canvas_height':
        case 'background_color':
        case 'show_metrics':
        case 'font_size':
        case 'line_width':
        case 'agent_colors':
          return z.any()
        default:
          return null
      }
    }

    // Agent type ratios
    if (parent === 'agent_type_ratios' && ['SystemAgent', 'IndependentAgent', 'ControlAgent'].includes(child)) {
      return z.number()
    }
  }

  if (pathSegments.length === 3) {
    const [parent, child, grandchild] = pathSegments

    // Agent parameters nested fields
    if (parent === 'agent_parameters' && ['SystemAgent', 'IndependentAgent', 'ControlAgent'].includes(child)) {
      switch (grandchild) {
        case 'target_update_freq':
          return z.number().min(1).max(1000).int()
        case 'memory_size':
          return z.number().min(1000).max(1000000).int()
        case 'learning_rate':
          return z.number().min(0.0001).max(1.0).positive()
        case 'gamma':
          return z.number().min(0.0).max(1.0)
        case 'epsilon_start':
        case 'epsilon_min':
          return z.number().min(0.0).max(1.0)
        case 'epsilon_decay':
          return z.number().min(0.9).max(0.9999)
        case 'dqn_hidden_size':
          return z.number().min(32).max(2048).int()
        case 'batch_size':
          return z.number().min(16).max(1024).int()
        case 'tau':
          return z.number().min(0.001).max(1.0)
        case 'success_reward':
          return z.number().min(0.1).max(100.0)
        case 'failure_penalty':
          return z.number().min(-100.0).max(-0.1)
        case 'base_cost':
          return z.number().min(0.0).max(10.0)
        default:
          return null
      }
    }

    // Agent colors
    if (parent === 'visualization' && child === 'agent_colors' &&
        ['SystemAgent', 'IndependentAgent', 'ControlAgent'].includes(grandchild)) {
      return z.string()
    }
  }

  return null
}

// Validate partial configuration (useful for form validation)
export function validatePartialConfig(partialConfig: Partial<SimulationConfigType>): ValidationResult {
  const errors: ValidationError[] = []
  const warnings: ValidationError[] = []

  // Validate each field in the partial config
  for (const [key, value] of Object.entries(partialConfig)) {
    const fieldErrors = validateField(key, value)

    if (fieldErrors.length > 0) {
      errors.push(...fieldErrors)
    }
  }

  return {
    success: errors.length === 0,
    errors,
    warnings
  }
}

// Create a safe validation function that doesn't throw
export function safeValidateConfig(config: unknown): {
  success: boolean
  data?: SimulationConfigType
  errors: ValidationError[]
} {
  try {
    const validatedData = SimulationConfigSchema.parse(config)

    return {
      success: true,
      data: validatedData,
      errors: []
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      const formattedErrors = formatZodError(error)

      return {
        success: false,
        errors: formattedErrors
      }
    }

    return {
      success: false,
      errors: [{
        path: 'root',
        message: 'Unexpected validation error occurred',
        code: 'unexpected_error'
      }]
    }
  }
}

// Utility to get all validation errors for a specific section
export function getValidationErrorsForSection(
  errors: ValidationError[],
  section: string
): ValidationError[] {
  return errors.filter(error =>
    error.path === section || error.path.startsWith(`${section}.`)
  )
}

// Utility to check if a specific field has validation errors
export function hasFieldError(errors: ValidationError[], fieldPath: string): boolean {
  return errors.some(error => error.path === fieldPath)
}

// Utility to get the first error for a specific field
export function getFieldError(errors: ValidationError[], fieldPath: string): ValidationError | undefined {
  return errors.find(error => error.path === fieldPath)
}

// Utility to get all errors for a field and its children
export function getFieldErrors(errors: ValidationError[], fieldPath: string): ValidationError[] {
  return errors.filter(error =>
    error.path === fieldPath || error.path.startsWith(`${fieldPath}.`)
  )
}

// Create default values for configuration fields
export function getDefaultConfig(): SimulationConfigType {
  return {
    width: 100,
    height: 100,
    position_discretization_method: 'floor',
    use_bilinear_interpolation: true,
    system_agents: 10,
    independent_agents: 15,
    control_agents: 5,
    agent_type_ratios: {
      SystemAgent: 0.3,
      IndependentAgent: 0.5,
      ControlAgent: 0.2
    },
    learning_rate: 0.001,
    epsilon_start: 1.0,
    epsilon_min: 0.01,
    epsilon_decay: 0.995,
    agent_parameters: {
      SystemAgent: getDefaultAgentParameters(),
      IndependentAgent: getDefaultAgentParameters(),
      ControlAgent: getDefaultAgentParameters()
    },
    visualization: getDefaultVisualizationConfig(),
    gather_parameters: getDefaultModuleParameters(),
    share_parameters: getDefaultModuleParameters(),
    move_parameters: getDefaultModuleParameters(),
    attack_parameters: getDefaultModuleParameters()
  }
}

function getDefaultAgentParameters(): AgentParameterType {
  return {
    target_update_freq: 100,
    memory_size: 10000,
    learning_rate: 0.001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_min: 0.01,
    epsilon_decay: 0.995,
    dqn_hidden_size: 128,
    batch_size: 32,
    tau: 0.005,
    success_reward: 1.0,
    failure_penalty: -1.0,
    base_cost: 0.1
  }
}

function getDefaultModuleParameters(): ModuleParameterType {
  return {
    target_update_freq: 50,
    memory_size: 5000,
    learning_rate: 0.001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_min: 0.01,
    epsilon_decay: 0.995,
    dqn_hidden_size: 64,
    batch_size: 16,
    tau: 0.005,
    success_reward: 0.5,
    failure_penalty: -0.5,
    base_cost: 0.05
  }
}

function getDefaultVisualizationConfig(): VisualizationConfigType {
  return {
    canvas_width: 800,
    canvas_height: 600,
    background_color: '#1a1a1a',
    agent_colors: {
      SystemAgent: '#ff6b6b',
      IndependentAgent: '#4ecdc4',
      ControlAgent: '#45b7d1'
    },
    show_metrics: true,
    font_size: 12,
    line_width: 2
  }
}