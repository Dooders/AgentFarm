import { ValidationError, ValidationResult } from '@/types/validation'
import { SimulationConfig } from '@/types/config'
import { validateSimulationConfig, validateField as zodValidateField } from '@/utils/validationUtils'

/**
 * Validation service for configuration validation using Zod schemas
 * This service provides validation logic without direct store dependencies
 */

export interface ValidationService {
  validateConfig: (config: SimulationConfig) => ValidationResult
  validateField: (path: string, value: any) => ValidationResult
  validateSingleField: (path: string, value: any) => ValidationError[]
}

export class ZodValidationService implements ValidationService {
  validateConfig(config: SimulationConfig): ValidationResult {
    // Use the Zod-based validation utility
    const zodResult = validateSimulationConfig(config)

    // Add any additional business logic validation if needed
    const additionalErrors = this.validateBusinessRules(config)

    return {
      success: zodResult.success && additionalErrors.length === 0,
      errors: [...zodResult.errors, ...additionalErrors],
      warnings: zodResult.warnings
    }
  }

  validateField(path: string, value: any): ValidationResult {
    const errors = zodValidateField(path, value)

    return {
      success: errors.length === 0,
      errors,
      warnings: []
    }
  }

  validateSingleField(path: string, value: any): ValidationError[] {
    return zodValidateField(path, value)
  }

  /**
   * Additional business rule validation that complements Zod schema validation
   */
  private validateBusinessRules(config: SimulationConfig): ValidationError[] {
    const errors: ValidationError[] = []

    // Performance warnings
    if (config.width * config.height > 50000) {
      errors.push({
        path: 'environment',
        message: 'Large environment size may impact performance',
        code: 'performance_warning'
      })
    }

    // Agent capacity validation
    const totalAgents = config.system_agents + config.independent_agents + config.control_agents
    const environmentCapacity = config.width * config.height

    if (totalAgents > environmentCapacity) {
      errors.push({
        path: 'agents',
        message: 'Total number of agents exceeds environment capacity',
        code: 'capacity_exceeded'
      })
    }

    // Memory efficiency checks
    const totalMemory = Object.values(config.agent_parameters).reduce((total, params) => {
      return total + params.memory_size * (config.system_agents + config.independent_agents + config.control_agents)
    }, 0)

    if (totalMemory > 10000000) { // 10M memory entries
      errors.push({
        path: 'memory',
        message: 'Total memory usage may be excessive for available system resources',
        code: 'memory_warning'
      })
    }

    return errors
  }
}

// Create singleton instance
export const validationService = new ZodValidationService()