import { ValidationError, ValidationResult } from '@/types/validation'
import { SimulationConfigType } from '@/types/config'
import { validateSimulationConfig, validateField as zodValidateField } from '@/utils/validationUtils'

/**
 * Validation service for configuration validation using Zod schemas
 * This service provides validation logic without direct store dependencies
 */

export interface ValidationService {
  validateConfig: (config: SimulationConfigType) => ValidationResult
  validateField: (path: string, value: any) => ValidationResult
  validateSingleField: (path: string, value: any) => ValidationError[]
}

export class ZodValidationService implements ValidationService {
  validateConfig(config: SimulationConfigType): ValidationResult {
    // Use the Zod-based validation utility
    const zodResult = validateSimulationConfig(config)

    // Add any additional business logic validation if needed
    const { errors: additionalErrors, warnings: additionalWarnings } = this.validateBusinessRules(config)

    return {
      success: zodResult.success && additionalErrors.length === 0,
      errors: [...zodResult.errors, ...additionalErrors],
      warnings: [...zodResult.warnings, ...additionalWarnings]
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
  private validateBusinessRules(config: SimulationConfigType): { errors: ValidationError[]; warnings: ValidationError[] } {
    const errors: ValidationError[] = []
    const warnings: ValidationError[] = []

    // Performance warnings
    if (config.width * config.height > 50000) {
      warnings.push({
        path: 'environment',
        message: 'Large environment size may impact performance',
        code: 'performance_warning'
      })
    }

    // Agent capacity validation (error)
    const totalAgents = config.system_agents + config.independent_agents + config.control_agents
    const environmentCapacity = config.width * config.height

    if (totalAgents > environmentCapacity) {
      errors.push({
        path: 'agents',
        message: 'Total number of agents exceeds environment capacity',
        code: 'capacity_exceeded'
      })
    }

    // Memory efficiency checks (warning)
    const totalMemory = Object.entries(config.agent_parameters).reduce((total, [agentType, params]) => {
      let agentCount = 0
      switch (agentType) {
        case 'SystemAgent':
          agentCount = config.system_agents
          break
        case 'IndependentAgent':
          agentCount = config.independent_agents
          break
        case 'ControlAgent':
          agentCount = config.control_agents
          break
      }
      const p = params as unknown as { memory_size: number }
      return total + p.memory_size * agentCount
    }, 0)

    if (totalMemory > 10000000) { // 10M memory entries
      warnings.push({
        path: 'memory',
        message: 'Total memory usage may be excessive for available system resources',
        code: 'memory_warning'
      })
    }

    return { errors, warnings }
  }
}

// Create singleton instance
export const validationService = new ZodValidationService()