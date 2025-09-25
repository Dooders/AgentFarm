import { ValidationError, ValidationResult } from '@/types/validation'
import { SimulationConfig } from '@/types/config'

/**
 * Validation service for configuration validation
 * This service provides validation logic without direct store dependencies
 */

export interface ValidationService {
  validateConfig: (config: SimulationConfig) => ValidationResult
  validateField: (path: string, value: any) => ValidationResult
  validateSingleField: (path: string, value: any) => ValidationError[]
}

export class ConfigValidationService implements ValidationService {
  validateConfig(config: SimulationConfig): ValidationResult {
    const errors: ValidationError[] = []
    const warnings: ValidationError[] = []

    // Validate basic config fields
    const fieldErrors = this.validateSingleField('width', config.width)
    errors.push(...fieldErrors)

    const heightErrors = this.validateSingleField('height', config.height)
    errors.push(...heightErrors)

    // Validate agent ratios sum to 1
    const ratioSum = Object.values(config.agent_type_ratios).reduce((sum, ratio) => sum + ratio, 0)
    if (Math.abs(ratioSum - 1.0) > 0.001) {
      errors.push({
        path: 'agent_type_ratios',
        message: 'Agent type ratios must sum to 1.0',
        code: 'invalid_sum'
      })
    }

    // Add warnings for performance considerations
    if (config.width * config.height > 50000) {
      warnings.push({
        path: 'environment',
        message: 'Large environment size may impact performance',
        code: 'performance_warning'
      })
    }

    return {
      success: errors.length === 0,
      errors,
      warnings
    }
  }

  validateField(path: string, value: any): ValidationResult {
    const errors = this.validateSingleField(path, value)

    return {
      success: errors.length === 0,
      errors,
      warnings: []
    }
  }

  validateSingleField(path: string, value: any): ValidationError[] {
    const errors: ValidationError[] = []

    // Field-specific validation rules
    switch (path) {
      case 'width':
      case 'height':
        if (typeof value !== 'number' || value < 10 || value > 1000) {
          errors.push({
            path,
            message: `${path} must be a number between 10 and 1000`,
            code: 'invalid_range'
          })
        }
        break

      case 'system_agents':
      case 'independent_agents':
      case 'control_agents':
        if (typeof value !== 'number' || value < 0 || value > 10000) {
          errors.push({
            path,
            message: `${path} must be a number between 0 and 10000`,
            code: 'invalid_range'
          })
        }
        break

      case 'learning_rate':
        if (typeof value !== 'number' || value <= 0 || value > 1) {
          errors.push({
            path,
            message: 'learning_rate must be a number between 0 and 1',
            code: 'invalid_range'
          })
        }
        break

      case 'epsilon_start':
      case 'epsilon_min':
        if (typeof value !== 'number' || value < 0 || value > 1) {
          errors.push({
            path,
            message: `${path} must be a number between 0 and 1`,
            code: 'invalid_range'
          })
        }
        break

      case 'epsilon_decay':
        if (typeof value !== 'number' || value <= 0 || value > 1) {
          errors.push({
            path,
            message: 'epsilon_decay must be a number between 0 and 1',
            code: 'invalid_range'
          })
        }
        break
    }

    return errors
  }
}

// Create singleton instance
export const validationService = new ConfigValidationService()