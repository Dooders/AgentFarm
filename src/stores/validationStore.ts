import { create } from 'zustand'
import { ValidationError, ValidationResult } from '@/types/validation'

interface ValidationStore {
  // Core state
  isValidating: boolean
  errors: ValidationError[]
  warnings: ValidationError[]
  lastValidationTime: number

  // Validation actions
  setValidating: (validating: boolean) => void
  addError: (error: ValidationError) => void
  addErrors: (errors: ValidationError[]) => void
  clearErrors: () => void
  addWarning: (warning: ValidationError) => void
  addWarnings: (warnings: ValidationError[]) => void
  clearWarnings: () => void
  setValidationResult: (result: ValidationResult) => void
  updateLastValidationTime: () => void

  // Field-specific validation
  getFieldError: (path: string) => ValidationError | undefined
  getFieldErrors: (path: string) => ValidationError[]
  hasFieldError: (path: string) => boolean

  // Validation utilities
  clearFieldErrors: (path: string) => void
  validateField: (path: string, value: any) => Promise<ValidationResult>

  // Computed selectors (these are computed from state)
  get isValid(): boolean
  get hasErrors(): boolean
  get hasWarnings(): boolean
  get errorCount(): number
  get warningCount(): number
}

export const useValidationStore = create<ValidationStore>((set, get) => ({
  // Initial state
  isValidating: false,
  errors: [],
  warnings: [],
  lastValidationTime: 0,

  // Computed properties
  get isValid() {
    return get().errors.length === 0
  },

  get hasErrors() {
    return get().errors.length > 0
  },

  get hasWarnings() {
    return get().warnings.length > 0
  },

  get errorCount() {
    return get().errors.length
  },

  get warningCount() {
    return get().warnings.length
  },

  // Actions
  setValidating: (validating: boolean) => {
    set({ isValidating: validating })
  },

  addError: (error: ValidationError) => {
    set(state => ({
      errors: [...state.errors, error]
    }))
  },

  addErrors: (errors: ValidationError[]) => {
    set(state => ({
      errors: [...state.errors, ...errors]
    }))
  },

  clearErrors: () => {
    set({ errors: [] })
  },

  addWarning: (warning: ValidationError) => {
    set(state => ({
      warnings: [...state.warnings, warning]
    }))
  },

  addWarnings: (warnings: ValidationError[]) => {
    set(state => ({
      warnings: [...state.warnings, ...warnings]
    }))
  },

  clearWarnings: () => {
    set({ warnings: [] })
  },

  setValidationResult: (result: ValidationResult) => {
    set({
      errors: result.errors,
      warnings: result.warnings,
      lastValidationTime: Date.now(),
      isValidating: false
    })
  },

  updateLastValidationTime: () => {
    set({ lastValidationTime: Date.now() })
  },

  // Field-specific methods
  getFieldError: (path: string): ValidationError | undefined => {
    const { errors } = get()
    return errors.find(error => error.path === path)
  },

  getFieldErrors: (path: string): ValidationError[] => {
    const { errors } = get()
    return errors.filter(error => error.path.startsWith(path))
  },

  hasFieldError: (path: string): boolean => {
    const { errors } = get()
    return errors.some(error => error.path === path)
  },

  clearFieldErrors: (path: string) => {
    set(state => ({
      errors: state.errors.filter(error => !error.path.startsWith(path))
    }))
  },

  validateField: async (path: string, _value: any): Promise<ValidationResult> => {
    const { setValidating, addError, clearFieldErrors } = get()

    setValidating(true)
    clearFieldErrors(path)

    try {
      // This would be implemented with actual validation logic
      // For now, return success - in real implementation, this would use Zod schemas
      await new Promise(resolve => setTimeout(resolve, 50)) // Simulate validation delay

      return {
        success: true,
        errors: [],
        warnings: []
      }
    } catch (error) {
      const validationError: ValidationError = {
        path,
        message: error instanceof Error ? error.message : 'Validation failed',
        code: 'validation_error'
      }

      addError(validationError)

      return {
        success: false,
        errors: [validationError],
        warnings: []
      }
    } finally {
      setValidating(false)
    }
  }
}))