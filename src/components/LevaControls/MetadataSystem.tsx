import React, { useMemo, createContext, useContext } from 'react'
import { InputMetadata, ValidationRule } from './ConfigInput'

// Global metadata registry
export interface ControlMetadata {
  [path: string]: InputMetadata
}

// Control grouping system
export interface ControlGroup {
  id: string
  label: string
  description?: string
  icon?: string
  controls: string[]
  collapsed?: boolean
  color?: string
  priority?: number
}

// Control category for organization
export interface ControlCategory {
  id: string
  label: string
  description?: string
  icon?: string
  groups: string[]
  collapsed?: boolean
}

// Metadata context for sharing control metadata across components
interface MetadataContextType {
  metadata: ControlMetadata
  groups: Record<string, ControlGroup>
  categories: Record<string, ControlCategory>
  registerControl: (path: string, metadata: InputMetadata) => void
  updateControl: (path: string, metadata: Partial<InputMetadata>) => void
  getControlMetadata: (path: string) => InputMetadata | undefined
  getGroupControls: (groupId: string) => string[]
  getCategoryGroups: (categoryId: string) => string[]
  validateControl: (path: string, value: any) => ValidationError[]
}

interface ValidationError {
  message: string
  severity: 'error' | 'warning' | 'info'
  rule?: string
}

// Create metadata context
const MetadataContext = createContext<MetadataContextType | null>(null)

/**
 * Metadata Provider component
 *
 * Provides global metadata management for all controls
 */
export const MetadataProvider: React.FC<{
  children: React.ReactNode
  initialMetadata?: ControlMetadata
  initialGroups?: Record<string, ControlGroup>
  initialCategories?: Record<string, ControlCategory>
}> = ({
  children,
  initialMetadata = {},
  initialGroups = {},
  initialCategories = {}
}) => {
  const contextValue = useMemo<MetadataContextType>(() => ({
    metadata: initialMetadata,
    groups: initialGroups,
    categories: initialCategories,

    registerControl: (path: string, metadata: InputMetadata) => {
      // This would typically update a global state or send to a store
      console.log(`Registering control: ${path}`, metadata)
    },

    updateControl: (path: string, metadata: Partial<InputMetadata>) => {
      // This would typically update a global state or send to a store
      console.log(`Updating control: ${path}`, metadata)
    },

    getControlMetadata: (path: string) => {
      return initialMetadata[path]
    },

    getGroupControls: (groupId: string) => {
      return initialGroups[groupId]?.controls || []
    },

    getCategoryGroups: (categoryId: string) => {
      return initialCategories[categoryId]?.groups || []
    },

    validateControl: (path: string, value: any) => {
      const metadata = initialMetadata[path]
      if (!metadata?.validationRules) return []

      const errors: ValidationError[] = []

      for (const rule of metadata.validationRules) {
        try {
          const isValid = rule.validator(value)
          if (!isValid) {
            errors.push({
              message: rule.errorMessage,
              severity: rule.severity,
              rule: rule.name
            })
          }
        } catch (error) {
          errors.push({
            message: `Validation error: ${error}`,
            severity: 'error',
            rule: rule.name
          })
        }
      }

      return errors
    }
  }), [initialMetadata, initialGroups, initialCategories])

  return (
    <MetadataContext.Provider value={contextValue}>
      {children}
    </MetadataContext.Provider>
  )
}

/**
 * Hook to use metadata context
 */
export const useMetadata = () => {
  const context = useContext(MetadataContext)
  if (!context) {
    throw new Error('useMetadata must be used within a MetadataProvider')
  }
  return context
}

/**
 * Hook to get control metadata with fallbacks
 */
export const useControlMetadata = (path: string) => {
  const { getControlMetadata } = useMetadata()

  return useMemo(() => {
    const metadata = getControlMetadata(path)

    return {
      ...metadata,
      // Default values
      category: metadata?.category || 'general',
      inputType: metadata?.inputType || 'text',
      required: metadata?.required || false,
      advanced: metadata?.advanced || false,
      format: metadata?.format || 'text'
    } as InputMetadata & {
      category: string
      inputType: 'text' | 'number' | 'boolean' | 'select' | 'object' | 'array' | 'vector2' | 'color' | 'file'
      required: boolean
      advanced: boolean
      format: 'text' | 'number' | 'percentage' | 'currency' | 'scientific' | 'bytes'
    }
  }, [path, getControlMetadata])
}

/**
 * Hook to get validation rules for a control
 */
export const useControlValidation = (path: string) => {
  const { getControlMetadata } = useMetadata()

  return useMemo(() => {
    const metadata = getControlMetadata(path)
    return metadata?.validationRules || []
  }, [path, getControlMetadata])
}

/**
 * Hook to get control grouping information
 */
export const useControlGrouping = (path: string) => {
  const { getControlMetadata, groups, categories } = useMetadata()

  return useMemo(() => {
    const metadata = getControlMetadata(path)
    const categoryId = metadata?.category || 'general'

    const category = categories[categoryId]
    const groupIds = category?.groups || []

    const controlGroup = groupIds.find(groupId =>
      groups[groupId]?.controls.includes(path)
    )

    return {
      category,
      group: controlGroup ? groups[controlGroup] : null,
      isGrouped: !!controlGroup
    }
  }, [path, getControlMetadata, groups, categories])
}

/**
 * Utility function to create control metadata
 */
export const createControlMetadata = (
  overrides: Partial<InputMetadata> = {}
): InputMetadata => ({
  category: 'general',
  inputType: 'text',
  required: false,
  advanced: false,
  format: 'text',
  ...overrides
})

/**
 * Utility function to create validation rules
 */
export const createValidationRule = (
  name: string,
  validator: (value: any) => boolean | Promise<boolean>,
  errorMessage: string,
  description?: string,
  severity: 'error' | 'warning' | 'info' = 'error'
): ValidationRule => ({
  name,
  description: description || errorMessage,
  validator,
  errorMessage,
  severity
})

/**
 * Common validation rules
 */
export const ValidationRules = {
  required: (value: any) =>
    createValidationRule(
      'required',
      (val) => val !== null && val !== undefined && val !== '',
      'This field is required',
      'Value cannot be empty or null'
    ),

  minLength: (min: number) =>
    createValidationRule(
      `min_length_${min}`,
      (val) => typeof val === 'string' && val.length >= min,
      `Value must be at least ${min} characters long`,
      `Minimum length is ${min} characters`
    ),

  maxLength: (max: number) =>
    createValidationRule(
      `max_length_${max}`,
      (val) => typeof val === 'string' && val.length <= max,
      `Value must be at most ${max} characters long`,
      `Maximum length is ${max} characters`
    ),

  range: (min: number, max: number) =>
    createValidationRule(
      `range_${min}_${max}`,
      (val) => typeof val === 'number' && val >= min && val <= max,
      `Value must be between ${min} and ${max}`,
      `Valid range is ${min} to ${max}`
    ),

  pattern: (pattern: RegExp, description: string) =>
    createValidationRule(
      'pattern',
      (val) => typeof val === 'string' && pattern.test(val),
      `Value does not match required pattern`,
      description
    ),

  email: () =>
    ValidationRules.pattern(
      /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
      'Must be a valid email address'
    ),

  url: () =>
    ValidationRules.pattern(
      /^https?:\/\/.+/,
      'Must be a valid URL starting with http:// or https://'
    ),

  numeric: () =>
    createValidationRule(
      'numeric',
      (val) => !isNaN(Number(val)),
      'Value must be a valid number',
      'Only numeric values are allowed'
    ),

  integer: () =>
    createValidationRule(
      'integer',
      (val) => Number.isInteger(Number(val)),
      'Value must be a whole number',
      'Only integer values are allowed'
    ),

  positive: () =>
    createValidationRule(
      'positive',
      (val) => Number(val) > 0,
      'Value must be positive',
      'Only positive values are allowed'
    ),

  nonNegative: () =>
    createValidationRule(
      'non_negative',
      (val) => Number(val) >= 0,
      'Value must be non-negative',
      'Only non-negative values are allowed'
    )
}

/**
 * Predefined metadata templates for common control types
 */
export const MetadataTemplates = {
  percentage: (): InputMetadata => ({
    category: 'display',
    inputType: 'number',
    format: 'percentage',
    validationRules: [ValidationRules.range(0, 1)],
    tooltip: 'Value as a percentage (0-100%)'
  }),

  ratio: (): InputMetadata => ({
    category: 'parameters',
    inputType: 'number',
    format: 'number',
    validationRules: [ValidationRules.range(0, 1)],
    tooltip: 'Value as a ratio (0-1)'
  }),

  coordinates: (): InputMetadata => ({
    category: 'position',
    inputType: 'vector2',
    format: 'number',
    tooltip: 'X, Y coordinate pair'
  }),

  color: (greyscaleOnly = false): InputMetadata => ({
    category: 'visualization',
    inputType: 'color',
    format: greyscaleOnly ? 'greyscale' : 'hex',
    tooltip: greyscaleOnly ? 'Greyscale color value' : 'Color value in hex format'
  }),

  filePath: (extensions?: string[]): InputMetadata => ({
    category: 'input',
    inputType: 'file',
    tooltip: extensions ? `File with extension: ${extensions.join(', ')}` : 'File path',
    validationRules: extensions ?
      [ValidationRules.pattern(new RegExp(`\\.(${extensions.join('|')})$`), 'Invalid file extension')] :
      []
  }),

  number: (min?: number, max?: number): InputMetadata => ({
    category: 'parameters',
    inputType: 'number',
    format: 'number',
    validationRules: [
      ...(min !== undefined ? [ValidationRules.range(min, max || Infinity)] : []),
      ValidationRules.numeric()
    ],
    tooltip: `Numeric value${min !== undefined && max !== undefined ? ` between ${min} and ${max}` : ''}`
  })
}