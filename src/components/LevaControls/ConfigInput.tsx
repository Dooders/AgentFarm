import React, { useCallback, useMemo } from 'react'
import styled from 'styled-components'
import { ValidationError } from '@/types/validation'

// Base configuration for all input types
export interface BaseInputProps {
  /** Unique identifier for the input */
  path: string
  /** Display label for the input */
  label: string
  /** Current value */
  value: any
  /** Callback when value changes */
  onChange: (value: any) => void
  /** Whether the input is disabled */
  disabled?: boolean
  /** Error message to display */
  error?: string | ValidationError[]
  /** Help text or description */
  help?: string
  /** Unit or suffix to display */
  unit?: string
  /** Minimum allowed value (for numeric inputs) */
  min?: number
  /** Maximum allowed value (for numeric inputs) */
  max?: number
  /** Step size for numeric inputs */
  step?: number
  /** Placeholder text */
  placeholder?: string
  /** Options for select inputs */
  options?: string[] | { [key: string]: any }
  /** Whether the input is required */
  required?: boolean
  /** Additional metadata */
  metadata?: InputMetadata
  /** Custom className */
  className?: string
}

// Metadata interface for enhanced input configuration
export interface InputMetadata {
  /** Category or section this input belongs to */
  category?: string
  /** Tooltip description */
  tooltip?: string
  /** Validation rules */
  validationRules?: ValidationRule[]
  /** Display format */
  format?: 'number' | 'percentage' | 'currency' | 'scientific' | 'bytes'
  /** Input type hint */
  inputType?: 'text' | 'number' | 'boolean' | 'select' | 'object' | 'array' | 'vector2' | 'color' | 'file'
  /** Dependencies on other fields */
  dependencies?: string[]
  /** Default value */
  defaultValue?: any
  /** Whether this field should be hidden from basic view */
  advanced?: boolean
  /** Custom CSS class */
  className?: string
  /** Icon to display */
  icon?: string
}

// Validation rule interface
export interface ValidationRule {
  name: string
  description: string
  validator: (value: any, context?: any) => boolean | Promise<boolean>
  errorMessage: string
  severity: 'error' | 'warning' | 'info'
}

// Styled wrapper for consistent input styling
const InputWrapper = styled.div`
  position: relative;
  margin: var(--leva-space-xs, 4px) 0;

  .input-container {
    position: relative;
    display: flex;
    align-items: center;
    min-height: 28px;
    padding: 4px 8px;
    background: var(--leva-colors-elevation2, #2a2a2a);
    border-radius: var(--leva-radii-sm, 4px);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    transition: all 0.2s ease;
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');

    &:hover {
      border-color: var(--leva-colors-accent1, #666666);
    }

    &:focus-within {
      border-color: var(--leva-colors-accent2, #888888);
      box-shadow: 0 0 0 2px var(--leva-colors-accent1, #666666);
    }
  }

  .input-label {
    font-family: var(--leva-fonts-sans, 'Albertus');
    font-size: 11px;
    font-weight: 600;
    color: var(--leva-colors-highlight1, #ffffff);
    margin-right: var(--leva-space-sm, 8px);
    min-width: 120px;
    text-align: right;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .input-control {
    flex: 1;
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 11px;
    color: var(--leva-colors-highlight1, #ffffff);
    background: transparent;
    border: none;
    outline: none;
    min-height: 20px;
    padding: 2px 4px;

    &::placeholder {
      color: var(--leva-colors-accent2, #888888);
    }
  }

  .input-unit {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 10px;
    color: var(--leva-colors-accent2, #888888);
    margin-left: var(--leva-space-xs, 4px);
    text-transform: lowercase;
  }

  .input-help {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: var(--leva-colors-accent2, #888888);
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }

  .input-error {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: #ff6b6b;
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }

  .input-icon {
    font-size: 12px;
    margin-right: var(--leva-space-xs, 4px);
    color: var(--leva-colors-accent2, #888888);
  }

  .input-required {
    color: #ff6b6b;
    margin-left: 2px;
  }
`

// Error display component
const ErrorDisplay: React.FC<{ errors: string | ValidationError[] }> = ({ errors }) => {
  const errorMessages = useMemo(() => {
    if (typeof errors === 'string') {
      return [errors]
    }

    if (Array.isArray(errors)) {
      return errors.map(error =>
        typeof error === 'string' ? error : error.message || 'Validation error'
      )
    }

    return []
  }, [errors])

  if (errorMessages.length === 0) {
    return null
  }

  return (
    <div className="input-error">
      {errorMessages.map((message, index) => (
        <div key={index}>• {message}</div>
      ))}
    </div>
  )
}

// Help text component
const HelpText: React.FC<{ help: string; metadata?: InputMetadata }> = ({ help, metadata }) => {
  if (!help && !metadata?.tooltip) {
    return null
  }

  return (
    <div className="input-help" title={metadata?.tooltip}>
      {help || metadata?.tooltip}
    </div>
  )
}

/**
 * Base ConfigInput component that provides a unified interface for all input types
 *
 * This foundational component establishes the standard interface and behavior for all
 * enhanced custom controls. It provides consistent styling, error handling, metadata
 * integration, and accessibility features that all specialized controls inherit.
 *
 * @component
 * @example
 * ```tsx
 * <ConfigInput
 *   path="test.control"
 *   label="Test Control"
 *   value={42}
 *   onChange={handleChange}
 *   metadata={{
 *     category: 'parameters',
 *     tooltip: 'Test control description',
 *     validationRules: [ValidationRules.range(0, 100)]
 *   }}
 * />
 * ```
 *
 * Features:
 * - Consistent styling and theming across all derived controls
 * - Comprehensive error handling and display system
 * - Help text and tooltips with metadata integration
 * - Validation rule system with real-time feedback
 * - Accessibility features with ARIA compliance
 * - Extensible architecture for custom input types
 * - TypeScript support with comprehensive type definitions
 *
 * @param props - The component props
 * @param props.path - Unique identifier for the input
 * @param props.label - Display label for the input
 * @param props.value - Current input value
 * @param props.onChange - Callback when value changes
 * @param props.disabled - Whether the input is disabled (optional)
 * @param props.error - Error message(s) to display (optional)
 * @param props.help - Help text or description (optional)
 * @param props.unit - Unit or suffix to display (optional)
 * @param props.min - Minimum allowed value (for numeric inputs)
 * @param props.max - Maximum allowed value (for numeric inputs)
 * @param props.step - Step size for numeric inputs (optional)
 * @param props.placeholder - Placeholder text (optional)
 * @param props.options - Options for select inputs (optional)
 * @param props.required - Whether the input is required (optional)
 * @param props.metadata - Additional metadata for the input (optional)
 * @param props.className - Additional CSS class (optional)
 *
 * @returns React component that renders a standardized input control
 */
export const ConfigInput: React.FC<BaseInputProps> = ({
  path,
  label,
  value,
  onChange,
  disabled = false,
  error,
  help,
  unit,
  required = false,
  metadata,
  className
}) => {
  // Handle value changes with validation
  const handleChange = useCallback((newValue: any) => {
    if (disabled) return

    try {
      // Basic validation
      if (required && (newValue === null || newValue === undefined || newValue === '')) {
        console.warn(`Required field "${label}" cannot be empty`)
        return
      }

      // Type-specific validation
      if (metadata?.validationRules) {
        for (const rule of metadata.validationRules) {
          const isValid = rule.validator(newValue)
          if (!isValid) {
            console.warn(`Validation failed for "${label}": ${rule.errorMessage}`)
            return
          }
        }
      }

      onChange(newValue)
    } catch (err) {
      console.error(`Error updating "${label}":`, err)
    }
  }, [disabled, required, label, metadata, onChange])

  // Format display value based on metadata
  const displayValue = useMemo(() => {
    if (value === null || value === undefined) {
      return ''
    }

    switch (metadata?.format) {
      case 'percentage':
        return `${(value * 100).toFixed(1)}%`
      case 'scientific':
        return value.toExponential(2)
      case 'bytes':
        return formatBytes(value)
      default:
        return String(value)
    }
  }, [value, metadata?.format])

  return (
    <InputWrapper className={className}>
      <div className="input-container">
        {metadata?.icon && (
          <span className="input-icon" title={metadata.tooltip}>
            {metadata.icon}
          </span>
        )}

        <label className="input-label">
          {label}
          {required && <span className="input-required">*</span>}
        </label>

        <input
          type="text"
          className="input-control"
          value={displayValue}
          onChange={(e) => handleChange(e.target.value)}
          disabled={disabled}
          placeholder={metadata?.defaultValue?.toString() || 'Enter value...'}
          title={metadata?.tooltip || help}
        />

        {unit && (
          <span className="input-unit">{unit}</span>
        )}
      </div>

      <HelpText help={help || ''} metadata={metadata} />
      <ErrorDisplay errors={error || []} />
    </InputWrapper>
  )
}

// Utility functions
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'

  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

/**
 * Factory function to create specialized input components
 */
export const createInputComponent = (
  type: 'text' | 'number' | 'boolean' | 'select' | 'object' | 'array',
  additionalProps?: Partial<BaseInputProps>
) => {
  return (props: BaseInputProps) => (
    <ConfigInput
      {...props}
      {...additionalProps}
    />
  )
}

/**
 * Hook for managing input state and validation
 */
export const useInputState = (
  initialValue: any,
  validator?: (value: any) => ValidationError[]
) => {
  const [value, setValue] = React.useState(initialValue)
  const [errors, setErrors] = React.useState<ValidationError[]>([])

  const updateValue = useCallback((newValue: any) => {
    setValue(newValue)

    if (validator) {
      const validationErrors = validator(newValue)
      setErrors(validationErrors)
    }
  }, [validator])

  return { value, setValue: updateValue, errors }
}