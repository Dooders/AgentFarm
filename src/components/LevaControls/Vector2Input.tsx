import React, { useCallback, useMemo, useState } from 'react'
import styled from 'styled-components'
import { ConfigInput, BaseInputProps, InputMetadata, ValidationRule } from './ConfigInput'
import { ValidationError } from '@/types/validation'

// Vector2 specific props
export interface Vector2Props extends Omit<BaseInputProps, 'value' | 'onChange' | 'min' | 'max' | 'step'> {
  /** Current vector2 value */
  value: { x: number; y: number } | [number, number] | null
  /** Callback when value changes */
  onChange: (value: { x: number; y: number } | [number, number]) => void
  /** Minimum allowed value for both x and y */
  min?: number
  /** Maximum allowed value for both x and y */
  max?: number
  /** Step size for both x and y */
  step?: number
  /** Whether to show coordinate labels */
  showLabels?: boolean
  /** Whether to allow negative values */
  allowNegative?: boolean
  /** Number of decimal places to display */
  precision?: number
  /** Whether to use compact layout */
  compact?: boolean
}

// Styled components for Vector2 input
const Vector2Wrapper = styled.div`
  position: relative;
  margin: var(--leva-space-xs, 4px) 0;

  .vector2-container {
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

  .vector2-label {
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

  .vector2-inputs {
    display: flex;
    gap: var(--leva-space-sm, 8px);
    flex: 1;
  }

  .vector2-input-group {
    display: flex;
    flex-direction: column;
    flex: 1;
  }

  .vector2-input-label {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: var(--leva-colors-accent2, #888888);
    margin-bottom: 2px;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .vector2-input {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 11px;
    color: var(--leva-colors-highlight1, #ffffff);
    background: var(--leva-colors-elevation3, #3a3a3a);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    border-radius: var(--leva-radii-xs, 2px);
    outline: none;
    padding: 2px 4px;
    text-align: center;
    min-height: 20px;
    transition: all 0.2s ease;

    &:hover {
      border-color: var(--leva-colors-accent1, #666666);
    }

    &:focus {
      border-color: var(--leva-colors-accent2, #888888);
      box-shadow: 0 0 0 1px var(--leva-colors-accent1, #666666);
    }

    &::placeholder {
      color: var(--leva-colors-accent2, #888888);
    }
  }

  .vector2-separator {
    font-size: 12px;
    color: var(--leva-colors-accent2, #888888);
    margin: 0 var(--leva-space-xs, 4px);
    user-select: none;
  }

  .vector2-error {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: #ff6b6b;
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }

  .vector2-help {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: var(--leva-colors-accent2, #888888);
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }
`

// Component for individual number input
const NumberInput: React.FC<{
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  precision?: number
  allowNegative?: boolean
  placeholder?: string
  disabled?: boolean
}> = ({
  value,
  onChange,
  min,
  max,
  step = 1,
  precision = 2,
  allowNegative = true,
  placeholder,
  disabled = false
}) => {
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value

    // Allow empty input
    if (inputValue === '') {
      return
    }

    // Parse the number
    const numValue = parseFloat(inputValue)

    // Check if it's a valid number
    if (isNaN(numValue)) {
      return
    }

    // Check negative values
    if (!allowNegative && numValue < 0) {
      return
    }

    // Check min/max constraints
    if (min !== undefined && numValue < min) {
      return
    }
    if (max !== undefined && numValue > max) {
      return
    }

    onChange(numValue)
  }, [onChange, min, max, allowNegative])

  const displayValue = value.toFixed(precision)

  return (
    <input
      type="number"
      className="vector2-input"
      value={displayValue}
      onChange={handleChange}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      placeholder={placeholder}
    />
  )
}

/**
 * Vector2Input component for coordinate pair inputs (x, y)
 *
 * A specialized input component for handling two-dimensional coordinate values.
 * Provides two synchronized numeric inputs for X and Y coordinates with comprehensive
 * validation, formatting, and visual feedback.
 *
 * @component
 * @example
 * ```tsx
 * <Vector2Input
 *   path="display.position"
 *   label="Display Position"
 *   value={{ x: 100, y: 200 }}
 *   onChange={(value) => console.log('Position:', value)}
 *   min={0}
 *   max={1000}
 *   step={1}
 *   showLabels={true}
 *   precision={0}
 *   help="X, Y coordinates for display positioning"
 * />
 * ```
 *
 * Features:
 * - Two numeric inputs for x and y coordinates with proper validation
 * - Min/max/step constraints and precision control for both coordinates
 * - Optional coordinate labels (X/Y) for better user guidance
 * - Negative value control with configurable allowance
 * - Compact layout option for space-constrained interfaces
 * - Real-time value synchronization between inputs
 * - Consistent theming with greyscale professional design
 * - Error handling and validation feedback
 *
 * @param props - The component props
 * @param props.path - Unique identifier for the input
 * @param props.label - Display label for the coordinate pair
 * @param props.value - Current coordinate values as object or array
 * @param props.onChange - Callback when coordinate values change
 * @param props.min - Minimum allowed value for both x and y (optional)
 * @param props.max - Maximum allowed value for both x and y (optional)
 * @param props.step - Step size for increments (default: 1)
 * @param props.showLabels - Show X/Y labels (default: true)
 * @param props.allowNegative - Allow negative coordinate values (default: true)
 * @param props.precision - Decimal places to display (default: 2)
 * @param props.compact - Use compact layout (default: false)
 * @param props.disabled - Disable input interaction (default: false)
 * @param props.error - Error message to display (optional)
 * @param props.help - Help text or description (optional)
 * @param props.metadata - Additional metadata for the input (optional)
 * @param props.className - Additional CSS class (optional)
 *
 * @returns React component that renders coordinate input controls
 */
export const Vector2Input: React.FC<Vector2Props> = ({
  path,
  label,
  value,
  onChange,
  disabled = false,
  error,
  help,
  metadata,
  min,
  max,
  step = 1,
  showLabels = true,
  allowNegative = true,
  precision = 2,
  required = false,
  compact = false,
  className
}) => {
  // Normalize value to consistent format
  const normalizedValue = useMemo(() => {
    if (!value) return { x: 0, y: 0 }

    if (Array.isArray(value)) {
      return { x: value[0] || 0, y: value[1] || 0 }
    }

    return value
  }, [value])

  // Handle individual coordinate changes
  const handleXChange = useCallback((newX: number) => {
    onChange({ x: newX, y: normalizedValue.y })
  }, [onChange, normalizedValue.y])

  const handleYChange = useCallback((newY: number) => {
    onChange({ x: normalizedValue.x, y: newY })
  }, [onChange, normalizedValue.x])

  // Validation rules
  const validationRules: ValidationRule[] = useMemo(() => [
    {
      name: 'valid_coordinates',
      description: 'Coordinates must be valid numbers',
      validator: (val) => {
        if (!val || typeof val !== 'object') return false
        return typeof val.x === 'number' && typeof val.y === 'number' &&
               !isNaN(val.x) && !isNaN(val.y)
      },
      errorMessage: 'Coordinates must be valid numbers',
      severity: 'error'
    },
    {
      name: 'within_bounds',
      description: 'Coordinates must be within specified bounds',
      validator: (val) => {
        if (!val || typeof val !== 'object') return false
        const x = val.x
        const y = val.y

        if (min !== undefined && (x < min || y < min)) return false
        if (max !== undefined && (x > max || y > max)) return false
        return true
      },
      errorMessage: `Coordinates must be between ${min || '-∞'} and ${max || '∞'}`,
      severity: 'error'
    }
  ], [min, max])

  // Enhanced metadata
  const enhancedMetadata: InputMetadata = useMemo(() => ({
    ...metadata,
    inputType: 'vector2',
    validationRules,
    format: 'number',
    tooltip: metadata?.tooltip || `Enter ${label.toLowerCase()} coordinates (x, y)`
  }), [metadata, validationRules, label])

  // Error messages
  const errorMessages = useMemo(() => {
    if (!error) return []

    if (typeof error === 'string') {
      return [error]
    }

    if (Array.isArray(error)) {
      return error.map(e => typeof e === 'string' ? e : e.message || 'Validation error')
    }

    return []
  }, [error])

  return (
    <Vector2Wrapper className={className}>
      <div className="vector2-container">
        <label className="vector2-label">
          {label}
          {required && <span style={{ color: '#ff6b6b', marginLeft: '2px' }}>*</span>}
        </label>

        <div className="vector2-inputs">
          <div className="vector2-input-group">
            {showLabels && <div className="vector2-input-label">X</div>}
            <NumberInput
              value={normalizedValue.x}
              onChange={handleXChange}
              min={min}
              max={max}
              step={step}
              precision={precision}
              allowNegative={allowNegative}
              disabled={disabled}
              placeholder="0"
            />
          </div>

          <div className="vector2-separator">×</div>

          <div className="vector2-input-group">
            {showLabels && <div className="vector2-input-label">Y</div>}
            <NumberInput
              value={normalizedValue.y}
              onChange={handleYChange}
              min={min}
              max={max}
              step={step}
              precision={precision}
              allowNegative={allowNegative}
              disabled={disabled}
              placeholder="0"
            />
          </div>
        </div>
      </div>

      {help && (
        <div className="vector2-help" title={enhancedMetadata.tooltip}>
          {help}
        </div>
      )}

      {errorMessages.length > 0 && (
        <div className="vector2-error">
          {errorMessages.map((message, index) => (
            <div key={index}>• {message}</div>
          ))}
        </div>
      )}
    </Vector2Wrapper>
  )
}

/**
 * Utility function to create Vector2Input with preset configuration
 */
export const createVector2Input = (
  label: string,
  config: Partial<Vector2Props>
) => {
  return (props: Vector2Props) => (
    <Vector2Input
      label={label}
      {...config}
      {...props}
    />
  )
}

/**
 * Hook for managing Vector2 state with validation
 */
export const useVector2State = (
  initialValue: { x: number; y: number } | [number, number] = { x: 0, y: 0 },
  validator?: (value: { x: number; y: number }) => ValidationError[]
) => {
  const [value, setValue] = useState<{ x: number; y: number }>(
    Array.isArray(initialValue) ? { x: initialValue[0], y: initialValue[1] } : initialValue
  )
  const [errors, setErrors] = useState<ValidationError[]>([])

  const updateValue = useCallback((newValue: { x: number; y: number }) => {
    setValue(newValue)

    if (validator) {
      const validationErrors = validator(newValue)
      setErrors(validationErrors)
    }
  }, [validator])

  return { value, setValue: updateValue, errors }
}