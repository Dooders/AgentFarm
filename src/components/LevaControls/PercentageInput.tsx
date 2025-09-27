import React, { useCallback, useMemo, useRef, useState } from 'react'
import styled from 'styled-components'
import { ConfigInput, BaseInputProps, InputMetadata, ValidationRule } from './ConfigInput'
import { ValidationError } from '@/types/validation'

// Percentage input specific props
export interface PercentageProps extends Omit<BaseInputProps, 'value' | 'onChange' | 'min' | 'max' | 'step'> {
  /** Current percentage value (0-1 or 0-100) */
  value: number | null
  /** Callback when value changes */
  onChange: (value: number) => void
  /** Minimum allowed value */
  min?: number
  /** Maximum allowed value */
  max?: number
  /** Step size for increments */
  step?: number
  /** Whether to show as percentage (0-100) or decimal (0-1) */
  asPercentage?: boolean
  /** Whether to show progress bar visualization */
  showProgress?: boolean
  /** Whether to show slider control */
  showSlider?: boolean
  /** Color scheme for progress bar */
  progressColor?: string
  /** Background color for progress bar */
  backgroundColor?: string
  /** Whether to show value labels on progress bar */
  showValueLabels?: boolean
  /** Number of decimal places to display */
  precision?: number
  /** Whether to use compact layout */
  compact?: boolean
}

// Styled components for Percentage input
const PercentageWrapper = styled.div`
  position: relative;
  margin: var(--leva-space-xs, 4px) 0;

  .percentage-container {
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

  .percentage-label {
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

  .percentage-controls {
    display: flex;
    align-items: center;
    gap: var(--leva-space-sm, 8px);
    flex: 1;
  }

  .percentage-input {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 11px;
    color: var(--leva-colors-highlight1, #ffffff);
    background: var(--leva-colors-elevation3, #3a3a3a);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    border-radius: var(--leva-radii-xs, 2px);
    outline: none;
    padding: 2px 4px;
    min-height: 20px;
    transition: all 0.2s ease;
    width: 80px;
    text-align: center;

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

  .percentage-slider {
    flex: 1;
    height: 6px;
    background: var(--leva-colors-elevation1, #1a1a1a);
    border-radius: 3px;
    position: relative;
    cursor: pointer;
    overflow: hidden;

    &:hover {
      background: var(--leva-colors-elevation2, #2a2a2a);
    }
  }

  .percentage-progress {
    height: 100%;
    background: var(--progress-color, var(--leva-colors-accent1, #666666));
    border-radius: 3px;
    transition: width 0.2s ease;
    position: relative;
  }

  .percentage-progress-value {
    position: absolute;
    right: 4px;
    top: 50%;
    transform: translateY(-50%);
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 8px;
    color: var(--leva-colors-highlight1, #ffffff);
    text-shadow: 0 0 2px rgba(0, 0, 0, 0.5);
  }

  .percentage-progress-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 2px;
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 7px;
    color: var(--leva-colors-accent2, #888888);
  }

  .percentage-progress-label {
    font-size: 7px;
    color: var(--leva-colors-accent2, #888888);
  }

  .percentage-unit {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 10px;
    color: var(--leva-colors-accent2, #888888);
    margin-left: var(--leva-space-xs, 4px);
  }

  .percentage-error {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: #ff6b6b;
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }

  .percentage-help {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: var(--leva-colors-accent2, #888888);
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }
`

// Slider component for percentage input
const PercentageSlider: React.FC<{
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  showValueLabels?: boolean
  progressColor?: string
  backgroundColor?: string
  asPercentage: boolean
  debounceMs?: number
  precision?: number
}> = ({
  value,
  min,
  max,
  step,
  onChange,
  showValueLabels = false,
  progressColor = 'var(--leva-colors-accent1, #666666)',
  backgroundColor = 'var(--leva-colors-elevation1, #1a1a1a)',
  asPercentage,
  debounceMs = 16,
  precision
}) => {
  const [isDragging, setIsDragging] = useState(false)
  const rafIdRef = useRef<number | null>(null)
  const pendingValueRef = useRef<number | null>(null)

  // Convert value to 0-1 range for slider
  const normalizedValue = (value - min) / (max - min)
  const clampedValue = Math.max(0, Math.min(1, normalizedValue))

  const flushPending = useCallback(() => {
    if (pendingValueRef.current == null) return
    onChange(pendingValueRef.current)
    pendingValueRef.current = null
    rafIdRef.current = null
  }, [onChange])

  const scheduleChange = useCallback((nextValue: number) => {
    pendingValueRef.current = nextValue
    if (rafIdRef.current == null) {
      rafIdRef.current = requestAnimationFrame(flushPending)
    }
  }, [flushPending])

  const updateValueFromMouse = useCallback((e: MouseEvent | React.MouseEvent, sliderElement: HTMLElement) => {
    const rect = sliderElement.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const newValue = min + x * (max - min)
    const steppedValue = Math.round(newValue / step) * step
    const clamped = Math.max(min, Math.min(max, steppedValue))
    scheduleChange(clamped)
  }, [min, max, step, scheduleChange])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true)
    updateValueFromMouse(e, e.currentTarget as HTMLElement)
  }, [updateValueFromMouse])

  const sliderRef = React.useRef<HTMLDivElement>(null)

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !sliderRef.current) return
    updateValueFromMouse(e, sliderRef.current)
  }, [isDragging, updateValueFromMouse])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
    flushPending()
  }, [])

  // Add/remove event listeners
  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      return () => {
        if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current)
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  const formatValue = (val: number) => {
    const usedPrecision = typeof precision === 'number' ? precision : (asPercentage ? 0 : 2)
    return asPercentage ? `${(val * 100).toFixed(usedPrecision)}%` : val.toFixed(usedPrecision)
  }

  return (
    <div
      ref={sliderRef}
      className="percentage-slider"
      style={{ backgroundColor }}
      role="slider"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-label="percentage slider"
      tabIndex={0}
      onMouseDown={handleMouseDown}
    >
      <div
        className="percentage-progress"
        style={{
          width: `${clampedValue * 100}%`,
          backgroundColor: progressColor
        }}
      >
        {showValueLabels && (
          <div className="percentage-progress-value">
            {formatValue(value)}
          </div>
        )}
      </div>

      {showValueLabels && (
        <div className="percentage-progress-labels">
          <span className="percentage-progress-label">
            {formatValue(min)}
          </span>
          <span className="percentage-progress-label">
            {formatValue(max)}
          </span>
        </div>
      )}
    </div>
  )
}

// Number input component for percentage
const PercentageNumberInput: React.FC<{
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  precision?: number
  asPercentage: boolean
  placeholder?: string
  disabled?: boolean
}> = ({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  precision = 2,
  asPercentage = true,
  placeholder,
  disabled = false
}) => {
  const [raw, setRaw] = useState<string>('')
  const [localError, setLocalError] = useState<string | null>(null)

  // Keep raw input in sync when external value changes
  React.useEffect(() => {
    const formatted = asPercentage ? `${(value * 100).toFixed(precision)}%` : value.toFixed(precision)
    setRaw(formatted)
  }, [value, precision, asPercentage])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value
    setRaw(inputValue)
    setLocalError(null)

    // Allow partial states such as '-', '1.', '50%'
    const cleaned = inputValue.trim()
    if (cleaned === '' || cleaned === '-' || cleaned.endsWith('.') || cleaned === '%') {
      return
    }

    // Parse the number component
    const numericPart = cleaned.replace('%', '')
    let numValue = parseFloat(numericPart)
    if (isNaN(numValue)) {
      setLocalError('Enter a valid number')
      return
    }
    if (asPercentage) {
      numValue = numValue / 100
    }

    if (numValue < min || numValue > max) {
      setLocalError(`Value must be between ${min} and ${max}`)
      return
    }

    onChange(numValue)
  }, [onChange, min, max, asPercentage])

  const displayValue = asPercentage
    ? `${(value * 100).toFixed(precision)}%`
    : value.toFixed(precision)

  return (
    <>
      <input
        type="text"
        className="percentage-input"
        value={raw}
        onChange={handleChange}
        disabled={disabled}
        placeholder={placeholder}
        aria-invalid={!!localError}
        aria-describedby={localError ? 'percentage-input-error' : undefined}
      />
      {localError && (
        <div id="percentage-input-error" className="percentage-error">{localError}</div>
      )}
    </>
  )
}

/**
 * PercentageInput component for ratio/percentage values with visual indicators
 *
 * Features:
 * - Numeric input with percentage formatting
 * - Optional slider control
 * - Optional progress bar visualization
 * - Min/max/step validation
 * - Visual feedback for value ranges
 * - Consistent theming
 */
export const PercentageInput: React.FC<PercentageProps> = ({
  path,
  label,
  value,
  onChange,
  disabled = false,
  error,
  help,
  metadata,
  min = 0,
  max = 1,
  step = 0.01,
  asPercentage = true,
  showProgress = true,
  showSlider = false,
  progressColor,
  backgroundColor,
  showValueLabels = false,
  precision = 2,
  required = false,
  compact = false,
  className
}) => {
  // Normalize value to 0-1 range
  const normalizedValue = useMemo(() => {
    if (value === null || value === undefined) return 0
    return Math.max(min, Math.min(max, value))
  }, [value, min, max])

  // Handle number input change
  const handleNumberChange = useCallback((newValue: number) => {
    onChange(newValue)
  }, [onChange])

  // Handle slider change
  const handleSliderChange = useCallback((newValue: number) => {
    onChange(newValue)
  }, [onChange])

  // Validation rules
  const validationRules: ValidationRule[] = useMemo(() => [
    {
      name: 'valid_percentage_range',
      description: `Value must be between ${min} and ${max}`,
      validator: (val) => {
        if (typeof val !== 'number') return false
        return val >= min && val <= max
      },
      errorMessage: `Value must be between ${min} and ${max}`,
      severity: 'error'
    },
    {
      name: 'valid_percentage_format',
      description: 'Value must be a valid number',
      validator: (val) => {
        return typeof val === 'number' && !isNaN(val) && isFinite(val)
      },
      errorMessage: 'Value must be a valid number',
      severity: 'error'
    }
  ], [min, max])

  // Enhanced metadata
  const enhancedMetadata: InputMetadata = useMemo(() => ({
    ...metadata,
    inputType: 'number',
    validationRules,
    format: 'percentage',
    tooltip: metadata?.tooltip || `${label} (${asPercentage ? '0-100%' : '0-1'})`
  }), [metadata, validationRules, label, asPercentage])

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
    <PercentageWrapper className={className}>
      <div className="percentage-container">
        <label className="percentage-label">
          {label}
          {required && <span style={{ color: '#ff6b6b', marginLeft: '2px' }}>*</span>}
        </label>

        <div className="percentage-controls">
          <PercentageNumberInput
            value={normalizedValue}
            onChange={handleNumberChange}
            min={min}
            max={max}
            step={step}
            precision={precision}
            asPercentage={asPercentage}
            disabled={disabled}
            placeholder={asPercentage ? "50%" : "0.5"}
          />

          <span className="percentage-unit">
            {asPercentage ? '%' : 'ratio'}
          </span>

          {showProgress && (
            <PercentageSlider
              value={normalizedValue}
              min={min}
              max={max}
              step={step}
              onChange={handleSliderChange}
              showValueLabels={showValueLabels}
              progressColor={progressColor}
              backgroundColor={backgroundColor}
              asPercentage={asPercentage}
              debounceMs={16}
            precision={precision}
            />
          )}
        </div>
      </div>

      {help && (
        <div className="percentage-help" title={enhancedMetadata.tooltip}>
          {help}
        </div>
      )}

      {errorMessages.length > 0 && (
        <div className="percentage-error">
          {errorMessages.map((message, index) => (
            <div key={index}>• {message}</div>
          ))}
        </div>
      )}
    </PercentageWrapper>
  )
}

/**
 * Utility function to create PercentageInput with preset configuration
 */
export const createPercentageInput = (
  label: string,
  config: Partial<PercentageProps>
) => {
  return (props: PercentageProps) => (
    <PercentageInput
      {...config}
      {...props}
    />
  )
}

/**
 * Hook for managing percentage state with validation
 */
export const usePercentageState = (
  initialValue: number = 0.5,
  validator?: (value: number) => ValidationError[]
) => {
  const [value, setValue] = useState<number>(initialValue)
  const [errors, setErrors] = useState<ValidationError[]>([])

  const updateValue = useCallback((newValue: number) => {
    setValue(newValue)

    if (validator) {
      const validationErrors = validator(newValue)
      setErrors(validationErrors)
    }
  }, [validator])

  return { value, setValue: updateValue, errors }
}