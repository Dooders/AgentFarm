import React, { useCallback, useMemo, useState } from 'react'
import styled from 'styled-components'
import { ConfigInput, BaseInputProps, InputMetadata, ValidationRule } from './ConfigInput'
import { ValidationError } from '@/types/validation'

// Color input specific props
export interface ColorProps extends Omit<BaseInputProps, 'value' | 'onChange'> {
  /** Current color value */
  value: string | { r: number; g: number; b: number; a?: number } | null
  /** Callback when value changes */
  onChange: (value: string | { r: number; g: number; b: number; a?: number }) => void
  /** Color format to use */
  format?: 'hex' | 'rgb' | 'rgba' | 'greyscale'
  /** Whether to show alpha channel */
  showAlpha?: boolean
  /** Whether to force greyscale mode */
  greyscaleOnly?: boolean
  /** Whether to show color preview */
  showPreview?: boolean
  /** Color presets for quick selection */
  presets?: string[]
  /** Whether to use compact layout */
  compact?: boolean
}

// Styled components for Color input
const ColorWrapper = styled.div`
  position: relative;
  margin: var(--leva-space-xs, 4px) 0;

  .color-container {
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

  .color-label {
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

  .color-controls {
    display: flex;
    align-items: center;
    gap: var(--leva-space-sm, 8px);
    flex: 1;
  }

  .color-input {
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
    flex: 1;

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

  .color-preview {
    width: 20px;
    height: 20px;
    border-radius: var(--leva-radii-xs, 2px);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    cursor: pointer;
    transition: all 0.2s ease;

    &:hover {
      transform: scale(1.1);
      box-shadow: 0 0 0 2px var(--leva-colors-accent1, #666666);
    }
  }

  .color-presets {
    display: flex;
    gap: var(--leva-space-xs, 4px);
    flex-wrap: wrap;
    max-width: 120px;
  }

  .color-preset {
    width: 16px;
    height: 16px;
    border-radius: var(--leva-radii-xs, 2px);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    cursor: pointer;
    transition: all 0.2s ease;

    &:hover {
      transform: scale(1.2);
      box-shadow: 0 0 0 2px var(--leva-colors-accent1, #666666);
    }
  }

  .color-error {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: #ff6b6b;
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }

  .color-help {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: var(--leva-colors-accent2, #888888);
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }
`

// Color preview component
const ColorPreview: React.FC<{
  color: string
  onClick?: () => void
}> = ({ color, onClick }) => {
  return (
    <div
      className="color-preview"
      style={{ backgroundColor: color }}
      onClick={onClick}
      title={`Current color: ${color}`}
    />
  )
}

// Color preset component
const ColorPreset: React.FC<{
  color: string
  onClick: (color: string) => void
  isSelected?: boolean
}> = ({ color, onClick, isSelected = false }) => {
  return (
    <div
      className="color-preset"
      style={{
        backgroundColor: color,
        borderColor: isSelected ? 'var(--leva-colors-accent2, #888888)' : undefined,
        boxShadow: isSelected ? '0 0 0 2px var(--leva-colors-accent1, #666666)' : undefined
      }}
      onClick={() => onClick(color)}
      title={color}
    />
  )
}

// Default greyscale presets
const GREYSCALE_PRESETS = [
  '#000000', '#1a1a1a', '#2a2a2a', '#3a3a3a', '#4a4a4a',
  '#5a5a5a', '#6a6a6a', '#7a7a7a', '#8a8a8a', '#9a9a9a',
  '#aaaaaa', '#bababa', '#cacaca', '#dadada', '#eaeaea',
  '#fafafa', '#ffffff'
]

// Default color presets (for non-greyscale mode)
const COLOR_PRESETS = [
  '#000000', '#1a1a1a', '#333333', '#555555', '#777777',
  '#999999', '#aaaaaa', '#cccccc', '#ffffff', '#ff0000',
  '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
  '#ffa500', '#800080', '#008000', '#ff69b4', '#4a9eff'
]

/**
 * ColorInput component for color parameters with greyscale compatibility
 *
 * Features:
 * - Multiple color formats (hex, rgb, rgba, greyscale)
 * - Greyscale-only mode for professional themes
 * - Color preview and presets
 * - Alpha channel support
 * - Validation for color values
 * - Consistent theming
 */
export const ColorInput: React.FC<ColorProps> = ({
  path,
  label,
  value,
  onChange,
  disabled = false,
  error,
  help,
  metadata,
  format = 'hex',
  showAlpha = false,
  greyscaleOnly = false,
  showPreview = true,
  presets,
  required = false,
  compact = false,
  className
}) => {
  // Normalize color value to consistent format
  const normalizedValue = useMemo(() => {
    if (!value) return '#1a1a1a'

    if (typeof value === 'string') {
      return value
    }

    if (typeof value === 'object' && 'r' in value) {
      const { r, g, b, a = 1 } = value
      if (greyscaleOnly || (r === g && g === b)) {
        // Convert to greyscale hex
        const grey = Math.round((r + g + b) / 3)
        return showAlpha && a < 1
          ? `rgba(${grey}, ${grey}, ${grey}, ${a})`
          : `#${grey.toString(16).padStart(2, '0').repeat(3)}`
      }

      return showAlpha && a < 1
        ? `rgba(${r}, ${g}, ${b}, ${a})`
        : `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
    }

    return '#1a1a1a'
  }, [value, greyscaleOnly, showAlpha])

  // Get presets based on mode
  const colorPresets = useMemo(() => {
    return presets || (greyscaleOnly ? GREYSCALE_PRESETS : COLOR_PRESETS)
  }, [presets, greyscaleOnly])

  // Handle color input change
  const handleColorChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value

    // Validate color format
    if (format === 'hex' && !isValidHexColor(inputValue)) {
      return
    }

    if (format === 'rgb' && !isValidRgbColor(inputValue)) {
      return
    }

    if (format === 'rgba' && !isValidRgbaColor(inputValue)) {
      return
    }

    onChange(inputValue)
  }, [format, onChange])

  // Handle preset selection
  const handlePresetSelect = useCallback((color: string) => {
    onChange(color)
  }, [onChange])

  // Validation rules
  const validationRules: ValidationRule[] = useMemo(() => [
    {
      name: 'valid_color_format',
      description: `Color must be a valid ${format} color`,
      validator: (val) => {
        if (!val || typeof val !== 'string') return false

        switch (format) {
          case 'hex':
            return isValidHexColor(val)
          case 'rgb':
            return isValidRgbColor(val)
          case 'rgba':
            return isValidRgbaColor(val)
          case 'greyscale':
            return isValidGreyscaleColor(val)
          default:
            return false
        }
      },
      errorMessage: `Invalid ${format} color format`,
      severity: 'error'
    }
  ], [format])

  // Enhanced metadata
  const enhancedMetadata: InputMetadata = useMemo(() => ({
    ...metadata,
    inputType: 'color',
    validationRules,
    format: greyscaleOnly ? 'greyscale' : format,
    tooltip: metadata?.tooltip || `Enter ${label.toLowerCase()} color (${format} format)`
  }), [metadata, validationRules, format, greyscaleOnly, label])

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
    <ColorWrapper className={className}>
      <div className="color-container">
        <label className="color-label">
          {label}
          {required && <span style={{ color: '#ff6b6b', marginLeft: '2px' }}>*</span>}
        </label>

        <div className="color-controls">
          {showPreview && (
            <ColorPreview
              color={normalizedValue}
              onClick={() => {/* Could open color picker modal */}}
            />
          )}

          <input
            type="text"
            className="color-input"
            value={normalizedValue}
            onChange={handleColorChange}
            disabled={disabled}
            placeholder={greyscaleOnly ? '#1a1a1a' : '#000000'}
            title={enhancedMetadata.tooltip}
          />

          <div className="color-presets">
            {colorPresets.slice(0, compact ? 8 : 12).map((color) => (
              <ColorPreset
                key={color}
                color={color}
                onClick={handlePresetSelect}
                isSelected={color === normalizedValue}
              />
            ))}
          </div>
        </div>
      </div>

      {help && (
        <div className="color-help" title={enhancedMetadata.tooltip}>
          {help}
        </div>
      )}

      {errorMessages.length > 0 && (
        <div className="color-error">
          {errorMessages.map((message, index) => (
            <div key={index}>• {message}</div>
          ))}
        </div>
      )}
    </ColorWrapper>
  )
}

// Color validation functions
function isValidHexColor(color: string): boolean {
  const hexPattern = /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/
  return hexPattern.test(color)
}

function isValidRgbColor(color: string): boolean {
  const rgbPattern = /^rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)$/
  const match = color.match(rgbPattern)

  if (!match) return false

  const [, r, g, b] = match.map(Number)
  return r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255
}

function isValidRgbaColor(color: string): boolean {
  const rgbaPattern = /^rgba\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3}),\s*(0?\.?\d{1,3}|1)\)$/
  const match = color.match(rgbaPattern)

  if (!match) return false

  const [, r, g, b, a] = match.map(Number)
  return r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255 &&
         a >= 0 && a <= 1
}

function isValidGreyscaleColor(color: string): boolean {
  if (isValidHexColor(color)) {
    const hex = color.substring(1)
    if (hex.length === 3) {
      return hex[0] === hex[1] && hex[1] === hex[2]
    }
    if (hex.length === 6) {
      return hex.substring(0, 2) === hex.substring(2, 4) &&
             hex.substring(2, 4) === hex.substring(4, 6)
    }
  }

  if (isValidRgbColor(color)) {
    const match = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/)
    if (match) {
      const [, r, g, b] = match.slice(1).map(Number)
      return r === g && g === b
    }
  }

  return false
}

/**
 * Utility function to create ColorInput with preset configuration
 */
export const createColorInput = (
  label: string,
  config: Partial<ColorProps>
) => {
  return (props: ColorProps) => (
    <ColorInput
      label={label}
      {...config}
      {...props}
    />
  )
}

/**
 * Hook for managing color state with validation
 */
export const useColorState = (
  initialValue: string = '#1a1a1a',
  validator?: (value: string) => ValidationError[]
) => {
  const [value, setValue] = useState<string>(initialValue)
  const [errors, setErrors] = useState<ValidationError[]>([])

  const updateValue = useCallback((newValue: string) => {
    setValue(newValue)

    if (validator) {
      const validationErrors = validator(newValue)
      setErrors(validationErrors)
    }
  }, [validator])

  return { value, setValue: updateValue, errors }
}