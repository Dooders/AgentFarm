import React from 'react'
import { NumberInputProps } from '@/types/leva'
import styled from 'styled-components'

const NumberInputContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`

const NumberInputLabel = styled.label`
  font-size: 11px;
  font-weight: 500;
  color: var(--leva-colors-highlight2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

const NumberInputField = styled.input`
  width: 100%;
  height: 28px;
  padding: 0 8px;
  background: var(--leva-colors-elevation2);
  border: 1px solid var(--leva-colors-accent1);
  border-radius: var(--leva-radii-sm);
  color: var(--leva-colors-highlight1);
  font-family: var(--leva-fonts-mono);
  font-size: 11px;

  &:focus {
    outline: none;
    border-color: var(--leva-colors-accent2);
    box-shadow: 0 0 0 1px var(--leva-colors-accent2);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`

const NumberInputError = styled.div`
  font-size: 10px;
  color: #ff6b6b;
  margin-top: 2px;
`

export const NumberInput: React.FC<NumberInputProps> = ({
  value,
  onChange,
  min,
  max,
  step = 1,
  label,
  error,
  disabled = false
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { value: inputValue } = e.target

    // Allow empty input (user clearing the field)
    if (inputValue === '') {
      onChange('')
      return
    }

    // Support scientific notation and standard floats
    const sciMatch = inputValue.match(/^[-+]?\d*\.?\d+(e[-+]?\d+)?$/i)
    const numValue = sciMatch ? Number(inputValue) : parseFloat(inputValue)

    // Allow intermediate values (e.g. '-', '.', '1.') for better UX
    if (isNaN(numValue)) {
      // Only allow valid intermediate characters
      if (inputValue === '-' || inputValue === '.' || inputValue.endsWith('.')) {
        onChange(inputValue)
      }
      return
    }

    // Validate against min/max if provided
    if (min !== undefined && numValue < min) return
    if (max !== undefined && numValue > max) return
    onChange(numValue)
  }

  const handleIncrement = () => {
    const currentValue = typeof value === 'number' ? value : 0
    const maxBoundary = max ?? Infinity
    const newValue = Math.min(currentValue + step, maxBoundary)
    onChange(newValue)
  }

  const handleDecrement = () => {
    const currentValue = typeof value === 'number' ? value : 0
    const minBoundary = min ?? -Infinity
    const newValue = Math.max(currentValue - step, minBoundary)
    onChange(newValue)
  }

  return (
    <NumberInputContainer>
      {label && <NumberInputLabel>{label}</NumberInputLabel>}
      <div style={{ position: 'relative' }}>
        <NumberInputField
          type="number"
          value={value}
          onChange={handleChange}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
        />
        <div style={{
          position: 'absolute',
          right: '4px',
          top: '50%',
          transform: 'translateY(-50%)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1px'
        }}>
          <button
            type="button"
            onClick={handleIncrement}
            disabled={disabled || (max !== undefined && value >= max)}
            style={{
              width: '12px',
              height: '10px',
              background: 'var(--leva-colors-accent2)',
              border: 'none',
              borderRadius: '2px',
              cursor: disabled || (max !== undefined && value >= max) ? 'not-allowed' : 'pointer',
              opacity: disabled || (max !== undefined && value >= max) ? 0.5 : 1,
              fontSize: '8px',
              color: 'var(--leva-colors-elevation1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            ▲
          </button>
          <button
            type="button"
            onClick={handleDecrement}
            disabled={disabled || (min !== undefined && value <= min)}
            style={{
              width: '12px',
              height: '10px',
              background: 'var(--leva-colors-accent2)',
              border: 'none',
              borderRadius: '2px',
              cursor: disabled || (min !== undefined && value <= min) ? 'not-allowed' : 'pointer',
              opacity: disabled || (min !== undefined && value <= min) ? 0.5 : 1,
              fontSize: '8px',
              color: 'var(--leva-colors-elevation1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            ▼
          </button>
        </div>
      </div>
      {error && <NumberInputError>{error}</NumberInputError>}
    </NumberInputContainer>
  )
}