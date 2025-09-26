import React from 'react'
import { StringInputProps } from '@/types/leva'
import styled from 'styled-components'

const StringInputContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`

const StringInputLabel = styled.label`
  font-size: 11px;
  font-weight: 500;
  color: var(--leva-colors-highlight2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

const StringInputField = styled.input`
  width: 100%;
  height: 24px;
  padding: 0 8px;
  background: var(--leva-colors-elevation2);
  border: 1px solid var(--leva-colors-accent1);
  border-radius: var(--leva-radii-sm);
  color: var(--leva-colors-highlight1);
  font-family: var(--leva-fonts-mono);
  font-size: 12px;

  &:focus {
    outline: none;
    border-color: var(--leva-colors-accent2);
    box-shadow: 0 0 0 1px var(--leva-colors-accent2);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  &::placeholder {
    color: var(--leva-colors-accent1);
    opacity: 0.7;
  }
`

const StringInputError = styled.div`
  font-size: 10px;
  color: #ff6b6b;
  margin-top: 2px;
`

export const StringInput: React.FC<StringInputProps> = ({
  value,
  onChange,
  placeholder,
  maxLength,
  label,
  error,
  disabled = false
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!disabled) {
      const newValue = e.target.value

      // Check maxLength if provided
      if (maxLength !== undefined && newValue.length > maxLength) {
        return
      }

      onChange(newValue)
    }
  }

  return (
    <StringInputContainer>
      {label && <StringInputLabel>{label}</StringInputLabel>}
      <StringInputField
        type="text"
        value={value || ''}
        onChange={handleChange}
        placeholder={placeholder}
        maxLength={maxLength}
        disabled={disabled}
      />
      {error && <StringInputError>{error}</StringInputError>}
    </StringInputContainer>
  )
}