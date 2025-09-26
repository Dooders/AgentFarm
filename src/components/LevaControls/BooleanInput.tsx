import React from 'react'
import { BooleanInputProps } from '@/types/leva'
import styled from 'styled-components'

const BooleanInputContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`

const BooleanInputLabel = styled.label`
  font-size: 11px;
  font-weight: 500;
  color: var(--leva-colors-highlight2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
`

const BooleanInputField = styled.input`
  width: 16px;
  height: 16px;
  background: var(--leva-colors-elevation2);
  border: 2px solid var(--leva-colors-accent1);
  border-radius: var(--leva-radii-sm);
  cursor: pointer;
  appearance: none;
  position: relative;

  &:checked {
    background: var(--leva-colors-accent2);
    border-color: var(--leva-colors-accent2);
  }

  &:checked::after {
    content: '✓';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: var(--leva-colors-elevation1);
    font-size: 12px;
    font-weight: bold;
  }

  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px var(--leva-colors-accent2);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`

const BooleanInputError = styled.div`
  font-size: 10px;
  color: #ff6b6b;
  margin-top: 2px;
`

export const BooleanInput: React.FC<BooleanInputProps> = ({
  value,
  onChange,
  label,
  error,
  disabled = false
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!disabled) {
      onChange(e.target.checked)
    }
  }

  return (
    <BooleanInputContainer>
      <BooleanInputLabel>
        <BooleanInputField
          type="checkbox"
          checked={!!value}
          onChange={handleChange}
          disabled={disabled}
        />
        {label && <span>{label}</span>}
      </BooleanInputLabel>
      {error && <BooleanInputError>{error}</BooleanInputError>}
    </BooleanInputContainer>
  )
}