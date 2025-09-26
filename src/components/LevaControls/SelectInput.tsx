import React, { useMemo, useState } from 'react'
import { SelectInputProps } from '../../types/leva'
import styled from 'styled-components'

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`

const Label = styled.label`
  font-size: 11px;
  font-weight: 500;
  color: var(--leva-colors-highlight2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

const SearchInput = styled.input`
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

  &::placeholder {
    color: var(--leva-colors-accent1);
    opacity: 0.7;
  }
`

const Select = styled.select`
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
`

const ClearButton = styled.button`
  height: 28px;
  padding: 0 10px;
  border: 1px solid var(--leva-colors-accent1);
  background: var(--leva-colors-elevation2);
  color: var(--leva-colors-highlight1);
  border-radius: var(--leva-radii-sm);
  font-size: 11px;
  font-family: var(--leva-fonts-sans);
`

const ErrorText = styled.div`
  font-size: 10px;
  color: #ff6b6b;
  margin-top: 2px;
`

type OptionItem = { label: string; value: any }

export const SelectInput: React.FC<SelectInputProps> = ({
  value,
  onChange,
  options,
  label,
  placeholder,
  searchable = true,
  multiple = false,
  clearable = false,
  error,
  disabled = false
}) => {
  const [query, setQuery] = useState('')

  const optionList = useMemo<OptionItem[]>(() => {
    if (Array.isArray(options)) return options.map((v): OptionItem => ({ label: String(v), value: v }))
    return Object.entries(options || {}).map(([k, v]): OptionItem => ({ label: String(k), value: v }))
  }, [options])

  const filtered = useMemo<OptionItem[]>(() => {
    if (!searchable || !query.trim()) return optionList
    const q = query.toLowerCase()
    return optionList.filter((o: OptionItem) => o.label.toLowerCase().includes(q))
  }, [optionList, query, searchable])

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (multiple) {
      const selectedIndices = Array.from(e.target.selectedOptions).map((o) => Number(o.value))
      const selectedValues = selectedIndices
        .map((idx) => filtered[idx])
        .filter(Boolean)
        .map((o) => o.value)
      onChange(selectedValues)
    } else {
      const idx = Number(e.target.value)
      const match = filtered[idx]
      onChange(match ? match.value : e.target.value)
    }
  }

  const handleClear = () => {
    if (multiple) onChange([])
    else onChange('')
  }

  // Compute controlled value(s) as index strings for the rendered options
  const toIndexValue = (val: any): string | '' => {
    const idx = filtered.findIndex((o) => o.value === val)
    return idx >= 0 ? String(idx) : ''
  }
  const controlledValue = multiple
    ? (Array.isArray(value) ? (value.map((v) => toIndexValue(v)).filter(Boolean)) : [])
    : toIndexValue(value)

  return (
    <Container>
      {label && <Label>{label}</Label>}
      {searchable && (
        <SearchInput
          value={query}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
          placeholder={placeholder || 'Search...'}
          disabled={disabled}
        />
      )}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <Select
          value={controlledValue}
          onChange={handleChange}
          multiple={multiple}
          disabled={disabled}
        >
          {filtered.map((o, idx) => (
            <option key={`${o.label}-${idx}`} value={String(idx)}>
              {o.label}
            </option>
          ))}
        </Select>
        {clearable && !disabled && (
          <ClearButton type="button" onClick={handleClear}>Clear</ClearButton>
        )}
      </div>
      {error && <ErrorText>{error}</ErrorText>}
    </Container>
  )
}

