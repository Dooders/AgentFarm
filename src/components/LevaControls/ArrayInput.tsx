import React from 'react'
import { ConfigInputProps } from '../../types/leva'
import styled from 'styled-components'
import { NumberInput } from './NumberInput'
import { StringInput } from './StringInput'
import { BooleanInput } from './BooleanInput'

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`

const LabelRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`

const Label = styled.label`
  font-size: 11px;
  font-weight: 500;
  color: var(--leva-colors-highlight2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

const Items = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`

const Row = styled.div`
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  align-items: center;
`

const Button = styled.button`
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

export const ArrayInput: React.FC<ConfigInputProps> = ({
  value,
  onChange,
  label,
  error,
  disabled = false
}) => {
  const items: any[] = Array.isArray(value) ? value : []

  const inferredType = (() => {
    if (items.length === 0) return 'string'
    const first = items[0]
    if (typeof first === 'number') return 'number'
    if (typeof first === 'boolean') return 'boolean'
    if (typeof first === 'string') return 'string'
    return 'string'
  })()

  const updateItem = (index: number, newVal: any) => {
    const next = items.slice()
    next[index] = newVal
    onChange(next)
  }

  const removeItem = (index: number) => {
    const next = items.slice()
    next.splice(index, 1)
    onChange(next)
  }

  const addItem = () => {
    const defaultVal = inferredType === 'number' ? 0 : inferredType === 'boolean' ? false : ''
    onChange([...(items || []), defaultVal])
  }

  return (
    <Container>
      <LabelRow>
        {label && <Label>{label}</Label>}
        <Button type="button" onClick={addItem} disabled={disabled}>Add</Button>
      </LabelRow>
      <Items>
        {items.map((item, idx) => (
          <Row key={idx}>
            {inferredType === 'number' && (
              <NumberInput
                path={`array.${idx}`}
                schema={{}}
                value={item as number}
                onChange={(v: number) => updateItem(idx, v)}
              />
            )}
            {inferredType === 'string' && (
              <StringInput
                path={`array.${idx}`}
                schema={{}}
                value={item as string}
                onChange={(v: string) => updateItem(idx, v)}
              />
            )}
            {inferredType === 'boolean' && (
              <BooleanInput
                path={`array.${idx}`}
                schema={{}}
                value={item as boolean}
                onChange={(v: boolean) => updateItem(idx, v)}
              />
            )}
            <Button type="button" onClick={() => removeItem(idx)} disabled={disabled}>Remove</Button>
          </Row>
        ))}
      </Items>
      {error && <ErrorText>{error}</ErrorText>}
    </Container>
  )
}

