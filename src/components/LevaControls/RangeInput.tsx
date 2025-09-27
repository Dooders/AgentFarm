import React, { useMemo } from 'react'
import { RangeInputProps } from '../../types/leva'
import styled from 'styled-components'

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
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

const Track = styled.div`
  position: relative;
  height: 6px;
  border-radius: 3px;
  background: var(--leva-colors-elevation2);
  border: 1px solid var(--leva-colors-accent1);
`

const ThumbInput = styled.input`
  position: absolute;
  left: 0;
  right: 0;
  width: 100%;
  -webkit-appearance: none;
  background: transparent;
  height: 28px;
  margin: 0;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 14px;
    width: 14px;
    border-radius: 50%;
    background: var(--leva-colors-accent2);
    border: 1px solid var(--leva-colors-accent1);
    cursor: pointer;
  }

  &::-moz-range-thumb {
    height: 14px;
    width: 14px;
    border-radius: 50%;
    background: var(--leva-colors-accent2);
    border: 1px solid var(--leva-colors-accent1);
    cursor: pointer;
  }
`

const ValueText = styled.div`
  font-family: var(--leva-fonts-mono);
  font-size: 11px;
  color: var(--leva-colors-highlight1);
`

const ErrorText = styled.div`
  font-size: 10px;
  color: #ff6b6b;
  margin-top: 2px;
`

type RangeTuple = [number, number]

export const RangeInput: React.FC<RangeInputProps> = ({
  value,
  onChange,
  label,
  min,
  max,
  step = 1,
  showValue = true,
  formatValue,
  error,
  disabled = false
}) => {
  const safeMin = typeof min === 'number' ? min : 0
  const safeMax = typeof max === 'number' ? max : 100

  const [start, end] = useMemo<RangeTuple>(() => {
    const isNumber = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n)
    const isTuple = (v: unknown): v is [number, number] => Array.isArray(v) && v.length === 2 && isNumber(v[0]) && isNumber(v[1])
    const isMinMax = (v: unknown): v is { min: number; max: number } =>
      !!v && typeof v === 'object' && 'min' in (v as any) && 'max' in (v as any) &&
      isNumber((v as any).min) && isNumber((v as any).max)

    if (isTuple(value)) return [value[0], value[1]]
    if (isMinMax(value)) return [(value as { min: number }).min, (value as { max: number }).max]
    return [safeMin, safeMax]
  }, [value, safeMin, safeMax])

  const clamp = (n: number) => Math.min(Math.max(n, safeMin), safeMax)

  const updateStart = (n: number) => {
    const s = clamp(Math.min(n, end))
    onChange([s, end])
  }
  const updateEnd = (n: number) => {
    const e = clamp(Math.max(n, start))
    onChange([start, e])
  }

  return (
    <Container>
      <LabelRow>
        {label && <Label>{label}</Label>}
        {showValue && (
          <ValueText>
            {formatValue ? `${formatValue(start)} — ${formatValue(end)}` : `${start} — ${end}`}
          </ValueText>
        )}
      </LabelRow>
      <div style={{ position: 'relative', height: '28px' }}>
        <Track />
        <ThumbInput
          type="range"
          min={safeMin}
          max={safeMax}
          step={step}
          value={start}
          aria-label={label ? `${label} start` : 'Range start'}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateStart(Number(e.target.value))}
          disabled={disabled}
        />
        <ThumbInput
          type="range"
          min={safeMin}
          max={safeMax}
          step={step}
          value={end}
          aria-label={label ? `${label} end` : 'Range end'}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateEnd(Number(e.target.value))}
          disabled={disabled}
        />
      </div>
      {error && <ErrorText>{error}</ErrorText>}
    </Container>
  )
}

