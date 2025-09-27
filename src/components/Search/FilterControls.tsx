import React from 'react'
import styled from 'styled-components'
import { useSearchStore } from '@/stores/searchStore'

const Container = styled.div`
  display: grid;
  grid-template-columns: repeat(3, minmax(160px, 1fr));
  gap: 8px 12px;
`

const Group = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`

const Label = styled.div`
  font-size: 12px;
  color: var(--text-secondary);
  font-weight: 600;
  text-transform: uppercase;
`

const Row = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px 12px;
  align-items: center;
`

const Checkbox = styled.label`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-primary);
  input { margin: 0; }
`

export const FilterControls: React.FC = () => {
  const filters = useSearchStore((s) => s.filters)
  const setFilters = useSearchStore((s) => s.setFilters)

  const toggleType = (t: string) => {
    const set = new Set(filters.parameterTypes || []) as Set<any>
    if (set.has(t)) set.delete(t)
    else set.add(t)
    setFilters({ parameterTypes: set.size > 0 ? set : null })
  }

  const toggleSection = (sec: string) => {
    const set = new Set(filters.sections || []) as Set<any>
    if (set.has(sec)) set.delete(sec)
    else set.add(sec)
    setFilters({ sections: set.size > 0 ? set : null })
  }

  return (
    <Container>
      <Group>
        <Label>Parameter Types</Label>
        <Row>
          {['number','string','boolean','object','array'].map(t => (
            <Checkbox key={t}><input type="checkbox" checked={!!filters.parameterTypes?.has(t as any)} onChange={() => toggleType(t)} /> {t}</Checkbox>
          ))}
        </Row>
      </Group>
      <Group>
        <Label>Validation</Label>
        <Row>
          {['any','valid','warning','error'].map(v => (
            <Checkbox key={v}><input type="radio" name="val" checked={filters.validationStatus === (v as any)} onChange={() => setFilters({ validationStatus: v as any })} /> {v}</Checkbox>
          ))}
        </Row>
      </Group>
      <Group>
        <Label>Modification</Label>
        <Row>
          {['any','changed','unchanged'].map(m => (
            <Checkbox key={m}><input type="radio" name="mod" checked={filters.modificationStatus === (m as any)} onChange={() => setFilters({ modificationStatus: m as any })} /> {m}</Checkbox>
          ))}
        </Row>
      </Group>
      <Group>
        <Label>Sections</Label>
        <Row>
          {['environment','agents','learning','agent_parameters','modules','visualization','other'].map(sec => (
            <Checkbox key={sec}><input type="checkbox" checked={!!filters.sections?.has(sec as any)} onChange={() => toggleSection(sec)} /> {sec}</Checkbox>
          ))}
        </Row>
      </Group>
    </Container>
  )
}

export default FilterControls

