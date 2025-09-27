import React, { useState } from 'react'
import styled from 'styled-components'
import { useSearchStore } from '@/stores/searchStore'

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`

const Row = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
`

const Input = styled.input`
  padding: 6px 10px;
  font-size: 12px;
  background: var(--background-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
`

const Button = styled.button`
  padding: 6px 10px;
  font-size: 12px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
`

const List = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`

const Item = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
  border: 1px solid var(--border-subtle);
  padding: 6px 8px;
  border-radius: 6px;
  background: var(--background-primary);
`

export const SavedSearches: React.FC = () => {
  const [name, setName] = useState('')
  const save = useSearchStore((s) => s.saveCurrentSearch)
  const apply = useSearchStore((s) => s.applySavedSearch)
  const del = useSearchStore((s) => s.deleteSavedSearch)
  const saved = useSearchStore((s) => s.saved)

  return (
    <Container>
      <Row>
        <Input value={name} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)} placeholder="Save current search as…" />
        <Button onClick={() => { if (name.trim()) { save(name.trim()); setName('') } }}>Save</Button>
      </Row>
      <List>
        {saved.length === 0 && (
          <div style={{ color: 'var(--text-secondary)', fontSize: 12 }}>No saved searches.</div>
        )}
        {saved.map(item => (
          <Item key={item.id}>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <strong style={{ fontSize: 12 }}>{item.name}</strong>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{item.query.text}</span>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <Button onClick={() => apply(item.id)}>Apply</Button>
              <Button onClick={() => del(item.id)}>Delete</Button>
            </div>
          </Item>
        ))}
      </List>
    </Container>
  )
}

export default SavedSearches

