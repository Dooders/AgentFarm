import React, { useEffect, useRef, useState } from 'react'
import styled from 'styled-components'
import { useSearchStore } from '@/stores/searchStore'

const Container = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
  width: 100%;
`

const Input = styled.input`
  flex: 1;
  padding: 6px 10px;
  font-size: 12px;
  background: var(--background-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
`

const Select = styled.select`
  padding: 6px 8px;
  font-size: 12px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
`

const Toggle = styled.button<{ active?: boolean }>`
  padding: 6px 8px;
  font-size: 12px;
  background: ${(p: { active?: boolean }) => p.active ? 'var(--accent-primary)' : 'var(--background-tertiary)'};
  color: ${(p: { active?: boolean }) => p.active ? 'white' : 'var(--text-primary)'};
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

export const SearchBar: React.FC<{ autoFocusId?: string }> = ({ autoFocusId = 'toolbar-search' }) => {
  const query = useSearchStore((s) => s.query)
  const setQuery = useSearchStore((s) => s.setQuery)
  const filters = useSearchStore((s) => s.filters)
  const setFilters = useSearchStore((s) => s.setFilters)
  const runSearch = useSearchStore((s) => s.runSearch)
  const isSearching = useSearchStore((s) => s.isSearching)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (autoFocusId) inputRef.current?.setAttribute('id', autoFocusId)
  }, [autoFocusId])

  // Debounce typing
  const [local, setLocal] = useState(query)
  useEffect(() => setLocal(query), [query])
  useEffect(() => {
    const t = setTimeout(() => setQuery(local), 200)
    return () => clearTimeout(t)
  }, [local, setQuery])

  return (
    <Container>
      <Input
        ref={inputRef}
        value={local}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setLocal(e.target.value)}
        placeholder="Search keys and values…"
        aria-label="Search"
        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter') runSearch() }}
      />
      <Select
        value={filters.scope}
        aria-label="Search scope"
        onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setFilters({ scope: e.target.value as any })}
      >
        <option value="both">Both</option>
        <option value="keys">Keys</option>
        <option value="values">Values</option>
      </Select>
      <Toggle active={!!filters.fuzzy} onClick={() => setFilters({ fuzzy: !filters.fuzzy })} aria-pressed={!!filters.fuzzy}>Fuzzy</Toggle>
      <Toggle active={!!filters.regex} onClick={() => setFilters({ regex: !filters.regex })} aria-pressed={!!filters.regex}>Regex</Toggle>
      <Toggle active={!!filters.caseSensitive} onClick={() => setFilters({ caseSensitive: !filters.caseSensitive })} aria-pressed={!!filters.caseSensitive}>Aa</Toggle>
      <Button onClick={() => runSearch()} disabled={isSearching}>Search</Button>
    </Container>
  )
}

export default SearchBar

