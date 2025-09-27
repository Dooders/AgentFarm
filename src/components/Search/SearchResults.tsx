import React from 'react'
import styled from 'styled-components'
import { useSearchStore } from '@/stores/searchStore'

const List = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`

const Row = styled.div`
  display: grid;
  grid-template-columns: auto 1fr auto auto;
  gap: 8px;
  align-items: center;
  padding: 8px;
  border: 1px solid var(--border-subtle);
  background: var(--background-primary);
  border-radius: 6px;
`

const Path = styled.div`
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  color: var(--text-secondary);
`

const Preview = styled.div`
  font-size: 12px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`

const Tag = styled.span<{ tone?: 'ok' | 'warn' | 'err' | 'muted' }>`
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 10px;
  border: 1px solid var(--border-subtle);
  background: ${(p: { tone?: 'ok' | 'warn' | 'err' | 'muted' }) => p.tone === 'ok' ? 'rgba(80, 160, 80, .1)' : p.tone === 'warn' ? 'rgba(160, 140, 60, .1)' : p.tone === 'err' ? 'rgba(180, 80, 80, .1)' : 'var(--background-secondary)'};
  color: var(--text-secondary);
`

const Button = styled.button`
  padding: 4px 8px;
  font-size: 12px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
`

export const SearchResults: React.FC = () => {
  const results = useSearchStore((s) => s.results)
  if (!results) return <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>No results yet.</div>
  return (
    <div>
      <div style={{ color: 'var(--text-secondary)', fontSize: '12px', marginBottom: '8px' }}>{results.total} results in {results.tookMs}ms</div>
      <List>
        {results.items.map((item) => (
          <Row key={item.path}>
            <Path title={item.path}>{item.path}</Path>
            <Preview title={item.valuePreview}>{item.valuePreview}</Preview>
            <Tag tone={item.validation === 'error' ? 'err' : item.validation === 'warning' ? 'warn' : 'ok'}>{item.validation}</Tag>
            <div style={{ display: 'flex', gap: 6 }}>
              <Tag tone={item.modified ? 'warn' : 'muted'}>{item.modified ? 'changed' : 'unchanged'}</Tag>
              <Tag tone='muted'>{item.parameterType}</Tag>
              <Tag tone='muted'>{item.section}</Tag>
              <Button onClick={async () => {
                try { await navigator.clipboard.writeText(item.path) } catch {}
              }}>Copy Path</Button>
              <Button onClick={() => {
                // Attempt to scroll left panel area into view as a basic navigation
                const el = document.getElementById('left-scroll-area')
                if (el) el.scrollTo({ top: 0, behavior: 'smooth' })
              }}>Jump</Button>
            </div>
          </Row>
        ))}
      </List>
    </div>
  )
}

export default SearchResults

