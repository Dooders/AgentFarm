import React, { useEffect, useMemo, useState } from 'react'
import styled from 'styled-components'
import { useConfigStore } from '@/stores/configStore'

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`

const Row = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
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

const Textarea = styled.textarea`
  padding: 6px 10px;
  font-size: 12px;
  background: var(--background-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  min-height: 60px;
`

const Button = styled.button`
  padding: 6px 10px;
  font-size: 12px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  cursor: pointer;
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`

const List = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
`

const Item = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px;
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  background: var(--background-primary);
`

const ItemHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
`

const Title = styled.div`
  font-weight: 600;
  color: var(--text-primary);
`

const Meta = styled.div`
  font-size: 11px;
  color: var(--text-secondary);
  display: flex;
  gap: 6px;
`

const Actions = styled.div`
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
`

export const PresetManager: React.FC = () => {
  const templates = useConfigStore((s) => s.templates)
  const listTemplates = useConfigStore((s) => (s as any).listTemplates)
  const loadTemplate = useConfigStore((s) => s.loadTemplate)
  const applyTemplatePartial = useConfigStore((s) => s.applyTemplatePartial)
  const saveTemplate = useConfigStore((s) => s.saveTemplate)
  const deleteTemplate = useConfigStore((s) => s.deleteTemplate)

  const [query, setQuery] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDescription, setNewDescription] = useState('')
  const [partial, setPartial] = useState<Record<'environment'|'agents'|'learning'|'modules'|'visualization', boolean>>({ environment: false, agents: false, learning: false, modules: false, visualization: false })

  useEffect(() => {
    void listTemplates({ includeSystem: true, includeUser: true })
  }, [listTemplates])

  useEffect(() => {
    // Subscribe to template create/delete events if available
    if (typeof window !== 'undefined' && (window as any).electronAPI) {
      const unsubCreate = (window as any).electronAPI.on?.('config:template:created', () => {
        void listTemplates({ includeSystem: true, includeUser: true })
      })
      const unsubDelete = (window as any).electronAPI.on?.('config:template:deleted', () => {
        void listTemplates({ includeSystem: true, includeUser: true })
      })
      return () => {
        try {
          ;(window as any).electronAPI?.removeListener?.('config:template:created', unsubCreate)
          ;(window as any).electronAPI?.removeListener?.('config:template:deleted', unsubDelete)
        } catch {}
      }
    }
  }, [listTemplates])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return templates
    return templates.filter(t =>
      t.name.toLowerCase().includes(q) ||
      (t.description || '').toLowerCase().includes(q) ||
      (t.category || '').toLowerCase().includes(q)
    )
  }, [templates, query])

  const onCreate = async () => {
    if (!newName.trim()) return
    await saveTemplate({ name: newName.trim(), description: newDescription.trim(), category: 'user', baseConfig: {}, tags: [] })
    setIsCreating(false)
    setNewName('')
    setNewDescription('')
  }

  return (
    <Container>
      <Row>
        <Input
          aria-label="Search presets"
          placeholder="Search presets..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ flex: 1 }}
        />
        <Button onClick={() => void listTemplates({ includeSystem: true, includeUser: true })}>Refresh</Button>
        <Button onClick={() => setIsCreating((v) => !v)} aria-expanded={isCreating} aria-controls="create-preset-form">
          {isCreating ? 'Cancel' : 'Create Preset'}
        </Button>
      </Row>

      {isCreating && (
        <Item id="create-preset-form" role="region" aria-label="Create new preset form">
          <Row>
            <Input placeholder="Preset name" value={newName} onChange={(e) => setNewName(e.target.value)} style={{ flex: 1 }} />
          </Row>
          <Row>
            <Textarea placeholder="Description (optional)" value={newDescription} onChange={(e) => setNewDescription(e.target.value)} style={{ width: '100%' }} />
          </Row>
          <Row>
            <Button onClick={onCreate} disabled={!newName.trim()}>Save Preset</Button>
          </Row>
        </Item>
      )}

      <List role="list" aria-label="Available presets">
        {filtered.map((t) => (
          <Item key={`${t.category}:${t.name}`} role="listitem">
            <ItemHeader>
              <Title>{t.name}</Title>
              <Meta>
                <span>{t.category || 'user'}</span>
              </Meta>
            </ItemHeader>
            {t.description && <div style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{t.description}</div>}
            <Actions>
              <Button onClick={() => void loadTemplate(t.name)}>Apply</Button>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Partial:</span>
                {(['environment','agents','learning','modules','visualization'] as const).map((k) => (
                  <label key={k} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--text-secondary)' }}>
                    <input type="checkbox" checked={partial[k]} onChange={(e) => setPartial(p => ({ ...p, [k]: e.target.checked }))} />
                    {k}
                  </label>
                ))}
                <Button onClick={() => {
                  const sections = (Object.entries(partial).filter(([,v]) => v).map(([k]) => k) as Array<'environment'|'agents'|'learning'|'visualization'|'modules'>)
                  if (sections.length > 0) void applyTemplatePartial(t.name, sections)
                }}>Apply Selected</Button>
              </div>
              {t.category !== 'system' && (
                <Button onClick={() => void deleteTemplate(t.name)}>Delete</Button>
              )}
            </Actions>
          </Item>
        ))}
      </List>
    </Container>
  )
}

export default PresetManager

