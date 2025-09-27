import React, { useMemo, useRef, useState } from 'react'
import styled from 'styled-components'
import { SimulationConfigType } from '@/types/config'
import { useConfigStore } from '@/stores/configStore'
import { configSelectors } from '@/stores/selectors'

const ContentSection = styled.div`
  margin-bottom: 24px;
  padding: 16px;
  background: var(--background-primary);
  border-radius: 8px;
  border: 1px solid var(--border-subtle);
`

const ConfigItem = styled.div`
  margin-bottom: 12px;
  padding: 8px;
  background: var(--background-secondary);
  border-radius: 4px;
  border-left: 3px solid var(--accent-primary);
`

const ConfigLabel = styled.span`
  font-weight: 600;
  color: var(--text-primary);
`

const ConfigValue = styled.span`
  color: var(--text-secondary);
  margin-left: 8px;
`

const Toolbar = styled.div`
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 12px;
`

const SmallButton = styled.button`
  padding: 6px 10px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  &:hover { background: var(--accent-primary); color: white; }
  &:disabled { opacity: 0.6; cursor: not-allowed; }
`

const DiffList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`

const DiffItem = styled.div<{ variant: 'added' | 'removed' | 'changed' }>`
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px;
  border-radius: 6px;
  background: var(--background-secondary);
  border: 1px solid var(--border-subtle);
  border-left: 4px solid
    ${p => p.variant === 'added' ? 'var(--success, #16a34a)' : p.variant === 'removed' ? 'var(--error-border, #dc2626)' : 'var(--accent-primary)'};
`

const PathText = styled.div`
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  color: var(--text-secondary);
`

const ValuesRow = styled.div`
  display: grid;
  grid-template-columns: 1fr auto 1fr auto;
  gap: 8px;
  align-items: center;
`

const ValueBox = styled.div`
  padding: 6px 8px;
  background: var(--background-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`

const Divider = styled.span`
  color: var(--text-secondary);
  font-size: 12px;
`

type ComparisonPanelProps = {
  compareConfig: SimulationConfigType | null
  comparisonFilePath?: string
  errorMessage?: string
}

export const ComparisonPanel: React.FC<ComparisonPanelProps> = React.memo(({ compareConfig, comparisonFilePath, errorMessage }) => {
  const diff = useConfigStore(configSelectors.getComparisonDiff)
  const copyFromComparison = useConfigStore(s => s.copyFromComparison)
  const removeConfigPath = useConfigStore(s => s.removeConfigPath)
  const applyAll = useConfigStore(s => s.applyAllDifferencesFromComparison)

  const [showAdded, setShowAdded] = useState(true)
  const [showRemoved, setShowRemoved] = useState(true)
  const [showChanged, setShowChanged] = useState(true)

  const addedAnchorRef = useRef<HTMLDivElement | null>(null)
  const removedAnchorRef = useRef<HTMLDivElement | null>(null)
  const changedAnchorRef = useRef<HTMLDivElement | null>(null)

  const comparisonSummary = useMemo(() => {
    if (!compareConfig) return null
    return [
      { label: 'Environment Settings', value: `Grid: ${compareConfig.width}x${compareConfig.height}, ${compareConfig.position_discretization_method} discretization` },
      { label: 'Agent Counts', value: `System: ${compareConfig.system_agents}, Independent: ${compareConfig.independent_agents}, Control: ${compareConfig.control_agents}` },
      { label: 'Learning Rate', value: String(compareConfig.learning_rate) },
      { label: 'Visualization', value: `Canvas: ${compareConfig.visualization.canvas_width}x${compareConfig.visualization.canvas_height}, Metrics: ${compareConfig.visualization.show_metrics ? 'Enabled' : 'Disabled'}` }
    ]
  }, [compareConfig])

  // Display helper: consistently stringify values for diff display
  const stringify = (v: unknown) => {
    try { return JSON.stringify(v) } catch { return String(v) }
  }

  const addedCount = diff ? Object.keys(diff.added).length : 0
  const removedCount = diff ? Object.keys(diff.removed).length : 0
  const changedCount = diff ? Object.keys(diff.changed).length : 0
  const hasAnyDiff = (addedCount + removedCount + changedCount) > 0

  return (
    <div>
      <div style={{ margin: '4px 0', color: 'var(--text-secondary)', fontSize: '12px' }}>
        {comparisonFilePath ? `Loaded from: ${comparisonFilePath}` : 'No file selected'}
      </div>
      {errorMessage && (
        <div role="alert" style={{ color: 'var(--error-text)', background: 'var(--error-bg)', border: '1px solid var(--error-border)', padding: '8px', borderRadius: '4px', marginBottom: '8px' }}>
          {errorMessage}
        </div>
      )}
      <ContentSection>
        {!compareConfig && (
          <div style={{ color: 'var(--text-secondary)' }}>No comparison configuration loaded.</div>
        )}
        {compareConfig && (
          <div role="list" aria-label="Comparison configuration parameters">
            {comparisonSummary?.map((row) => (
              <ConfigItem role="listitem" key={row.label} aria-readonly="true">
                <ConfigLabel>{row.label}:</ConfigLabel>
                <ConfigValue>{row.value}</ConfigValue>
              </ConfigItem>
            ))}
          </div>
        )}
      </ContentSection>

      {compareConfig && (
        <ContentSection>
          <Toolbar role="toolbar" aria-label="Diff filters and actions">
            <SmallButton onClick={() => setShowAdded(v => !v)} aria-pressed={showAdded}>{showAdded ? 'Hide' : 'Show'} Added</SmallButton>
            <SmallButton onClick={() => setShowRemoved(v => !v)} aria-pressed={showRemoved}>{showRemoved ? 'Hide' : 'Show'} Removed</SmallButton>
            <SmallButton onClick={() => setShowChanged(v => !v)} aria-pressed={showChanged}>{showChanged ? 'Hide' : 'Show'} Changed</SmallButton>
            <SmallButton onClick={() => applyAll()} disabled={!hasAnyDiff}>Apply All Differences</SmallButton>
            <SmallButton onClick={() => addedAnchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })} disabled={addedCount === 0}>Jump to Added</SmallButton>
            <SmallButton onClick={() => removedAnchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })} disabled={removedCount === 0}>Jump to Removed</SmallButton>
            <SmallButton onClick={() => changedAnchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })} disabled={changedCount === 0}>Jump to Changed</SmallButton>
          </Toolbar>

          {!hasAnyDiff && (
            <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>No differences detected.</div>
          )}

          {hasAnyDiff && (
            <DiffList aria-label="Differences list">
              {showAdded && addedCount > 0 && <div ref={addedAnchorRef} />}
              {showAdded && Object.entries(diff.added).map(([path, to]) => (
                <DiffItem key={`added-${path}`} variant="added">
                  <PathText>{path}</PathText>
                  <ValuesRow>
                    <ValueBox title="Current value">(missing)</ValueBox>
                    <Divider>→</Divider>
                    <ValueBox title="Comparison value">{stringify(to)}</ValueBox>
                    <SmallButton onClick={() => copyFromComparison(path)} aria-label={`Copy ${path} from comparison`}>
                      Copy
                    </SmallButton>
                  </ValuesRow>
                </DiffItem>
              ))}

              {showRemoved && removedCount > 0 && <div ref={removedAnchorRef} />}
              {showRemoved && Object.entries(diff.removed).map(([path, from]) => (
                <DiffItem key={`removed-${path}`} variant="removed">
                  <PathText>{path}</PathText>
                  <ValuesRow>
                    <ValueBox title="Current value">{stringify(from)}</ValueBox>
                    <Divider>→</Divider>
                    <ValueBox title="Comparison value">(missing)</ValueBox>
                    <SmallButton onClick={() => removeConfigPath(path)} aria-label={`Remove ${path} from current`}>
                      Remove
                    </SmallButton>
                  </ValuesRow>
                </DiffItem>
              ))}

              {showChanged && changedCount > 0 && <div ref={changedAnchorRef} />}
              {showChanged && Object.entries(diff.changed).map(([path, pair]) => (
                <DiffItem key={`changed-${path}`} variant="changed">
                  <PathText>{path}</PathText>
                  <ValuesRow>
                    <ValueBox title="Current value">{stringify(pair.from)}</ValueBox>
                    <Divider>→</Divider>
                    <ValueBox title="Comparison value">{stringify(pair.to)}</ValueBox>
                    <SmallButton onClick={() => copyFromComparison(path)} aria-label={`Copy ${path} from comparison`}>
                      Copy
                    </SmallButton>
                  </ValuesRow>
                </DiffItem>
              ))}
            </DiffList>
          )}
        </ContentSection>
      )}
    </div>
  )
})

