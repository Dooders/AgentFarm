import React, { useMemo } from 'react'
import styled from 'styled-components'
import { SimulationConfigType } from '@/types/config'

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

type ComparisonPanelProps = {
  compareConfig: SimulationConfigType | null
  comparisonFilePath?: string
  errorMessage?: string
}

export const ComparisonPanel: React.FC<ComparisonPanelProps> = ({ compareConfig, comparisonFilePath, errorMessage }) => {
  const comparisonSummary = useMemo(() => {
    if (!compareConfig) return null
    return [
      { label: 'Environment Settings', value: `Grid: ${compareConfig.width}x${compareConfig.height}, ${compareConfig.position_discretization_method} discretization` },
      { label: 'Agent Counts', value: `System: ${compareConfig.system_agents}, Independent: ${compareConfig.independent_agents}, Control: ${compareConfig.control_agents}` },
      { label: 'Learning Rate', value: String(compareConfig.learning_rate) },
      { label: 'Visualization', value: `Canvas: ${compareConfig.visualization.canvas_width}x${compareConfig.visualization.canvas_height}, Metrics: ${compareConfig.visualization.show_metrics ? 'Enabled' : 'Disabled'}` }
    ]
  }, [compareConfig])

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
    </div>
  )
}

