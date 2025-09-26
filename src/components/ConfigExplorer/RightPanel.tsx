import React from 'react'
import styled from 'styled-components'
import { ValidationSummary } from '@/components/Validation/ValidationSummary'
import { useAccessibility } from '@/components/UI/AccessibilityProvider'

const RightPanelContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--background-primary);
`

const PanelHeader = styled.div`
  padding: 16px;
  border-bottom: 1px solid var(--border-subtle);
  background: var(--background-secondary);
`

const PanelTitle = styled.h2`
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
`

const ContentArea = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  background: var(--background-secondary);
`

const SectionTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

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

const ActionButton = styled.button`
  padding: 8px 16px;
  margin: 4px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;

  &:hover {
    background: var(--accent-primary);
    color: white;
  }
`

export const RightPanel: React.FC = () => {
  const { announceToScreenReader } = useAccessibility()

  const handleButtonClick = (action: string) => {
    announceToScreenReader(`Activated ${action}`, 'polite')
  }

  return (
    <RightPanelContainer role="complementary" aria-label="Configuration comparison and validation panel">
      <PanelHeader>
        <PanelTitle>Configuration Comparison</PanelTitle>
      </PanelHeader>

      <ContentArea id="comparison-content" tabIndex={-1}>
        <section aria-labelledby="current-config-title">
          <SectionTitle id="current-config-title">Current Configuration</SectionTitle>

          <ContentSection role="list" aria-label="Current configuration parameters">
            <ConfigItem role="listitem">
              <ConfigLabel>Environment Settings:</ConfigLabel>
              <ConfigValue>Grid: 100x100, Floor discretization</ConfigValue>
            </ConfigItem>
            <ConfigItem role="listitem">
              <ConfigLabel>Agent Counts:</ConfigLabel>
              <ConfigValue>System: 20, Independent: 20, Control: 10</ConfigValue>
            </ConfigItem>
            <ConfigItem role="listitem">
              <ConfigLabel>Learning Rate:</ConfigLabel>
              <ConfigValue>0.001 (Adaptive)</ConfigValue>
            </ConfigItem>
            <ConfigItem role="listitem">
              <ConfigLabel>Visualization:</ConfigLabel>
              <ConfigValue>Canvas: 800x600, Metrics: Enabled</ConfigValue>
            </ConfigItem>
          </ContentSection>
        </section>

        <section aria-labelledby="comparison-tools-title">
          <SectionTitle id="comparison-tools-title">Comparison Tools</SectionTitle>

          <ContentSection role="group" aria-label="Configuration comparison actions">
            <div style={{ marginBottom: '16px' }}>
              <ActionButton
                onClick={() => handleButtonClick('Load Base Config')}
                aria-describedby="base-config-desc"
              >
                Load Base Config
              </ActionButton>
              <ActionButton
                onClick={() => handleButtonClick('Load Compare Config')}
                aria-describedby="compare-config-desc"
              >
                Load Compare Config
              </ActionButton>
              <ActionButton
                onClick={() => handleButtonClick('Generate Diff Report')}
                aria-describedby="diff-report-desc"
              >
                Generate Diff Report
              </ActionButton>
            </div>

            <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
              <p>Compare different configuration versions to identify changes and optimize settings.</p>
              <ul style={{ marginTop: '8px', paddingLeft: '16px' }} role="list">
                <li role="listitem">Highlight parameter differences</li>
                <li role="listitem">Track configuration evolution</li>
                <li role="listitem">Validate configuration integrity</li>
                <li role="listitem">Export comparison results</li>
              </ul>
            </div>
          </ContentSection>
        </section>

        <section aria-labelledby="validation-status-title">
          <SectionTitle id="validation-status-title">Validation Status</SectionTitle>

          <ContentSection id="validation-content" tabIndex={-1}>
            <ValidationSummary />
          </ContentSection>
        </section>
      </ContentArea>
    </RightPanelContainer>
  )
}