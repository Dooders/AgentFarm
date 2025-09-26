import React from 'react'
import styled from 'styled-components'

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
  return (
    <RightPanelContainer>
      <PanelHeader>
        <PanelTitle>Configuration Comparison</PanelTitle>
      </PanelHeader>

      <ContentArea>
        <SectionTitle>Current Configuration</SectionTitle>

        <ContentSection>
          <ConfigItem>
            <ConfigLabel>Environment Settings:</ConfigLabel>
            <ConfigValue>Grid: 100x100, Floor discretization</ConfigValue>
          </ConfigItem>
          <ConfigItem>
            <ConfigLabel>Agent Counts:</ConfigLabel>
            <ConfigValue>System: 20, Independent: 20, Control: 10</ConfigValue>
          </ConfigItem>
          <ConfigItem>
            <ConfigLabel>Learning Rate:</ConfigLabel>
            <ConfigValue>0.001 (Adaptive)</ConfigValue>
          </ConfigItem>
          <ConfigItem>
            <ConfigLabel>Visualization:</ConfigLabel>
            <ConfigValue>Canvas: 800x600, Metrics: Enabled</ConfigValue>
          </ConfigItem>
        </ContentSection>

        <SectionTitle>Comparison Tools</SectionTitle>

        <ContentSection>
          <div style={{ marginBottom: '16px' }}>
            <ActionButton>Load Base Config</ActionButton>
            <ActionButton>Load Compare Config</ActionButton>
            <ActionButton>Generate Diff Report</ActionButton>
          </div>

          <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
            <p>Compare different configuration versions to identify changes and optimize settings.</p>
            <ul style={{ marginTop: '8px', paddingLeft: '16px' }}>
              <li>Highlight parameter differences</li>
              <li>Track configuration evolution</li>
              <li>Validate configuration integrity</li>
              <li>Export comparison results</li>
            </ul>
          </div>
        </ContentSection>

        <SectionTitle>Validation Status</SectionTitle>

        <ContentSection>
          <div style={{ color: 'var(--success-color, #4caf50)', marginBottom: '8px' }}>
            ✓ All parameters validated successfully
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
            <p>Configuration passes all validation rules:</p>
            <ul style={{ marginTop: '8px', paddingLeft: '16px' }}>
              <li>Parameter ranges: Valid</li>
              <li>Type consistency: Valid</li>
              <li>Dependency rules: Valid</li>
              <li>Performance constraints: Valid</li>
            </ul>
          </div>
        </ContentSection>
      </ContentArea>
    </RightPanelContainer>
  )
}