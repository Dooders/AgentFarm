import React from 'react'
import { LevaControls } from '@/components/LevaControls/LevaControls'
import { ConfigFolder } from '@/components/LevaControls/ConfigFolder'
import { useLevaStore } from '@/stores/levaStore'
import { useConfigStore } from '@/stores/configStore'
import styled from 'styled-components'
import { ValidationDisplay } from '@/components/Validation/ValidationDisplay'

const LeftPanelContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--background-primary);
  border-right: 1px solid var(--border-subtle);
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

const ControlsSection = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
`

const SectionTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

const ControlGroup = styled.div`
  margin-bottom: 16px;
  padding: 12px;
  background: var(--background-secondary);
  border-radius: 8px;
  border: 1px solid var(--border-subtle);
`

export const LeftPanel: React.FC = () => {
  const levaStore = useLevaStore()
  const configStore = useConfigStore()

  return (
    <LeftPanelContainer>
      <PanelHeader>
        <PanelTitle>Configuration Explorer</PanelTitle>
      </PanelHeader>

      <ControlsSection>
        <SectionTitle>Leva Controls</SectionTitle>

        <ConfigFolder
          label="Environment Settings"
          collapsed={levaStore.isFolderCollapsed('environment')}
          onToggle={() => levaStore.toggleFolder('environment')}
        >
          <ControlGroup>
            <LevaControls />
            <ValidationDisplay prefixPaths={['width','height','position_discretization_method','use_bilinear_interpolation','visualization']}
                                compact
                                title="Environment Issues"
            />
          </ControlGroup>
        </ConfigFolder>

        <ConfigFolder
          label="Agent Parameters"
          collapsed={levaStore.isFolderCollapsed('agents')}
          onToggle={() => levaStore.toggleFolder('agents')}
        >
          <ControlGroup>
            <div style={{ padding: '8px', color: 'var(--text-secondary)' }}>
              Agent configuration controls will be displayed here
            </div>
            <ValidationDisplay prefixPaths={['agent_parameters','agent_type_ratios','system_agents','independent_agents','control_agents']}
                                compact
                                title="Agent Issues"
            />
          </ControlGroup>
        </ConfigFolder>

        <ConfigFolder
          label="Learning Configuration"
          collapsed={levaStore.isFolderCollapsed('learning')}
          onToggle={() => levaStore.toggleFolder('learning')}
        >
          <ControlGroup>
            <div style={{ padding: '8px', color: 'var(--text-secondary)' }}>
              Learning parameter controls will be displayed here
            </div>
            <ValidationDisplay prefixPaths={['learning_rate','epsilon_start','epsilon_min','epsilon_decay']}
                                compact
                                title="Learning Issues"
            />
          </ControlGroup>
        </ConfigFolder>

        <ConfigFolder
          label="Visualization Settings"
          collapsed={levaStore.isFolderCollapsed('visualization')}
          onToggle={() => levaStore.toggleFolder('visualization')}
        >
          <ControlGroup>
            <div style={{ padding: '8px', color: 'var(--text-secondary)' }}>
              Visualization controls will be displayed here
            </div>
            <ValidationDisplay prefixPaths={['visualization']}
                                compact
                                title="Visualization Issues"
            />
          </ControlGroup>
        </ConfigFolder>

        <SectionTitle>Panel Controls</SectionTitle>
        <ControlGroup>
          <button
            onClick={() => levaStore.setPanelVisible(!levaStore.isVisible)}
            style={{
              padding: '8px 16px',
              background: levaStore.isVisible ? 'var(--accent-primary)' : 'var(--background-tertiary)',
              color: levaStore.isVisible ? 'white' : 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '100%'
            }}
          >
            {levaStore.isVisible ? 'Hide Leva Panel' : 'Show Leva Panel'}
          </button>

          <button
            onClick={() => levaStore.setPanelCollapsed(!levaStore.isCollapsed)}
            style={{
              padding: '8px 16px',
              background: 'var(--background-tertiary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '100%',
              marginTop: '8px'
            }}
          >
            {levaStore.isCollapsed ? 'Expand Panel' : 'Collapse Panel'}
          </button>
        </ControlGroup>
      </ControlsSection>
    </LeftPanelContainer>
  )
}