import React from 'react'
import { LevaControls } from '@/components/LevaControls/LevaControls'
import { ConfigFolder } from '@/components/LevaControls/ConfigFolder'
import { useLevaStore } from '@/stores/levaStore'
import { useConfigStore } from '@/stores/configStore'
import { useAccessibility } from '@/components/UI/AccessibilityProvider'
import { useKeyboardNavigation } from '@/hooks/useKeyboardNavigation'
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

// Expose an id for scroll sync
const ScrollableArea = styled(ControlsSection).attrs({ id: 'left-scroll-area' })``

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

export const LeftPanel: React.FC = React.memo(() => {
  const levaStore = useLevaStore()
  const configStore = useConfigStore()
  const { announceToScreenReader } = useAccessibility()

  // Keyboard navigation for the left panel
  const { ref, focusElement } = useKeyboardNavigation({
    onEscape: () => {
      // Collapse all folders on Escape
      announceToScreenReader('All folders collapsed', 'polite')
    },
    onTab: () => {
      // Handle tab navigation
    },
    preventDefault: false // Allow native browser functionality
  })

  const handleFolderToggle = (folderId: string, isCollapsed: boolean) => {
    levaStore.toggleFolder(folderId)
    announceToScreenReader(
      `${isCollapsed ? 'Expanded' : 'Collapsed'} ${folderId} folder`,
      'polite'
    )
  }

  const handlePanelToggle = (isVisible: boolean) => {
    levaStore.setPanelVisible(!isVisible)
    announceToScreenReader(
      `Leva panel ${!isVisible ? 'shown' : 'hidden'}`,
      'polite'
    )
  }

  const containerRef = React.useRef<HTMLDivElement>(null)

  return (
    <LeftPanelContainer
      ref={(node) => {
        containerRef.current = node as HTMLDivElement | null
        // Attach to keyboard nav ref if compatible
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const anyRef = ref as any
        if (anyRef && typeof anyRef === 'object') {
          anyRef.current = node
        }
      }}
      role="navigation"
      aria-label="Configuration sections navigation"
      tabIndex={-1}
      onFocus={focusElement}
    >
      <PanelHeader>
        <PanelTitle>Configuration Explorer</PanelTitle>
      </PanelHeader>

      <ScrollableArea>
        <SectionTitle>Leva Controls</SectionTitle>

        <ConfigFolder
          label="Environment Settings"
          collapsed={levaStore.isFolderCollapsed('environment')}
          onToggle={() => handleFolderToggle('environment', levaStore.isFolderCollapsed('environment'))}
          id="environment"
        >
          <ControlGroup>
            <LevaControls />
            <ValidationDisplay
              prefixPaths={['width','height','position_discretization_method','use_bilinear_interpolation','visualization']}
              compact
              title="Environment Issues"
            />
          </ControlGroup>
        </ConfigFolder>

        <ConfigFolder
          label="Agent Parameters"
          collapsed={levaStore.isFolderCollapsed('agents')}
          onToggle={() => handleFolderToggle('agents', levaStore.isFolderCollapsed('agents'))}
          id="agents"
        >
          <ControlGroup>
            <div style={{ padding: '8px', color: 'var(--text-secondary)' }}>
              Agent configuration controls will be displayed here
            </div>
            <ValidationDisplay
              prefixPaths={['agent_parameters','agent_type_ratios','system_agents','independent_agents','control_agents']}
              compact
              title="Agent Issues"
            />
          </ControlGroup>
        </ConfigFolder>

        <ConfigFolder
          label="Learning Configuration"
          collapsed={levaStore.isFolderCollapsed('learning')}
          onToggle={() => handleFolderToggle('learning', levaStore.isFolderCollapsed('learning'))}
          id="learning"
        >
          <ControlGroup>
            <div style={{ padding: '8px', color: 'var(--text-secondary)' }}>
              Learning parameter controls will be displayed here
            </div>
            <ValidationDisplay
              prefixPaths={['learning_rate','epsilon_start','epsilon_min','epsilon_decay']}
              compact
              title="Learning Issues"
            />
          </ControlGroup>
        </ConfigFolder>

        <ConfigFolder
          label="Visualization Settings"
          collapsed={levaStore.isFolderCollapsed('visualization')}
          onToggle={() => handleFolderToggle('visualization', levaStore.isFolderCollapsed('visualization'))}
          id="visualization"
        >
          <ControlGroup>
            <div style={{ padding: '8px', color: 'var(--text-secondary)' }}>
              Visualization controls will be displayed here
            </div>
            <ValidationDisplay
              prefixPaths={['visualization']}
              compact
              title="Visualization Issues"
            />
          </ControlGroup>
        </ConfigFolder>

        <SectionTitle>Panel Controls</SectionTitle>
        <ControlGroup>
          <button
            onClick={() => handlePanelToggle(levaStore.isVisible)}
            aria-expanded={levaStore.isVisible}
            aria-controls="leva-panel"
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
            onClick={() => {
              levaStore.setPanelCollapsed(!levaStore.isCollapsed)
              announceToScreenReader(
                `Panel ${!levaStore.isCollapsed ? 'collapsed' : 'expanded'}`,
                'polite'
              )
            }}
            aria-expanded={!levaStore.isCollapsed}
            aria-controls="controls-section"
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
      </ScrollableArea>
    </LeftPanelContainer>
  )
})