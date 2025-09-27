// @ts-nocheck
import React, { useEffect } from 'react'
import { ResizablePanels } from './ResizablePanels'
import { LeftPanel } from '../ConfigExplorer/LeftPanel'
import { RightPanel } from '../ConfigExplorer/RightPanel'
import { useConfigStore } from '@/stores/configStore'
import { YamlPreview } from '@/components/Preview/YamlPreview'
import { Toolbar } from './Toolbar'
import { StatusBar } from './StatusBar'

export const DualPanelLayout: React.FC = () => {
  const { leftPanelWidth, restoreUIState } = useConfigStore()

  // Restore UI state on component mount
  useEffect(() => {
    restoreUIState()
  }, [restoreUIState])

  return (
    <div className="dual-panel-layout" data-testid="dual-panel-layout" style={{ height: '100vh', width: '100vw', display: 'flex', flexDirection: 'column' }}>
      <div style={{ flex: '0 0 auto' }}>
        <Toolbar />
      </div>
      <div style={{ flex: '1 1 auto', minHeight: 0 }}>
        <ResizablePanels
        direction="horizontal"
        defaultSizes={[leftPanelWidth * 100, (1 - leftPanelWidth) * 100]}
        minSizes={[20, 20]}
        persistKey="layout:main-horizontal"
      >
        <div data-testid="left-panel">
          <LeftPanel />
        </div>
        <div data-testid="right-panel">
          {/* Example nested vertical split inside right panel for advanced layout */}
          <ResizablePanels
            direction="vertical"
            defaultSizes={[60, 40]}
            minSizes={[30, 20]}
            persistKey="layout:right-vertical"
          >
            <div>
              <RightPanel />
            </div>
            <div>
              <YamlPreview />
            </div>
          </ResizablePanels>
        </div>
        </ResizablePanels>
      </div>
      <div style={{ flex: '0 0 auto' }}>
        <StatusBar />
      </div>
    </div>
  )
}