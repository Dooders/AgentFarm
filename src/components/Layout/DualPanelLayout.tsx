import React, { useEffect } from 'react'
import { ResizablePanels } from './ResizablePanels'
import { LeftPanel } from '../ConfigExplorer/LeftPanel'
import { RightPanel } from '../ConfigExplorer/RightPanel'
import { useConfigStore } from '@/stores/configStore'

export const DualPanelLayout: React.FC = () => {
  const { leftPanelWidth, restoreUIState } = useConfigStore()

  // Restore UI state on component mount
  useEffect(() => {
    restoreUIState()
  }, [restoreUIState])

  return (
    <div className="dual-panel-layout" style={{ height: '100vh', width: '100vw' }}>
      <ResizablePanels
        leftPanel={<LeftPanel />}
        rightPanel={<RightPanel />}
        defaultSplit={leftPanelWidth}
      />
    </div>
  )
}