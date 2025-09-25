import React from 'react'
import { ResizablePanels } from './ResizablePanels'
import { LeftPanel } from '../ConfigExplorer/LeftPanel'
import { RightPanel } from '../ConfigExplorer/RightPanel'

export const DualPanelLayout: React.FC = () => {
  return (
    <div className="dual-panel-layout" style={{ height: '100vh', width: '100vw' }}>
      <ResizablePanels
        leftPanel={<LeftPanel />}
        rightPanel={<RightPanel />}
        defaultSplit={0.6}
      />
    </div>
  )
}