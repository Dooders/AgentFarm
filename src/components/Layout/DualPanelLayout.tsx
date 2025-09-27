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
    <div className="dual-panel-layout" data-testid="dual-panel-layout" style={{ height: '100vh', width: '100vw' }}>
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
            {/* Bottom area reserved for future YAML preview (#18) */}
            <div>
              <div style={{ padding: '12px', borderTop: '1px solid var(--border-subtle)' }}>
                <h3 style={{ margin: 0, fontSize: 14, color: 'var(--text-secondary)' }}>Preview Area</h3>
                <p style={{ marginTop: 8, color: 'var(--text-muted)' }}>YAML preview will appear here (Issue #18).</p>
              </div>
            </div>
          </ResizablePanels>
        </div>
      </ResizablePanels>
    </div>
  )
}