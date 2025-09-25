import React from 'react'

export const LeftPanel: React.FC = () => {
  return (
    <div className="left-panel" style={{ padding: '16px', background: 'var(--background-primary)' }}>
      <h2 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>
        Configuration Explorer
      </h2>
      <div style={{ color: 'var(--text-secondary)' }}>
        <p>Left panel content will be implemented in subsequent issues.</p>
        <p>This will contain:</p>
        <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
          <li>Navigation tree</li>
          <li>Leva controls</li>
          <li>Validation display</li>
          <li>YAML preview</li>
        </ul>
      </div>
    </div>
  )
}