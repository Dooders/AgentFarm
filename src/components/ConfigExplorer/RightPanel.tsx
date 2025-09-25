import React from 'react'

export const RightPanel: React.FC = () => {
  return (
    <div className="right-panel" style={{ padding: '16px', background: 'var(--background-secondary)' }}>
      <h2 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>
        Comparison Panel
      </h2>
      <div style={{ color: 'var(--text-secondary)' }}>
        <p>Right panel content will be implemented in subsequent issues.</p>
        <p>This will contain:</p>
        <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
          <li>Comparison config panel</li>
          <li>Read-only comparison view</li>
          <li>Diff highlighting</li>
          <li>Copy controls</li>
        </ul>
      </div>
    </div>
  )
}