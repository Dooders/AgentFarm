import React, { useState, useRef, useCallback } from 'react'

interface ResizablePanelsProps {
  leftPanel: React.ReactNode
  rightPanel: React.ReactNode
  defaultSplit: number
}

export const ResizablePanels: React.FC<ResizablePanelsProps> = ({
  leftPanel,
  rightPanel,
  defaultSplit = 0.5
}) => {
  const [splitPosition, setSplitPosition] = useState(defaultSplit)
  const containerRef = useRef<HTMLDivElement>(null)
  const isDragging = useRef(false)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging.current || !containerRef.current) return

    const containerRect = containerRef.current.getBoundingClientRect()
    const newPosition = (e.clientX - containerRect.left) / containerRect.width
    const clampedPosition = Math.max(0.2, Math.min(0.8, newPosition))
    setSplitPosition(clampedPosition)
  }, [])

  const handleMouseUp = useCallback(() => {
    isDragging.current = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }, [])

  React.useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [handleMouseMove, handleMouseUp])

  return (
    <div
      ref={containerRef}
      className="resizable-panels"
      style={{
        display: 'flex',
        height: '100%',
        width: '100%',
        position: 'relative'
      }}
    >
      <div
        className="left-panel"
        style={{
          width: `${splitPosition * 100}%`,
          overflow: 'hidden'
        }}
      >
        {leftPanel}
      </div>

      <div
        className="split-handle"
        style={{
          width: '8px',
          cursor: 'col-resize',
          backgroundColor: 'var(--border-subtle)',
          borderLeft: '1px solid var(--border-medium)',
          borderRight: '1px solid var(--border-medium)',
          position: 'relative'
        }}
        onMouseDown={handleMouseDown}
      >
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '2px',
            height: '32px',
            backgroundColor: 'var(--border-medium)'
          }}
        />
      </div>

      <div
        className="right-panel"
        style={{
          width: `${(1 - splitPosition) * 100}%`,
          overflow: 'hidden'
        }}
      >
        {rightPanel}
      </div>
    </div>
  )
}