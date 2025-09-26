import React, { useState, useRef, useCallback, useEffect } from 'react'
import { useConfigStore } from '@/stores/configStore'

interface ResizablePanelsProps {
  leftPanel: React.ReactNode
  rightPanel: React.ReactNode
  defaultSplit: number
}

/**
 * ResizablePanels - Desktop-focused dual panel layout component
 *
 * Optimized for desktop use with:
 * - Horizontal side-by-side panel layout
 * - Drag-to-resize functionality
 * - Persistent panel sizing via localStorage
 * - Minimal responsive behavior for ultra-wide monitors only
 */
export const ResizablePanels: React.FC<ResizablePanelsProps> = ({
  leftPanel,
  rightPanel,
  defaultSplit = 0.5
}) => {
  const { leftPanelWidth, setPanelWidths } = useConfigStore()
  const [splitPosition, setSplitPosition] = useState(leftPanelWidth || defaultSplit)
  const containerRef = useRef<HTMLDivElement>(null)
  const isDragging = useRef(false)

  // Sync with store state when it changes
  useEffect(() => {
    setSplitPosition(leftPanelWidth)
  }, [leftPanelWidth])

  // Handle desktop-focused layout changes - simplified for desktop use
  useEffect(() => {
    // Desktop-focused: No responsive state management needed

    // Optional: Add minimal responsive behavior for very large desktop screens
    const handleResize = () => {
      const width = window.innerWidth

      // Only apply constraints for very wide screens (ultra-wide monitors)
      if (width > 1920) {
        // For ultra-wide screens, ensure panels don't get too extreme
        if (leftPanelWidth < 0.3 || leftPanelWidth > 0.7) {
          setPanelWidths(0.5, 0.5) // Reset to balanced 50/50 split
        }
      }
    }

    // Set initial layout
    handleResize()

    // Minimal resize listener - only for ultra-wide screens
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [leftPanelWidth, setPanelWidths])

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

    // Update store with new panel sizes
    const leftWidth = clampedPosition
    const rightWidth = 1 - clampedPosition
    setPanelWidths(leftWidth, rightWidth)
  }, [setPanelWidths])

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

  // Desktop-focused layout rendering

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
          overflow: 'hidden',
          minWidth: '200px',
          maxWidth: '80%'
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
          overflow: 'hidden',
          minWidth: '200px',
          maxWidth: '80%'
        }}
      >
        {rightPanel}
      </div>
    </div>
  )
}