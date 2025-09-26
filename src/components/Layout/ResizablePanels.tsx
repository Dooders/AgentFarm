import React, { useState, useRef, useCallback, useEffect } from 'react'
import { useConfigStore } from '@/stores/configStore'

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
  const { leftPanelWidth, setPanelWidths } = useConfigStore()
  const [splitPosition, setSplitPosition] = useState(leftPanelWidth || defaultSplit)
  const [isMobile, setIsMobile] = useState(false)
  const [isTablet, setIsTablet] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const isDragging = useRef(false)

  // Sync with store state when it changes
  useEffect(() => {
    setSplitPosition(leftPanelWidth)
  }, [leftPanelWidth])

  // Handle responsive layout changes with debouncing and breakpoint transition detection
  useEffect(() => {
    let timeoutId: NodeJS.Timeout
    let currentBreakpoint: string

    const getBreakpoint = (width: number): string => {
      if (width <= 768) return 'mobile'
      if (width <= 1024) return 'tablet'
      return 'desktop'
    }

    const handleResize = () => {
      const width = window.innerWidth
      const newBreakpoint = getBreakpoint(width)

      // Only update if we've transitioned between breakpoints
      if (newBreakpoint !== currentBreakpoint) {
        currentBreakpoint = newBreakpoint

        // Check if mobile (stack panels vertically)
        if (width <= 768) {
          setIsMobile(true)
          setIsTablet(false)
          // On mobile, reset to equal split only when transitioning TO mobile
          if (leftPanelWidth !== 0.5) {
            setPanelWidths(0.5, 0.5)
          }
        } else if (width <= 1024) {
          setIsMobile(false)
          setIsTablet(true)
          // On tablet, use a more balanced split only when transitioning TO tablet
          if (leftPanelWidth < 0.4 || leftPanelWidth > 0.6) {
            setPanelWidths(0.5, 0.5)
          }
        } else {
          setIsMobile(false)
          setIsTablet(false)
        }
      }
    }

    // Set initial state and breakpoint
    currentBreakpoint = getBreakpoint(window.innerWidth)
    handleResize()

    // Debounced resize listener to prevent excessive updates
    const debouncedResize = () => {
      clearTimeout(timeoutId)
      timeoutId = setTimeout(handleResize, 150) // 150ms debounce
    }

    window.addEventListener('resize', debouncedResize)

    return () => {
      clearTimeout(timeoutId)
      window.removeEventListener('resize', debouncedResize)
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

  // Responsive layout rendering
  if (isMobile) {
    // Mobile: Stack panels vertically
    return (
      <div
        ref={containerRef}
        className="resizable-panels mobile"
        style={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          width: '100%',
          position: 'relative'
        }}
      >
        <div
          className="left-panel"
          style={{
            height: '50%',
            overflow: 'hidden',
            borderBottom: '1px solid var(--border-medium)'
          }}
        >
          {leftPanel}
        </div>

        <div
          className="right-panel"
          style={{
            height: '50%',
            overflow: 'hidden'
          }}
        >
          {rightPanel}
        </div>
      </div>
    )
  }

  // Desktop/Tablet: Side-by-side layout
  return (
    <div
      ref={containerRef}
      className={`resizable-panels ${isTablet ? 'tablet' : 'desktop'}`}
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
          minWidth: isTablet ? '300px' : '200px',
          maxWidth: isTablet ? '70%' : '80%'
        }}
      >
        {leftPanel}
      </div>

      <div
        className="split-handle"
        style={{
          width: isTablet ? '12px' : '8px',
          cursor: isMobile ? 'default' : 'col-resize',
          backgroundColor: 'var(--border-subtle)',
          borderLeft: '1px solid var(--border-medium)',
          borderRight: '1px solid var(--border-medium)',
          position: 'relative',
          display: isMobile ? 'none' : 'block'
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
          minWidth: isTablet ? '250px' : '200px'
        }}
      >
        {rightPanel}
      </div>
    </div>
  )
}