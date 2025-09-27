import React, { useState, useRef, useCallback, useEffect, Children } from 'react'
import { useConfigStore } from '@/stores/configStore'

type Direction = 'horizontal' | 'vertical'

interface ResizablePanelsProps {
  direction?: Direction
  defaultSizes?: number[] // percentages that sum to ~100
  minSizes?: number[] // min percentages per panel
  gutterSize?: number // px
  persistKey?: string // key to persist/restore sizes
  onSizesChange?: (sizes: number[]) => void
  children: React.ReactNode
}

// Utility to normalize sizes to sum to 100
function normalizePercentages(values: number[]): number[] {
  const total = values.reduce((a, b) => a + b, 0)
  if (total === 0) {
    const n = values.length
    return n === 0 ? [] : Array(n).fill(100 / n)
  }
  return values.map((v: number) => (v / total) * 100)
}

export const ResizablePanels: React.FC<ResizablePanelsProps> = (props: ResizablePanelsProps) => {
  const {
    direction = 'horizontal',
    defaultSizes,
    minSizes,
    gutterSize = 8,
    persistKey,
    onSizesChange,
    children
  } = props
  const childArray = Children.toArray(children)
  const panelCount = childArray.length

  const { getLayoutSizes, setLayoutSizes } = useConfigStore()

  // Initialize sizes
  const persisted = persistKey ? getLayoutSizes(persistKey) : undefined
  const initial = panelCount === 0
    ? []
    : normalizePercentages(
        (persisted && persisted.length === panelCount ? persisted : defaultSizes) ||
          Array(panelCount).fill(100 / panelCount)
      )
  const [sizes, setSizes] = useState<number[]>(initial)
  const [isDragging, setIsDragging] = useState<boolean>(false)

  useEffect(() => {
    // If persistKey changes or store updates externally, sync sizes
    if (persistKey) {
      const latest = getLayoutSizes(persistKey)
      if (latest && latest.length === panelCount) {
        setSizes(normalizePercentages(latest))
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [persistKey, panelCount])

  const containerRef = useRef<HTMLDivElement>(null)
  const dragInfo = useRef<{ active: boolean; gutterIndex: number; startPos: number; startSizes: number[] } | null>(null)

  // Persist helper
  const persistSizes = useCallback(
    (next: number[]) => {
      if (persistKey) {
        setLayoutSizes(persistKey, next)
      }
      onSizesChange?.(next)
    },
    [persistKey, setLayoutSizes, onSizesChange]
  )

  const clampSizes = useCallback(
    (next: number[]): number[] => {
      const result = [...next]
      // Apply minimums if provided
      if (minSizes && minSizes.length === panelCount) {
        for (let i = 0; i < panelCount; i++) {
          result[i] = Math.max(minSizes[i] as number, result[i])
        }
      } else {
        // Ensure non-negative by default
        for (let i = 0; i < panelCount; i++) {
          result[i] = Math.max(0, result[i])
        }
      }
      // Ensure the sizes sum to exactly 100 by normalizing proportionally
      return normalizePercentages(result)
    },
    [minSizes, panelCount]
  )

  const onPointerDown = useCallback((index: number, clientX: number, clientY: number) => {
    if (!containerRef.current) return
    dragInfo.current = {
      active: true,
      gutterIndex: index,
      startPos: direction === 'horizontal' ? clientX : clientY,
      startSizes: [...sizes]
    }
    setIsDragging(true)
    document.body.style.userSelect = 'none'
    document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize'
  }, [direction, sizes])

  const onPointerMove = useCallback((clientX: number, clientY: number) => {
    if (!dragInfo.current || !dragInfo.current.active || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const totalPx = direction === 'horizontal' ? rect.width : rect.height
    const currentPos = direction === 'horizontal' ? clientX : clientY
    const deltaPx = currentPos - dragInfo.current.startPos
    const deltaPct = (deltaPx / Math.max(totalPx, 1)) * 100

    const i = dragInfo.current.gutterIndex
    const next = [...dragInfo.current.startSizes]
    // Adjust two adjacent panels
    next[i] = next[i] + deltaPct
    next[i + 1] = next[i + 1] - deltaPct

    // Enforce minimums
    const normalized = clampSizes(next)
    setSizes(normalized)
  }, [clampSizes, direction])

  const onPointerUp = useCallback(() => {
    if (!dragInfo.current) return
    dragInfo.current.active = false
    setIsDragging(false)
    document.body.style.userSelect = ''
    document.body.style.cursor = ''
    // Persist
    persistSizes(sizes)
  }, [persistSizes, sizes])

  // Mouse handlers
  const onMouseDown = useCallback((index: number) => (e: React.MouseEvent) => {
    e.preventDefault()
    onPointerDown(index, e.clientX, e.clientY)
  }, [onPointerDown])

  const onMouseMove = useCallback((e: MouseEvent) => {
    onPointerMove(e.clientX, e.clientY)
  }, [onPointerMove])

  const onMouseUp = useCallback(() => {
    onPointerUp()
  }, [onPointerUp])

  // Touch handlers
  const onTouchStart = useCallback((index: number) => (e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      const t = e.touches[0]
      onPointerDown(index, t.clientX, t.clientY)
    }
  }, [onPointerDown])

  const onTouchMove = useCallback((e: TouchEvent) => {
    if (e.touches.length > 0) {
      const t = e.touches[0]
      onPointerMove(t.clientX, t.clientY)
    }
  }, [onPointerMove])

  const onTouchEnd = useCallback(() => {
    onPointerUp()
  }, [onPointerUp])

  useEffect(() => {
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
    document.addEventListener('touchmove', onTouchMove, { passive: false })
    document.addEventListener('touchend', onTouchEnd)
    return () => {
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
      document.removeEventListener('touchmove', onTouchMove)
      document.removeEventListener('touchend', onTouchEnd)
    }
  }, [onMouseMove, onMouseUp, onTouchMove, onTouchEnd])

  // Ultra-wide sanity adjustment for main horizontal split (optional)
  useEffect(() => {
    if (direction === 'horizontal' && panelCount === 2) {
      const width = window.innerWidth
      if (width > 1920 && (sizes[0] < 30 || sizes[0] > 70)) {
        const balanced = [50, 50]
        setSizes(balanced)
        persistSizes(balanced)
      }
    }
  }, [direction, panelCount, sizes, persistSizes])

  // Keyboard handler for handle at index
  const onHandleKeyDown = useCallback((index: number) => (e: React.KeyboardEvent) => {
    const key = e.key
    const step = e.shiftKey ? 5 : 1
    let handled = false
    const next = [...sizes]
    if (direction === 'horizontal') {
      if (key === 'ArrowLeft') {
        next[index] = next[index] + step
        next[index + 1] = next[index + 1] - step
        handled = true
      } else if (key === 'ArrowRight') {
        next[index] = next[index] - step
        next[index + 1] = next[index + 1] + step
        handled = true
      }
    } else {
      if (key === 'ArrowUp') {
        next[index] = next[index] + step
        next[index + 1] = next[index + 1] - step
        handled = true
      } else if (key === 'ArrowDown') {
        next[index] = next[index] - step
        next[index + 1] = next[index + 1] + step
        handled = true
      }
    }
    if (handled) {
      e.preventDefault()
      const normalized = clampSizes(next)
      setSizes(normalized)
      persistSizes(normalized)
    }
  }, [sizes, direction, clampSizes, persistSizes])

  const onHandleDoubleClick = useCallback((index: number) => () => {
    let reset: number[]
    if (defaultSizes && defaultSizes.length === panelCount) {
      reset = normalizePercentages(defaultSizes)
    } else {
      reset = Array(panelCount).fill(100 / Math.max(panelCount, 1))
    }
    setSizes(reset)
    persistSizes(reset)
  }, [defaultSizes, panelCount, persistSizes])

  // Render
  return (
    <div
      ref={containerRef}
      className={`resizable-panels ${direction}${isDragging ? ' dragging' : ''}`}
      style={{
        display: 'flex',
        flexDirection: direction === 'horizontal' ? 'row' : 'column',
        height: '100%',
        width: '100%',
        position: 'relative'
      }}
    >
      {childArray.map((child: React.ReactNode, i: number) => (
        <React.Fragment key={`panel-${i}`}>
          <div
            className={`panel panel-${i}${panelCount === 2 && i === 0 ? ' left-panel' : ''}${panelCount === 2 && i === 1 ? ' right-panel' : ''}`}
            data-testid={`panel-${i}`}
            style={{
              [direction === 'horizontal' ? 'width' : 'height']: `${sizes[i]}%`,
              [direction === 'horizontal' ? 'height' : 'width']: '100%',
              overflow: 'hidden',
              transition: isDragging ? 'none' : 'width 0.15s ease, height 0.15s ease'
            } as React.CSSProperties}
          >
            {child}
          </div>
          {i < panelCount - 1 && (
            <div
              className="split-handle resize-handle"
              data-testid="panel-resizer"
              role="separator"
              aria-orientation={direction === 'horizontal' ? 'vertical' : 'horizontal'}
              tabIndex={0}
              onKeyDown={onHandleKeyDown(i)}
              onMouseDown={onMouseDown(i)}
              onTouchStart={onTouchStart(i)}
              onDoubleClick={onHandleDoubleClick(i)}
              style={{
                [direction === 'horizontal' ? 'width' : 'height']: `${gutterSize}px`,
                [direction === 'horizontal' ? 'height' : 'width']: '100%',
                cursor: direction === 'horizontal' ? 'col-resize' : 'row-resize',
                backgroundColor: 'var(--border-subtle)',
                borderLeft: direction === 'horizontal' ? '1px solid var(--border-medium)' : undefined,
                borderRight: direction === 'horizontal' ? '1px solid var(--border-medium)' : undefined,
                borderTop: direction === 'vertical' ? '1px solid var(--border-medium)' : undefined,
                borderBottom: direction === 'vertical' ? '1px solid var(--border-medium)' : undefined,
                position: 'relative'
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: direction === 'horizontal' ? '2px' : '32px',
                  height: direction === 'horizontal' ? '32px' : '2px',
                  backgroundColor: 'var(--border-medium)'
                }}
              />
            </div>
          )}
        </React.Fragment>
      ))}
    </div>
  )
}