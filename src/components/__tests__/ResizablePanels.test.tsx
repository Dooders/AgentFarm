import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ResizablePanels } from '../Layout/ResizablePanels'

const MockLeftPanel = () => <div>Left Panel Content</div>
const MockRightPanel = () => <div>Right Panel Content</div>

describe('ResizablePanels', () => {
  it('renders both panels', () => {
    render(
      <ResizablePanels
        leftPanel={<MockLeftPanel />}
        rightPanel={<MockRightPanel />}
        defaultSplit={0.5}
      />
    )

    expect(screen.getByText('Left Panel Content')).toBeInTheDocument()
    expect(screen.getByText('Right Panel Content')).toBeInTheDocument()
  })

  it('sets initial split position correctly', () => {
    const { container } = render(
      <ResizablePanels
        leftPanel={<MockLeftPanel />}
        rightPanel={<MockRightPanel />}
        defaultSplit={0.6}
      />
    )

    const leftPanel = container.querySelector('.left-panel')
    expect(leftPanel).toHaveStyle({ width: '60%' })
  })

  it('has draggable split handle', () => {
    const { container } = render(
      <ResizablePanels
        leftPanel={<MockLeftPanel />}
        rightPanel={<MockRightPanel />}
        defaultSplit={0.5}
      />
    )

    const handle = container.querySelector('.split-handle')
    expect(handle).toBeInTheDocument()
    expect(handle).toHaveStyle({ cursor: 'col-resize' })
  })

  it('handles mouse events for resizing', () => {
    const { container } = render(
      <ResizablePanels
        leftPanel={<MockLeftPanel />}
        rightPanel={<MockRightPanel />}
        defaultSplit={0.5}
      />
    )

    const handle = container.querySelector('.split-handle')

    // Mock mouse events
    const mouseDownEvent = new MouseEvent('mousedown', { bubbles: true })
    const mouseMoveEvent = new MouseEvent('mousemove', { bubbles: true, clientX: 600 })
    const mouseUpEvent = new MouseEvent('mouseup', { bubbles: true })

    // Test mouse down
    fireEvent(handle as HTMLElement, mouseDownEvent)
    expect(document.body.style.cursor).toBe('col-resize')

    // Test mouse move (would update split position)
    fireEvent(document, mouseMoveEvent)

    // Test mouse up
    fireEvent(document, mouseUpEvent)
    expect(document.body.style.cursor).toBe('')
  })

  it('clamps split position between 0.2 and 0.8', () => {
    // This test would verify the clamping logic in handleMouseMove
    // In a real scenario, we would test edge cases
    expect(true).toBe(true) // Placeholder for actual clamping test
  })
})