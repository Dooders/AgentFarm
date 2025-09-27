import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { DualPanelLayout } from '../Layout/DualPanelLayout'

describe('DualPanelLayout', () => {
  it('renders the complete layout structure', () => {
    render(<DualPanelLayout />)

    // Check that both panels are rendered
    expect(screen.getByText('Configuration Explorer')).toBeInTheDocument()
    expect(screen.getByText('Configuration Comparison')).toBeInTheDocument()

    // Toolbar should be present
    expect(screen.getByRole('toolbar', { name: /application toolbar/i })).toBeInTheDocument()
  })

  it('uses ResizablePanels as the layout container', () => {
    const { container } = render(<DualPanelLayout />)

    // Check for the resizable panels structure
    const resizableContainer = container.querySelector('.dual-panel-layout')
    expect(resizableContainer).toBeInTheDocument()

    // Check for split handle
    const splitHandle = container.querySelector('.split-handle')
    expect(splitHandle).toBeInTheDocument()
  })

  it('applies correct styling to the layout', () => {
    const { container } = render(<DualPanelLayout />)

    const layout = container.querySelector('.dual-panel-layout')
    expect(layout).toHaveStyle({
      height: '100vh',
      width: '100vw'
    })
  })

  it('passes default split ratio to ResizablePanels', () => {
    const { container } = render(<DualPanelLayout />)

    const leftPanel = container.querySelector('.left-panel')
    // Should be 50% width based on desktop-focused default split of 0.5
    expect(leftPanel).toHaveStyle({ width: '50%' })
  })

  it('contains both LeftPanel and RightPanel components', () => {
    render(<DualPanelLayout />)

    // Left panel content
    expect(screen.getByText('Configuration Explorer')).toBeInTheDocument()
    expect(screen.getByText('Leva Controls')).toBeInTheDocument()

    // Right panel content
    expect(screen.getByText('Configuration Comparison')).toBeInTheDocument()
  })

  it('maintains desktop-focused layout structure', () => {
    const { container } = render(<DualPanelLayout />)

    // Check that the layout is properly structured
    const layout = container.querySelector('.dual-panel-layout')
    const leftPanel = container.querySelector('.left-panel')
    const rightPanel = container.querySelector('.right-panel')
    const splitHandle = container.querySelector('.split-handle')

    expect(layout).toContainElement(leftPanel as HTMLElement)
    expect(layout).toContainElement(splitHandle as HTMLElement)
    expect(layout).toContainElement(rightPanel as HTMLElement)
  })
})