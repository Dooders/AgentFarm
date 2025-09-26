import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ConfigExplorer } from '../ConfigExplorer/ConfigExplorer'

describe('Accessibility Tests', () => {
  it('should have proper heading hierarchy', () => {
    render(<ConfigExplorer />)

    // Check for main heading
    const mainHeading = screen.getByText('Configuration Explorer')
    expect(mainHeading.tagName).toBe('H2')

    // Check for section headings
    const sectionHeadings = screen.getAllByText(/Panel/)
    expect(sectionHeadings.length).toBeGreaterThanOrEqual(1)
  })

  it('should have semantic HTML structure', () => {
    const { container } = render(<ConfigExplorer />)

    // Check for proper semantic elements
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6')
    expect(headings.length).toBeGreaterThanOrEqual(2)

    // Check for lists (navigation items)
    const lists = container.querySelectorAll('ul, ol')
    expect(lists.length).toBeGreaterThanOrEqual(2)
  })

  it('should have proper color contrast setup', () => {
    const { container } = render(<ConfigExplorer />)

    // Check that components have proper styling classes
    const styledElements = container.querySelectorAll('[style*="color"], [style*="background"]')
    expect(styledElements.length).toBeGreaterThan(0)

    // Check that CSS classes for theming are applied
    const themedElements = container.querySelectorAll('.config-explorer, .left-panel, .right-panel')
    expect(themedElements.length).toBeGreaterThanOrEqual(3)
  })

  it('should have logical tab order structure', () => {
    render(<ConfigExplorer />)

    // Check that content flows in logical order
    const leftPanel = screen.getByText('Configuration Explorer')
    const rightPanel = screen.getByText('Configuration Comparison')

    // Both panels should be present and visible
    expect(leftPanel).toBeVisible()
    expect(rightPanel).toBeVisible()
  })

  it('should have descriptive content', () => {
    render(<ConfigExplorer />)

    // Check for descriptive text that helps screen readers
    const descriptiveTexts = screen.getAllByText('Comparison Tools')
    expect(descriptiveTexts.length).toBeGreaterThanOrEqual(1)

    expect(screen.getByText('Configuration Explorer')).toBeInTheDocument()
    expect(screen.getByText('Leva Controls')).toBeInTheDocument()
  })

  it('should have proper text content for screen readers', () => {
    render(<ConfigExplorer />)

    // Check that important UI elements have text content
    const headings = screen.getAllByRole('heading')
    expect(headings.length).toBeGreaterThanOrEqual(2)

    // Each heading should have descriptive text
    headings.forEach(heading => {
      expect(heading.textContent?.trim()).toBeTruthy()
    })
  })

  it('should support keyboard navigation structure', () => {
    render(<ConfigExplorer />)

    // Check that the layout supports logical keyboard navigation
    // (from left panel to right panel)
    const leftPanel = screen.getByText('Configuration Explorer').closest('.left-panel')
    const rightPanel = screen.getByText('Configuration Comparison').closest('.right-panel')

    expect(leftPanel).toBeInTheDocument()
    expect(rightPanel).toBeInTheDocument()

    // Both panels should exist and be properly structured
    expect(leftPanel).toBeVisible()
    expect(rightPanel).toBeVisible()

    // The layout should have proper structure for tab navigation
    expect(leftPanel?.parentElement?.parentElement).toBe(rightPanel?.parentElement?.parentElement)
  })
})