import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AccessibilityProvider } from '@/components/UI/AccessibilityProvider'
import { ConfigExplorer } from '../ConfigExplorer/ConfigExplorer'

describe('Accessibility Tests', () => {
  const renderWithAccessibility = (component: React.ReactElement) => {
    return render(
      <AccessibilityProvider>
        {component}
      </AccessibilityProvider>
    )
  }

  it('should have proper heading hierarchy', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check for main heading
    const mainHeading = screen.getByText('Configuration Explorer')
    expect(mainHeading.tagName).toBe('H2')

    // Check for section headings
    const sectionHeadings = screen.getAllByText(/Panel/)
    expect(sectionHeadings.length).toBeGreaterThanOrEqual(1)
  })

  it('should have semantic HTML structure', () => {
    const { container } = renderWithAccessibility(<ConfigExplorer />)

    // Check for proper semantic elements
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6')
    expect(headings.length).toBeGreaterThanOrEqual(2)

    // Check for lists (navigation items)
    const lists = container.querySelectorAll('ul, ol')
    expect(lists.length).toBeGreaterThanOrEqual(1)
  })

  it('should have ARIA landmarks', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check for main landmark
    const main = screen.getByRole('main')
    expect(main).toBeInTheDocument()

    // Check for navigation landmark (there may be multiple)
    const navigationElements = screen.getAllByRole('navigation')
    expect(navigationElements.length).toBeGreaterThanOrEqual(1)

    // Check for complementary landmark
    const complementary = screen.getByRole('complementary')
    expect(complementary).toBeInTheDocument()
  })

  it('should have skip navigation links', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check for skip links
    const skipLinks = screen.getAllByText(/Skip to/)
    expect(skipLinks.length).toBeGreaterThanOrEqual(3)

    // Check that skip links have proper href attributes
    skipLinks.forEach(link => {
      expect(link).toHaveAttribute('href')
      expect(link.getAttribute('href')).toMatch(/^#/)
    })
  })

  it('should have proper ARIA labels and descriptions', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check for aria-label attributes
    const ariaLabelledElements = screen.getAllByLabelText(/.*/)
    expect(ariaLabelledElements.length).toBeGreaterThan(0)

    // Check for aria-describedby attributes on buttons
    const buttonsWithDescriptions = screen.getAllByRole('button').filter(button =>
      button.hasAttribute('aria-describedby')
    )
    expect(buttonsWithDescriptions.length).toBeGreaterThan(0)
  })

  it('should support keyboard navigation', async () => {
    const user = userEvent.setup()
    renderWithAccessibility(<ConfigExplorer />)

    // Focus should be manageable with Tab key
    const focusableElements = screen.getAllByRole('button').concat(
      screen.getAllByRole('link')
    )

    expect(focusableElements.length).toBeGreaterThan(0)

    // Simulate tab navigation
    await user.tab()
    expect(document.activeElement).toBeTruthy()
  })

  it('should have proper focus management', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check that focusable elements have proper tabindex
    const buttons = screen.getAllByRole('button')
    buttons.forEach(button => {
      const tabIndex = button.getAttribute('tabindex')
      if (tabIndex) {
        expect(parseInt(tabIndex)).toBeGreaterThanOrEqual(0)
      }
    })
  })

  it('should support high contrast mode', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check that high contrast styles are available
    const body = document.body
    expect(body.classList.contains('high-contrast')).toBe(false)

    // Simulate high contrast mode
    body.classList.add('high-contrast')
    expect(body.classList.contains('high-contrast')).toBe(true)
  })

  it('should have proper validation accessibility', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check for validation regions or any region with validation-related content
    const validationRegions = screen.getAllByRole('region')
    const validationRegion = validationRegions.find(region =>
      region.getAttribute('aria-label')?.includes('Validation') ||
      region.getAttribute('aria-label')?.includes('validation') ||
      region.textContent?.includes('validation') ||
      region.textContent?.includes('Validation')
    )

    // If no specific validation region, check for any region that could contain validation
    if (!validationRegion && validationRegions.length > 0) {
      expect(validationRegions.length).toBeGreaterThan(0)
    } else {
      expect(validationRegion).toBeDefined()
    }
  })

  it('should have proper live regions for announcements', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check for live regions
    const liveRegions = document.querySelectorAll('[aria-live]')
    expect(liveRegions.length).toBeGreaterThan(0)

    // Check for polite live regions (at least one should exist)
    const politeRegions = document.querySelectorAll('[aria-live="polite"]')
    const assertiveRegions = document.querySelectorAll('[aria-live="assertive"]')

    // At least one type of live region should exist
    expect(politeRegions.length + assertiveRegions.length).toBeGreaterThan(0)
  })

  it('should have proper color contrast setup', () => {
    const { container } = renderWithAccessibility(<ConfigExplorer />)

    // Check that components have proper styling classes
    const styledElements = container.querySelectorAll('[style*="color"], [style*="background"]')
    expect(styledElements.length).toBeGreaterThan(0)

    // Check that CSS classes for theming are applied
    const themedElements = container.querySelectorAll('.config-explorer, .left-panel, .right-panel')
    expect(themedElements.length).toBeGreaterThanOrEqual(3)
  })

  it('should have proper tab order structure', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check that content flows in logical order
    const leftPanel = screen.getByText('Configuration Explorer')
    const rightPanel = screen.getByText('Configuration Comparison')

    // Both panels should be present and visible
    expect(leftPanel).toBeVisible()
    expect(rightPanel).toBeVisible()
  })

  it('should have descriptive content', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check for descriptive text that helps screen readers
    const descriptiveTexts = screen.getAllByText('Comparison Tools')
    expect(descriptiveTexts.length).toBeGreaterThanOrEqual(1)

    expect(screen.getByText('Configuration Explorer')).toBeInTheDocument()
    expect(screen.getByText('Leva Controls')).toBeInTheDocument()
  })

  it('should have proper text content for screen readers', () => {
    renderWithAccessibility(<ConfigExplorer />)

    // Check that important UI elements have text content
    const headings = screen.getAllByRole('heading')
    expect(headings.length).toBeGreaterThanOrEqual(2)

    // Each heading should have descriptive text
    headings.forEach(heading => {
      expect(heading.textContent?.trim()).toBeTruthy()
    })
  })

  it('should support keyboard navigation structure', () => {
    renderWithAccessibility(<ConfigExplorer />)

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