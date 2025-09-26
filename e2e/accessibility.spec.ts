import { test, expect } from '@playwright/test'
import {
  checkPageAccessibility,
  checkWCAG2AACompliance,
  checkKeyboardNavigation,
  checkColorContrast,
  checkScreenReaderCompatibility,
  checkMobileAccessibility,
} from '../src/test/accessibility'

test.describe('Accessibility Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should pass basic accessibility checks', async ({ page }) => {
    await checkPageAccessibility(page)
  })

  test('should comply with WCAG 2.1 AA standards', async ({ page }) => {
    await checkWCAG2AACompliance(page)
  })

  test('should have proper keyboard navigation', async ({ page }) => {
    await checkKeyboardNavigation(page)
  })

  test('should have sufficient color contrast', async ({ page }) => {
    await checkColorContrast(page)
  })

  test('should be compatible with screen readers', async ({ page }) => {
    await checkScreenReaderCompatibility(page)
  })

  test('should be accessible on mobile devices', async ({ page }) => {
    await checkMobileAccessibility(page)
  })

  test('should have proper ARIA labels and roles', async ({ page }) => {
    // Check main layout components
    const dualPanelLayout = page.locator('[data-testid="dual-panel-layout"]')
    await expect(dualPanelLayout).toHaveAttribute('role', /main|application|complementary/)

    const leftPanel = page.locator('[data-testid="left-panel"]')
    await expect(leftPanel).toHaveAttribute('aria-label')

    const rightPanel = page.locator('[data-testid="right-panel"]')
    await expect(rightPanel).toHaveAttribute('aria-label')

    // Check configuration controls
    const controls = page.locator('[data-testid="leva-controls"]')
    if (await controls.isVisible()) {
      await expect(controls).toHaveAttribute('aria-label')
    }
  })

  test('should have proper heading structure', async ({ page }) => {
    const headings = await page.$$eval('h1, h2, h3, h4, h5, h6', headings =>
      headings.map(h => ({
        level: h.tagName,
        text: h.textContent?.trim(),
      }))
    )

    // Should have at least one h1
    const h1Headings = headings.filter(h => h.level === 'H1')
    expect(h1Headings.length).toBeGreaterThan(0)

    // Check heading hierarchy
    const headingLevels = headings.map(h => parseInt(h.level.charAt(1)))
    for (let i = 1; i < headingLevels.length; i++) {
      expect(headingLevels[i]).toBeLessThanOrEqual(headingLevels[i - 1] + 1)
    }
  })

  test('should have accessible form controls', async ({ page }) => {
    // Check for form inputs and labels
    const inputs = await page.$$eval('input, select, textarea', inputs =>
      inputs.map(input => ({
        type: input.type || input.tagName.toLowerCase(),
        id: input.id,
        ariaLabel: input.getAttribute('aria-label'),
        ariaLabelledBy: input.getAttribute('aria-labelledby'),
      }))
    )

    // Each input should have some form of label
    inputs.forEach(input => {
      const hasLabel = input.id || input.ariaLabel || input.ariaLabelledBy
      expect(hasLabel).toBeTruthy()
    })
  })

  test('should have accessible buttons and links', async ({ page }) => {
    // Check buttons
    const buttons = await page.$$eval('button', buttons =>
      buttons.map(button => ({
        text: button.textContent?.trim(),
        ariaLabel: button.getAttribute('aria-label'),
        disabled: button.disabled,
      }))
    )

    buttons.forEach(button => {
      if (!button.disabled) {
        const hasLabel = button.text || button.ariaLabel
        expect(hasLabel).toBeTruthy()
      }
    })

    // Check links
    const links = await page.$$eval('a', links =>
      links.map(link => ({
        text: link.textContent?.trim(),
        href: link.href,
      }))
    )

    links.forEach(link => {
      expect(link.text).toBeTruthy()
      expect(link.href).toBeTruthy()
    })
  })

  test('should have proper focus management', async ({ page }) => {
    // Focus first focusable element
    await page.keyboard.press('Tab')
    let focusedElement = await page.evaluate(() => document.activeElement?.tagName)

    // Should focus a real element
    expect(focusedElement).toBeTruthy()

    // Tab through several elements
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Tab')
      focusedElement = await page.evaluate(() => document.activeElement?.tagName)
      expect(focusedElement).toBeTruthy()
    }

    // Shift+Tab should go backwards
    await page.keyboard.press('Shift+Tab')
    const previousElement = await page.evaluate(() => document.activeElement?.tagName)
    expect(previousElement).toBeTruthy()
  })

  test('should handle dynamic content accessibility', async ({ page }) => {
    // This test would depend on specific dynamic content in the app
    // For now, just check that the page remains accessible after interactions

    // Perform some interactions
    const inputs = await page.$$('input[type="number"]')
    if (inputs.length > 0) {
      await inputs[0].fill('50')
      await inputs[0].press('Enter')
    }

    // Check accessibility after interaction
    await checkPageAccessibility(page)
  })

  test('should have accessible error states', async ({ page }) => {
    // Try to trigger validation errors if possible
    // This depends on the specific validation logic in the app

    const errorElements = await page.$$('[role="alert"], [aria-live="assertive"], .error')
    // If there are error states, they should be properly announced
    errorElements.forEach(async (element) => {
      const ariaLive = await element.getAttribute('aria-live')
      expect(['assertive', 'polite', null]).toContain(ariaLive)
    })
  })

  test('should maintain accessibility across different viewport sizes', async ({ page }) => {
    const viewports = [
      { width: 1920, height: 1080, name: 'desktop' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 375, height: 667, name: 'mobile' },
    ]

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      await page.waitForTimeout(300)

      await checkPageAccessibility(page)
    }
  })
})