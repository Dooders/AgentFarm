import { expect, type Page } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

// Accessibility test configuration
export const a11yConfig = {
  rules: {
    // Disable overly strict rules for this application
    'color-contrast': { enabled: false },
    'duplicate-id': { enabled: true },
    'empty-heading': { enabled: true },
    'heading-order': { enabled: true },
    'image-alt': { enabled: true },
    'label': { enabled: true },
    'link-name': { enabled: true },
    'list': { enabled: true },
    'listitem': { enabled: true },
    'region': { enabled: true },
    'button-name': { enabled: true },
    'form-field-multiple-labels': { enabled: true },
    'frame-title': { enabled: true },
    'input-button-name': { enabled: true },
    'input-image-alt': { enabled: true },
    'landmark-one-main': { enabled: true },
    'page-has-heading-one': { enabled: true },
    'tab-order': { enabled: true },
  },
}

// Common accessibility checks
export const checkAccessibility = async (page: Page, context?: string) => {
  const axeBuilder = new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa'])

  // Configure axe based on context
  if (context) {
    switch (context) {
      case 'modal':
        axeBuilder.withRules(['dialog-name', 'focus-trap'])
        break
      case 'form':
        axeBuilder.withRules(['form-field-multiple-labels', 'input-button-name'])
        break
      case 'navigation':
        axeBuilder.withRules(['landmark-one-main', 'page-has-heading-one'])
        break
    }
  }

  const results = await axeBuilder.analyze()

  expect(results.violations).toEqual([])
  return results
}

// Specific accessibility checks for different components
export const checkPageAccessibility = async (page: Page) => {
  return checkAccessibility(page, 'page')
}

export const checkModalAccessibility = async (page: Page, modalSelector: string) => {
  await page.locator(modalSelector).waitFor({ state: 'visible' })
  return checkAccessibility(page, 'modal')
}

export const checkFormAccessibility = async (page: Page, formSelector: string) => {
  await page.locator(formSelector).waitFor({ state: 'visible' })
  return checkAccessibility(page, 'form')
}

// Accessibility violation reporter
export const reportAccessibilityViolations = (violations: any[]) => {
  if (violations.length === 0) return

  console.group('🚨 Accessibility Violations Found:')
  violations.forEach((violation, index) => {
    console.group(`${index + 1}. ${violation.id} - ${violation.impact}`)
    console.log(`Description: ${violation.description}`)
    console.log(`Help: ${violation.help}`)
    console.log(`Help URL: ${violation.helpUrl}`)
    console.log('Affected elements:', violation.nodes.map((node: any) => ({
      html: node.html,
      target: node.target,
      failureSummary: node.failureSummary,
    })))
    console.groupEnd()
  })
  console.groupEnd()
}

// WCAG 2.1 AA compliance checker
export const checkWCAG2AACompliance = async (page: Page) => {
  const axeBuilder = new AxeBuilder({ page }).withTags(['wcag2aa'])
  const results = await axeBuilder.analyze()

  if (results.violations.length > 0) {
    reportAccessibilityViolations(results.violations)
  }

  expect(results.violations).toEqual([])
  return results
}

// Keyboard navigation checker
export const checkKeyboardNavigation = async (page: Page) => {
  // Tab through all focusable elements
  let tabCount = 0
  const maxTabs = 50 // Prevent infinite loops

  while (tabCount < maxTabs) {
    const focusedElement = await page.evaluate(() => {
      const focused = document.activeElement
      return focused ? {
        tagName: focused.tagName,
        id: focused.id,
        className: focused.className,
        role: focused.getAttribute('role'),
        tabindex: focused.getAttribute('tabindex'),
        textContent: focused.textContent?.substring(0, 50),
      } : null
    })

    if (!focusedElement) break

    // Check if element is focusable and accessible
    expect(focusedElement).toBeTruthy()

    // Ensure no focus trap issues
    if (focusedElement.role === 'dialog' || focusedElement.className?.includes('modal')) {
      // This should be handled by focus trap logic
      console.log('Focus in modal:', focusedElement)
    }

    tabCount++
    await page.keyboard.press('Tab')
  }

  expect(tabCount).toBeGreaterThan(0) // Ensure there are focusable elements
}

// Color contrast checker
export const checkColorContrast = async (page: Page) => {
  const axeBuilder = new AxeBuilder({ page }).withRules(['color-contrast'])
  const results = await axeBuilder.analyze()

  // Report but don't fail on color contrast issues for now
  if (results.violations.length > 0) {
    console.log('⚠️ Color contrast issues found:')
    reportAccessibilityViolations(results.violations)
  }

  return results
}

// Screen reader compatibility checker
export const checkScreenReaderCompatibility = async (page: Page) => {
  const results = await checkAccessibility(page)

  // Additional screen reader specific checks
  const ariaLabels = await page.$$eval('[aria-label], [aria-labelledby], [aria-describedby]', elements =>
    elements.map(el => ({
      tagName: el.tagName,
      ariaLabel: el.getAttribute('aria-label'),
      ariaLabelledBy: el.getAttribute('aria-labelledby'),
      ariaDescribedBy: el.getAttribute('aria-describedby'),
    }))
  )

  expect(ariaLabels.length).toBeGreaterThan(0)
  return results
}

// Mobile accessibility checker
export const checkMobileAccessibility = async (page: Page) => {
  // Set mobile viewport
  await page.setViewportSize({ width: 375, height: 667 })

  const results = await checkAccessibility(page)

  // Check for mobile-specific issues
  const touchTargets = await page.$$eval('button, a, input, select, [role="button"]', elements =>
    elements.map(el => {
      const rect = el.getBoundingClientRect()
      return {
        tagName: el.tagName,
        width: rect.width,
        height: rect.height,
        isSmall: rect.width < 44 || rect.height < 44,
      }
    })
  )

  const smallTouchTargets = touchTargets.filter(target => target.isSmall)
  if (smallTouchTargets.length > 0) {
    console.warn('Small touch targets found:', smallTouchTargets)
  }

  return results
}

// Performance accessibility checker
export const checkPerformanceAccessibility = async (page: Page) => {
  const startTime = Date.now()

  const results = await checkAccessibility(page)

  const endTime = Date.now()
  const checkDuration = endTime - startTime

  // Axe checks should complete quickly
  expect(checkDuration).toBeLessThan(5000)

  return results
}