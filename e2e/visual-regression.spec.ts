import { test, expect } from '@playwright/test'

test.describe('Visual Regression Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should match initial layout screenshot', async ({ page }) => {
    // Wait for the page to fully load
    await page.waitForLoadState('networkidle')

    // Take a screenshot of the full page
    await expect(page).toHaveScreenshot('initial-layout.png', {
      fullPage: true,
      animations: 'disabled',
    })
  })

  test('should match dual panel layout', async ({ page }) => {
    // Wait for layout to stabilize
    await page.waitForSelector('[data-testid="dual-panel-layout"]')

    // Take screenshot of specific element
    const layout = page.locator('[data-testid="dual-panel-layout"]')
    await expect(layout).toHaveScreenshot('dual-panel-layout.png', {
      animations: 'disabled',
    })
  })

  test('should match mobile layout', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })

    // Wait for responsive layout to apply
    await page.waitForTimeout(500)

    await expect(page).toHaveScreenshot('mobile-layout.png', {
      fullPage: true,
      animations: 'disabled',
    })
  })

  test('should match tablet layout', async ({ page }) => {
    // Set tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 })

    // Wait for responsive layout to apply
    await page.waitForTimeout(500)

    await expect(page).toHaveScreenshot('tablet-layout.png', {
      fullPage: true,
      animations: 'disabled',
    })
  })

  test('should match configuration controls', async ({ page }) => {
    // Wait for Leva controls to load
    await page.waitForSelector('[data-testid="leva-controls"]', { timeout: 10000 })

    const controls = page.locator('[data-testid="leva-controls"]')
    await expect(controls).toHaveScreenshot('configuration-controls.png', {
      animations: 'disabled',
    })
  })

  test('should match resizable panels interaction', async ({ page }) => {
    const resizer = page.locator('[data-testid="panel-resizer"], .resize-handle').first()

    if (await resizer.isVisible()) {
      // Hover over resizer to show visual feedback
      await resizer.hover()

      // Wait for visual feedback
      await page.waitForTimeout(200)

      // Take screenshot during interaction
      await expect(page).toHaveScreenshot('resizable-panels-hover.png', {
        fullPage: true,
        animations: 'disabled',
      })
    }
  })

  test('should match different themes', async ({ page }) => {
    // Test with different viewport sizes and orientations
    const viewports = [
      { width: 1920, height: 1080, name: 'desktop-wide' },
      { width: 1366, height: 768, name: 'desktop-narrow' },
      { width: 1024, height: 768, name: 'tablet-landscape' },
      { width: 768, height: 1024, name: 'tablet-portrait' },
    ]

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      await page.waitForTimeout(300)

      await expect(page).toHaveScreenshot(`${viewport.name}-layout.png`, {
        fullPage: true,
        animations: 'disabled',
      })
    }
  })
})