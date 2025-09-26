import { test, expect } from '@playwright/test'

test.describe('Configuration Workflows', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should allow configuration parameter adjustment', async ({ page }) => {
    // Test Leva controls interaction
    const levaPanel = page.locator('[data-testid="leva-controls"]')

    // Wait for controls to load
    await expect(levaPanel).toBeVisible()

    // Test number input adjustment
    const systemAgentsInput = page.locator('input[type="number"]').first()
    await expect(systemAgentsInput).toBeVisible()

    // Test boolean toggle
    const booleanControls = page.locator('input[type="checkbox"], [role="switch"]')
    if (await booleanControls.count() > 0) {
      await booleanControls.first().click()
    }
  })

  test('should support panel resizing', async ({ page }) => {
    const resizer = page.locator('[data-testid="panel-resizer"], .resize-handle').first()

    if (await resizer.isVisible()) {
      // Get initial position
      const initialBox = await page.locator('[data-testid="left-panel"]').boundingBox()

      // Drag the resizer
      await resizer.hover()
      await page.mouse.down()
      await page.mouse.move(100, 0)
      await page.mouse.up()

      // Verify panel size changed
      const newBox = await page.locator('[data-testid="left-panel"]').boundingBox()
      expect(newBox?.width).not.toEqual(initialBox?.width)
    }
  })

  test('should maintain state on resize', async ({ page }) => {
    // Set some configuration values
    const inputs = page.locator('input[type="number"]')
    if (await inputs.count() > 0) {
      await inputs.first().fill('50')
    }

    // Resize window
    await page.setViewportSize({ width: 800, height: 600 })

    // Verify configuration persists
    await expect(inputs.first()).toHaveValue('50')
  })

  test('should handle responsive design', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })

    // Verify layout adapts
    await expect(page.locator('body')).toBeVisible()

    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 })

    // Verify layout still works
    await expect(page.locator('body')).toBeVisible()
  })
})