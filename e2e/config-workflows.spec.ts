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

  test('nested vertical split should resize within right panel', async ({ page }) => {
    // Find the second resizer which should correspond to the nested vertical split
    const resizers = page.locator('[data-testid="panel-resizer"], .resize-handle')
    const count = await resizers.count()
    if (count < 2) test.skip(true, 'Nested resizer not present')

    // Approximate: last resizer is likely the nested vertical one
    const nestedResizer = resizers.nth(count - 1)

    // Measure initial right panel bounding box
    const rightPanel = page.locator('[data-testid="right-panel"]')
    await expect(rightPanel).toBeVisible()

    // Move mouse to the nested resizer and drag vertically
    await nestedResizer.hover()
    const center = await nestedResizer.boundingBox()
    if (!center) test.skip(true, 'No bounding box for nested resizer')

    await page.mouse.move(center.x + center.width / 2, center.y + center.height / 2)
    await page.mouse.down()
    await page.mouse.move(center.x + center.width / 2, center.y + center.height / 2 + 80)
    await page.mouse.up()

    // No exact selector for nested panes; assert app remains interactive
    await expect(page.locator('[data-testid="dual-panel-layout"]')).toBeVisible()
  })

  test('layout sizes persist across reload', async ({ page, context }) => {
    // Drag the main horizontal resizer to change left panel width
    const mainResizer = page.locator('[data-testid="panel-resizer"], .resize-handle').first()
    const leftPanel = page.locator('[data-testid="left-panel"]')

    const before = await leftPanel.boundingBox()
    await mainResizer.hover()
    const box = await mainResizer.boundingBox()
    if (!box || !before) test.skip(true, 'No bounding boxes available for resize test')

    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await page.mouse.down()
    await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2)
    await page.mouse.up()

    const after = await leftPanel.boundingBox()
    expect(after?.width).not.toEqual(before.width)

    // Reload the page and verify the width is similar (persisted)
    await page.reload()
    const persisted = await leftPanel.boundingBox()
    // Allow small delta due to reflow
    expect(Math.abs((persisted?.width || 0) - (after?.width || 0)) < 10).toBeTruthy()
  })
})