import { test, expect } from '@playwright/test'

test.describe('Live Simulation Config Explorer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should load the application successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Live Simulation Config Explorer/)
    await expect(page.locator('body')).toBeVisible()
  })

  test('should display dual panel layout', async ({ page }) => {
    // Check for main layout components
    await expect(page.locator('[data-testid="dual-panel-layout"]')).toBeVisible()
    await expect(page.locator('[data-testid="left-panel"]')).toBeVisible()
    await expect(page.locator('[data-testid="right-panel"]')).toBeVisible()
  })

  test('should have resizable panels', async ({ page }) => {
    const leftPanel = page.locator('[data-testid="left-panel"]')
    const rightPanel = page.locator('[data-testid="right-panel"]')

    // Get initial sizes
    const initialLeftWidth = await leftPanel.boundingBox().then(box => box?.width || 0)
    const initialRightWidth = await rightPanel.boundingBox().then(box => box?.width || 0)

    // Verify panels have reasonable sizes
    expect(initialLeftWidth).toBeGreaterThan(100)
    expect(initialRightWidth).toBeGreaterThan(200)
  })

  test('should display configuration explorer', async ({ page }) => {
    await expect(page.locator('[data-testid="config-explorer"]')).toBeVisible()
  })

  test('should have Leva controls panel', async ({ page }) => {
    await expect(page.locator('[data-testid="leva-controls"]')).toBeVisible()
  })
})