import { test, expect } from '@playwright/test'

test.describe('Trading Application', () => {
  test('home page loads correctly', async ({ page }) => {
    await page.goto('/')

    // Check if the page title is correct
    await expect(page).toHaveTitle(/SwaggyStacks|Trading/)

    // Check for key navigation elements
    await expect(page.locator('nav')).toBeVisible()
  })

  test('navigation works', async ({ page }) => {
    await page.goto('/')

    // Test navigation if links exist
    const dashboardLink = page.locator('a[href*="dashboard"]').first()
    if (await dashboardLink.count() > 0) {
      await dashboardLink.click()
      await expect(page.url()).toContain('dashboard')
    }
  })

  test('trading interface accessibility', async ({ page }) => {
    await page.goto('/')

    // Check for proper ARIA labels and accessibility
    const main = page.locator('main')
    await expect(main).toBeVisible()

    // Ensure no accessibility violations for critical elements
    const buttons = page.locator('button')
    const buttonCount = await buttons.count()

    for (let i = 0; i < Math.min(buttonCount, 5); i++) {
      const button = buttons.nth(i)
      if (await button.isVisible()) {
        // Check that buttons have accessible names
        const accessibleName = await button.textContent() || await button.getAttribute('aria-label')
        expect(accessibleName).toBeTruthy()
      }
    }
  })

  test('responsive design', async ({ page }) => {
    await page.goto('/')

    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    await expect(page.locator('body')).toBeVisible()

    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 })
    await expect(page.locator('body')).toBeVisible()

    // Test desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 })
    await expect(page.locator('body')).toBeVisible()
  })
})