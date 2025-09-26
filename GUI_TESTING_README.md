# Comprehensive GUI Testing Suite

This document describes the comprehensive GUI testing suite implemented for the Live Simulation Config Explorer application. The suite includes visual testing, end-to-end testing, accessibility validation, performance monitoring, and advanced testing utilities.

## Overview

The GUI testing suite is built on top of modern testing tools and follows best practices for:

- **Component Development**: Storybook for isolated component development
- **End-to-End Testing**: Playwright for comprehensive user workflow testing
- **Visual Regression**: Screenshot-based testing for UI consistency
- **Accessibility**: WCAG 2.1 AA compliance with axe-core and comprehensive a11y testing
- **Enhanced Accessibility Features**: Advanced focus management, keyboard navigation, and screen reader support
- **API Mocking**: MSW for reliable external service simulation
- **Cross-Browser Testing**: Multi-browser compatibility validation
- **Performance Monitoring**: Load testing and performance validation

## Testing Infrastructure

### 1. Storybook Setup

**Location**: `.storybook/`
**Purpose**: Component development and visual testing

#### Configuration Files:
- `main.ts` - Storybook configuration with Vite integration
- `preview.ts` - Global settings for stories and viewports

#### Key Features:
- Custom theme integration
- Multiple viewport configurations (mobile, tablet, desktop)
- Accessibility addon integration
- Dark/light theme support

#### Usage:
```bash
# Start Storybook development server
npm run storybook

# Build Storybook for production
npm run storybook:build

# Run Storybook accessibility tests
npm run storybook:test
```

### 2. Playwright E2E Testing

**Location**: `e2e/`
**Purpose**: End-to-end user workflow testing

#### Configuration:
- `playwright.config.ts` - Main Playwright configuration
- Multiple browser support (Chromium, Firefox, WebKit)
- Mobile and tablet viewport testing
- Screenshot-based visual regression

#### Test Categories:
- `basic.spec.ts` - Fundamental layout and functionality tests
- `config-workflows.spec.ts` - Configuration parameter workflows
- `visual-regression.spec.ts` - Visual consistency tests
- `accessibility.spec.ts` - WCAG 2.1 AA compliance tests

#### Usage:
```bash
# Run all E2E tests
npm run test:e2e

# Run tests with UI
npm run test:e2e:ui

# Run tests in headed mode (visible browser)
npm run test:e2e:headed
```

### 3. Advanced Accessibility Testing

**Location**: `src/test/accessibility.ts` and `src/components/__tests__/Accessibility.test.tsx`
**Purpose**: WCAG 2.1 AA compliance validation with enhanced accessibility features

#### Enhanced Features (Issue #13 Implementation):
- **Comprehensive Focus Management**: High-contrast focus rings, focus trapping, and restoration
- **Advanced Keyboard Navigation**: Arrow key navigation, skip links, Enter/Space activation
- **ARIA Compliance**: Proper landmarks, live regions, and screen reader announcements
- **High Contrast Mode**: Enhanced visibility testing for visual impairments
- **Screen Reader Optimization**: Live regions, announcements, and semantic structure

#### Core Testing Functions:
- `checkPageAccessibility()` - Basic accessibility checks
- `checkWCAG2AACompliance()` - Full WCAG 2.1 AA validation
- `checkKeyboardNavigation()` - Keyboard-only navigation testing
- `checkMobileAccessibility()` - Mobile-specific accessibility
- `checkScreenReaderCompatibility()` - Screen reader specific validation

#### New Testing Features:
- **Focus State Testing**: Enhanced focus indicators and high contrast mode validation
- **Skip Navigation Testing**: Skip link functionality and focus management
- **Live Region Testing**: Dynamic content announcements for screen readers
- **ARIA Landmark Testing**: Proper semantic structure validation

#### Accessibility Components Testing:
- **`AccessibilityProvider`**: Global accessibility context and state management
- **`SkipNavigation`**: Skip link functionality with focus management
- **`useKeyboardNavigation`**: Custom hook for keyboard navigation patterns
- **`focusManagement`**: Focus trapping, restoration, and navigation utilities

#### Enhanced Component Testing:
- **Comprehensive ARIA testing** for landmarks, labels, and roles
- **Keyboard navigation simulation** with user event testing
- **High contrast mode validation** and color contrast analysis
- **Focus management verification** for proper tab order and focus trapping

### 4. API Mocking with MSW

**Location**: `src/test/mocks/`
**Purpose**: Reliable external service simulation

#### Components:
- `handlers.ts` - Mock API endpoint definitions
- `server.ts` - MSW server setup and control
- `test-utils.ts` - Mock data generators and utilities

#### Features:
- RESTful API endpoint mocking
- Error simulation endpoints
- Performance testing endpoints
- Large dataset simulation
- Custom response builders

#### Usage:
```typescript
import { startMSW, stopMSW } from '@/test/mocks/server'

// In test setup
beforeAll(async () => {
  await startMSW('worker')
})

afterAll(async () => {
  await stopMSW('worker')
})
```

### 5. Test Utilities

**Location**: `src/test/test-helpers.ts`
**Purpose**: Common testing patterns and utilities

#### Categories:
- **Panel Interactions**: Resizing, dragging, layout manipulation
- **Form Controls**: Input manipulation, validation testing
- **Navigation**: Menu navigation, section switching
- **State Management**: Zustand store testing utilities
- **Performance**: Render time measurement, load testing
- **Responsive Design**: Viewport testing, layout validation
- **Accessibility**: ARIA testing, focus management

## Test Organization

### Directory Structure
```
tests/
├── e2e/                    # End-to-end tests
│   ├── basic.spec.ts
│   ├── config-workflows.spec.ts
│   ├── visual-regression.spec.ts
│   └── accessibility.spec.ts
├── src/
│   ├── test/
│   │   ├── mocks/         # MSW mock definitions
│   │   ├── accessibility.ts # Axe integration
│   │   ├── test-helpers.ts # Common utilities
│   │   ├── setup.ts       # Test environment setup
│   │   └── test-utils.tsx # React Testing Library utils
│   ├── components/
│   │   └── */*.stories.tsx # Storybook stories
│   └── __tests__/         # Unit tests
└── .storybook/            # Storybook configuration
```

## Running Tests

### Development Workflow
```bash
# 1. Start Storybook for component development
npm run storybook

# 2. Run unit tests with coverage
npm run test:coverage

# 3. Run E2E tests
npm run test:e2e

# 4. Run accessibility tests
npm run test:e2e # (includes accessibility checks)

# 5. Run all tests
npm run test:all
```

### CI/CD Pipeline
```bash
# Headless testing for CI
npm run test:run          # Unit tests
npm run test:e2e          # E2E tests
npm run test:coverage:all # Coverage reports
```

## Best Practices

### 1. Writing Tests

#### Component Tests
```typescript
// Use descriptive test names
test('should display configuration controls when panel is expanded', async () => {
  // Arrange
  render(<LevaControls />)

  // Act
  await expandFolder('Agent Parameters')

  // Assert
  expect(screen.getByText('System Agent')).toBeInTheDocument()
})
```

#### E2E Tests
```typescript
test('should allow user to modify system agent parameters', async ({ page }) => {
  // Navigate
  await page.goto('/')

  // Interact
  await page.fill('[data-testid="system-agents-input"]', '50')

  // Verify
  await expect(page.locator('[data-testid="system-agents-value"]')).toHaveText('50')
})
```

### 2. Accessibility Testing
```typescript
// Always include accessibility checks
test('should be accessible to screen readers', async ({ page }) => {
  await checkPageAccessibility(page)
  await checkKeyboardNavigation(page)
})
```

### 3. Visual Regression Testing
```typescript
// Use meaningful screenshot names
test('should display consistent layout on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 })
  await expect(page).toHaveScreenshot('mobile-layout.png')
})
```

### 4. API Mocking
```typescript
// Mock external dependencies
test('should handle API errors gracefully', async ({ page }) => {
  await page.route('/api/config', route => route.abort())
  await expect(page.locator('[data-testid="error-message"]')).toBeVisible()
})
```

## Performance Testing

### Load Testing
```typescript
test('should handle large configuration files', async ({ page }) => {
  const startTime = performance.now()

  // Load large configuration
  await page.goto('/?config=large-dataset')

  // Measure load time
  const loadTime = performance.now() - startTime
  expect(loadTime).toBeLessThan(3000) // 3 seconds max
})
```

### Memory Leak Detection
```typescript
test('should not leak memory during navigation', async ({ page }) => {
  const initialMemory = await page.evaluate(() => performance.memory.usedJSHeapSize)

  // Perform navigation multiple times
  for (let i = 0; i < 10; i++) {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  }

  const finalMemory = await page.evaluate(() => performance.memory.usedJSHeapSize)
  expect(finalMemory - initialMemory).toBeLessThan(1024 * 1024) // Less than 1MB increase
})
```

## Cross-Browser Testing

### Browser Matrix
- **Chromium**: Latest stable version
- **Firefox**: Latest stable version
- **WebKit**: Latest stable version (Safari)
- **Mobile Chrome**: Pixel 5 emulation
- **Mobile Safari**: iPhone 12 emulation

### Device Testing
```typescript
const devices = [
  { name: 'iPhone 12', viewport: { width: 390, height: 844 } },
  { name: 'iPad', viewport: { width: 768, height: 1024 } },
  { name: 'Desktop', viewport: { width: 1920, height: 1080 } },
]
```

## Continuous Integration

### GitHub Actions Setup
```yaml
# .github/workflows/gui-tests.yml
name: GUI Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm run test:run

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
```

## Debugging Tests

### Visual Debugging
```bash
# Run tests in headed mode to see what's happening
npm run test:e2e:headed

# Debug specific test
npm run test:e2e -- --grep "should display configuration"
```

### Accessibility Debugging
```typescript
// Add detailed logging to see accessibility violations
const results = await checkAccessibility(page)
if (results.violations.length > 0) {
  console.log('Accessibility violations:', results.violations)
}
```

### Performance Debugging
```typescript
// Measure specific operations
const startTime = performance.now()
await page.click('[data-testid="submit-button"]')
const clickTime = performance.now() - startTime

const loadTime = await page.evaluate(() => {
  return performance.getEntriesByType('navigation')[0]?.loadEventEnd || 0
})
```

## Maintenance

### Adding New Tests
1. Follow the existing naming conventions
2. Include accessibility checks
3. Add appropriate data-testid attributes
4. Update this documentation
5. Ensure CI/CD pipeline includes new tests

### Updating Dependencies
```bash
# Update testing dependencies
npm update @playwright/test @axe-core/playwright msw

# Update Storybook
npm update storybook @storybook/react-vite
```

### Test Data Management
- Keep mock data in `src/test/mocks/handlers.ts`
- Use factories for complex test data
- Maintain realistic data sizes for performance testing
- Update mock responses when APIs change

## Troubleshooting

### Common Issues

#### Playwright Browser Issues
```bash
# Reinstall browsers
npx playwright install

# Install system dependencies (Linux)
sudo apt-get install -y libgconf-2-4 libxss1 libxtst6 libxrandr2 libasound2 libpangocairo-1.0-0 libatk1.0-0 libcairo-gobject2 libgtk-3-0 libgdk-pixbuf2.0-0
```

#### MSW Issues
```typescript
// Ensure MSW is properly started
beforeAll(async () => {
  await startMSW('worker')
})

// Reset handlers between tests
afterEach(async () => {
  await resetMSW('worker')
})
```

#### Accessibility Test Failures
```typescript
// Temporarily disable specific rules for debugging
const axeBuilder = new AxeBuilder({ page })
  .withRules(['color-contrast']) // Enable only specific rule
```

## Contributing

### Guidelines
1. Write tests for new features before implementation
2. Ensure all tests pass before submitting PR
3. Include accessibility considerations
4. Add visual regression tests for UI changes
5. Update documentation for new testing utilities

### Code Review Checklist
- [ ] Tests follow naming conventions
- [ ] Accessibility checks included
- [ ] Visual regression tests added
- [ ] Performance impact considered
- [ ] Cross-browser compatibility verified
- [ ] Documentation updated

## Resources

### Official Documentation
- [Playwright](https://playwright.dev/)
- [Storybook](https://storybook.js.org/)
- [MSW](https://mswjs.io/)
- [axe-core](https://github.com/dequelabs/axe-core)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)

### Learning Resources
- [Testing JavaScript](https://testingjavascript.com/)
- [Web Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Component-Driven Development](https://www.componentdriven.org/)

---

This testing suite provides comprehensive coverage for ensuring high-quality user experience and maintainable codebase through automated testing, accessibility validation, and performance monitoring.