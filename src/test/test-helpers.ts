import { screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// User event setup
export const user = userEvent.setup()

// Common test data
export const testConfig = {
  width: 100,
  height: 100,
  system_agents: 20,
  independent_agents: 20,
  control_agents: 10,
  learning_rate: 0.001,
  epsilon_start: 1.0,
  epsilon_min: 0.1,
  epsilon_decay: 0.995,
}

// Panel interaction utilities
export const dragPanel = async (resizer: HTMLElement, deltaX: number, deltaY: number) => {
  fireEvent.mouseDown(resizer)
  fireEvent.mouseMove(document, { clientX: deltaX, clientY: deltaY })
  fireEvent.mouseUp(document)
}

export const resizePanel = async (panel: HTMLElement, newSize: number, direction: 'horizontal' | 'vertical' = 'horizontal') => {
  const resizer = panel.querySelector('[data-testid="panel-resizer"]') || panel.parentElement?.querySelector('.resize-handle')
  if (resizer) {
    const rect = panel.getBoundingClientRect()
    const delta = direction === 'horizontal' ? newSize - rect.width : newSize - rect.height
    await dragPanel(resizer as HTMLElement, delta, 0)
  }
}

// Configuration control utilities
export const setNumberInput = async (label: string, value: number) => {
  const input = screen.getByLabelText(label)
  await user.clear(input)
  await user.type(input, value.toString())
}

export const setBooleanControl = async (label: string, checked: boolean) => {
  const control = screen.getByLabelText(label)
  if (control) {
    if (checked !== control.checked) {
      await user.click(control)
    }
  }
}

export const setSelectControl = async (label: string, value: string) => {
  const select = screen.getByLabelText(label)
  await user.selectOptions(select, value)
}

export const setSliderControl = async (label: string, value: number) => {
  const slider = screen.getByLabelText(label)
  await user.clear(slider)
  await user.type(slider, value.toString())
  fireEvent.change(slider, { target: { value } })
}

// Form utilities
export const fillConfigForm = async (config: Partial<typeof testConfig>) => {
  if (config.width !== undefined) await setNumberInput(/width/i, config.width)
  if (config.height !== undefined) await setNumberInput(/height/i, config.height)
  if (config.system_agents !== undefined) await setNumberInput(/system agents/i, config.system_agents)
  if (config.independent_agents !== undefined) await setNumberInput(/independent agents/i, config.independent_agents)
  if (config.control_agents !== undefined) await setNumberInput(/control agents/i, config.control_agents)
  if (config.learning_rate !== undefined) await setNumberInput(/learning rate/i, config.learning_rate)
}

export const submitForm = async (buttonText: string = 'Save') => {
  const submitButton = screen.getByRole('button', { name: new RegExp(buttonText, 'i') })
  await user.click(submitButton)
}

// Navigation utilities
export const navigateToSection = async (sectionName: string) => {
  const sectionLink = screen.getByRole('link', { name: new RegExp(sectionName, 'i') }) ||
                     screen.getByText(new RegExp(sectionName, 'i'))
  await user.click(sectionLink)
}

export const expandFolder = async (folderName: string) => {
  const folder = screen.getByText(folderName).closest('[data-testid="folder"]') ||
                 screen.getByText(new RegExp(folderName, 'i'))
  const expandButton = folder?.querySelector('[data-testid="expand-button"]')
  if (expandButton && !folder?.getAttribute('data-expanded')) {
    await user.click(expandButton)
  }
}

// Validation utilities
export const expectFieldError = async (fieldLabel: string, errorMessage: string) => {
  const field = screen.getByLabelText(new RegExp(fieldLabel, 'i'))
  const errorElement = field?.parentElement?.querySelector('[data-testid="error-message"]') ||
                      document.querySelector(`[data-error-for="${field?.id}"]`)
  await waitFor(() => {
    expect(errorElement).toBeInTheDocument()
    expect(errorElement).toHaveTextContent(errorMessage)
  })
}

export const expectNoFieldErrors = async () => {
  const errorMessages = screen.queryAllByTestId('error-message')
  expect(errorMessages).toHaveLength(0)
}

// Loading state utilities
export const waitForLoadingToComplete = async (timeout = 5000) => {
  await waitFor(() => {
    expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument()
  }, { timeout })
}

export const expectLoadingState = async (elementCount: number = 1) => {
  const loadingElements = screen.getAllByTestId('loading-spinner')
  expect(loadingElements).toHaveLength(elementCount)
}

// Modal and dialog utilities
export const openModal = async (triggerText: string) => {
  const trigger = screen.getByRole('button', { name: new RegExp(triggerText, 'i') })
  await user.click(trigger)
  await waitFor(() => screen.getByRole('dialog'))
}

export const closeModal = async (closeButtonText: string = 'Close') => {
  const closeButton = screen.getByRole('button', { name: new RegExp(closeButtonText, 'i') })
  await user.click(closeButton)
  await waitFor(() => {
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
}

// Toast and notification utilities
export const expectToastMessage = async (message: string, type: 'success' | 'error' | 'warning' = 'success') => {
  await waitFor(() => {
    const toast = screen.getByTestId(`toast-${type}`)
    expect(toast).toHaveTextContent(message)
  })
}

export const dismissToast = async () => {
  const dismissButton = screen.getByTestId('toast-dismiss')
  await user.click(dismissButton)
}

// File upload utilities
export const uploadFile = async (fileInput: HTMLElement, file: File) => {
  const dataTransfer = new DataTransfer()
  dataTransfer.items.add(file)
  fireEvent.drop(fileInput, { dataTransfer })
}

// Theme utilities
export const toggleTheme = async () => {
  const themeToggle = screen.getByTestId('theme-toggle')
  await user.click(themeToggle)
}

export const expectDarkTheme = async () => {
  await waitFor(() => {
    expect(document.documentElement).toHaveClass('dark')
  })
}

export const expectLightTheme = async () => {
  await waitFor(() => {
    expect(document.documentElement).not.toHaveClass('dark')
  })
}

// Responsive utilities
export const setViewport = (width: number, height: number) => {
  Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: width })
  Object.defineProperty(window, 'innerHeight', { writable: true, configurable: true, value: height })
  window.dispatchEvent(new Event('resize'))
}

export const expectMobileLayout = async () => {
  const mobileElements = screen.getAllByTestId('mobile-only')
  expect(mobileElements.length).toBeGreaterThan(0)
}

export const expectDesktopLayout = async () => {
  const desktopElements = screen.getAllByTestId('desktop-only')
  expect(desktopElements.length).toBeGreaterThan(0)
}

// Animation and transition utilities
export const waitForAnimation = async (element: HTMLElement, timeout = 1000) => {
  await waitFor(() => {
    const computedStyle = window.getComputedStyle(element)
    return computedStyle.animationName !== 'none' || computedStyle.transitionDuration !== '0s'
  }, { timeout })
}

export const waitForAnimationEnd = async (element: HTMLElement) => {
  return new Promise(resolve => {
    const handleAnimationEnd = () => {
      element.removeEventListener('animationend', handleAnimationEnd)
      element.removeEventListener('transitionend', handleAnimationEnd)
      resolve(void 0)
    }
    element.addEventListener('animationend', handleAnimationEnd)
    element.addEventListener('transitionend', handleAnimationEnd)
  })
}

// Performance utilities
export const measurePerformance = async (operation: () => Promise<void>, threshold: number = 100) => {
  const start = performance.now()
  await operation()
  const end = performance.now()
  const duration = end - start
  expect(duration).toBeLessThan(threshold)
  return duration
}

// Accessibility utilities
export const expectAccessible = async (element: HTMLElement) => {
  expect(element).toHaveAttribute('aria-label')
  expect(element).toHaveAttribute('role')
}

export const expectKeyboardNavigable = async (container: HTMLElement) => {
  const focusableElements = container.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )
  expect(focusableElements.length).toBeGreaterThan(0)
}

// State management utilities
export const waitForStateUpdate = async (checkFn: () => boolean, timeout = 1000) => {
  await waitFor(checkFn, { timeout })
}

export const expectStateChange = async (initialValue: any, expectedValue: any) => {
  expect(initialValue).not.toEqual(expectedValue)
  await waitFor(() => expect(expectedValue).toBeDefined())
}