/**
 * Focus management utilities for accessibility
 */

const focusStack: HTMLElement[] = []

/**
 * Save the current focus to the stack
 */
export const saveFocus = (): void => {
  const activeElement = document.activeElement as HTMLElement
  if (activeElement) {
    focusStack.push(activeElement)
  }
}

/**
 * Restore the last saved focus from the stack
 */
export const restoreFocus = (): void => {
  const previousElement = focusStack.pop()
  if (previousElement) {
    previousElement.focus()
  }
}

/**
 * Trap focus within a container (for modals)
 */
export const trapFocus = (container: HTMLElement): void => {
  const focusableElements = container.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )

  const firstElement = focusableElements[0] as HTMLElement
  const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement

  const handleTabKey = (e: KeyboardEvent): void => {
    if (e.key !== 'Tab') return

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault()
        lastElement.focus()
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault()
        firstElement.focus()
      }
    }
  }

  container.addEventListener('keydown', handleTabKey)

  // Store the cleanup function
  ;(container as any).__focusTrapCleanup = () => {
    container.removeEventListener('keydown', handleTabKey)
    delete (container as any).__focusTrapCleanup
  }
}

/**
 * Remove focus trap from a container
 */
export const removeFocusTrap = (container: HTMLElement): void => {
  const cleanup = (container as any).__focusTrapCleanup
  if (cleanup && typeof cleanup === 'function') {
    cleanup()
  }
}

/**
 * Focus the first focusable element in a container
 */
export const focusFirstFocusable = (container: HTMLElement): void => {
  const focusableElements = container.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )

  const firstElement = focusableElements[0] as HTMLElement
  if (firstElement) {
    firstElement.focus()
  }
}

/**
 * Focus the next focusable element after the current one
 */
export const focusNextFocusable = (currentElement: HTMLElement): void => {
  const allFocusable = Array.from(document.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  ))

  const currentIndex = allFocusable.indexOf(currentElement)
  const nextIndex = currentIndex + 1

  if (nextIndex < allFocusable.length) {
    allFocusable[nextIndex].focus()
  } else {
    allFocusable[0]?.focus()
  }
}

/**
 * Focus the previous focusable element before the current one
 */
export const focusPreviousFocusable = (currentElement: HTMLElement): void => {
  const allFocusable = Array.from(document.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  ))

  const currentIndex = allFocusable.indexOf(currentElement)
  const prevIndex = currentIndex - 1

  if (prevIndex >= 0) {
    allFocusable[prevIndex].focus()
  } else {
    allFocusable[allFocusable.length - 1]?.focus()
  }
}

/**
 * Check if an element is focusable
 */
export const isFocusable = (element: HTMLElement): boolean => {
  const focusableSelectors = [
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    'a[href]',
    '[tabindex]:not([tabindex="-1"])'
  ]

  return focusableSelectors.some(selector => element.matches(selector))
}

/**
 * Get all focusable elements within a container
 */
export const getFocusableElements = (container: HTMLElement): HTMLElement[] => {
  const focusableSelectors = [
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    'a[href]',
    '[tabindex]:not([tabindex="-1"])'
  ].join(', ')

  return Array.from(container.querySelectorAll(focusableSelectors))
}