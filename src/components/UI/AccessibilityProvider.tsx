import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'

interface AccessibilityContextType {
  highContrast: boolean
  setHighContrast: (enabled: boolean) => void
  focusElement: (selector: string) => void
  announceToScreenReader: (message: string, priority: 'polite' | 'assertive') => void
  keyboardNavigationActive: boolean
  setKeyboardNavigationActive: (active: boolean) => void
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined)

interface AccessibilityProviderProps {
  children: ReactNode
}

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({ children }) => {
  const [highContrast, setHighContrastState] = useState(false)
  const [keyboardNavigationActive, setKeyboardNavigationActive] = useState(false)

  // Load accessibility preferences from localStorage
  useEffect(() => {
    const savedHighContrast = localStorage.getItem('high-contrast') === 'true'
    setHighContrastState(savedHighContrast)

    if (savedHighContrast) {
      document.body.classList.add('high-contrast')
    }
  }, [])

  // Handle high contrast toggle
  const setHighContrast = (enabled: boolean) => {
    setHighContrastState(enabled)
    localStorage.setItem('high-contrast', enabled.toString())

    if (enabled) {
      document.body.classList.add('high-contrast')
    } else {
      document.body.classList.remove('high-contrast')
    }
  }

  // Focus management utility
  const focusElement = (selector: string) => {
    const element = document.querySelector(selector) as HTMLElement
    if (element) {
      element.focus()
    }
  }

  // Screen reader announcements
  const announceToScreenReader = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const announcement = document.createElement('div')
    announcement.setAttribute('aria-live', priority)
    announcement.setAttribute('aria-atomic', 'true')
    announcement.style.position = 'absolute'
    announcement.style.left = '-10000px'
    announcement.style.width = '1px'
    announcement.style.height = '1px'
    announcement.style.overflow = 'hidden'
    announcement.textContent = message

    document.body.appendChild(announcement)

    setTimeout(() => {
      document.body.removeChild(announcement)
    }, 1000)
  }

  // Keyboard navigation detection
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        setKeyboardNavigationActive(true)
      }
    }

    const handleMouseDown = () => {
      setKeyboardNavigationActive(false)
    }

    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('mousedown', handleMouseDown)

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('mousedown', handleMouseDown)
    }
  }, [])

  const contextValue: AccessibilityContextType = {
    highContrast,
    setHighContrast,
    focusElement,
    announceToScreenReader,
    keyboardNavigationActive,
    setKeyboardNavigationActive,
  }

  return (
    <AccessibilityContext.Provider value={contextValue}>
      {children}
    </AccessibilityContext.Provider>
  )
}

export const useAccessibility = (): AccessibilityContextType => {
  const context = useContext(AccessibilityContext)
  if (context === undefined) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider')
  }
  return context
}