import React from 'react'
import { useAccessibility } from './AccessibilityProvider'

interface SkipNavigationProps {
  mainContentId?: string
  validationContentId?: string
  comparisonContentId?: string
}

export const SkipNavigation: React.FC<SkipNavigationProps> = ({
  mainContentId = 'main-content',
  validationContentId = 'validation-content',
  comparisonContentId = 'comparison-content'
}) => {
  const { announceToScreenReader } = useAccessibility()

  const handleSkipToMain = () => {
    const mainElement = document.getElementById(mainContentId)
    if (mainElement) {
      mainElement.focus()
      announceToScreenReader('Skipped to main content', 'polite')
    }
  }

  const handleSkipToValidation = () => {
    const validationElement = document.getElementById(validationContentId)
    if (validationElement) {
      validationElement.focus()
      announceToScreenReader('Skipped to validation content', 'polite')
    }
  }

  const handleSkipToComparison = () => {
    const comparisonElement = document.getElementById(comparisonContentId)
    if (comparisonElement) {
      comparisonElement.focus()
      announceToScreenReader('Skipped to comparison content', 'polite')
    }
  }

  return (
    <nav className="skip-navigation" aria-label="Skip navigation">
      <a
        href={`#${mainContentId}`}
        className="skip-link"
        onClick={(e) => {
          e.preventDefault()
          handleSkipToMain()
        }}
      >
        Skip to main content
      </a>
      <a
        href={`#${validationContentId}`}
        className="skip-link"
        onClick={(e) => {
          e.preventDefault()
          handleSkipToValidation()
        }}
      >
        Skip to validation errors
      </a>
      <a
        href={`#${comparisonContentId}`}
        className="skip-link"
        onClick={(e) => {
          e.preventDefault()
          handleSkipToComparison()
        }}
      >
        Skip to comparison panel
      </a>
    </nav>
  )
}