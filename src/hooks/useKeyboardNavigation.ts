import { useEffect, useRef, useCallback, useState } from 'react'

interface KeyboardNavigationOptions {
  onEnter?: () => void
  onSpace?: () => void
  onEscape?: () => void
  onArrowUp?: () => void
  onArrowDown?: () => void
  onArrowLeft?: () => void
  onArrowRight?: () => void
  onTab?: () => void
  preventDefault?: boolean
}

export const useKeyboardNavigation = (options: KeyboardNavigationOptions = {}) => {
  const elementRef = useRef<HTMLElement>(null)

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    const {
      onEnter,
      onSpace,
      onEscape,
      onArrowUp,
      onArrowDown,
      onArrowLeft,
      onArrowRight,
      onTab,
      preventDefault = true
    } = options

    if (preventDefault) {
      event.preventDefault()
    }

    switch (event.key) {
      case 'Enter':
        onEnter?.()
        break
      case ' ':
        onSpace?.()
        break
      case 'Escape':
        onEscape?.()
        break
      case 'ArrowUp':
        onArrowUp?.()
        break
      case 'ArrowDown':
        onArrowDown?.()
        break
      case 'ArrowLeft':
        onArrowLeft?.()
        break
      case 'ArrowRight':
        onArrowRight?.()
        break
      case 'Tab':
        onTab?.()
        break
    }
  }, [options])

  useEffect(() => {
    const element = elementRef.current
    if (!element) return

    element.addEventListener('keydown', handleKeyDown)

    return () => {
      element.removeEventListener('keydown', handleKeyDown)
    }
  }, [handleKeyDown])

  const focusElement = useCallback(() => {
    if (elementRef.current) {
      elementRef.current.focus()
    }
  }, [])

  return {
    ref: elementRef,
    focusElement
  }
}

// Folder navigation hook for navigating through collapsible sections
export const useFolderNavigation = (folderIds: string[]) => {
  const [currentIndex, setCurrentIndex] = useState(0)

  const navigateToFolder = useCallback((index: number) => {
    if (index >= 0 && index < folderIds.length) {
      setCurrentIndex(index)
      const folderElement = document.querySelector(`[data-folder-id="${folderIds[index]}"]`)
      if (folderElement instanceof HTMLElement) {
        folderElement.focus()
      }
    }
  }, [folderIds])

  const navigateNext = useCallback(() => {
    navigateToFolder(currentIndex + 1)
  }, [currentIndex, navigateToFolder])

  const navigatePrevious = useCallback(() => {
    navigateToFolder(currentIndex - 1)
  }, [currentIndex, navigateToFolder])

  const keyboardOptions: KeyboardNavigationOptions = {
    onArrowDown: navigateNext,
    onArrowUp: navigatePrevious,
    onEnter: () => {
      const currentFolder = document.querySelector(`[data-folder-id="${folderIds[currentIndex]}"]`)
      if (currentFolder) {
        const button = currentFolder.querySelector('button')
        if (button instanceof HTMLButtonElement) {
          button.click()
        }
      }
    },
    preventDefault: true
  }

  return {
    currentIndex,
    navigateToFolder,
    navigateNext,
    navigatePrevious,
    keyboardOptions,
    folderIds
  }
}