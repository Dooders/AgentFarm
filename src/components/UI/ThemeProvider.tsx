import React, { useEffect } from 'react'

interface ThemeProviderProps {
  children: React.ReactNode
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  useEffect(() => {
    const root = document.documentElement
    root.setAttribute('data-theme', 'custom')

    // Apply grayscale preference using data attribute
    try {
      const grayscalePref = localStorage.getItem('ui:grayscale')
      const enabled = grayscalePref === 'true' || grayscalePref === '1'
      document.body.setAttribute('data-mode', enabled ? 'grayscale' : 'default')
    } catch (error) {
      console.warn('Failed to read grayscale preference:', error)
    }
  }, [])

  return (
    <>{children}</>
  )
}

