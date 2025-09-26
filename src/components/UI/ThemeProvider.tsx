import React, { useEffect } from 'react'

interface ThemeProviderProps {
  children: React.ReactNode
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  useEffect(() => {
    const root = document.documentElement
    root.setAttribute('data-theme', 'custom')

    // Optional global grayscale toggle based on persisted preference
    try {
      const grayscalePref = localStorage.getItem('ui:grayscale')
      if (grayscalePref === 'true') {
        document.body.classList.add('grayscale')
      } else {
        document.body.classList.remove('grayscale')
      }
    } catch (error) {
      console.warn('Failed to read grayscale preference:', error)
    }
  }, [])

  return (
    <>{children}</>
  )
}

