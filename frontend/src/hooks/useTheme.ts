/**
 * Theme management hook for dark/light mode
 */

import { useState, useEffect, createContext, useContext } from 'react'

export type Theme = 'light' | 'dark' | 'system'

export interface ThemeContextType {
  theme: Theme
  actualTheme: 'light' | 'dark'
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

const THEME_STORAGE_KEY = 'swaggy-stacks-theme'

export const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

export function useThemeState(): ThemeContextType {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window === 'undefined') return 'system'
    return (localStorage.getItem(THEME_STORAGE_KEY) as Theme) || 'system'
  })

  const [actualTheme, setActualTheme] = useState<'light' | 'dark'>('dark')

  // Update actual theme based on theme preference and system preference
  useEffect(() => {
    const updateActualTheme = () => {
      if (theme === 'system') {
        const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
        setActualTheme(systemTheme)
      } else {
        setActualTheme(theme as 'light' | 'dark')
      }
    }

    updateActualTheme()

    // Listen for system theme changes
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      const handleChange = () => updateActualTheme()
      
      mediaQuery.addEventListener('change', handleChange)
      return () => mediaQuery.removeEventListener('change', handleChange)
    }
  }, [theme])

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(actualTheme)
    root.style.colorScheme = actualTheme
  }, [actualTheme])

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
    localStorage.setItem(THEME_STORAGE_KEY, newTheme)
  }

  const toggleTheme = () => {
    if (theme === 'light') {
      setTheme('dark')
    } else if (theme === 'dark') {
      setTheme('system')
    } else {
      setTheme('light')
    }
  }

  return {
    theme,
    actualTheme,
    setTheme,
    toggleTheme,
  }
}