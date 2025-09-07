/**
 * Theme provider for managing dark/light mode throughout the application
 */

'use client'

import React, { createContext } from 'react'
import { ThemeContext, ThemeContextType, useThemeState } from '@/hooks/useTheme'

interface ThemeProviderProps {
  children: React.ReactNode
  defaultTheme?: 'light' | 'dark' | 'system'
  storageKey?: string
}

export function ThemeProvider({ 
  children, 
  defaultTheme = 'system',
  ...props 
}: ThemeProviderProps) {
  const themeState = useThemeState()

  return (
    <ThemeContext.Provider value={themeState}>
      {children}
    </ThemeContext.Provider>
  )
}