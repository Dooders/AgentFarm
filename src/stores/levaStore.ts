import { create } from 'zustand'
import { SimulationConfig } from '@/types/config'
import { persistState, retrieveState } from './persistence'

export interface LevaStore {
  // Leva panel state
  isVisible: boolean
  isCollapsed: boolean
  panelPosition: 'left' | 'right'
  panelWidth: number

  // Control state
  activeControls: string[]
  disabledControls: Set<string>
  hiddenControls: Set<string>

  // Folder state
  collapsedFolders: Set<string>
  expandedFolders: Set<string>

  // Theme state
  theme: 'dark' | 'light' | 'custom'
  customTheme: Record<string, any>

  // Pending updates for config synchronization
  pendingUpdates: Array<{ path: string; value: any }>

  // Actions
  setPanelVisible: (visible: boolean) => void
  setPanelCollapsed: (collapsed: boolean) => void
  setPanelPosition: (position: 'left' | 'right') => void
  setPanelWidth: (width: number) => void

  toggleControl: (controlPath: string) => void
  setControlEnabled: (controlPath: string, enabled: boolean) => void
  setControlVisible: (controlPath: string, visible: boolean) => void

  toggleFolder: (folderPath: string) => void
  setFolderCollapsed: (folderPath: string, collapsed: boolean) => void
  expandAllFolders: () => void
  collapseAllFolders: () => void

  setTheme: (theme: 'dark' | 'light' | 'custom') => void
  setCustomTheme: (theme: Record<string, any>) => void

  // Leva integration
  syncWithConfig: (config: SimulationConfig) => void
  updateFromLeva: (path: string, value: any) => void

  // Utilities
  resetToDefaults: () => void
  exportSettings: () => Record<string, any>
  importSettings: (settings: Record<string, any>) => void

  // Selectors
  getActiveControls: () => string[]
  getVisibleControls: () => string[]
  isControlEnabled: (controlPath: string) => boolean
  isControlVisible: (controlPath: string) => boolean
  isFolderCollapsed: (folderPath: string) => boolean
  isFolderExpanded: (folderPath: string) => boolean

  // Theme methods
  getCurrentTheme: () => Record<string, any>

  // Persistence methods
  persistSettings: () => void
  restoreSettings: () => void
  clearPersistedSettings: () => void

  // Update processing
  processPendingUpdates: () => void
}

const defaultState = {
  isVisible: true,
  isCollapsed: false,
  panelPosition: 'right' as const,
  panelWidth: 300,

  activeControls: [],
  disabledControls: new Set<string>(),
  hiddenControls: new Set<string>(),

  collapsedFolders: new Set<string>(),
  expandedFolders: new Set<string>([
    'environment',
    'agents',
    'learning',
    'visualization',
    'modules'
  ]),

  theme: 'custom' as const,
  customTheme: {
    colors: {
      elevation1: '#1a1a1a',
      elevation2: '#2a2a2a',
      elevation3: '#3a3a3a',
      accent1: '#666666',
      accent2: '#888888',
      accent3: '#aaaaaa',
      highlight1: '#ffffff',
      highlight2: '#ffffff',
      highlight3: '#ffffff',
    },
    fonts: {
      mono: 'JetBrains Mono',
      sans: 'Albertus',
    },
    radii: {
      xs: '2px',
      sm: '4px',
      md: '8px',
      lg: '12px',
    },
    space: {
      xs: '4px',
      sm: '8px',
      md: '16px',
      lg: '24px',
    }
  },
  pendingUpdates: [] as Array<{ path: string; value: any }>,
}

export const useLevaStore = create<LevaStore>((set, get) => ({
  ...defaultState,

  // Panel actions
  setPanelVisible: (visible: boolean) => {
    set({ isVisible: visible })
  },

  setPanelCollapsed: (collapsed: boolean) => {
    set({ isCollapsed: collapsed })
  },

  setPanelPosition: (position: 'left' | 'right') => {
    set({ panelPosition: position })
  },

  setPanelWidth: (width: number) => {
    set({ panelWidth: Math.max(200, Math.min(800, width)) })
  },

  // Control actions
  toggleControl: (controlPath: string) => {
    const { activeControls } = get()
    const newActiveControls = activeControls.includes(controlPath)
      ? activeControls.filter(path => path !== controlPath)
      : [...activeControls, controlPath]

    set({ activeControls: newActiveControls })
  },

  setControlEnabled: (controlPath: string, enabled: boolean) => {
    set(state => {
      const newDisabledControls = new Set(state.disabledControls)
      if (enabled) {
        newDisabledControls.delete(controlPath)
      } else {
        newDisabledControls.add(controlPath)
      }
      return { disabledControls: newDisabledControls }
    })
  },

  setControlVisible: (controlPath: string, visible: boolean) => {
    set(state => {
      const newHiddenControls = new Set(state.hiddenControls)
      if (visible) {
        newHiddenControls.delete(controlPath)
      } else {
        newHiddenControls.add(controlPath)
      }
      return { hiddenControls: newHiddenControls }
    })
  },

  // Folder actions
  toggleFolder: (folderPath: string) => {
    set(state => {
      const newCollapsedFolders = new Set(state.collapsedFolders)
      const newExpandedFolders = new Set(state.expandedFolders)

      if (newCollapsedFolders.has(folderPath)) {
        newCollapsedFolders.delete(folderPath)
        newExpandedFolders.add(folderPath)
      } else {
        newCollapsedFolders.add(folderPath)
        newExpandedFolders.delete(folderPath)
      }

      return {
        collapsedFolders: newCollapsedFolders,
        expandedFolders: newExpandedFolders
      }
    })
  },

  setFolderCollapsed: (folderPath: string, collapsed: boolean) => {
    set(state => {
      const newCollapsedFolders = new Set(state.collapsedFolders)
      const newExpandedFolders = new Set(state.expandedFolders)

      if (collapsed) {
        newCollapsedFolders.add(folderPath)
        newExpandedFolders.delete(folderPath)
      } else {
        newCollapsedFolders.delete(folderPath)
        newExpandedFolders.add(folderPath)
      }

      return {
        collapsedFolders: newCollapsedFolders,
        expandedFolders: newExpandedFolders
      }
    })
  },

  expandAllFolders: () => {
    set(state => ({
      collapsedFolders: new Set(),
      expandedFolders: new Set([...state.expandedFolders, ...state.collapsedFolders])
    }))
  },

  collapseAllFolders: () => {
    set(state => ({
      collapsedFolders: new Set([...state.expandedFolders, ...state.collapsedFolders]),
      expandedFolders: new Set()
    }))
  },

  // Theme actions
  setTheme: (theme: 'dark' | 'light' | 'custom') => {
    set({ theme })
  },

  setCustomTheme: (customTheme: Record<string, any>) => {
    set({ customTheme })
  },

  // Leva integration
  syncWithConfig: (config: SimulationConfig) => {
    // Sync Leva controls with the current configuration
    const configPaths = extractConfigPaths(config)
    set({ activeControls: configPaths })

    // Update expanded folders based on config structure
    const configSections = ['environment', 'agents', 'learning', 'visualization', 'modules']
    const newExpandedFolders = new Set([
      ...get().expandedFolders,
      ...configSections
    ])

    set({ expandedFolders: newExpandedFolders })
  },

  updateFromLeva: (path: string, value: any) => {
    // This would be called when Leva controls are updated
    // and would need to update the config store
    console.log('Leva update:', path, value)

    // Note: This method is deprecated in favor of the callback approach
    // The actual integration is now handled in the LevaControls component
    // through the onChange callback

    // For backward compatibility, we can still call the config store
    // but this should be avoided in favor of the callback approach
    try {
      // We'll store the update to be processed later to avoid circular dependencies
      set(state => ({
        pendingUpdates: [...state.pendingUpdates, { path, value }]
      }))
    } catch (error) {
      console.error('Failed to queue Leva update:', error)
    }
  },

  // Add method to process pending updates
  processPendingUpdates: () => {
    set(state => {
      if (state.pendingUpdates.length === 0) return state

      // Import config store dynamically only when needed
      import('./configStore').then(({ useConfigStore }) => {
        const configStore = useConfigStore.getState()
        state.pendingUpdates.forEach(({ path, value }) => {
          try {
            configStore.updateConfig(path, value)
          } catch (error) {
            console.error(`Failed to process update for ${path}:`, error)
          }
        })
      })

      return { pendingUpdates: [] }
    })
  },

  // State for pending updates
  pendingUpdates: [] as Array<{ path: string; value: any }>,

  // Enhanced integration methods
  bindConfigValue: (path: string, value: any) => {
    const currentControls = get().activeControls
    if (!currentControls.includes(path)) {
      set({ activeControls: [...currentControls, path] })
    }
  },

  unbindConfigValue: (path: string) => {
    const currentControls = get().activeControls
    const newControls = currentControls.filter(control => control !== path)
    set({ activeControls: newControls })
  },

  // Get configuration value for a specific path
  getConfigValue: (path: string, config: SimulationConfig) => {
    return getNestedValue(config, path)
  },

  // Set configuration value for a specific path
  setConfigValue: (path: string, value: any, config: SimulationConfig) => {
    return setNestedValue(config, path, value)
  },

  // Utilities
  resetToDefaults: () => {
    set(defaultState)
  },

  exportSettings: () => {
    const state = get()
    return {
      isVisible: state.isVisible,
      isCollapsed: state.isCollapsed,
      panelPosition: state.panelPosition,
      panelWidth: state.panelWidth,
      activeControls: state.activeControls,
      disabledControls: Array.from(state.disabledControls),
      hiddenControls: Array.from(state.hiddenControls),
      collapsedFolders: Array.from(state.collapsedFolders),
      expandedFolders: Array.from(state.expandedFolders),
      theme: state.theme,
      customTheme: state.customTheme
    }
  },

  importSettings: (settings: Record<string, any>) => {
    set({
      isVisible: settings.isVisible ?? defaultState.isVisible,
      isCollapsed: settings.isCollapsed ?? defaultState.isCollapsed,
      panelPosition: settings.panelPosition ?? defaultState.panelPosition,
      panelWidth: settings.panelWidth ?? defaultState.panelWidth,
      activeControls: settings.activeControls ?? defaultState.activeControls,
      disabledControls: new Set(settings.disabledControls ?? []),
      hiddenControls: new Set(settings.hiddenControls ?? []),
      collapsedFolders: new Set(settings.collapsedFolders ?? []),
      expandedFolders: new Set(settings.expandedFolders ?? defaultState.expandedFolders),
      theme: settings.theme ?? defaultState.theme,
      customTheme: settings.customTheme ?? defaultState.customTheme
    })
  },

  // Selectors
  getActiveControls: () => {
    const { activeControls, disabledControls, hiddenControls } = get()
    return activeControls.filter(control =>
      !disabledControls.has(control) && !hiddenControls.has(control)
    )
  },

  getVisibleControls: () => {
    const { activeControls, hiddenControls } = get()
    return activeControls.filter(control => !hiddenControls.has(control))
  },

  isControlEnabled: (controlPath: string) => {
    const { disabledControls } = get()
    return !disabledControls.has(controlPath)
  },

  isControlVisible: (controlPath: string) => {
    const { hiddenControls } = get()
    return !hiddenControls.has(controlPath)
  },

  isFolderCollapsed: (folderPath: string) => {
    const { collapsedFolders } = get()
    return collapsedFolders.has(folderPath)
  },

  isFolderExpanded: (folderPath: string) => {
    const { expandedFolders } = get()
    return expandedFolders.has(folderPath)
  },

  // Theme methods
  getCurrentTheme: () => {
    const { theme, customTheme } = get()

    if (theme === 'custom') {
      return customTheme
    }

    // Return default themes for built-in options
    return {
      colors: {
        elevation1: theme === 'light' ? '#ffffff' : '#1a1a1a',
        elevation2: theme === 'light' ? '#f5f5f5' : '#2a2a2a',
        elevation3: theme === 'light' ? '#e5e5e5' : '#3a3a3a',
        accent1: '#666666',
        accent2: '#888888',
        accent3: '#aaaaaa',
        highlight1: theme === 'light' ? '#000000' : '#ffffff',
        highlight2: theme === 'light' ? '#333333' : '#ffffff',
        highlight3: theme === 'light' ? '#666666' : '#ffffff',
      },
      fonts: {
        mono: 'JetBrains Mono',
        sans: 'Albertus',
      },
      radii: {
        xs: '2px',
        sm: '4px',
        md: '8px',
        lg: '12px',
      },
      space: {
        xs: '4px',
        sm: '8px',
        md: '16px',
        lg: '24px',
      }
    }
  },

  // Persistence methods
  persistSettings: () => {
    const state = get()
    const settings = {
      isVisible: state.isVisible,
      isCollapsed: state.isCollapsed,
      panelPosition: state.panelPosition,
      panelWidth: state.panelWidth,
      theme: state.theme,
      customTheme: state.customTheme,
      expandedFolders: Array.from(state.expandedFolders),
      collapsedFolders: Array.from(state.collapsedFolders),
      activeControls: state.activeControls,
      disabledControls: Array.from(state.disabledControls),
      hiddenControls: Array.from(state.hiddenControls),
      pendingUpdates: state.pendingUpdates // Include pending updates in persistence
    }

    persistState(settings, {
      name: 'leva-settings',
      version: 1,
      onError: (error) => console.error('Failed to persist Leva settings:', error)
    })
  },

  restoreSettings: () => {
    const persistedSettings = retrieveState({
      name: 'leva-settings',
      version: 1,
      onError: (error) => console.warn('Failed to restore Leva settings:', error)
    })

    if (persistedSettings && typeof persistedSettings === 'object' && persistedSettings !== null) {
      const settings = persistedSettings as Record<string, unknown>
      set({
        isVisible: (settings.isVisible as boolean) ?? defaultState.isVisible,
        isCollapsed: (settings.isCollapsed as boolean) ?? defaultState.isCollapsed,
        panelPosition: (settings.panelPosition as 'left' | 'right') ?? defaultState.panelPosition,
        panelWidth: (settings.panelWidth as number) ?? defaultState.panelWidth,
        theme: (settings.theme as 'dark' | 'light' | 'custom') ?? defaultState.theme,
        customTheme: (settings.customTheme as Record<string, any>) ?? defaultState.customTheme,
        expandedFolders: new Set((settings.expandedFolders as string[]) ?? Array.from(defaultState.expandedFolders)),
        collapsedFolders: new Set((settings.collapsedFolders as string[]) ?? []),
        activeControls: (settings.activeControls as string[]) ?? defaultState.activeControls,
        disabledControls: new Set((settings.disabledControls as string[]) ?? []),
        hiddenControls: new Set((settings.hiddenControls as string[]) ?? []),
        pendingUpdates: (settings.pendingUpdates as Array<{ path: string; value: any }>) ?? []
      })

      // Process any pending updates after restoring
      const { processPendingUpdates } = get()
      processPendingUpdates()
    }
  },

  clearPersistedSettings: () => {
    try {
      const storage = typeof window !== 'undefined' && window.localStorage
        ? window.localStorage
        : null

      if (storage) {
        storage.removeItem('leva-settings')
      }
    } catch (error) {
      console.warn('Failed to clear persisted Leva settings:', error)
    }
  },

  // Process pending updates
  processPendingUpdates: () => {
    set(state => {
      if (state.pendingUpdates.length === 0) return state

      // Import config store dynamically only when needed
      import('./configStore').then(({ useConfigStore }) => {
        const configStore = useConfigStore.getState()
        state.pendingUpdates.forEach(({ path, value }) => {
          try {
            configStore.updateConfig(path, value)
          } catch (error) {
            console.error(`Failed to process update for ${path}:`, error)
          }
        })
      }).catch(error => {
        console.error('Failed to import config store:', error)
      })

      return { pendingUpdates: [] }
    })
  }
}))

// Helper functions for Leva integration
function extractConfigPaths(config: SimulationConfig): string[] {
  const paths: string[] = []

  function extractPaths(obj: any, prefix = ''): void {
    for (const [key, value] of Object.entries(obj)) {
      const path = prefix ? `${prefix}.${key}` : key

      if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
        extractPaths(value, path)
      } else {
        paths.push(path)
      }
    }
  }

  extractPaths(config)
  return paths
}

function getNestedValue(obj: any, path: string): any {
  const keys = path.split('.')
  let current = obj

  for (const key of keys) {
    if (current === null || current === undefined || typeof current !== 'object') {
      return undefined
    }
    current = current[key]
  }

  return current
}

function setNestedValue(obj: any, path: string, value: any): any {
  const keys = path.split('.')
  const result = { ...obj }

  let current = result
  for (let i = 0; i < keys.length - 1; i++) {
    if (!current[keys[i]] || typeof current[keys[i]] !== 'object' || Array.isArray(current[keys[i]])) {
      current[keys[i]] = {}
    } else {
      current[keys[i]] = { ...current[keys[i]] }
    }
    current = current[keys[i]]
  }

  current[keys[keys.length - 1]] = value
  return result
}