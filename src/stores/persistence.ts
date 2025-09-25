/**
 * State persistence utilities for Zustand stores
 * Handles localStorage/sessionStorage with error handling and type safety
 */

export interface PersistOptions<T> {
  name: string
  storage?: 'localStorage' | 'sessionStorage'
  version?: number
  migrate?: (persistedState: any, version: number) => T
  onError?: (error: Error) => void
}

export interface PersistedState<T> {
  state: T
  version: number
  timestamp: number
}

/**
 * Creates a storage adapter for localStorage or sessionStorage
 */
function createStorage(storage: 'localStorage' | 'sessionStorage') {
  return {
    getItem: (key: string): string | null => {
      try {
        return window[storage].getItem(key)
      } catch (error) {
        console.warn(`Failed to read from ${storage}:`, error)
        return null
      }
    },

    setItem: (key: string, value: string): void => {
      try {
        window[storage].setItem(key, value)
      } catch (error) {
        console.error(`Failed to write to ${storage}:`, error)
      }
    },

    removeItem: (key: string): void => {
      try {
        window[storage].removeItem(key)
      } catch (error) {
        console.warn(`Failed to remove from ${storage}:`, error)
      }
    }
  }
}

/**
 * Persists state to storage with versioning and error handling
 */
export function persistState<T>(
  state: T,
  options: PersistOptions<T>
): void {
  const {
    name,
    storage = 'localStorage',
    version = 1,
    onError
  } = options

  try {
    const storageAdapter = createStorage(storage)
    const persistedState: PersistedState<T> = {
      state,
      version,
      timestamp: Date.now()
    }

    storageAdapter.setItem(name, JSON.stringify(persistedState))
  } catch (error) {
    const persistError = error instanceof Error ? error : new Error('Unknown persistence error')
    console.error(`Failed to persist state for ${name}:`, persistError)
    onError?.(persistError)
  }
}

/**
 * Retrieves persisted state from storage with migration support
 */
export function retrieveState<T>(
  options: PersistOptions<T>
): T | null {
  const {
    name,
    storage = 'localStorage',
    version = 1,
    migrate,
    onError
  } = options

  try {
    const storedValue = createStorage(storage).getItem(name)

    if (!storedValue) {
      return null
    }

    const parsed: PersistedState<T> = JSON.parse(storedValue)

    // Check if migration is needed
    if (parsed.version !== version && migrate) {
      const migratedState = migrate(parsed.state, parsed.version)
      persistState(migratedState, options)
      return migratedState
    }

    // Return state if versions match
    if (parsed.version === version) {
      return parsed.state
    }

    // If no migration function and versions don't match, return null
    console.warn(`Version mismatch for ${name}: stored ${parsed.version}, expected ${version}`)
    return null

  } catch (error) {
    const retrieveError = error instanceof Error ? error : new Error('Unknown retrieval error')
    console.error(`Failed to retrieve state for ${name}:`, retrieveError)
    onError?.(retrieveError)
    return null
  }
}

/**
 * Removes persisted state from storage
 */
export function clearPersistedState(name: string, storage: 'localStorage' | 'sessionStorage' = 'localStorage'): void {
  createStorage(storage).removeItem(name)
}

/**
 * Checks if storage is available and has sufficient space
 */
export function isStorageAvailable(storage: 'localStorage' | 'sessionStorage' = 'localStorage'): boolean {
  try {
    const testKey = '__storage_test__'
    window[storage].setItem(testKey, 'test')
    window[storage].removeItem(testKey)
    return true
  } catch {
    return false
  }
}

/**
 * Gets storage usage information
 */
export function getStorageUsage(storage: 'localStorage' | 'sessionStorage' = 'localStorage'): {
  used: number
  available: number
  percentage: number
} {
  try {
    const used = JSON.stringify(window[storage]).length

    // Estimate available space (rough approximation)
    const available = 5 * 1024 * 1024 // 5MB estimate
    const percentage = Math.min((used / available) * 100, 100)

    return { used, available, percentage }
  } catch {
    return { used: 0, available: 0, percentage: 0 }
  }
}