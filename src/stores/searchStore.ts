import { create, StateCreator } from 'zustand'
import { persistState, retrieveState } from './persistence'
import { SearchFilters, SearchQuery, SearchResult, SavedSearch, ParameterType, SearchSection, ModificationStatus, ValidationStatus } from '@/types/search'
import { useConfigStore } from './configStore'
import { useValidationStore } from './validationStore'
import { searchService } from '@/services/searchService'

export interface SearchState {
  query: string
  filters: SearchFilters
  results: SearchResult | null
  isSearching: boolean
  history: string[]
  saved: SavedSearch[]
  suggestionIndex: number

  setQuery: (q: string) => void
  setFilters: (f: Partial<SearchFilters>) => void
  runSearch: () => void
  clearResults: () => void
  nextSuggestion: () => void
  prevSuggestion: () => void
  saveCurrentSearch: (name: string) => void
  deleteSavedSearch: (id: string) => void
  applySavedSearch: (id: string) => void
}

const defaultFilters: SearchFilters = {
  scope: 'both',
  parameterTypes: null,
  validationStatus: 'any',
  modificationStatus: 'any',
  sections: null,
  regex: false,
  caseSensitive: false,
  fuzzy: true,
  searchWithin: false
}

const creator: StateCreator<SearchState> = (set, get) => ({
  query: '',
  filters: defaultFilters,
  results: null,
  isSearching: false,
  history: [],
  saved: [],
  suggestionIndex: -1,

  setQuery: (q: string) => set({ query: q }),

  setFilters: (f: Partial<SearchFilters>) => {
    const next = { ...get().filters, ...f }
    set({ filters: next })
    // Persist preferences
    try {
      persistState({ filters: next }, { name: 'search-preferences', version: 1 })
    } catch {}
  },

  runSearch: () => {
    const { query, filters } = get()
    const config = useConfigStore.getState().config
    const original = useConfigStore.getState().originalConfig
    const validation = useValidationStore.getState()

    const context = {
      getValidationForPath: (path: string) => {
        const hasErr = validation.errors.some(e => path.startsWith(e.path))
        if (hasErr) return 'error'
        const hasWarn = validation.warnings.some(w => path.startsWith(w.path))
        if (hasWarn) return 'warning'
        return 'valid'
      },
      isPathModified: (path: string) => {
        const getNested = (obj: Record<string, unknown>, p: string): unknown => {
          return p.split('.').reduce((acc: any, k: string) => (acc && acc[k] !== undefined ? acc[k] : undefined), obj as any)
        }
        const a = getNested(config as unknown as Record<string, unknown>, path)
        const b = getNested(original as unknown as Record<string, unknown>, path)
        try { return JSON.stringify(a) !== JSON.stringify(b) } catch { return a !== b }
      }
    }

    set({ isSearching: true })
    const base = filters.searchWithin && get().results ? get().results!.items : undefined
    const sr = searchService.search(config, { text: query, filters }, context, base)
    set({ results: sr, isSearching: false })

    // update history
    if (query && !get().history.includes(query)) {
      const nextHistory = [query, ...get().history].slice(0, 20)
      set({ history: nextHistory })
      try { persistState({ history: nextHistory }, { name: 'search-history', version: 1 }) } catch {}
    }
  },

  clearResults: () => set({ results: null }),

  nextSuggestion: () => set({ suggestionIndex: Math.min(get().suggestionIndex + 1, Math.max(0, get().history.length - 1)) }),
  prevSuggestion: () => set({ suggestionIndex: Math.max(get().suggestionIndex - 1, -1) }),

  saveCurrentSearch: (name: string) => {
    const id = `saved_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    const saved: SavedSearch = {
      id,
      name,
      query: { text: get().query, filters: get().filters },
      createdAt: Date.now(),
      usageCount: 0
    }
    const next = [saved, ...get().saved].slice(0, 50)
    set({ saved: next })
    try { persistState({ saved: next }, { name: 'search-saved', version: 1 }) } catch {}
  },

  deleteSavedSearch: (id: string) => {
    const next = get().saved.filter(s => s.id !== id)
    set({ saved: next })
    try { persistState({ saved: next }, { name: 'search-saved', version: 1 }) } catch {}
  },

  applySavedSearch: (id: string) => {
    const found = get().saved.find(s => s.id === id)
    if (!found) return
    set({ query: found.query.text, filters: found.query.filters })
    get().runSearch()
  }
})

export const useSearchStore = create<SearchState>(creator)

// Restore persisted preferences/history on module load
;(function restore() {
  try {
    const pref = retrieveState<{ filters: SearchFilters }>({ name: 'search-preferences', version: 1 })
    const hist = retrieveState<{ history: string[] }>({ name: 'search-history', version: 1 })
    const saved = retrieveState<{ saved: SavedSearch[] }>({ name: 'search-saved', version: 1 })
    if (pref?.filters) useSearchStore.setState({ filters: { ...defaultFilters, ...pref.filters } })
    if (hist?.history) useSearchStore.setState({ history: hist.history })
    if (saved?.saved) useSearchStore.setState({ saved: saved.saved })
  } catch {}
})()

