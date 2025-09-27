export type SearchScope = 'keys' | 'values' | 'both'

export type ParameterType = 'number' | 'string' | 'boolean' | 'object' | 'array'

export type ValidationStatus = 'any' | 'valid' | 'error' | 'warning'

export type ModificationStatus = 'any' | 'changed' | 'unchanged'

export type SearchSection = 'environment' | 'agents' | 'learning' | 'agent_parameters' | 'visualization' | 'modules' | 'other'

export interface SearchFilters {
  scope: SearchScope
  parameterTypes: Set<ParameterType> | null
  validationStatus: ValidationStatus
  modificationStatus: ModificationStatus
  sections: Set<SearchSection> | null
  regex?: boolean
  caseSensitive?: boolean
  fuzzy?: boolean
  searchWithin?: boolean
}

export interface SearchQuery {
  text: string
  filters: SearchFilters
}

export interface SearchResultItem {
  path: string
  key: string
  valuePreview: string
  parameterType: ParameterType
  section: SearchSection
  score: number
  modified: boolean
  validation: 'error' | 'warning' | 'valid'
}

export interface SearchResult {
  query: SearchQuery
  items: SearchResultItem[]
  total: number
  tookMs: number
}

export interface SavedSearch {
  id: string
  name: string
  query: SearchQuery
  createdAt: number
  usageCount: number
}

