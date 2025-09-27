import { SimulationConfigType } from '@/types/config'
import { SearchQuery, SearchResult, SearchResultItem, ParameterType, SearchSection } from '@/types/search'

type FlatEntry = {
  path: string
  key: string
  value: unknown
  parameterType: ParameterType
  section: SearchSection
}

function detectType(value: unknown): ParameterType {
  if (Array.isArray(value)) return 'array'
  const t = typeof value
  if (t === 'number') return 'number'
  if (t === 'boolean') return 'boolean'
  if (t === 'string') return 'string'
  if (value !== null && t === 'object') return 'object'
  return 'string'
}

function classifySection(path: string): SearchSection {
  const head = path.split('.')[0]
  if (!head) return 'other'
  if (head === 'visualization') return 'visualization'
  if (head === 'agent_parameters') return 'agent_parameters'
  if (head === 'learning_rate' || head.startsWith('epsilon_')) return 'learning'
  if (head === 'width' || head === 'height' || head === 'position_discretization_method' || head === 'use_bilinear_interpolation') return 'environment'
  if (head === 'system_agents' || head === 'independent_agents' || head === 'control_agents' || head === 'agent_type_ratios') return 'agents'
  if (head.endsWith('_parameters')) return 'modules'
  return 'other'
}

function flattenConfig(config: SimulationConfigType): FlatEntry[] {
  const entries: FlatEntry[] = []
  const walk = (obj: unknown, prefix = ''): void => {
    if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
      const path = prefix || ''
      const key = prefix.split('.').pop() || ''
      entries.push({
        path,
        key,
        value: obj as unknown,
        parameterType: detectType(obj),
        section: classifySection(path)
      })
      return
    }
    const record = obj as Record<string, unknown>
    for (const [k, v] of Object.entries(record)) {
      const p = prefix ? `${prefix}.${k}` : k
      if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
        walk(v, p)
      } else {
        entries.push({
          path: p,
          key: k,
          value: v,
          parameterType: detectType(v),
          section: classifySection(p)
        })
      }
    }
  }
  walk(config as unknown as Record<string, unknown>)
  return entries
}

function previewValue(value: unknown, maxLen = 80): string {
  let text: string
  try {
    if (typeof value === 'string') text = value
    else text = JSON.stringify(value)
  } catch {
    text = String(value)
  }
  if (text.length > maxLen) return text.slice(0, maxLen - 1) + '…'
  return text
}

function scoreMatch(haystack: string, needle: string, fuzzy: boolean, caseSensitive: boolean): number {
  const h = caseSensitive ? haystack : haystack.toLowerCase()
  const n = caseSensitive ? needle : needle.toLowerCase()
  if (!n) return 0
  if (h === n) return 200
  if (h.startsWith(n)) return 150
  if (h.includes(n)) return 100
  if (!fuzzy) return 0
  // Simple subsequence fuzzy score
  let i = 0, j = 0, score = 0
  while (i < h.length && j < n.length) {
    if (h[i] === n[j]) { score += 2; j++ }
    i++
  }
  return j === n.length ? 50 + score : 0
}

function valueToString(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  try { return JSON.stringify(value) } catch { return String(value) }
}

export type SearchContext = {
  getValidationForPath?: (path: string) => 'error' | 'warning' | 'valid'
  isPathModified?: (path: string) => boolean
}

export const searchService = {
  search(config: SimulationConfigType, query: SearchQuery, context?: SearchContext): SearchResult {
    const start = performance.now ? performance.now() : Date.now()
    const items: SearchResultItem[] = []
    const filters = query.filters
    const regex = filters.regex && query.text ? new RegExp(query.text, filters.caseSensitive ? '' : 'i') : null

    const entries = flattenConfig(config)
    const text = query.text || ''
    for (const e of entries) {
      // Filter by section
      if (filters.sections && filters.sections.size > 0 && !filters.sections.has(e.section)) continue
      // Filter by type
      if (filters.parameterTypes && filters.parameterTypes.size > 0 && !filters.parameterTypes.has(e.parameterType)) continue

      const keyStr = e.key
      const valStr = valueToString(e.value)
      let match = false
      let score = 0

      if (!text) {
        match = true
        score = 1
      } else if (regex) {
        if (filters.scope !== 'values') {
          if (regex.test(keyStr)) { match = true; score = Math.max(score, 120) }
        }
        if (filters.scope !== 'keys') {
          if (regex.test(valStr)) { match = true; score = Math.max(score, 110) }
        }
      } else {
        if (filters.scope !== 'values') {
          const s = scoreMatch(keyStr, text, !!filters.fuzzy, !!filters.caseSensitive)
          if (s > 0) { match = true; score = Math.max(score, s) }
        }
        if (filters.scope !== 'keys') {
          const s = scoreMatch(valStr, text, !!filters.fuzzy, !!filters.caseSensitive)
          if (s > 0) { match = true; score = Math.max(score, s - 10) }
        }
      }

      if (!match) continue

      // Validation filter
      const validation = context?.getValidationForPath ? context.getValidationForPath(e.path) : 'valid'
      if (filters.validationStatus === 'error' && validation !== 'error') continue
      if (filters.validationStatus === 'warning' && validation !== 'warning') continue
      if (filters.validationStatus === 'valid' && validation !== 'valid') continue

      // Modification filter
      const modified = !!(context?.isPathModified && context.isPathModified(e.path))
      if (filters.modificationStatus === 'changed' && !modified) continue
      if (filters.modificationStatus === 'unchanged' && modified) continue

      items.push({
        path: e.path,
        key: e.key,
        valuePreview: previewValue(e.value),
        parameterType: e.parameterType,
        section: e.section,
        score,
        modified,
        validation
      })
    }

    items.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path))
    const end = performance.now ? performance.now() : Date.now()
    return { query, items, total: items.length, tookMs: Math.round((end as number) - (start as number)) }
  }
}

