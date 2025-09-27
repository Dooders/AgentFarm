import { SimulationConfigType } from '@/types/config'
import { SearchQuery, SearchResult, SearchResultItem, ParameterType, SearchSection } from '@/types/search'
import { time } from '@/utils/perf'

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
  search(config: SimulationConfigType, query: SearchQuery, context?: SearchContext, baseItems?: SearchResultItem[]): SearchResult {
    const start = performance.now ? performance.now() : Date.now()
    const items: SearchResultItem[] = []
    const filters = query.filters
    const regex = filters.regex && query.text ? new RegExp(query.text, filters.caseSensitive ? '' : 'i') : null

    const entriesAll = getFlattened(config)
    const entries = baseItems && baseItems.length > 0
      ? entriesAll.filter(e => baseItems.some(b => b.path === e.path))
      : entriesAll

    const text = (query.text || '').trim()

    // Basic boolean logic parsing (no parentheses): NOT, AND, OR keywords, space = AND
    type Clause = { type: 'term' | 'regex' | 'range' | 'keyed'; value: string; key?: string; op?: string; min?: number; max?: number }
    type Expr = { kind: 'NOT' | 'AND' | 'OR' | 'TERM'; term?: Clause; left?: Expr; right?: Expr }

    function parseQuery(q: string): Expr | null {
      if (!q) return null
      const tokens = q.split(/\s+/).filter(Boolean)
      const terms: (Expr | 'AND' | 'OR' | 'NOT')[] = []
      const opRangeRegex = /^([^:]+):(>=|<=|>|<|=)([-+]?[0-9]*\.?[0-9]+)$/
      const intervalRangeRegex = /^([^:]+):\[\s*([-+]?[0-9]*\.?[0-9]+)\.\.([-+]?[0-9]*\.?[0-9]+)\s*\]$/
      for (const tok of tokens) {
        const upper = tok.toUpperCase()
        if (upper === 'AND' || upper === 'OR' || upper === 'NOT') { terms.push(upper as any); continue }
        let rangeMatch = tok.match(opRangeRegex)
        if (rangeMatch) {
          const [, k, op, val] = rangeMatch
          terms.push({ kind: 'TERM', term: { type: 'range', value: tok, key: k, op, min: Number(val) } })
          continue
        }
        rangeMatch = tok.match(intervalRangeRegex)
        if (rangeMatch) {
          const [, k, minS, maxS] = rangeMatch
          terms.push({ kind: 'TERM', term: { type: 'range', value: tok, key: k, min: Number(minS), max: Number(maxS) } })
          continue
        }
        // keyed term: field:value
        const keyed = tok.match(/^([^:]+):(.+)$/)
        if (keyed) {
          terms.push({ kind: 'TERM', term: { type: regex ? 'regex' : 'keyed', key: keyed[1], value: keyed[2] } })
          continue
        }
        terms.push({ kind: 'TERM', term: { type: regex ? 'regex' : 'term', value: tok } })
      }

      // Build expression with NOT having highest precedence, then AND, then OR
      function reduceNot(list: (Expr | 'AND' | 'OR' | 'NOT')[]): (Expr | 'AND' | 'OR')[] {
        const out: (Expr | 'AND' | 'OR')[] = []
        for (let i = 0; i < list.length; i++) {
          const t = list[i]
          if (t === 'NOT') {
            const next = list[i + 1]
            if (next && next !== 'AND' && next !== 'OR' && next !== 'NOT') {
              out.push({ kind: 'NOT', left: next as Expr })
              i++
            }
          } else out.push(t as any)
        }
        return out
      }
      function reduceAnd(list: (Expr | 'AND' | 'OR')[]): (Expr | 'OR')[] {
        const out: (Expr | 'OR')[] = []
        let buffer: Expr | null = null
        let expectingAnd = false
        for (const t of list) {
          if (t === 'AND') { expectingAnd = true; continue }
          if (t === 'OR') { if (buffer) out.push(buffer); out.push('OR'); buffer = null; expectingAnd = false; continue }
          const expr = t as Expr
          if (buffer && expectingAnd) {
            buffer = { kind: 'AND', left: buffer, right: expr }
            expectingAnd = false
          } else if (!buffer) {
            buffer = expr
          } else {
            // implicit AND for space-separated terms
            buffer = { kind: 'AND', left: buffer, right: expr }
          }
        }
        if (buffer) out.push(buffer)
        return out
      }
      function reduceOr(list: (Expr | 'OR')[]): Expr {
        if (list.length === 0) return { kind: 'TERM', term: { type: 'term', value: '' } }
        let buffer = list[0] as Expr
        for (let i = 1; i < list.length; i += 2) {
          const right = list[i + 1]
          if (right && right !== 'OR') {
            buffer = { kind: 'OR', left: buffer, right: right as Expr }
          }
        }
        return buffer
      }

      const step1 = reduceNot(terms)
      const step2 = reduceAnd(step1)
      return reduceOr(step2)
    }

    const expr = parseQuery(text)

    function evalClause(e: FlatEntry, c: Clause): number {
      const keyStr = e.key
      const valStr = valueToString(e.value)
      if (c.type === 'range' && c.key) {
        if (!e.path.startsWith(c.key) && e.key !== c.key) return 0
        const valNum = Number(e.value)
        if (Number.isNaN(valNum)) return 0
        if (typeof c.min === 'number' && typeof c.max === 'number') {
          return valNum >= c.min && valNum <= c.max ? 120 : 0
        }
        if (c.op && typeof c.min === 'number') {
          switch (c.op) {
            case '>': return valNum > c.min ? 120 : 0
            case '>=': return valNum >= c.min ? 120 : 0
            case '<': return valNum < c.min ? 120 : 0
            case '<=': return valNum <= c.min ? 120 : 0
            case '=': return valNum === c.min ? 140 : 0
          }
        }
        return 0
      }
      if (c.type === 'keyed' && c.key) {
        if (!e.path.startsWith(c.key) && e.key !== c.key) return 0
        return scoreMatch(valStr, c.value, !!filters.fuzzy, !!filters.caseSensitive)
      }
      if (c.type === 'regex') {
        const r = new RegExp(c.value, filters.caseSensitive ? '' : 'i')
        if (filters.scope !== 'values' && r.test(keyStr)) return 120
        if (filters.scope !== 'keys' && r.test(valStr)) return 110
        return 0
      }
      // plain term
      let s = 0
      if (!filters.regex && filters.scope !== 'values') s = Math.max(s, scoreMatch(keyStr, c.value, !!filters.fuzzy, !!filters.caseSensitive))
      if (!filters.regex && filters.scope !== 'keys') s = Math.max(s, scoreMatch(valStr, c.value, !!filters.fuzzy, !!filters.caseSensitive) - 10)
      return s
    }

    function evalExpr(e: FlatEntry, ex: Expr | null): number {
      if (!ex) return 1
      switch (ex.kind) {
        case 'TERM': return ex.term ? evalClause(e, ex.term) : 0
        case 'NOT': return ex.left ? (evalExpr(e, ex.left) > 0 ? 0 : 100) : 0
        case 'AND': return (ex.left && ex.right) ? Math.min(evalExpr(e, ex.left), evalExpr(e, ex.right)) : 0
        case 'OR': return (ex.left && ex.right) ? Math.max(evalExpr(e, ex.left), evalExpr(e, ex.right)) : 0
      }
      return 0
    }

    // Attempt query-level caching only when not scoped to baseItems (searchWithin)
    const cacheKey = baseItems && baseItems.length > 0 ? null : buildQueryKey(query)
    const cached = cacheKey ? getQueryCache(config).get(cacheKey) : null
    if (cached) {
      return cached
    }

    for (const e of entries) {
      // Filter by section
      if (filters.sections && filters.sections.size > 0 && !filters.sections.has(e.section)) continue
      // Filter by type
      if (filters.parameterTypes && filters.parameterTypes.size > 0 && !filters.parameterTypes.has(e.parameterType)) continue
      let score = 0
      if (!text) score = 1
      else if (regex) {
        const keyStr = e.key
        const valStr = valueToString(e.value)
        if (filters.scope !== 'values' && regex.test(keyStr)) score = Math.max(score, 120)
        if (filters.scope !== 'keys' && regex.test(valStr)) score = Math.max(score, 110)
      } else {
        score = evalExpr(e, expr)
      }
      if (score <= 0) continue

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
    const result: SearchResult = { query, items, total: items.length, tookMs: Math.round((end as number) - (start as number)) }
    if (cacheKey) {
      const qc = getQueryCache(config)
      setLimited(qc, cacheKey, result, 50)
    }
    return result
  }
}
// ---------------------------------------------
// Caching & Utilities
// ---------------------------------------------

const flattenCache = new WeakMap<SimulationConfigType, FlatEntry[]>()
const queryCache = new WeakMap<SimulationConfigType, Map<string, SearchResult>>()

function getFlattened(config: SimulationConfigType): FlatEntry[] {
  const hit = flattenCache.get(config)
  if (hit) return hit
  const computed = time('search.flatten', () => flattenConfig(config))
  flattenCache.set(config, computed)
  // Invalidate any stale query cache for this config reference
  if (!queryCache.has(config)) queryCache.set(config, new Map())
  return computed
}

function getQueryCache(config: SimulationConfigType): Map<string, SearchResult> {
  let m = queryCache.get(config)
  if (!m) { m = new Map(); queryCache.set(config, m) }
  return m
}

function buildQueryKey(q: SearchQuery): string {
  const f = q.filters
  const types = f.parameterTypes ? Array.from(f.parameterTypes).sort().join(',') : 'null'
  const sections = f.sections ? Array.from(f.sections).sort().join(',') : 'null'
  return [
    q.text || '',
    f.scope,
    types,
    sections,
    f.validationStatus,
    f.modificationStatus,
    f.regex ? 'r1' : 'r0',
    f.caseSensitive ? 'c1' : 'c0',
    f.fuzzy ? 'f1' : 'f0'
  ].join('|')
}

function setLimited<K, V>(m: Map<K, V>, k: K, v: V, maxSize: number): void {
  if (m.size >= maxSize) {
    const first = m.keys().next()
    if (!first.done) m.delete(first.value)
  }
  m.set(k, v)
}

