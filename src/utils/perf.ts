export function time<T>(label: string, fn: () => T): T {
  const shouldLog = typeof process !== 'undefined' && (process.env?.PERF_LOG === '1' || process.env?.PERF_LOG === 'true')
  if (!shouldLog) return fn()
  const start = (globalThis.performance && performance.now) ? performance.now() : Date.now()
  try {
    return fn()
  } finally {
    const end = (globalThis.performance && performance.now) ? performance.now() : Date.now()
    // eslint-disable-next-line no-console
    console.log(`[perf] ${label}: ${Math.round((end as number) - (start as number))}ms`)
  }
}

export function mark(label: string): void {
  const shouldLog = typeof process !== 'undefined' && (process.env?.PERF_LOG === '1' || process.env?.PERF_LOG === 'true')
  if (!shouldLog) return
  const now = (globalThis.performance && performance.now) ? performance.now() : Date.now()
  // eslint-disable-next-line no-console
  console.log(`[perf] ${label}: ${Math.round(now as number)}ms`)
}

type RumEvent = {
  type: 'nav' | 'mark' | 'error'
  name: string
  value?: number
  ts: number
  meta?: Record<string, unknown>
}

let rumBuffer: RumEvent[] = []
let rumFlushTimer: number | undefined

function flushRum() {
  if (typeof navigator === 'undefined') return
  const endpoint = (globalThis as any).__RUM_ENDPOINT__ || (typeof process !== 'undefined' && process.env?.VITE_RUM_ENDPOINT)
  if (!endpoint || rumBuffer.length === 0) return
  const payload = rumBuffer.slice()
  rumBuffer = []
  fetch(endpoint as string, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    keepalive: true,
    body: JSON.stringify({ events: payload })
  }).catch(() => {})
}

function enqueueRum(event: RumEvent) {
  rumBuffer.push(event)
  if (rumBuffer.length >= 20) {
    flushRum()
    return
  }
  if (rumFlushTimer !== undefined) return
  rumFlushTimer = window.setTimeout(() => {
    rumFlushTimer = undefined
    flushRum()
  }, 5000)
}

export function rumNav(name: string, value?: number, meta?: Record<string, unknown>) {
  if (typeof window === 'undefined') return
  enqueueRum({ type: 'nav', name, value, ts: Date.now(), meta })
}

export function rumMark(name: string, value?: number, meta?: Record<string, unknown>) {
  if (typeof window === 'undefined') return
  enqueueRum({ type: 'mark', name, value, ts: Date.now(), meta })
}

export function rumError(name: string, meta?: Record<string, unknown>) {
  if (typeof window === 'undefined') return
  enqueueRum({ type: 'error', name, ts: Date.now(), meta })
}

