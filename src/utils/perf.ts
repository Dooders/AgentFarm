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

