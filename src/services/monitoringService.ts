type Breadcrumb = { message: string; level: 'info'|'warn'|'error'; ts: number; data?: Record<string, unknown> }

class MonitoringService {
  private breadcrumbs: Breadcrumb[] = []
  private errorEndpoint?: string
  private rumEndpoint?: string

  constructor() {
    if (typeof process !== 'undefined') {
      this.errorEndpoint = process.env.VITE_ERROR_ENDPOINT
      this.rumEndpoint = process.env.VITE_RUM_ENDPOINT
    }
    if (typeof window !== 'undefined') {
      window.addEventListener('error', (e) => {
        this.captureError(e.message, { source: 'window', filename: (e as any).filename, lineno: (e as any).lineno })
      })
      window.addEventListener('unhandledrejection', (e) => {
        this.captureError('unhandledrejection', { reason: (e as any).reason })
      })
    }
  }

  setEndpoints(errorEndpoint?: string, rumEndpoint?: string) {
    this.errorEndpoint = errorEndpoint
    this.rumEndpoint = rumEndpoint
  }

  addBreadcrumb(message: string, level: Breadcrumb['level'] = 'info', data?: Record<string, unknown>) {
    const crumb: Breadcrumb = { message, level, ts: Date.now(), data }
    this.breadcrumbs.push(crumb)
    if (this.breadcrumbs.length > 100) this.breadcrumbs.shift()
  }

  captureError(message: string, data?: Record<string, unknown>) {
    this.addBreadcrumb(message, 'error', data)
    if (typeof fetch === 'undefined' || !this.errorEndpoint) return
    const payload = {
      message,
      data,
      ts: Date.now(),
      breadcrumbs: this.breadcrumbs.slice(-20)
    }
    fetch(this.errorEndpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      keepalive: true,
      body: JSON.stringify(payload)
    }).catch(() => {})
  }
}

export const monitoringService = new MonitoringService()

