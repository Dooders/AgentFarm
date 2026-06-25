## Monitoring & Performance

This project includes lightweight Real User Monitoring (RUM) and error reporting.

### RUM
- Functions: `rumNav`, `rumMark`, `rumError` in `src/utils/perf.ts`
- Buffer flushes automatically (20 events or 5s)
- Configure endpoint via `VITE_RUM_ENDPOINT`
- Payload shape:
```
{
  events: [
    { type: 'nav'|'mark'|'error', name: string, value?: number, ts: number, meta?: object }
  ]
}
```

### Error Reporting
- `monitoringService` in `src/services/monitoringService.ts` captures `window.error` and `unhandledrejection`
- Configure endpoint via `VITE_ERROR_ENDPOINT`
- Breadcrumbs (last 20) are included with error submissions

### Perf Logging
- Set `PERF_LOG=1` to enable console perf logs from `src/utils/perf.ts`

