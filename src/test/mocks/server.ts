import { setupServer } from 'msw/node'
import { setupWorker } from 'msw'
import { handlers } from './handlers'

// This configures a Service Worker with the given request handlers.
export const worker = setupWorker(...handlers)

// This configures a request mocking server with the given request handlers.
export const server = setupServer(...handlers)

// Helper function to start MSW
export const startMSW = async (mode: 'worker' | 'server' = 'worker') => {
  if (mode === 'worker') {
    // Start the Service Worker for browser environment
    await worker.start({
      onUnhandledRequest: 'warn', // Warn about unhandled requests
      serviceWorker: {
        url: '/mockServiceWorker.js',
      },
    })
    return worker
  } else {
    // Start the server for Node.js environment
    server.listen({
      onUnhandledRequest: 'warn',
    })
    return server
  }
}

// Helper function to stop MSW
export const stopMSW = async (mode: 'worker' | 'server' = 'worker') => {
  if (mode === 'worker') {
    await worker.stop()
  } else {
    server.close()
  }
}

// Helper function to reset MSW handlers
export const resetMSW = async (mode: 'worker' | 'server' = 'worker') => {
  if (mode === 'worker') {
    await worker.resetHandlers()
  } else {
    server.resetHandlers()
  }
}