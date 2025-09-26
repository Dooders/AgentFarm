import { setupServer } from 'msw/node'
import { handlers } from './handlers'

// This configures a request mocking server with the given request handlers.
export const server = setupServer(...handlers)

// Helper function to start MSW
export const startMSW = async () => {
  server.listen({
    onUnhandledRequest: 'warn',
  })
  return server
}

// Helper function to stop MSW
export const stopMSW = async () => {
  server.close()
}

// Helper function to reset MSW handlers
export const resetMSW = async () => {
  server.resetHandlers()
}