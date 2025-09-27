// eslint-disable-next-line import/no-extraneous-dependencies
import { http } from 'msw'
import { beforeAll, afterEach, afterAll, expect } from 'vitest'
import { server } from './server'
import { mockConfig } from './handlers'

// Mock data generators
export const generateMockConfig = (overrides: Partial<typeof mockConfig> = {}) => ({
  ...mockConfig,
  ...overrides,
})

export const generateMockConfigs = (count: number, overrides: Partial<typeof mockConfig> = {}) => {
  return Array.from({ length: count }, (_, i) => ({
    id: `config-${i}`,
    name: `Configuration ${i + 1}`,
    ...generateMockConfig(overrides),
  }))
}

// Custom handler creators
export const createErrorHandler = (status: number, error: any) => {
  return http.get('/api/test-error', () => {
    return Response.json(error, { status })
  })
}

export const createDelayHandler = (delay: number) => {
  return http.get('/api/test-delay', async () => {
    await new Promise(resolve => setTimeout(resolve, delay))
    return Response.json({ message: 'Delayed response' })
  })
}

export const createLargeDataHandler = (itemCount: number) => {
  return http.get('/api/test-large-data', () => {
    const data = Array.from({ length: itemCount }, (_, i) => ({
      id: i,
      name: `Item ${i}`,
      value: Math.random(),
      data: 'x'.repeat(100),
      nested: {
        level1: {
          level2: {
            level3: {
              value: `deep value ${i}`,
            },
          },
        },
      },
    }))
    return Response.json(data)
  })
}

// Server control utilities
export const setupMockServer = () => {
  beforeAll(() => server.listen({ onUnhandledRequest: 'warn' }))
  afterEach(() => server.resetHandlers())
  afterAll(() => server.close())
}

export const createMockResponse = (data: any, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: () => Promise.resolve(data),
  text: () => Promise.resolve(JSON.stringify(data)),
  clone() { return { ...this } },
  headers: new Headers(),
})

export const createMockError = (message: string, status = 500) => {
  const error = new Error(message)
  ;(error as any).status = status
  return error
}

// Test data factories
export const configFactory = {
  minimal: () => ({
    width: 50,
    height: 50,
    system_agents: 5,
    independent_agents: 5,
    control_agents: 5,
  }),

  withValidationErrors: () => ({
    width: -10,
    height: 0,
    system_agents: 0,
    independent_agents: -5,
    control_agents: 1000,
    learning_rate: 2.0,
  }),

  large: () => ({
    ...mockConfig,
    system_agents: 1000,
    independent_agents: 1000,
    control_agents: 500,
  }),

  complex: () => ({
    ...mockConfig,
    agent_parameters: {
      ...mockConfig.agent_parameters,
      SystemAgent: {
        ...mockConfig.agent_parameters.SystemAgent,
        memory_size: 10000,
        dqn_hidden_size: 256,
        batch_size: 128,
      },
    },
  }),
}

// Test assertion helpers
export const expectToBeValidConfig = (config: any) => {
  expect(config).toBeDefined()
  expect(config.width).toBeGreaterThan(0)
  expect(config.height).toBeGreaterThan(0)
  expect(config.system_agents).toBeGreaterThan(0)
  expect(config.independent_agents).toBeGreaterThan(0)
  expect(config.control_agents).toBeGreaterThan(0)
  expect(config.learning_rate).toBeGreaterThan(0)
  expect(config.learning_rate).toBeLessThanOrEqual(1)
}

export const expectToHaveValidationErrors = (config: any) => {
  expect(config).toBeDefined()
  // This would be used with validation result objects
  // Implementation depends on your validation structure
}