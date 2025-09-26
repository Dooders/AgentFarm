import { rest } from 'msw'
import { SimulationConfigType } from '@/types/config'

// Mock configuration data
export const mockConfig: SimulationConfigType = {
  width: 100,
  height: 100,
  position_discretization_method: 'floor',
  use_bilinear_interpolation: true,
  system_agents: 20,
  independent_agents: 20,
  control_agents: 10,
  agent_type_ratios: {
    SystemAgent: 0.4,
    IndependentAgent: 0.4,
    ControlAgent: 0.2,
  },
  learning_rate: 0.001,
  epsilon_start: 1.0,
  epsilon_min: 0.1,
  epsilon_decay: 0.995,
  agent_parameters: {
    SystemAgent: {
      target_update_freq: 100,
      memory_size: 1000,
      learning_rate: 0.001,
      gamma: 0.99,
      epsilon_start: 1.0,
      epsilon_min: 0.1,
      epsilon_decay: 0.995,
      dqn_hidden_size: 64,
      batch_size: 32,
      tau: 0.01,
      success_reward: 1.0,
      failure_penalty: -0.1,
      base_cost: 0.01,
    },
    IndependentAgent: {
      target_update_freq: 100,
      memory_size: 1000,
      learning_rate: 0.001,
      gamma: 0.99,
      epsilon_start: 1.0,
      epsilon_min: 0.1,
      epsilon_decay: 0.995,
      dqn_hidden_size: 64,
      batch_size: 32,
      tau: 0.01,
      success_reward: 1.0,
      failure_penalty: -0.1,
      base_cost: 0.01,
    },
    ControlAgent: {
      target_update_freq: 100,
      memory_size: 1000,
      learning_rate: 0.001,
      gamma: 0.99,
      epsilon_start: 1.0,
      epsilon_min: 0.1,
      epsilon_decay: 0.995,
      dqn_hidden_size: 64,
      batch_size: 32,
      tau: 0.01,
      success_reward: 1.0,
      failure_penalty: -0.1,
      base_cost: 0.01,
    },
  },
  visualization: {
    canvas_width: 800,
    canvas_height: 600,
    background_color: '#000000',
    agent_colors: {
      SystemAgent: '#ff6b6b',
      IndependentAgent: '#4ecdc4',
      ControlAgent: '#45b7d1',
    },
    show_metrics: true,
    font_size: 12,
    line_width: 1,
  },
  gather_parameters: {
    target_update_freq: 100,
    memory_size: 1000,
    learning_rate: 0.001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_min: 0.1,
    epsilon_decay: 0.995,
    dqn_hidden_size: 64,
    batch_size: 32,
    tau: 0.01,
    success_reward: 1.0,
    failure_penalty: -0.1,
    base_cost: 0.01,
  },
  share_parameters: {
    target_update_freq: 100,
    memory_size: 1000,
    learning_rate: 0.001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_min: 0.1,
    epsilon_decay: 0.995,
    dqn_hidden_size: 64,
    batch_size: 32,
    tau: 0.01,
    success_reward: 1.0,
    failure_penalty: -0.1,
    base_cost: 0.01,
  },
  move_parameters: {
    target_update_freq: 100,
    memory_size: 1000,
    learning_rate: 0.001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_min: 0.1,
    epsilon_decay: 0.995,
    dqn_hidden_size: 64,
    batch_size: 32,
    tau: 0.01,
    success_reward: 1.0,
    failure_penalty: -0.1,
    base_cost: 0.01,
  },
  attack_parameters: {
    target_update_freq: 100,
    memory_size: 1000,
    learning_rate: 0.001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_min: 0.1,
    epsilon_decay: 0.995,
    dqn_hidden_size: 64,
    batch_size: 32,
    tau: 0.01,
    success_reward: 1.0,
    failure_penalty: -0.1,
    base_cost: 0.01,
  },
}

// API handlers for mocking
export const handlers = [
  // Configuration endpoints
  rest.get('/api/config', (req, res, ctx) => {
    return res(ctx.json(mockConfig))
  }),

  rest.post('/api/config', (req, res, ctx) => {
    return res(ctx.json({ success: true, message: 'Configuration saved successfully' }))
  }),

  rest.put('/api/config', (req, res, ctx) => {
    return res(ctx.json({ success: true, message: 'Configuration updated successfully' }))
  }),

  rest.delete('/api/config', (req, res, ctx) => {
    return res(ctx.json({ success: true, message: 'Configuration deleted successfully' }))
  }),

  // File operations
  rest.post('/api/files/load', (req, res, ctx) => {
    return res(ctx.json({ success: true, config: mockConfig }))
  }),

  rest.post('/api/files/save', (req, res, ctx) => {
    return res(ctx.json({ success: true, message: 'File saved successfully' }))
  }),

  rest.get('/api/files/recent', (req, res, ctx) => {
    return res(ctx.json([
      { name: 'config1.json', path: '/path/to/config1.json', lastModified: Date.now() },
      { name: 'config2.yaml', path: '/path/to/config2.yaml', lastModified: Date.now() - 3600000 },
    ]))
  }),

  // Export endpoints
  rest.post('/api/export/yaml', (req, res, ctx) => {
    return res(ctx.json({ success: true, data: 'exported YAML content' }))
  }),

  rest.post('/api/export/json', (req, res, ctx) => {
    return res(ctx.json({ success: true, data: mockConfig }))
  }),

  rest.post('/api/export/toml', (req, res, ctx) => {
    return res(ctx.json({ success: true, data: 'exported TOML content' }))
  }),

  // Validation endpoints
  rest.post('/api/validate', (req, res, ctx) => {
    return res(ctx.json({
      valid: true,
      errors: [],
      warnings: [],
      summary: {
        totalFields: 42,
        validFields: 42,
        invalidFields: 0,
        warnings: 0,
      }
    }))
  }),

  // Preset endpoints
  rest.get('/api/presets', (req, res, ctx) => {
    return res(ctx.json([
      {
        id: 'preset1',
        name: 'Default Configuration',
        description: 'Standard simulation setup',
        config: mockConfig,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }
    ]))
  }),

  rest.post('/api/presets', (req, res, ctx) => {
    return res(ctx.json({ success: true, id: 'new-preset-id' }))
  }),

  // Search endpoints
  rest.get('/api/search', (req, res, ctx) => {
    const query = req.url.searchParams.get('q')
    return res(ctx.json({
      results: [
        {
          path: 'system_agents',
          value: 20,
          type: 'number',
          description: 'Number of system agents in simulation'
        }
      ],
      total: 1,
      query: query || ''
    }))
  }),

  // IPC simulation for Electron
  rest.post('/api/ipc/config/load', (req, res, ctx) => {
    return res(ctx.json({ success: true, config: mockConfig }))
  }),

  rest.post('/api/ipc/config/save', (req, res, ctx) => {
    return res(ctx.json({ success: true, message: 'Configuration saved via IPC' }))
  }),

  rest.get('/api/ipc/system/info', (req, res, ctx) => {
    return res(ctx.json({
      platform: 'test',
      version: '1.0.0',
      electronVersion: '28.0.0',
      userDataPath: '/test/path',
    }))
  }),

  // Error simulation endpoints
  rest.get('/api/error/network', (req, res, ctx) => {
    return res(ctx.status(500), ctx.json({ error: 'Network error occurred' }))
  }),

  rest.get('/api/error/validation', (req, res, ctx) => {
    return res(ctx.status(400), ctx.json({
      valid: false,
      errors: [
        { field: 'system_agents', message: 'Must be greater than 0' },
        { field: 'learning_rate', message: 'Must be between 0 and 1' }
      ]
    }))
  }),

  // Performance simulation endpoints
  rest.get('/api/performance/slow', async (req, res, ctx) => {
    // Simulate slow response
    await new Promise(resolve => setTimeout(resolve, 2000))
    return res(ctx.json({ message: 'Slow response completed' }))
  }),

  rest.get('/api/performance/large-data', (req, res, ctx) => {
    // Generate large dataset for testing
    const largeData = Array.from({ length: 1000 }, (_, i) => ({
      id: i,
      name: `Item ${i}`,
      value: Math.random(),
      data: 'x'.repeat(100), // 100 character string
    }))
    return res(ctx.json(largeData))
  }),
]