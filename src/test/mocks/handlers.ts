import { http } from 'msw'
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
  http.get('/api/config', () => {
    return Response.json(mockConfig)
  }),

  http.post('/api/config', async () => {
    return Response.json({ success: true, message: 'Configuration saved successfully' })
  }),

  http.put('/api/config', async () => {
    return Response.json({ success: true, message: 'Configuration updated successfully' })
  }),

  http.delete('/api/config', async () => {
    return Response.json({ success: true, message: 'Configuration deleted successfully' })
  }),

  // File operations
  http.post('/api/files/load', async () => {
    return Response.json({ success: true, config: mockConfig })
  }),

  http.post('/api/files/save', async () => {
    return Response.json({ success: true, message: 'File saved successfully' })
  }),

  http.get('/api/files/recent', () => {
    return Response.json([
      { name: 'config1.json', path: '/path/to/config1.json', lastModified: Date.now() },
      { name: 'config2.yaml', path: '/path/to/config2.yaml', lastModified: Date.now() - 3600000 },
    ])
  }),

  // Export endpoints
  http.post('/api/export/yaml', async () => {
    return Response.json({ success: true, data: 'exported YAML content' })
  }),

  http.post('/api/export/json', async () => {
    return Response.json({ success: true, data: mockConfig })
  }),

  http.post('/api/export/toml', async () => {
    return Response.json({ success: true, data: 'exported TOML content' })
  }),

  // Validation endpoints
  http.post('/api/validate', async () => {
    return Response.json({
      valid: true,
      errors: [],
      warnings: [],
      summary: {
        totalFields: 42,
        validFields: 42,
        invalidFields: 0,
        warnings: 0,
      }
    })
  }),

  // Preset endpoints
  http.get('/api/presets', () => {
    return Response.json([
      {
        id: 'preset1',
        name: 'Default Configuration',
        description: 'Standard simulation setup',
        config: mockConfig,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }
    ])
  }),

  http.post('/api/presets', async () => {
    return Response.json({ success: true, id: 'new-preset-id' })
  }),

  // Search endpoints
  http.get('/api/search', ({ request }) => {
    const url = new URL(request.url)
    const query = url.searchParams.get('q')
    return Response.json({
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
    })
  }),

  // IPC simulation for Electron
  http.post('/api/ipc/config/load', async () => {
    return Response.json({ success: true, config: mockConfig })
  }),

  http.post('/api/ipc/config/save', async () => {
    return Response.json({ success: true, message: 'Configuration saved via IPC' })
  }),

  http.get('/api/ipc/system/info', () => {
    return Response.json({
      platform: 'test',
      version: '1.0.0',
      electronVersion: '28.0.0',
      userDataPath: '/test/path',
    })
  }),

  // Error simulation endpoints
  http.get('/api/error/network', () => {
    return Response.json({ error: 'Network error occurred' }, { status: 500 })
  }),

  http.get('/api/error/validation', () => {
    return Response.json({
      valid: false,
      errors: [
        { field: 'system_agents', message: 'Must be greater than 0' },
        { field: 'learning_rate', message: 'Must be between 0 and 1' }
      ]
    }, { status: 400 })
  }),

  // Performance simulation endpoints
  http.get('/api/performance/slow', async () => {
    // Simulate slow response
    await new Promise(resolve => setTimeout(resolve, 2000))
    return Response.json({ message: 'Slow response completed' })
  }),

  http.get('/api/performance/large-data', () => {
    // Generate large dataset for testing
    const largeData = Array.from({ length: 1000 }, (_, i) => ({
      id: i,
      name: `Item ${i}`,
      value: Math.random(),
      data: 'x'.repeat(100), // 100 character string
    }))
    return Response.json(largeData)
  }),
]