import { describe, it, expect } from 'vitest'
import { searchService } from '@/services/searchService'
import { SimulationConfigType } from '@/types/config'

const config: SimulationConfigType = {
  width: 100,
  height: 50,
  position_discretization_method: 'floor',
  use_bilinear_interpolation: true,
  system_agents: 10,
  independent_agents: 5,
  control_agents: 2,
  agent_type_ratios: { SystemAgent: 0.6, IndependentAgent: 0.3, ControlAgent: 0.1 },
  learning_rate: 0.01,
  epsilon_start: 1,
  epsilon_min: 0.1,
  epsilon_decay: 0.99,
  agent_parameters: {
    SystemAgent: {
      target_update_freq: 100,
      memory_size: 1000,
      learning_rate: 0.01,
      gamma: 0.9,
      epsilon_start: 1,
      epsilon_min: 0.1,
      epsilon_decay: 0.99,
      dqn_hidden_size: 64,
      batch_size: 32,
      tau: 0.01,
      success_reward: 1,
      failure_penalty: -0.1,
      base_cost: 0.01
    },
    IndependentAgent: {
      target_update_freq: 100,
      memory_size: 1000,
      learning_rate: 0.01,
      gamma: 0.9,
      epsilon_start: 1,
      epsilon_min: 0.1,
      epsilon_decay: 0.99,
      dqn_hidden_size: 64,
      batch_size: 32,
      tau: 0.01,
      success_reward: 1,
      failure_penalty: -0.1,
      base_cost: 0.01
    },
    ControlAgent: {
      target_update_freq: 100,
      memory_size: 1000,
      learning_rate: 0.01,
      gamma: 0.9,
      epsilon_start: 1,
      epsilon_min: 0.1,
      epsilon_decay: 0.99,
      dqn_hidden_size: 64,
      batch_size: 32,
      tau: 0.01,
      success_reward: 1,
      failure_penalty: -0.1,
      base_cost: 0.01
    }
  },
  visualization: {
    canvas_width: 800,
    canvas_height: 600,
    background_color: '#000000',
    agent_colors: { SystemAgent: '#ff6b6b', IndependentAgent: '#4ecdc4', ControlAgent: '#45b7d1' },
    show_metrics: true,
    font_size: 12,
    line_width: 1
  },
  gather_parameters: {
    target_update_freq: 100,
    memory_size: 1000,
    learning_rate: 0.01,
    gamma: 0.9,
    epsilon_start: 1,
    epsilon_min: 0.1,
    epsilon_decay: 0.99,
    dqn_hidden_size: 64,
    batch_size: 32,
    tau: 0.01,
    success_reward: 1,
    failure_penalty: -0.1,
    base_cost: 0.01
  },
  share_parameters: {
    target_update_freq: 100,
    memory_size: 1000,
    learning_rate: 0.01,
    gamma: 0.9,
    epsilon_start: 1,
    epsilon_min: 0.1,
    epsilon_decay: 0.99,
    dqn_hidden_size: 64,
    batch_size: 32,
    tau: 0.01,
    success_reward: 1,
    failure_penalty: -0.1,
    base_cost: 0.01
  },
  move_parameters: {
    target_update_freq: 100,
    memory_size: 1000,
    learning_rate: 0.01,
    gamma: 0.9,
    epsilon_start: 1,
    epsilon_min: 0.1,
    epsilon_decay: 0.99,
    dqn_hidden_size: 64,
    batch_size: 32,
    tau: 0.01,
    success_reward: 1,
    failure_penalty: -0.1,
    base_cost: 0.01
  },
  attack_parameters: {
    target_update_freq: 100,
    memory_size: 1000,
    learning_rate: 0.01,
    gamma: 0.9,
    epsilon_start: 1,
    epsilon_min: 0.1,
    epsilon_decay: 0.99,
    dqn_hidden_size: 64,
    batch_size: 32,
    tau: 0.01,
    success_reward: 1,
    failure_penalty: -0.1,
    base_cost: 0.01
  }
}

const filters = {
  scope: 'both' as const,
  parameterTypes: null,
  validationStatus: 'any' as const,
  modificationStatus: 'any' as const,
  sections: null,
  regex: false,
  caseSensitive: false,
  fuzzy: true
}

describe('searchService', () => {
  it('finds keys and values', () => {
    const res = searchService.search(config, { text: 'width', filters })
    expect(res.items.some(i => i.path.includes('width'))).toBe(true)
  })

  it('supports numeric range queries', () => {
    const res = searchService.search(config, { text: 'width:[50..150]', filters })
    expect(res.items.some(i => i.path === 'width')).toBe(true)
  })

  it('supports boolean logic AND/OR/NOT', () => {
    const res = searchService.search(config, { text: 'width AND NOT height:200', filters })
    expect(res.items.some(i => i.path === 'width')).toBe(true)
  })
})

