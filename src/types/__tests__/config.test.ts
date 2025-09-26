import { describe, it, expect } from 'vitest'
import type { SimulationConfigType, AgentParameters, ModuleParameters } from '../config'

describe('Config Types', () => {
  it('creates valid SimulationConfig object', () => {
    const config: SimulationConfigType = {
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
        ControlAgent: 0.2
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
          base_cost: 0.01
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
          base_cost: 0.01
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
          base_cost: 0.01
        }
      },
      visualization: {
        canvas_width: 800,
        canvas_height: 600,
        background_color: '#000000',
        agent_colors: {
          SystemAgent: '#ff6b6b',
          IndependentAgent: '#4ecdc4',
          ControlAgent: '#45b7d1'
        },
        show_metrics: true,
        font_size: 12,
        line_width: 1
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
        base_cost: 0.01
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
        base_cost: 0.01
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
        base_cost: 0.01
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
        base_cost: 0.01
      }
    }

    expect(config).toBeDefined()
    expect(config.width).toBe(100)
    expect(config.agent_type_ratios.SystemAgent).toBe(0.4)
    expect(config.visualization.canvas_width).toBe(800)
  })

  it('validates required AgentParameters interface', () => {
    const agentParams: AgentParameters = {
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
      base_cost: 0.01
    }

    expect(agentParams).toBeDefined()
    expect(typeof agentParams.learning_rate).toBe('number')
    expect(typeof agentParams.gamma).toBe('number')
    expect(typeof agentParams.epsilon_start).toBe('number')
  })

  it('validates required ModuleParameters interface', () => {
    const moduleParams: ModuleParameters = {
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
      base_cost: 0.01
    }

    expect(moduleParams).toBeDefined()
    expect(typeof moduleParams.success_reward).toBe('number')
    expect(typeof moduleParams.failure_penalty).toBe('number')
    expect(typeof moduleParams.base_cost).toBe('number')
  })

  it('validates enum types', () => {
    const validMethods = ['floor', 'round', 'ceil'] as const

    validMethods.forEach(method => {
      expect(method).toBeDefined()
    })

    expect(validMethods).toContain('floor')
    expect(validMethods).toContain('round')
    expect(validMethods).toContain('ceil')
  })
})