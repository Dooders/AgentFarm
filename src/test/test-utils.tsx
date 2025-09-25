import React, { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'

// Custom render function that includes providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <BrowserRouter>
      {children}
    </BrowserRouter>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options })

// Re-export everything
export * from '@testing-library/react'

// Override render method
export { customRender as render }

// Create a test wrapper for Zustand stores
export const createStoreWrapper = (initialState?: any) => {
  return ({ children }: { children: React.ReactNode }) => (
    <div data-testid="store-wrapper" data-initial-state={JSON.stringify(initialState)}>
      {children}
    </div>
  )
}

// Mock data for tests
export const mockConfig = {
  width: 100,
  height: 100,
  position_discretization_method: 'floor' as const,
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