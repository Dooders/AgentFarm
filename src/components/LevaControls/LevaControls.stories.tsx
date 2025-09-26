import type { Meta, StoryObj } from '@storybook/react'
import { LevaControls } from './LevaControls'

const meta: Meta<typeof LevaControls> = {
  title: 'LevaControls/LevaControls',
  component: LevaControls,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Custom Leva controls for configuration parameters with folder structure.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    config: {
      control: { type: 'object' },
    },
  },
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => <LevaControls />,
}

export const WithCustomConfig: Story = {
  args: {
    config: {
      system_agents: 20,
      independent_agents: 20,
      control_agents: 10,
      learning_rate: 0.001,
      epsilon_start: 1.0,
      epsilon_min: 0.1,
      epsilon_decay: 0.995,
    },
  },
}

export const AgentParameters: Story = {
  render: () => (
    <LevaControls
      config={{
        agent_type_ratios: {
          SystemAgent: 0.4,
          IndependentAgent: 0.4,
          ControlAgent: 0.2,
        },
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
          },
        },
      }}
    />
  ),
}

export const VisualizationSettings: Story = {
  render: () => (
    <LevaControls
      config={{
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
      }}
    />
  ),
}