import type { Meta, StoryObj } from '@storybook/react'
import { DualPanelLayout } from './DualPanelLayout'
import { ConfigExplorer } from '../ConfigExplorer/ConfigExplorer'

const meta: Meta<typeof DualPanelLayout> = {
  title: 'Layout/DualPanelLayout',
  component: DualPanelLayout,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'The main dual-panel layout component with resizable panels for the configuration explorer.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    defaultLayout: {
      control: { type: 'select' },
      options: ['horizontal', 'vertical'],
    },
  },
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    defaultLayout: 'horizontal',
  },
  render: (args) => (
    <DualPanelLayout {...args}>
      <ConfigExplorer />
    </DualPanelLayout>
  ),
}

export const VerticalLayout: Story = {
  args: {
    defaultLayout: 'vertical',
  },
  render: (args) => (
    <DualPanelLayout {...args}>
      <ConfigExplorer />
    </DualPanelLayout>
  ),
}

export const WithCustomContent: Story = {
  render: () => (
    <DualPanelLayout defaultLayout="horizontal">
      <div style={{ padding: '20px', backgroundColor: '#1a1a1a', color: 'white' }}>
        <h2>Left Panel</h2>
        <p>Navigation tree and controls would go here</p>
      </div>
      <div style={{ padding: '20px', backgroundColor: '#2a2a2a', color: 'white' }}>
        <h2>Right Panel</h2>
        <p>Configuration content would go here</p>
      </div>
    </DualPanelLayout>
  ),
}