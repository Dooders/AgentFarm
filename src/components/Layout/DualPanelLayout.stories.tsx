import type { Meta, StoryObj } from '@storybook/react'
import { DualPanelLayout } from './DualPanelLayout'

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
  argTypes: {},
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <DualPanelLayout />
  ),
}

export const Alternate: Story = {
  render: () => (
    <DualPanelLayout />
  ),
}

export const WithCustomContent: Story = {
  render: () => (
    <DualPanelLayout />
  ),
}