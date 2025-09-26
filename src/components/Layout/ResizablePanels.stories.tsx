import type { Meta, StoryObj } from '@storybook/react'
import { ResizablePanels } from './ResizablePanels'

const meta: Meta<typeof ResizablePanels> = {
  title: 'Layout/ResizablePanels',
  component: ResizablePanels,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Resizable panel component that allows users to adjust panel sizes.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    direction: {
      control: { type: 'select' },
      options: ['horizontal', 'vertical'],
    },
    defaultSizes: {
      control: { type: 'object' },
    },
    minSizes: {
      control: { type: 'object' },
    },
  },
}

export default meta
type Story = StoryObj<typeof meta>

export const HorizontalPanels: Story = {
  args: {
    direction: 'horizontal',
    defaultSizes: [30, 70],
    minSizes: [20, 40],
    children: [
      <div key="left" style={{ padding: '20px', backgroundColor: '#1a1a1a', color: 'white' }}>
        <h3>Left Panel (30%)</h3>
        <p>Navigation and controls</p>
      </div>,
      <div key="right" style={{ padding: '20px', backgroundColor: '#2a2a2a', color: 'white' }}>
        <h3>Right Panel (70%)</h3>
        <p>Main content area</p>
      </div>
    ],
  },
}

export const VerticalPanels: Story = {
  args: {
    direction: 'vertical',
    defaultSizes: [60, 40],
    minSizes: [40, 20],
    children: [
      <div key="top" style={{ padding: '20px', backgroundColor: '#1a1a1a', color: 'white' }}>
        <h3>Top Panel (60%)</h3>
        <p>Main workspace</p>
      </div>,
      <div key="bottom" style={{ padding: '20px', backgroundColor: '#2a2a2a', color: 'white' }}>
        <h3>Bottom Panel (40%)</h3>
        <p>Controls and output</p>
      </div>
    ],
  },
}

export const ThreePanels: Story = {
  args: {
    direction: 'horizontal',
    defaultSizes: [25, 50, 25],
    minSizes: [15, 30, 15],
    children: [
      <div key="left" style={{ padding: '20px', backgroundColor: '#1a1a1a', color: 'white' }}>
        <h3>Navigation (25%)</h3>
      </div>,
      <div key="center" style={{ padding: '20px', backgroundColor: '#2a2a2a', color: 'white' }}>
        <h3>Main Content (50%)</h3>
      </div>,
      <div key="right" style={{ padding: '20px', backgroundColor: '#1a1a1a', color: 'white' }}>
        <h3>Properties (25%)</h3>
      </div>
    ],
  },
}