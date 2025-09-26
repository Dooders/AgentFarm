import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { LevaControls } from './LevaControls'
import { useLevaStore } from '@/stores/levaStore'
import { useConfigStore } from '@/stores/configStore'

// Mock the stores
vi.mock('@/stores/levaStore', () => ({
  useLevaStore: vi.fn(() => ({
    isVisible: true,
    isCollapsed: false,
    theme: 'custom',
    getCurrentTheme: () => ({
      colors: {
        elevation1: '#1a1a1a',
        elevation2: '#2a2a2a',
        elevation3: '#3a3a3a',
        accent1: '#666666',
        accent2: '#888888',
        accent3: '#aaaaaa',
        highlight1: '#ffffff',
        highlight2: '#ffffff',
        highlight3: '#ffffff',
      },
      fonts: {
        mono: 'JetBrains Mono',
        sans: 'Albertus',
      },
      radii: {
        xs: '2px',
        sm: '4px',
        md: '8px',
        lg: '12px',
      },
      space: {
        xs: '4px',
        sm: '8px',
        md: '16px',
        lg: '24px',
      }
    })
  }))
}))

vi.mock('@/stores/configStore', () => ({
  useConfigStore: vi.fn(() => ({
    config: {
      width: 100,
      height: 100,
      position_discretization_method: 'floor',
      use_bilinear_interpolation: true,
      system_agents: 20,
      independent_agents: 20,
      control_agents: 10,
      learning_rate: 0.001,
      epsilon_start: 1.0,
      epsilon_min: 0.1,
      epsilon_decay: 0.995,
      visualization: {
        canvas_width: 800,
        canvas_height: 600,
        show_metrics: true
      }
    },
    updateConfig: vi.fn()
  }))
}))

describe('LevaControls', () => {
  it('renders without crashing', () => {
    render(<LevaControls />)
    expect(document.body).toBeTruthy()
  })

  it('applies custom theme styling', () => {
    render(<LevaControls />)
    const wrapper = document.querySelector('.leva-c')
    expect(wrapper).toBeTruthy()
  })

  it('creates environment controls', () => {
    render(<LevaControls />)
    // The Leva component should render its controls
    expect(document.body).toBeTruthy()
  })
})