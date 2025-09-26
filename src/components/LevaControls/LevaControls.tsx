import React, { useEffect } from 'react'
import { useLevaStore } from '@/stores/levaStore'
import { useConfigStore } from '@/stores/configStore'
import { Leva, useControls, folder } from 'leva'
import styled from 'styled-components'
import { SimulationConfig } from '@/types/config'

// Styled wrapper for the Leva panel
const LevaWrapper = styled.div`
  .leva-c {
    --leva-colors-elevation1: var(--leva-elevation1, #1a1a1a) !important;
    --leva-colors-elevation2: var(--leva-elevation2, #2a2a2a) !important;
    --leva-colors-elevation3: var(--leva-elevation3, #3a3a3a) !important;
    --leva-colors-accent1: var(--leva-accent1, #666666) !important;
    --leva-colors-accent2: var(--leva-accent2, #888888) !important;
    --leva-colors-accent3: var(--leva-accent3, #aaaaaa) !important;
    --leva-colors-highlight1: var(--leva-highlight1, #ffffff) !important;
    --leva-colors-highlight2: var(--leva-highlight2, #ffffff) !important;
    --leva-colors-highlight3: var(--leva-highlight3, #ffffff) !important;
    --leva-fonts-mono: var(--leva-font-mono, 'JetBrains Mono') !important;
    --leva-fonts-sans: var(--leva-font-sans, 'Albertus') !important;
    --leva-radii-xs: var(--leva-radii-xs, 2px) !important;
    --leva-radii-sm: var(--leva-radii-sm, 4px) !important;
    --leva-radii-md: var(--leva-radii-md, 8px) !important;
    --leva-radii-lg: var(--leva-radii-lg, 12px) !important;
    --leva-space-xs: var(--leva-space-xs, 4px) !important;
    --leva-space-sm: var(--leva-space-sm, 8px) !important;
    --leva-space-md: var(--leva-space-md, 16px) !important;
    --leva-space-lg: var(--leva-space-lg, 24px) !important;
  }
`

export const LevaControls: React.FC = () => {
  const levaStore = useLevaStore()
  const configStore = useConfigStore()

  // Initialize Leva controls based on current config
  useEffect(() => {
    levaStore.syncWithConfig(configStore.config)
  }, [configStore.config, levaStore])

  // Environment controls
  const environmentControls = useControls('Environment', {
    width: {
      value: configStore.config.width,
      min: 10,
      max: 1000,
      step: 1,
      label: 'Grid Width'
    },
    height: {
      value: configStore.config.height,
      min: 10,
      max: 1000,
      step: 1,
      label: 'Grid Height'
    },
    position_discretization_method: {
      value: configStore.config.position_discretization_method,
      options: ['floor', 'ceil', 'round'],
      label: 'Discretization Method'
    },
    use_bilinear_interpolation: {
      value: configStore.config.use_bilinear_interpolation,
      label: 'Bilinear Interpolation'
    }
  })

  // Agent controls
  const agentControls = useControls('Agents', {
    system_agents: {
      value: configStore.config.system_agents,
      min: 1,
      max: 1000,
      step: 1,
      label: 'System Agents'
    },
    independent_agents: {
      value: configStore.config.independent_agents,
      min: 1,
      max: 1000,
      step: 1,
      label: 'Independent Agents'
    },
    control_agents: {
      value: configStore.config.control_agents,
      min: 1,
      max: 1000,
      step: 1,
      label: 'Control Agents'
    }
  })

  // Learning controls
  const learningControls = useControls('Learning', {
    learning_rate: {
      value: configStore.config.learning_rate,
      min: 0.0001,
      max: 0.1,
      step: 0.0001,
      label: 'Learning Rate'
    },
    epsilon_start: {
      value: configStore.config.epsilon_start,
      min: 0.1,
      max: 1.0,
      step: 0.01,
      label: 'Epsilon Start'
    },
    epsilon_min: {
      value: configStore.config.epsilon_min,
      min: 0.01,
      max: 0.5,
      step: 0.01,
      label: 'Epsilon Min'
    },
    epsilon_decay: {
      value: configStore.config.epsilon_decay,
      min: 0.9,
      max: 0.999,
      step: 0.001,
      label: 'Epsilon Decay'
    }
  })

  // Visualization controls
  const visualizationControls = useControls('Visualization', {
    canvas_width: {
      value: configStore.config.visualization.canvas_width,
      min: 400,
      max: 1920,
      step: 10,
      label: 'Canvas Width'
    },
    canvas_height: {
      value: configStore.config.visualization.canvas_height,
      min: 300,
      max: 1080,
      step: 10,
      label: 'Canvas Height'
    },
    show_metrics: {
      value: configStore.config.visualization.show_metrics,
      label: 'Show Metrics'
    }
  })

  // Update config store when Leva controls change
  useEffect(() => {
    // Update environment config
    configStore.updateConfig('width', environmentControls.width)
    configStore.updateConfig('height', environmentControls.height)
    configStore.updateConfig('position_discretization_method', environmentControls.position_discretization_method)
    configStore.updateConfig('use_bilinear_interpolation', environmentControls.use_bilinear_interpolation)

    // Update agent config
    configStore.updateConfig('system_agents', agentControls.system_agents)
    configStore.updateConfig('independent_agents', agentControls.independent_agents)
    configStore.updateConfig('control_agents', agentControls.control_agents)

    // Update learning config
    configStore.updateConfig('learning_rate', learningControls.learning_rate)
    configStore.updateConfig('epsilon_start', learningControls.epsilon_start)
    configStore.updateConfig('epsilon_min', learningControls.epsilon_min)
    configStore.updateConfig('epsilon_decay', learningControls.epsilon_decay)

    // Update visualization config
    configStore.updateConfig('visualization.canvas_width', visualizationControls.canvas_width)
    configStore.updateConfig('visualization.canvas_height', visualizationControls.canvas_height)
    configStore.updateConfig('visualization.show_metrics', visualizationControls.show_metrics)
  }, [
    environmentControls,
    agentControls,
    learningControls,
    visualizationControls,
    configStore
  ])

  // Get current theme from Leva store
  const currentTheme = levaStore.getCurrentTheme()

  return (
    <LevaWrapper>
      <Leva
        collapsed={levaStore.isCollapsed}
        hidden={!levaStore.isVisible}
        theme={currentTheme}
      />
    </LevaWrapper>
  )
}