import React, { useEffect, useCallback, useMemo } from 'react'
import { useLevaStore } from '@/stores/levaStore'
import { useConfigStore } from '@/stores/configStore'
import { Leva, useControls, folder } from 'leva'
import styled from 'styled-components'
import { SimulationConfig } from '@/types/config'

// Define proper interface for config store
interface ConfigStoreInterface {
  updateConfig: (path: string, value: any) => void
}

// Utility function to safely update config
const safeConfigUpdate = (
  configStore: ConfigStoreInterface,
  path: string,
  value: any,
  fallbackValue?: any
): boolean => {
  try {
    // Validate the path and value before updating
    if (!path || path.trim() === '') {
      console.warn('Invalid config path provided:', path)
      return false
    }

    // Ensure value is valid (not undefined or null for required fields)
    if (value === undefined) {
      console.warn(`Undefined value for path: ${path}`)
      if (fallbackValue !== undefined) {
        value = fallbackValue
      } else {
        return false
      }
    }

    configStore.updateConfig(path, value)
    return true
  } catch (error) {
    console.error(`Failed to update config at path "${path}":`, error)
    return false
  }
}

// Utility function to validate control configuration
const validateControlConfig = (config: any): boolean => {
  if (!config || typeof config !== 'object') {
    console.warn('Invalid control configuration provided')
    return false
  }
  return true
}

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

  // Initialize Leva controls based on current config - only on mount
  useEffect(() => {
    if (configStore && configStore.config) {
      console.log('Initializing Leva controls with config:', configStore.config)
      levaStore.syncWithConfig(configStore.config)
    } else {
      console.warn('Config store or config is not available during initialization')
    }
    // Only run once on mount
  }, [])

  // Note: Bidirectional sync is handled by the fact that useControls
  // automatically re-renders when configStore.config changes, and
  // the onChange callbacks update the config store when Leva controls change

  // Callback to handle Leva control updates with error handling
  const handleLevaUpdate = useCallback((path: string, value: any) => {
    console.log('Leva update received:', path, value)

    if (!path || typeof path !== 'string') {
      console.error('Invalid path received from Leva control:', path)
      return
    }

    // Use the safe config update utility
    const success = safeConfigUpdate(configStore, path, value)

    if (success) {
      console.log(`Successfully updated config at path: ${path}`)
    } else {
      console.warn(`Failed to update config at path: ${path}, value:`, value)
    }

    // Update the Leva store to track active controls
    levaStore.bindConfigValue(path, value)
  }, [configStore, levaStore])

  // Environment controls with validation
  const environmentConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      console.warn('Invalid environment config detected, using defaults')
      return {
        width: { value: 100, min: 10, max: 1000, step: 1, label: 'Grid Width' },
        height: { value: 100, min: 10, max: 1000, step: 1, label: 'Grid Height' },
        position_discretization_method: { value: 'floor', options: ['floor', 'ceil', 'round'], label: 'Discretization Method' },
        use_bilinear_interpolation: { value: true, label: 'Bilinear Interpolation' }
      }
    }
    return {
      width: {
        value: config.width ?? 100,
        min: 10,
        max: 1000,
        step: 1,
        label: 'Grid Width'
      },
      height: {
        value: config.height ?? 100,
        min: 10,
        max: 1000,
        step: 1,
        label: 'Grid Height'
      },
      position_discretization_method: {
        value: config.position_discretization_method ?? 'floor',
        options: ['floor', 'ceil', 'round'],
        label: 'Discretization Method'
      },
      use_bilinear_interpolation: {
        value: config.use_bilinear_interpolation ?? true,
        label: 'Bilinear Interpolation'
      }
    }
  }, [configStore?.config])

  const environmentControls = useControls('Environment', environmentConfig, { onChange: handleLevaUpdate })

  // Agent controls with validation
  const agentConfig = useMemo(() => {
    const config = configStore?.config
    if (!config) {
      return {
        system_agents: { value: 20, min: 1, max: 1000, step: 1, label: 'System Agents' },
        independent_agents: { value: 20, min: 1, max: 1000, step: 1, label: 'Independent Agents' },
        control_agents: { value: 10, min: 1, max: 1000, step: 1, label: 'Control Agents' }
      }
    }
    return {
      system_agents: {
        value: config.system_agents ?? 20,
        min: 1,
        max: 1000,
        step: 1,
        label: 'System Agents'
      },
      independent_agents: {
        value: config.independent_agents ?? 20,
        min: 1,
        max: 1000,
        step: 1,
        label: 'Independent Agents'
      },
      control_agents: {
        value: config.control_agents ?? 10,
        min: 1,
        max: 1000,
        step: 1,
        label: 'Control Agents'
      }
    }
  }, [configStore?.config])

  const agentControls = useControls('Agents', agentConfig, { onChange: handleLevaUpdate })

  // Learning controls with validation
  const learningConfig = useMemo(() => {
    const config = configStore?.config
    if (!config) {
      return {
        learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Learning Rate' },
        epsilon_start: { value: 1.0, min: 0.1, max: 1.0, step: 0.01, label: 'Epsilon Start' },
        epsilon_min: { value: 0.1, min: 0.01, max: 0.5, step: 0.01, label: 'Epsilon Min' },
        epsilon_decay: { value: 0.995, min: 0.9, max: 0.999, step: 0.001, label: 'Epsilon Decay' }
      }
    }
    return {
      learning_rate: {
        value: config.learning_rate ?? 0.001,
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        label: 'Learning Rate'
      },
      epsilon_start: {
        value: config.epsilon_start ?? 1.0,
        min: 0.1,
        max: 1.0,
        step: 0.01,
        label: 'Epsilon Start'
      },
      epsilon_min: {
        value: config.epsilon_min ?? 0.1,
        min: 0.01,
        max: 0.5,
        step: 0.01,
        label: 'Epsilon Min'
      },
      epsilon_decay: {
        value: config.epsilon_decay ?? 0.995,
        min: 0.9,
        max: 0.999,
        step: 0.001,
        label: 'Epsilon Decay'
      }
    }
  }, [configStore?.config])

  const learningControls = useControls('Learning', learningConfig, { onChange: handleLevaUpdate })

  // Visualization controls with validation
  const visualizationConfig = useMemo(() => {
    const config = configStore?.config
    if (!config) {
      return {
        canvas_width: { value: 800, min: 400, max: 1920, step: 10, label: 'Canvas Width' },
        canvas_height: { value: 600, min: 300, max: 1080, step: 10, label: 'Canvas Height' },
        show_metrics: { value: true, label: 'Show Metrics' }
      }
    }
    return {
      canvas_width: {
        value: config.visualization?.canvas_width ?? 800,
        min: 400,
        max: 1920,
        step: 10,
        label: 'Canvas Width'
      },
      canvas_height: {
        value: config.visualization?.canvas_height ?? 600,
        min: 300,
        max: 1080,
        step: 10,
        label: 'Canvas Height'
      },
      show_metrics: {
        value: config.visualization?.show_metrics ?? true,
        label: 'Show Metrics'
      }
    }
  }, [configStore?.config])

  const visualizationControls = useControls('Visualization', visualizationConfig, { onChange: handleLevaUpdate })

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