import React, { useEffect, useCallback, useMemo } from 'react'
import { useLevaStore } from '@/stores/levaStore'
import { useConfigStore } from '@/stores/configStore'
import { Leva, useControls, folder } from 'leva'
import styled from 'styled-components'
// removed unused SimulationConfig import
import { useValidationStore } from '@/stores/validationStore'
import { validationService } from '@/services/validationService'

// Import enhanced custom controls (Issue #14)
import {
  LevaFolder,
  createSectionFolder,
  Vector2Input,
  ColorInput,
  FilePathInput,
  PercentageInput,
  MetadataProvider,
  ControlGroup,
  createControlGroup,
  MetadataTemplates
} from './index'

// Static path mapping - moved outside component to avoid unnecessary recreation
const PATH_MAPPING: Record<string, string> = {
  // Environment folder mappings
  'Environment/World Settings.width': 'width',
  'Environment/World Settings.height': 'height',
  'Environment/World Settings.position_discretization_method': 'position_discretization_method',
  'Environment/World Settings.use_bilinear_interpolation': 'use_bilinear_interpolation',
  'Environment/World Settings.grid_type': 'grid_type',
  'Environment/World Settings.wrap_around': 'wrap_around',

  // Population folder mappings
  'Environment/Population.system_agents': 'system_agents',
  'Environment/Population.independent_agents': 'independent_agents',
  'Environment/Population.control_agents': 'control_agents',
  'Environment/Population.agent_type_ratios.SystemAgent': 'agent_type_ratios.SystemAgent',
  'Environment/Population.agent_type_ratios.IndependentAgent': 'agent_type_ratios.IndependentAgent',
  'Environment/Population.agent_type_ratios.ControlAgent': 'agent_type_ratios.ControlAgent',

  // Resource Management folder mappings
  'Environment/Resource Management.resource_regeneration_rate': 'resource_regeneration_rate',
  'Environment/Resource Management.resource_max_level': 'resource_max_level',
  'Environment/Resource Management.resource_consumption_rate': 'resource_consumption_rate',
  'Environment/Resource Management.resource_spawn_chance': 'resource_spawn_chance',
  'Environment/Resource Management.resource_scarcity_factor': 'resource_scarcity_factor',

  // Agent Behavior folder mappings
  'Agent Behavior/Movement Parameters.move_target_update_freq': 'move_parameters.target_update_freq',
  'Agent Behavior/Movement Parameters.move_memory_size': 'move_parameters.memory_size',
  'Agent Behavior/Movement Parameters.move_learning_rate': 'move_parameters.learning_rate',
  'Agent Behavior/Movement Parameters.move_gamma': 'move_parameters.gamma',

  'Agent Behavior/Gathering Parameters.gather_target_update_freq': 'gather_parameters.target_update_freq',
  'Agent Behavior/Gathering Parameters.gather_memory_size': 'gather_parameters.memory_size',
  'Agent Behavior/Gathering Parameters.gather_learning_rate': 'gather_parameters.learning_rate',
  'Agent Behavior/Gathering Parameters.gather_success_reward': 'gather_parameters.success_reward',
  'Agent Behavior/Gathering Parameters.gather_failure_penalty': 'gather_parameters.failure_penalty',
  'Agent Behavior/Gathering Parameters.gather_base_cost': 'gather_parameters.base_cost',

  'Agent Behavior/Combat Parameters.attack_target_update_freq': 'attack_parameters.target_update_freq',
  'Agent Behavior/Combat Parameters.attack_memory_size': 'attack_parameters.memory_size',
  'Agent Behavior/Combat Parameters.attack_learning_rate': 'attack_parameters.learning_rate',
  'Agent Behavior/Combat Parameters.attack_success_reward': 'attack_parameters.success_reward',
  'Agent Behavior/Combat Parameters.attack_failure_penalty': 'attack_parameters.failure_penalty',
  'Agent Behavior/Combat Parameters.attack_base_cost': 'attack_parameters.base_cost',

  'Agent Behavior/Sharing Parameters.share_target_update_freq': 'share_parameters.target_update_freq',
  'Agent Behavior/Sharing Parameters.share_memory_size': 'share_parameters.memory_size',
  'Agent Behavior/Sharing Parameters.share_learning_rate': 'share_parameters.learning_rate',
  'Agent Behavior/Sharing Parameters.share_success_reward': 'share_parameters.success_reward',
  'Agent Behavior/Sharing Parameters.share_failure_penalty': 'share_parameters.failure_penalty',
  'Agent Behavior/Sharing Parameters.share_base_cost': 'share_parameters.base_cost',

  // Learning & AI folder mappings
  'Learning & AI/General Learning.learning_rate': 'learning_rate',
  'Learning & AI/General Learning.epsilon_start': 'epsilon_start',
  'Learning & AI/General Learning.epsilon_min': 'epsilon_min',
  'Learning & AI/General Learning.epsilon_decay': 'epsilon_decay',
  'Learning & AI/General Learning.batch_size': 'batch_size',

  // Module-specific learning mappings
  'Learning & AI/Module-Specific Learning.module_specific_learning.Movement.learning_rate': 'move_parameters.learning_rate',
  'Learning & AI/Module-Specific Learning.module_specific_learning.Movement.batch_size': 'move_parameters.batch_size',
  'Learning & AI/Module-Specific Learning.module_specific_learning.Gathering.learning_rate': 'gather_parameters.learning_rate',
  'Learning & AI/Module-Specific Learning.module_specific_learning.Gathering.batch_size': 'gather_parameters.batch_size',
  'Learning & AI/Module-Specific Learning.module_specific_learning.Combat.learning_rate': 'attack_parameters.learning_rate',
  'Learning & AI/Module-Specific Learning.module_specific_learning.Combat.batch_size': 'attack_parameters.batch_size',
  'Learning & AI/Module-Specific Learning.module_specific_learning.Sharing.learning_rate': 'share_parameters.learning_rate',
  'Learning & AI/Module-Specific Learning.module_specific_learning.Sharing.batch_size': 'share_parameters.batch_size',

  // Visualization folder mappings
  'Visualization/Display Settings.canvas_width': 'visualization.canvas_width',
  'Visualization/Display Settings.canvas_height': 'visualization.canvas_height',
  'Visualization/Display Settings.background_color': 'visualization.background_color',
  'Visualization/Display Settings.line_width': 'visualization.line_width',

  'Visualization/Animation Settings.max_frames': 'max_frames',
  'Visualization/Animation Settings.frame_delay': 'frame_delay',
  'Visualization/Animation Settings.animation_speed': 'animation_speed',
  'Visualization/Animation Settings.smooth_transitions': 'smooth_transitions',

  'Visualization/Metrics Display.show_metrics': 'visualization.show_metrics',
  'Visualization/Metrics Display.font_size': 'visualization.font_size',
  'Visualization/Metrics Display.agent_colors.SystemAgent': 'visualization.agent_colors.SystemAgent',
  'Visualization/Metrics Display.agent_colors.IndependentAgent': 'visualization.agent_colors.IndependentAgent',
  'Visualization/Metrics Display.agent_colors.ControlAgent': 'visualization.agent_colors.ControlAgent',
  'Visualization/Metrics Display.metrics_position': 'metrics_position'
}

import { MODULE_NAME_MAPPING } from '@/constants/moduleMapping'

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


  // Convert hierarchical Leva path to actual config path
  const convertLevaPathToConfigPath = useCallback((levaPath: string): string => {
    // First check if there's a direct mapping using the static PATH_MAPPING
    if (PATH_MAPPING[levaPath]) {
      return PATH_MAPPING[levaPath]
    }

    // If no direct mapping, try to extract the config path from the hierarchical path
    // This handles cases where the path structure might be different
    const pathParts = levaPath.split('.')
    if (pathParts.length >= 2) {
      // For paths like "Environment/World Settings.width", extract "width"
      // For paths like "agent_parameters.SystemAgent.target_update_freq", extract "agent_parameters.SystemAgent.target_update_freq"
      const lastPart = pathParts[pathParts.length - 1]
      const secondLastPart = pathParts[pathParts.length - 2]

      // Check if this is a nested agent parameter
      if (secondLastPart && ['SystemAgent', 'IndependentAgent', 'ControlAgent'].includes(secondLastPart)) {
        return `agent_parameters.${secondLastPart}.${lastPart}`
      }

      // Check if this is a direct agent parameter path
      if (levaPath.startsWith('agent_parameters.')) {
        return levaPath
      }

      // Check if this is a module parameter (move_parameters, gather_parameters, etc.)
      if (secondLastPart && ['move_parameters', 'gather_parameters', 'attack_parameters', 'share_parameters'].includes(secondLastPart)) {
        return `${secondLastPart}.${lastPart}`
      }

      // Check if this is a visualization parameter
      if (secondLastPart === 'visualization') {
        return `visualization.${lastPart}`
      }

      // Check if this is an agent type ratio
      if (secondLastPart === 'agent_type_ratios') {
        return `agent_type_ratios.${lastPart}`
      }

      // Check if this is a module-specific learning parameter - FIXED: use proper mapping instead of toLowerCase()
      if (levaPath.includes('module_specific_learning')) {
        const moduleIndex = pathParts.findIndex(part => part === 'module_specific_learning')
        if (moduleIndex !== -1 && pathParts.length > moduleIndex + 2) {
          const moduleName = pathParts[moduleIndex + 1]
          const paramName = pathParts[moduleIndex + 2]

          // Use the proper module name mapping instead of toLowerCase()
          let modulePrefix = MODULE_NAME_MAPPING[moduleName]
          if (!modulePrefix) {
            modulePrefix = `${moduleName.toLowerCase()}_parameters`
            console.warn(
              `[LevaControls] MODULE_NAME_MAPPING is missing an entry for module "${moduleName}". Falling back to "${modulePrefix}". This may cause inconsistent naming.`
            )
          }
          return `${modulePrefix}.${paramName}`
        }
      }
    }

    // Default: return the original path if no mapping found
    console.warn(`No mapping found for Leva path: ${levaPath}, using as-is`)
    return levaPath
  }, [])

  // Callback to handle Leva control updates with error handling
  const handleLevaUpdate = useCallback((path: string, value: any) => {
    console.log('Leva update received:', path, value)

    if (!path || typeof path !== 'string') {
      console.error('Invalid path received from Leva control:', path)
      return
    }

    // Convert the hierarchical Leva path to the actual config path
    const configPath = convertLevaPathToConfigPath(path)
    console.log(`Converted Leva path "${path}" to config path "${configPath}"`)

    // Use the safe config update utility with the actual config path
    const success = safeConfigUpdate(configStore, configPath, value)

    if (success) {
      console.log(`Successfully updated config at path: ${configPath}`)
    } else {
      console.warn(`Failed to update config at path: ${configPath}, value:`, value)
    }

    // Update the Leva store to track active controls
    levaStore.bindConfigValue(configPath, value)

    // Debounced field validation to avoid thrashing
    debounceValidate(configPath, value)
  }, [configStore, levaStore, convertLevaPathToConfigPath])

  // Simple debounce map scoped to component instance
  const debounceMap = useMemo(() => new Map<string, any>(), [])

  const debounceValidate = useCallback((path: string, value: any) => {
    const key = path
    const existing = debounceMap.get(key)
    if (existing) {
      clearTimeout(existing)
    }
    const timeout = setTimeout(() => {
      const result = validationService.validateField(path, value)
      if (result.errors.length > 0) {
        useValidationStore.getState().addErrors(result.errors)
      } else {
        useValidationStore.getState().clearFieldErrors(path)
      }
    }, 200)
    debounceMap.set(key, timeout)
  }, [debounceMap])

  // ========================================
  // ENVIRONMENT FOLDER STRUCTURE
  // ========================================

  // World Settings sub-folder
  const worldSettingsConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        width: { value: 100, min: 10, max: 1000, step: 1, label: 'Grid Width' },
        height: { value: 100, min: 10, max: 1000, step: 1, label: 'Grid Height' },
        position_discretization_method: { value: 'floor', options: ['floor', 'ceil', 'round'], label: 'Discretization Method' },
        use_bilinear_interpolation: { value: true, label: 'Bilinear Interpolation' },
        grid_type: { value: 'hex', options: ['square', 'hex', 'triangle'], label: 'Grid Type' },
        wrap_around: { value: true, label: 'Wrap Around Edges' }
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
      },
      grid_type: {
        value: config.grid_type ?? 'hex',
        options: ['square', 'hex', 'triangle'],
        label: 'Grid Type'
      },
      wrap_around: {
        value: config.wrap_around ?? true,
        label: 'Wrap Around Edges'
      }
    }
  }, [configStore?.config])

  const worldSettingsControls = useControls('Environment/World Settings', worldSettingsConfig, { onChange: handleLevaUpdate })

  // Population sub-folder
  const populationConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        system_agents: { value: 20, min: 0, max: 10000, step: 1, label: 'System Agents' },
        independent_agents: { value: 20, min: 0, max: 10000, step: 1, label: 'Independent Agents' },
        control_agents: { value: 10, min: 0, max: 10000, step: 1, label: 'Control Agents' },
        agent_type_ratios: folder({
          SystemAgent: { value: 0.4, min: 0, max: 1, step: 0.01, label: 'System Agent Ratio' },
          IndependentAgent: { value: 0.4, min: 0, max: 1, step: 0.01, label: 'Independent Agent Ratio' },
          ControlAgent: { value: 0.2, min: 0, max: 1, step: 0.01, label: 'Control Agent Ratio' }
        }, { collapsed: false })
      }
    }
    return {
      system_agents: {
        value: config.system_agents ?? 20,
        min: 0,
        max: 10000,
        step: 1,
        label: 'System Agents'
      },
      independent_agents: {
        value: config.independent_agents ?? 20,
        min: 0,
        max: 10000,
        step: 1,
        label: 'Independent Agents'
      },
      control_agents: {
        value: config.control_agents ?? 10,
        min: 0,
        max: 10000,
        step: 1,
        label: 'Control Agents'
      },
      agent_type_ratios: folder({
        SystemAgent: {
          value: config.agent_type_ratios?.SystemAgent ?? 0.4,
          min: 0,
          max: 1,
          step: 0.01,
          label: 'System Agent Ratio'
        },
        IndependentAgent: {
          value: config.agent_type_ratios?.IndependentAgent ?? 0.4,
          min: 0,
          max: 1,
          step: 0.01,
          label: 'Independent Agent Ratio'
        },
        ControlAgent: {
          value: config.agent_type_ratios?.ControlAgent ?? 0.2,
          min: 0,
          max: 1,
          step: 0.01,
          label: 'Control Agent Ratio'
        }
      }, { collapsed: false })
    }
  }, [configStore?.config])

  const populationControls = useControls('Environment/Population', populationConfig, { onChange: handleLevaUpdate })

  // Resource Management sub-folder
  const resourceConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        resource_regeneration_rate: { value: 0.1, min: 0, max: 1, step: 0.01, label: 'Regeneration Rate' },
        resource_max_level: { value: 100, min: 1, max: 1000, step: 1, label: 'Max Resource Level' },
        resource_consumption_rate: { value: 0.5, min: 0, max: 10, step: 0.1, label: 'Consumption Rate' },
        resource_spawn_chance: { value: 0.05, min: 0, max: 1, step: 0.01, label: 'Spawn Chance' },
        resource_scarcity_factor: { value: 1.0, min: 0.1, max: 5.0, step: 0.1, label: 'Scarcity Factor' }
      }
    }
    return {
      resource_regeneration_rate: {
        value: config.resource_regeneration_rate ?? 0.1,
        min: 0,
        max: 1,
        step: 0.01,
        label: 'Regeneration Rate'
      },
      resource_max_level: {
        value: config.resource_max_level ?? 100,
        min: 1,
        max: 1000,
        step: 1,
        label: 'Max Resource Level'
      },
      resource_consumption_rate: {
        value: config.resource_consumption_rate ?? 0.5,
        min: 0,
        max: 10,
        step: 0.1,
        label: 'Consumption Rate'
      },
      resource_spawn_chance: {
        value: config.resource_spawn_chance ?? 0.05,
        min: 0,
        max: 1,
        step: 0.01,
        label: 'Spawn Chance'
      },
      resource_scarcity_factor: {
        value: config.resource_scarcity_factor ?? 1.0,
        min: 0.1,
        max: 5.0,
        step: 0.1,
        label: 'Scarcity Factor'
      }
    }
  }, [configStore?.config])

  const resourceControls = useControls('Environment/Resource Management', resourceConfig, { onChange: handleLevaUpdate })

  // ========================================
  // AGENT BEHAVIOR FOLDER STRUCTURE
  // ========================================

  // Movement Parameters sub-folder
  const movementConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        move_target_update_freq: { value: 10, min: 1, max: 100, step: 1, label: 'Target Update Frequency' },
        move_memory_size: { value: 10000, min: 1000, max: 100000, step: 100, label: 'Memory Size' },
        move_learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Learning Rate' },
        move_gamma: { value: 0.99, min: 0, max: 1, step: 0.01, label: 'Discount Factor' }
      }
    }
    return {
      move_target_update_freq: {
        value: config.move_parameters?.target_update_freq ?? 10,
        min: 1,
        max: 100,
        step: 1,
        label: 'Target Update Frequency'
      },
      move_memory_size: {
        value: config.move_parameters?.memory_size ?? 10000,
        min: 1000,
        max: 100000,
        step: 100,
        label: 'Memory Size'
      },
      move_learning_rate: {
        value: config.move_parameters?.learning_rate ?? 0.001,
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        label: 'Learning Rate'
      },
      move_gamma: {
        value: config.move_parameters?.gamma ?? 0.99,
        min: 0,
        max: 1,
        step: 0.01,
        label: 'Discount Factor'
      }
    }
  }, [configStore?.config])

  const movementControls = useControls('Agent Behavior/Movement Parameters', movementConfig, { onChange: handleLevaUpdate })

  // Gathering Parameters sub-folder
  const gatheringConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        gather_target_update_freq: { value: 5, min: 1, max: 50, step: 1, label: 'Target Update Frequency' },
        gather_memory_size: { value: 5000, min: 500, max: 50000, step: 50, label: 'Memory Size' },
        gather_learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Learning Rate' },
        gather_success_reward: { value: 1.0, min: 0, max: 10, step: 0.1, label: 'Success Reward' },
        gather_failure_penalty: { value: -0.1, min: -1, max: 0, step: 0.01, label: 'Failure Penalty' },
        gather_base_cost: { value: 0.1, min: 0, max: 1, step: 0.01, label: 'Base Cost' }
      }
    }
    return {
      gather_target_update_freq: {
        value: config.gather_parameters?.target_update_freq ?? 5,
        min: 1,
        max: 50,
        step: 1,
        label: 'Target Update Frequency'
      },
      gather_memory_size: {
        value: config.gather_parameters?.memory_size ?? 5000,
        min: 500,
        max: 50000,
        step: 50,
        label: 'Memory Size'
      },
      gather_learning_rate: {
        value: config.gather_parameters?.learning_rate ?? 0.001,
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        label: 'Learning Rate'
      },
      gather_success_reward: {
        value: config.gather_parameters?.success_reward ?? 1.0,
        min: 0,
        max: 10,
        step: 0.1,
        label: 'Success Reward'
      },
      gather_failure_penalty: {
        value: config.gather_parameters?.failure_penalty ?? -0.1,
        min: -1,
        max: 0,
        step: 0.01,
        label: 'Failure Penalty'
      },
      gather_base_cost: {
        value: config.gather_parameters?.base_cost ?? 0.1,
        min: 0,
        max: 1,
        step: 0.01,
        label: 'Base Cost'
      }
    }
  }, [configStore?.config])

  const gatheringControls = useControls('Agent Behavior/Gathering Parameters', gatheringConfig, { onChange: handleLevaUpdate })

  // Combat Parameters sub-folder
  const combatConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        attack_target_update_freq: { value: 15, min: 1, max: 100, step: 1, label: 'Target Update Frequency' },
        attack_memory_size: { value: 8000, min: 1000, max: 50000, step: 100, label: 'Memory Size' },
        attack_learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Learning Rate' },
        attack_success_reward: { value: 2.0, min: 0, max: 20, step: 0.1, label: 'Success Reward' },
        attack_failure_penalty: { value: -0.5, min: -5, max: 0, step: 0.1, label: 'Failure Penalty' },
        attack_base_cost: { value: 0.2, min: 0, max: 2, step: 0.01, label: 'Base Cost' }
      }
    }
    return {
      attack_target_update_freq: {
        value: config.attack_parameters?.target_update_freq ?? 15,
        min: 1,
        max: 100,
        step: 1,
        label: 'Target Update Frequency'
      },
      attack_memory_size: {
        value: config.attack_parameters?.memory_size ?? 8000,
        min: 1000,
        max: 50000,
        step: 100,
        label: 'Memory Size'
      },
      attack_learning_rate: {
        value: config.attack_parameters?.learning_rate ?? 0.001,
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        label: 'Learning Rate'
      },
      attack_success_reward: {
        value: config.attack_parameters?.success_reward ?? 2.0,
        min: 0,
        max: 20,
        step: 0.1,
        label: 'Success Reward'
      },
      attack_failure_penalty: {
        value: config.attack_parameters?.failure_penalty ?? -0.5,
        min: -5,
        max: 0,
        step: 0.1,
        label: 'Failure Penalty'
      },
      attack_base_cost: {
        value: config.attack_parameters?.base_cost ?? 0.2,
        min: 0,
        max: 2,
        step: 0.01,
        label: 'Base Cost'
      }
    }
  }, [configStore?.config])

  const combatControls = useControls('Agent Behavior/Combat Parameters', combatConfig, { onChange: handleLevaUpdate })

  // Sharing Parameters sub-folder
  const sharingConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        share_target_update_freq: { value: 8, min: 1, max: 50, step: 1, label: 'Target Update Frequency' },
        share_memory_size: { value: 6000, min: 500, max: 30000, step: 50, label: 'Memory Size' },
        share_learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Learning Rate' },
        share_success_reward: { value: 1.5, min: 0, max: 15, step: 0.1, label: 'Success Reward' },
        share_failure_penalty: { value: -0.2, min: -2, max: 0, step: 0.01, label: 'Failure Penalty' },
        share_base_cost: { value: 0.15, min: 0, max: 1.5, step: 0.01, label: 'Base Cost' }
      }
    }
    return {
      share_target_update_freq: {
        value: config.share_parameters?.target_update_freq ?? 8,
        min: 1,
        max: 50,
        step: 1,
        label: 'Target Update Frequency'
      },
      share_memory_size: {
        value: config.share_parameters?.memory_size ?? 6000,
        min: 500,
        max: 30000,
        step: 50,
        label: 'Memory Size'
      },
      share_learning_rate: {
        value: config.share_parameters?.learning_rate ?? 0.001,
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        label: 'Learning Rate'
      },
      share_success_reward: {
        value: config.share_parameters?.success_reward ?? 1.5,
        min: 0,
        max: 15,
        step: 0.1,
        label: 'Success Reward'
      },
      share_failure_penalty: {
        value: config.share_parameters?.failure_penalty ?? -0.2,
        min: -2,
        max: 0,
        step: 0.01,
        label: 'Failure Penalty'
      },
      share_base_cost: {
        value: config.share_parameters?.base_cost ?? 0.15,
        min: 0,
        max: 1.5,
        step: 0.01,
        label: 'Base Cost'
      }
    }
  }, [configStore?.config])

  const sharingControls = useControls('Agent Behavior/Sharing Parameters', sharingConfig, { onChange: handleLevaUpdate })

  // ========================================
  // LEARNING & AI FOLDER STRUCTURE
  // ========================================

  // General Learning sub-folder
  const generalLearningConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        learning_rate: { value: 0.001, min: 0.0001, max: 1.0, step: 0.0001, label: 'Global Learning Rate' },
        epsilon_start: { value: 1.0, min: 0.1, max: 1.0, step: 0.01, label: 'Epsilon Start' },
        epsilon_min: { value: 0.1, min: 0.01, max: 0.5, step: 0.01, label: 'Epsilon Min' },
        epsilon_decay: { value: 0.995, min: 0.9, max: 0.999, step: 0.001, label: 'Epsilon Decay' },
        batch_size: { value: 32, min: 16, max: 512, step: 1, label: 'Global Batch Size' }
      }
    }
    return {
      learning_rate: {
        value: config.learning_rate ?? 0.001,
        min: 0.0001,
        max: 1.0,
        step: 0.0001,
        label: 'Global Learning Rate'
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
      },
      batch_size: {
        value: config.batch_size ?? 32,
        min: 16,
        max: 512,
        step: 1,
        label: 'Global Batch Size'
      }
    }
  }, [configStore?.config])

  const generalLearningControls = useControls('Learning & AI/General Learning', generalLearningConfig, { onChange: handleLevaUpdate })

  // Module-Specific Learning sub-folder
  const moduleLearningConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        module_specific_learning: folder({
          Movement: folder({
            learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Movement Learning Rate' },
            batch_size: { value: 32, min: 16, max: 512, step: 1, label: 'Movement Batch Size' }
          }, { collapsed: false }),
          Gathering: folder({
            learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Gathering Learning Rate' },
            batch_size: { value: 32, min: 16, max: 512, step: 1, label: 'Gathering Batch Size' }
          }, { collapsed: false }),
          Combat: folder({
            learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Combat Learning Rate' },
            batch_size: { value: 32, min: 16, max: 512, step: 1, label: 'Combat Batch Size' }
          }, { collapsed: false }),
          Sharing: folder({
            learning_rate: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Sharing Learning Rate' },
            batch_size: { value: 32, min: 16, max: 512, step: 1, label: 'Sharing Batch Size' }
          }, { collapsed: false })
        }, { collapsed: false })
      }
    }
    return {
      module_specific_learning: folder({
        Movement: folder({
          learning_rate: {
            value: config.move_parameters?.learning_rate ?? 0.001,
            min: 0.0001,
            max: 0.1,
            step: 0.0001,
            label: 'Movement Learning Rate'
          },
          batch_size: {
            value: config.move_parameters?.batch_size ?? 32,
            min: 16,
            max: 512,
            step: 1,
            label: 'Movement Batch Size'
          }
        }, { collapsed: false }),
        Gathering: folder({
          learning_rate: {
            value: config.gather_parameters?.learning_rate ?? 0.001,
            min: 0.0001,
            max: 0.1,
            step: 0.0001,
            label: 'Gathering Learning Rate'
          },
          batch_size: {
            value: config.gather_parameters?.batch_size ?? 32,
            min: 16,
            max: 512,
            step: 1,
            label: 'Gathering Batch Size'
          }
        }, { collapsed: false }),
        Combat: folder({
          learning_rate: {
            value: config.attack_parameters?.learning_rate ?? 0.001,
            min: 0.0001,
            max: 0.1,
            step: 0.0001,
            label: 'Combat Learning Rate'
          },
          batch_size: {
            value: config.attack_parameters?.batch_size ?? 32,
            min: 16,
            max: 512,
            step: 1,
            label: 'Combat Batch Size'
          }
        }, { collapsed: false }),
        Sharing: folder({
          learning_rate: {
            value: config.share_parameters?.learning_rate ?? 0.001,
            min: 0.0001,
            max: 0.1,
            step: 0.0001,
            label: 'Sharing Learning Rate'
          },
          batch_size: {
            value: config.share_parameters?.batch_size ?? 32,
            min: 16,
            max: 512,
            step: 1,
            label: 'Sharing Batch Size'
          }
        }, { collapsed: false })
      }, { collapsed: false })
    }
  }, [configStore?.config])

  const moduleLearningControls = useControls('Learning & AI/Module-Specific Learning', moduleLearningConfig, { onChange: handleLevaUpdate })

  // ========================================
  // VISUALIZATION FOLDER STRUCTURE
  // ========================================

  // Display Settings sub-folder
  const displayConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        canvas_width: { value: 800, min: 400, max: 1920, step: 10, label: 'Canvas Width' },
        canvas_height: { value: 600, min: 300, max: 1080, step: 10, label: 'Canvas Height' },
        background_color: { value: '#1a1a1a', label: 'Background Color' },
        line_width: { value: 2, min: 1, max: 10, step: 1, label: 'Line Width' }
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
      background_color: {
        value: config.visualization?.background_color ?? '#1a1a1a',
        label: 'Background Color'
      },
      line_width: {
        value: config.visualization?.line_width ?? 2,
        min: 1,
        max: 10,
        step: 1,
        label: 'Line Width'
      }
    }
  }, [configStore?.config])

  const displayControls = useControls('Visualization/Display Settings', displayConfig, { onChange: handleLevaUpdate })

  // Animation Settings sub-folder
  const animationConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        max_frames: { value: 1000, min: 100, max: 10000, step: 10, label: 'Max Frames' },
        frame_delay: { value: 16, min: 1, max: 100, step: 1, label: 'Frame Delay (ms)' },
        animation_speed: { value: 1.0, min: 0.1, max: 5.0, step: 0.1, label: 'Animation Speed' },
        smooth_transitions: { value: true, label: 'Smooth Transitions' }
      }
    }
    return {
      max_frames: {
        value: config.max_frames ?? 1000,
        min: 100,
        max: 10000,
        step: 10,
        label: 'Max Frames'
      },
      frame_delay: {
        value: config.frame_delay ?? 16,
        min: 1,
        max: 100,
        step: 1,
        label: 'Frame Delay (ms)'
      },
      animation_speed: {
        value: config.animation_speed ?? 1.0,
        min: 0.1,
        max: 5.0,
        step: 0.1,
        label: 'Animation Speed'
      },
      smooth_transitions: {
        value: config.smooth_transitions ?? true,
        label: 'Smooth Transitions'
      }
    }
  }, [configStore?.config])

  const animationControls = useControls('Visualization/Animation Settings', animationConfig, { onChange: handleLevaUpdate })

  // Metrics Display sub-folder
  const metricsConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        show_metrics: { value: true, label: 'Show Metrics' },
        font_size: { value: 12, min: 8, max: 24, step: 1, label: 'Font Size' },
        agent_colors: folder({
          SystemAgent: { value: '#4a9eff', label: 'System Agent Color' },
          IndependentAgent: { value: '#ff6b4a', label: 'Independent Agent Color' },
          ControlAgent: { value: '#4ade80', label: 'Control Agent Color' }
        }, { collapsed: false }),
        metrics_position: { value: 'top-right', options: ['top-left', 'top-right', 'bottom-left', 'bottom-right'], label: 'Metrics Position' }
      }
    }
    return {
      show_metrics: {
        value: config.visualization?.show_metrics ?? true,
        label: 'Show Metrics'
      },
      font_size: {
        value: config.visualization?.font_size ?? 12,
        min: 8,
        max: 24,
        step: 1,
        label: 'Font Size'
      },
      agent_colors: folder({
        SystemAgent: {
          value: config.visualization?.agent_colors?.SystemAgent ?? '#4a9eff',
          label: 'System Agent Color'
        },
        IndependentAgent: {
          value: config.visualization?.agent_colors?.IndependentAgent ?? '#ff6b4a',
          label: 'Independent Agent Color'
        },
        ControlAgent: {
          value: config.visualization?.agent_colors?.ControlAgent ?? '#4ade80',
          label: 'Control Agent Color'
        }
      }, { collapsed: false }),
      metrics_position: {
        value: config.metrics_position ?? 'top-right',
        options: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
        label: 'Metrics Position'
      }
    }
  }, [configStore?.config])

  const metricsControls = useControls('Visualization/Metrics Display', metricsConfig, { onChange: handleLevaUpdate })

  // ========================================
  // ENHANCED CUSTOM CONTROLS DEMONSTRATION (Issue #14)
  // ========================================

  // Example metadata for enhanced controls
  const enhancedControlsMetadata = useMemo(() => ({
    'enhanced_position': MetadataTemplates.coordinates(),
    'enhanced_background': MetadataTemplates.color(true), // Greyscale only
    'enhanced_config_file': MetadataTemplates.filePath(['json', 'yaml', 'yml']),
    'enhanced_learning_rate': MetadataTemplates.percentage(),
    'enhanced_memory_size': MetadataTemplates.number(1000, 100000)
  }), [])

  // Example control groups
  const enhancedGroups = useMemo(() => ({
    display_settings: createControlGroup(
      'display_settings',
      'Display Configuration',
      ['enhanced_position', 'enhanced_background'],
      {
        description: 'Visual display and rendering settings',
        icon: '👁️',
        color: '#4a9eff'
      }
    ),
    performance_settings: createControlGroup(
      'performance_settings',
      'Performance Tuning',
      ['enhanced_learning_rate', 'enhanced_memory_size'],
      {
        description: 'AI learning and memory configuration',
        icon: '⚡',
        color: '#ff6b4a'
      }
    )
  }), [])

  // Enhanced custom controls configuration
  const enhancedControlsConfig = useMemo(() => {
    const config = configStore?.config
    if (!config || !validateControlConfig(config)) {
      return {
        // Position settings with Vector2Input
        enhanced_position: {
          value: { x: 100, y: 100 },
          min: 0,
          max: 1000,
          step: 1,
          showLabels: true,
          label: 'Display Position',
          help: 'X, Y coordinates for display positioning'
        },
        // Color settings with ColorInput (greyscale mode)
        enhanced_background: {
          value: '#1a1a1a',
          format: 'hex',
          greyscaleOnly: true,
          showPreview: true,
          label: 'Background Color',
          help: 'Background color (greyscale only)'
        },
        // File path with FilePathInput
        enhanced_config_file: {
          value: null,
          mode: 'file',
          filters: ['json', 'yaml'],
          allowRelative: true,
          label: 'Configuration File',
          help: 'Path to configuration file (JSON or YAML)'
        },
        // Percentage with PercentageInput
        enhanced_learning_rate: {
          value: 0.001,
          min: 0.0001,
          max: 0.1,
          step: 0.0001,
          asPercentage: false,
          showProgress: true,
          label: 'Learning Rate',
          help: 'Neural network learning rate (0-1 range)'
        },
        // Numeric input with metadata
        enhanced_memory_size: {
          value: 10000,
          min: 1000,
          max: 100000,
          step: 100,
          label: 'Memory Size',
          help: 'Memory buffer size for experience replay'
        }
      }
    }

    return {
      enhanced_position: {
        value: { x: config.width || 100, y: config.height || 100 },
        min: 0,
        max: 1000,
        step: 1,
        showLabels: true,
        label: 'Display Position',
        help: 'X, Y coordinates for display positioning'
      },
      enhanced_background: {
        value: config.visualization?.background_color || '#1a1a1a',
        format: 'hex',
        greyscaleOnly: true,
        showPreview: true,
        label: 'Background Color',
        help: 'Background color (greyscale only)'
      },
      enhanced_config_file: {
        value: null, // Would typically come from config
        mode: 'file',
        filters: ['json', 'yaml'],
        allowRelative: true,
        label: 'Configuration File',
        help: 'Path to configuration file (JSON or YAML)'
      },
      enhanced_learning_rate: {
        value: config.learning_rate || 0.001,
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        asPercentage: false,
        showProgress: true,
        label: 'Learning Rate',
        help: 'Neural network learning rate (0-1 range)'
      },
      enhanced_memory_size: {
        value: config.move_parameters?.memory_size || 10000,
        min: 1000,
        max: 100000,
        step: 100,
        label: 'Memory Size',
        help: 'Memory buffer size for experience replay'
      }
    }
  }, [configStore?.config])

  // Custom controls using enhanced components
  const handleEnhancedUpdate = useCallback((path: string, value: any) => {
    console.log('Enhanced control update:', path, value)

    // Convert enhanced control paths to config paths
    const configPathMap: Record<string, string> = {
      'enhanced_position': 'visualization.position',
      'enhanced_background': 'visualization.background_color',
      'enhanced_config_file': 'config_file_path',
      'enhanced_learning_rate': 'learning_rate',
      'enhanced_memory_size': 'move_parameters.memory_size'
    }

    const configPath = configPathMap[path] || path
    const success = safeConfigUpdate(configStore, configPath, value)

    if (success) {
      console.log(`Successfully updated enhanced config at path: ${configPath}`)
    } else {
      console.warn(`Failed to update enhanced config at path: ${configPath}`)
    }
  }, [configStore])

  const enhancedControls = useControls('Enhanced Controls', enhancedControlsConfig, { onChange: handleEnhancedUpdate })

  // Get current theme from Leva store
  const currentTheme = levaStore.getCurrentTheme()

  return (
    <LevaWrapper>
      <Leva
        collapsed={levaStore.isCollapsed}
        hidden={!levaStore.isVisible}
        theme={currentTheme}
      />

      {/* Enhanced Controls Demo Panel */}
      <MetadataProvider
        initialMetadata={enhancedControlsMetadata}
        initialGroups={enhancedGroups}
      >
        <div style={{
          position: 'fixed',
          top: '10px',
          right: levaStore.isVisible ? '350px' : '10px',
          width: '320px',
          maxHeight: '80vh',
          overflow: 'auto',
          background: 'var(--leva-colors-elevation1, #1a1a1a)',
          borderRadius: 'var(--leva-radii-md, 8px)',
          border: '1px solid var(--leva-colors-elevation2, #2a2a2a)',
          zIndex: 1000,
          transition: 'right 0.3s ease'
        }}>
          <div style={{ padding: '16px' }}>
            <h3 style={{
              margin: '0 0 16px 0',
              color: 'var(--leva-colors-highlight1, #ffffff)',
              fontFamily: 'var(--leva-fonts-sans, Albertus)',
              fontSize: '14px'
            }}>
              🎛️ Enhanced Controls Demo
            </h3>

            <ControlGroup
              group={{
                id: 'demo_display',
                label: 'Display Settings',
                description: 'Enhanced visual configuration controls',
                icon: '👁️',
                controls: ['enhanced_position', 'enhanced_background'],
                color: '#4a9eff'
              }}
            >
              <Vector2Input
                path="enhanced_position"
                label="Display Position"
                value={enhancedControls.enhanced_position || { x: 100, y: 100 }}
                onChange={(value) => handleEnhancedUpdate('enhanced_position', value)}
                min={0}
                max={1000}
                showLabels={true}
                help="X, Y coordinates for display positioning"
              />

              <ColorInput
                path="enhanced_background"
                label="Background Color"
                value={enhancedControls.enhanced_background || '#1a1a1a'}
                onChange={(value) => handleEnhancedUpdate('enhanced_background', value)}
                greyscaleOnly={true}
                showPreview={true}
                help="Background color (greyscale only)"
              />
            </ControlGroup>

            <ControlGroup
              group={{
                id: 'demo_performance',
                label: 'Performance Settings',
                description: 'AI and performance tuning parameters',
                icon: '⚡',
                controls: ['enhanced_learning_rate', 'enhanced_memory_size'],
                color: '#ff6b4a'
              }}
            >
              <PercentageInput
                path="enhanced_learning_rate"
                label="Learning Rate"
                value={enhancedControls.enhanced_learning_rate || 0.001}
                onChange={(value) => handleEnhancedUpdate('enhanced_learning_rate', value)}
                min={0.0001}
                max={0.1}
                asPercentage={false}
                showProgress={true}
                help="Neural network learning rate (0-1 range)"
              />

              <PercentageInput
                path="enhanced_memory_size"
                label="Memory Size"
                value={(enhancedControls.enhanced_memory_size || 10000) / 100000}
                onChange={(value) => handleEnhancedUpdate('enhanced_memory_size', value * 100000)}
                min={1000 / 100000}
                max={100000 / 100000}
                asPercentage={true}
                showProgress={true}
                help="Memory buffer size for experience replay"
              />
            </ControlGroup>

            <FilePathInput
              path="enhanced_config_file"
              label="Configuration File"
              value={enhancedControls.enhanced_config_file}
              onChange={(value) => handleEnhancedUpdate('enhanced_config_file', value)}
              mode="file"
              filters={['json', 'yaml']}
              showBrowser={true}
              allowRelative={true}
              help="Path to configuration file (JSON or YAML)"
            />
          </div>
        </div>
      </MetadataProvider>
    </LevaWrapper>
  )
}