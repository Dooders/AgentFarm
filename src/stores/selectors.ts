/**
 * Store selectors for optimized state access and derived state
 * These selectors are memoized and provide efficient access to derived state
 */

import { ValidationError } from '@/types/validation'

// Re-export stores for convenience
export { useConfigStore } from './configStore'
export { useValidationStore } from './validationStore'
export { useLevaStore } from './levaStore'

// Config Store Selectors
export const configSelectors = {
  // Basic state selectors
  getConfig: (state: any) => state.config,
  getOriginalConfig: (state: any) => state.originalConfig,
  getIsDirty: (state: any) => state.isDirty,
  getCompareConfig: (state: any) => state.compareConfig,
  getShowComparison: (state: any) => state.showComparison,
  getSelectedSection: (state: any) => state.selectedSection,
  getExpandedFolders: (state: any) => state.expandedFolders,
  getValidationErrors: (state: any) => state.validationErrors,

  // Derived state selectors
  getEnvironmentConfig: (state: any) => ({
    width: state.config.width,
    height: state.config.height,
    position_discretization_method: state.config.position_discretization_method,
    use_bilinear_interpolation: state.config.use_bilinear_interpolation
  }),

  getAgentConfig: (state: any) => ({
    system_agents: state.config.system_agents,
    independent_agents: state.config.independent_agents,
    control_agents: state.config.control_agents,
    agent_type_ratios: state.config.agent_type_ratios
  }),

  getLearningConfig: (state: any) => ({
    learning_rate: state.config.learning_rate,
    epsilon_start: state.config.epsilon_start,
    epsilon_min: state.config.epsilon_min,
    epsilon_decay: state.config.epsilon_decay
  }),

  getVisualizationConfig: (state: any) => state.config.visualization,

  getModuleConfigs: (state: any) => ({
    gather: state.config.gather_parameters,
    share: state.config.share_parameters,
    move: state.config.move_parameters,
    attack: state.config.attack_parameters
  }),

  // Computed selectors
  getTotalAgents: (state: any) => {
    return state.config.system_agents +
           state.config.independent_agents +
           state.config.control_agents
  },

  getAgentTypePercentages: (state: any) => {
    const ratios = state.config.agent_type_ratios
    return {
      SystemAgent: Math.round(ratios.SystemAgent * 100),
      IndependentAgent: Math.round(ratios.IndependentAgent * 100),
      ControlAgent: Math.round(ratios.ControlAgent * 100)
    }
  },

  getHasUnsavedChanges: (state: any) => state.isDirty,

  getIsValid: (state: any) => state.validationErrors.length === 0,

  getErrorCount: (state: any) => state.validationErrors.length,

  getFieldError: (path: string) => (state: any) => {
    return state.validationErrors.find((error: ValidationError) => error.path === path)
  },

  getFieldErrors: (path: string) => (state: any) => {
    return state.validationErrors.filter((error: ValidationError) =>
      error.path.startsWith(path)
    )
  },

  // Comparison selectors
  getComparisonData: (state: any) => {
    if (!state.compareConfig) return null

    const current = state.config
    const comparison = state.compareConfig

    return {
      width: { current: current.width, comparison: comparison.width },
      height: { current: current.height, comparison: comparison.height },
      system_agents: { current: current.system_agents, comparison: comparison.system_agents },
      independent_agents: { current: current.independent_agents, comparison: comparison.independent_agents },
      control_agents: { current: current.control_agents, comparison: comparison.control_agents }
    }
  }
}

// Validation Store Selectors
export const validationSelectors = {
  // Basic state selectors
  getIsValidating: (state: any) => state.isValidating,
  getErrors: (state: any) => state.errors,
  getWarnings: (state: any) => state.warnings,
  getLastValidationTime: (state: any) => state.lastValidationTime,

  // Derived state selectors
  getErrorCount: (state: any) => state.errors.length,
  getWarningCount: (state: any) => state.warnings.length,
  getTotalIssues: (state: any) => state.errors.length + state.warnings.length,

  getIsValid: (state: any) => state.errors.length === 0,
  getHasWarnings: (state: any) => state.warnings.length > 0,
  getHasErrors: (state: any) => state.errors.length > 0,

  getFieldError: (path: string) => (state: any) => {
    return state.errors.find((error: ValidationError) => error.path === path)
  },

  getFieldErrors: (path: string) => (state: any) => {
    return state.errors.filter((error: ValidationError) =>
      error.path.startsWith(path)
    )
  },

  getFieldWarnings: (path: string) => (state: any) => {
    return state.warnings.filter((warning: ValidationError) =>
      warning.path.startsWith(path)
    )
  },

  // Validation result selectors
  getValidationResult: (state: any) => ({
    success: state.errors.length === 0,
    errors: state.errors,
    warnings: state.warnings
  }),

  getValidationSummary: (state: any) => ({
    isValidating: state.isValidating,
    totalErrors: state.errors.length,
    totalWarnings: state.warnings.length,
    lastValidationTime: state.lastValidationTime,
    hasIssues: state.errors.length > 0 || state.warnings.length > 0
  })
}

// Leva Store Selectors
export const levaSelectors = {
  // Basic state selectors
  getPanelState: (state: any) => ({
    isVisible: state.isVisible,
    isCollapsed: state.isCollapsed,
    panelPosition: state.panelPosition,
    panelWidth: state.panelWidth
  }),

  getControlState: (state: any) => ({
    activeControls: state.activeControls,
    disabledControls: state.disabledControls,
    hiddenControls: state.hiddenControls
  }),

  getFolderState: (state: any) => ({
    collapsedFolders: state.collapsedFolders,
    expandedFolders: state.expandedFolders
  }),

  getThemeState: (state: any) => ({
    theme: state.theme,
    customTheme: state.customTheme
  }),

  // Derived state selectors
  getActiveControls: (state: any) => {
    return state.activeControls.filter((control: string) =>
      !state.disabledControls.has(control) && !state.hiddenControls.has(control)
    )
  },

  getVisibleControls: (state: any) => {
    return state.activeControls.filter((control: string) =>
      !state.hiddenControls.has(control)
    )
  },

  isControlEnabled: (controlPath: string) => (state: any) => {
    return !state.disabledControls.has(controlPath)
  },

  isControlVisible: (controlPath: string) => (state: any) => {
    return !state.hiddenControls.has(controlPath)
  },

  isFolderCollapsed: (folderPath: string) => (state: any) => {
    return state.collapsedFolders.has(folderPath)
  },

  isFolderExpanded: (folderPath: string) => (state: any) => {
    return state.expandedFolders.has(folderPath)
  },

  // Panel selectors
  getPanelSettings: (state: any) => ({
    visible: state.isVisible,
    collapsed: state.isCollapsed,
    position: state.panelPosition,
    width: state.panelWidth
  }),

  // Theme selectors
  getCurrentTheme: (state: any) => {
    if (state.theme === 'custom') {
      return state.customTheme
    }

    // Return default themes for built-in options
    return {
      colors: {
        elevation1: state.theme === 'light' ? '#ffffff' : '#1a1a1a',
        elevation2: state.theme === 'light' ? '#f5f5f5' : '#2a2a2a',
        elevation3: state.theme === 'light' ? '#e5e5e5' : '#3a3a3a',
        accent1: '#666666',
        accent2: '#888888',
        accent3: '#aaaaaa',
        highlight1: state.theme === 'light' ? '#000000' : '#ffffff',
        highlight2: state.theme === 'light' ? '#333333' : '#ffffff',
        highlight3: state.theme === 'light' ? '#666666' : '#ffffff',
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
    }
  }
}

// Combined selectors for multiple stores
export const combinedSelectors = {
  // Get complete UI state
  getUIState: (configState: any, validationState: any, levaState: any) => ({
    config: {
      selectedSection: configState.selectedSection,
      expandedFolders: configState.expandedFolders,
      isDirty: configState.isDirty,
      showComparison: configState.showComparison
    },
    validation: {
      hasErrors: validationState.errors.length > 0,
      hasWarnings: validationState.warnings.length > 0,
      errorCount: validationState.errors.length,
      warningCount: validationState.warnings.length
    },
    leva: {
      panelVisible: levaState.isVisible,
      panelCollapsed: levaState.isCollapsed,
      panelPosition: levaState.panelPosition,
      panelWidth: levaState.panelWidth
    }
  }),

  // Get validation status for current config section
  getSectionValidationStatus: (section: string) => (_configState: any, validationState: any) => {
    const sectionErrors = validationState.errors.filter((error: ValidationError) =>
      error.path.startsWith(section)
    )
    const sectionWarnings = validationState.warnings.filter((warning: ValidationError) =>
      warning.path.startsWith(section)
    )

    return {
      hasErrors: sectionErrors.length > 0,
      hasWarnings: sectionWarnings.length > 0,
      errorCount: sectionErrors.length,
      warningCount: sectionWarnings.length,
      errors: sectionErrors,
      warnings: sectionWarnings
    }
  },

  // Get complete application state snapshot
  getAppState: (configState: any, validationState: any, levaState: any) => ({
    config: {
      current: configState.config,
      original: configState.originalConfig,
      isDirty: configState.isDirty,
      comparison: configState.compareConfig,
      showComparison: configState.showComparison,
      selectedSection: configState.selectedSection,
      expandedFolders: configState.expandedFolders
    },
    validation: {
      errors: validationState.errors,
      warnings: validationState.warnings,
      isValidating: validationState.isValidating,
      lastValidationTime: validationState.lastValidationTime,
      isValid: validationState.errors.length === 0
    },
    leva: {
      isVisible: levaState.isVisible,
      isCollapsed: levaState.isCollapsed,
      panelPosition: levaState.panelPosition,
      panelWidth: levaState.panelWidth,
      theme: levaState.theme,
      activeControls: levaState.activeControls,
      expandedFolders: levaState.expandedFolders
    }
  })
}