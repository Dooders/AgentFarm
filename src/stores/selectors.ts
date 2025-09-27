/**
 * Store selectors for optimized state access and derived state
 * These selectors are memoized and provide efficient access to derived state
 */

import { ValidationError } from '@/types/validation'
import { ConfigStore } from '@/types/config'
import { ValidationState } from '@/types/validation'
import { LevaStore } from './levaStore'

// Re-export stores for convenience
export { useConfigStore } from './configStore'
export { useValidationStore } from './validationStore'
export { useLevaStore } from './levaStore'

// Config Store Selectors
export const configSelectors = {
  // Basic state selectors
  getConfig: (state: ConfigStore) => state.config,
  getOriginalConfig: (state: ConfigStore) => state.originalConfig,
  getIsDirty: (state: ConfigStore) => state.isDirty,
  getCompareConfig: (state: ConfigStore) => state.compareConfig,
  getShowComparison: (state: ConfigStore) => state.showComparison,
  getSelectedSection: (state: ConfigStore) => state.selectedSection,
  getExpandedFolders: (state: ConfigStore) => state.expandedFolders,
  getValidationErrors: (state: ConfigStore) => state.validationErrors,
  getCurrentFilePath: (state: ConfigStore) => state.currentFilePath,
  getLastSaveTime: (state: ConfigStore) => state.lastSaveTime,
  getLastLoadTime: (state: ConfigStore) => state.lastLoadTime,

  // Derived state selectors
  getEnvironmentConfig: (state: ConfigStore) => ({
    width: state.config.width,
    height: state.config.height,
    position_discretization_method: state.config.position_discretization_method,
    use_bilinear_interpolation: state.config.use_bilinear_interpolation
  }),

  getAgentConfig: (state: ConfigStore) => ({
    system_agents: state.config.system_agents,
    independent_agents: state.config.independent_agents,
    control_agents: state.config.control_agents,
    agent_type_ratios: state.config.agent_type_ratios
  }),

  getLearningConfig: (state: ConfigStore) => ({
    learning_rate: state.config.learning_rate,
    epsilon_start: state.config.epsilon_start,
    epsilon_min: state.config.epsilon_min,
    epsilon_decay: state.config.epsilon_decay
  }),

  getVisualizationConfig: (state: ConfigStore) => state.config.visualization,

  getModuleConfigs: (state: ConfigStore) => ({
    gather: state.config.gather_parameters,
    share: state.config.share_parameters,
    move: state.config.move_parameters,
    attack: state.config.attack_parameters
  }),

  // Computed selectors
  getTotalAgents: (state: ConfigStore) => {
    return state.config.system_agents +
           state.config.independent_agents +
           state.config.control_agents
  },

  getAgentTypePercentages: (state: ConfigStore) => {
    const ratios = state.config.agent_type_ratios
    return {
      SystemAgent: Math.round(ratios.SystemAgent * 100),
      IndependentAgent: Math.round(ratios.IndependentAgent * 100),
      ControlAgent: Math.round(ratios.ControlAgent * 100)
    }
  },

  getHasUnsavedChanges: (state: ConfigStore) => state.isDirty,

  getIsValid: (state: ConfigStore) => state.validationErrors.length === 0,

  getErrorCount: (state: ConfigStore) => state.validationErrors.length,

  getFieldError: (path: string) => (state: ConfigStore) => {
    return state.validationErrors.find((error: ValidationError) => error.path === path)
  },

  getFieldErrors: (path: string) => (state: ConfigStore) => {
    return state.validationErrors.filter((error: ValidationError) =>
      error.path.startsWith(path)
    )
  },

  // Comparison selectors
  getComparisonData: (state: ConfigStore) => {
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
  },

  // Full diff between current and comparison
  getComparisonDiff: (state: ConfigStore) => state.getComparisonDiff(),

  // Diff statistics
  getComparisonStats: (state: ConfigStore) => {
    const diff = state.getComparisonDiff()
    const added = Object.keys(diff.added).length
    const removed = Object.keys(diff.removed).length
    const changed = Object.keys(diff.changed).length
    const unchanged = Object.keys(diff.unchanged).length
    const total = added + removed + changed + unchanged
    const percentChanged = total > 0 ? Math.round(((added + removed + changed) / total) * 100) : 0
    return { added, removed, changed, unchanged, total, percentChanged }
  },

  // Filtered diff by section or predicate
  getFilteredComparisonDiff:
    (filter?: { section?: string; predicate?: (path: string) => boolean }) =>
    (state: ConfigStore) => {
      const diff = state.getComparisonDiff()
      const match = (path: string) => {
        if (filter?.predicate && !filter.predicate(path)) return false
        if (filter?.section && !path.startsWith(filter.section)) return false
        return true
      }

      return {
        added: Object.fromEntries(Object.entries(diff.added).filter(([k]) => match(k))),
        removed: Object.fromEntries(Object.entries(diff.removed).filter(([k]) => match(k))),
        changed: Object.fromEntries(Object.entries(diff.changed).filter(([k]) => match(k))),
        unchanged: Object.fromEntries(Object.entries(diff.unchanged).filter(([k]) => match(k)))
      }
    }
}

// Validation Store Selectors
export const validationSelectors = {
  // Basic state selectors
  getIsValidating: (state: ValidationState) => state.isValidating,
  getErrors: (state: ValidationState) => state.errors,
  getWarnings: (state: ValidationState) => state.warnings,
  getLastValidationTime: (state: ValidationState) => state.lastValidationTime,

  // Derived state selectors
  getErrorCount: (state: ValidationState) => state.errors.length,
  getWarningCount: (state: ValidationState) => state.warnings.length,
  getTotalIssues: (state: ValidationState) => state.errors.length + state.warnings.length,

  getIsValid: (state: ValidationState) => state.errors.length === 0,
  getHasWarnings: (state: ValidationState) => state.warnings.length > 0,
  getHasErrors: (state: ValidationState) => state.errors.length > 0,

  getFieldError: (path: string) => (state: ValidationState) => {
    return state.errors.find((error: ValidationError) => error.path === path)
  },

  getFieldErrors: (path: string) => (state: ValidationState) => {
    return state.errors.filter((error: ValidationError) =>
      error.path.startsWith(path)
    )
  },

  getFieldWarnings: (path: string) => (state: ValidationState) => {
    return state.warnings.filter((warning: ValidationError) =>
      warning.path.startsWith(path)
    )
  },

  // Validation result selectors
  getValidationResult: (state: ValidationState) => ({
    success: state.errors.length === 0,
    errors: state.errors,
    warnings: state.warnings
  }),

  getValidationSummary: (state: ValidationState) => ({
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
  getPanelState: (state: LevaStore) => ({
    isVisible: state.isVisible,
    isCollapsed: state.isCollapsed,
    panelPosition: state.panelPosition,
    panelWidth: state.panelWidth
  }),

  getControlState: (state: LevaStore) => ({
    activeControls: state.activeControls,
    disabledControls: state.disabledControls,
    hiddenControls: state.hiddenControls
  }),

  getFolderState: (state: LevaStore) => ({
    collapsedFolders: state.collapsedFolders,
    expandedFolders: state.expandedFolders
  }),

  getThemeState: (state: LevaStore) => ({
    theme: state.theme,
    customTheme: state.customTheme
  }),

  // Derived state selectors
  getActiveControls: (state: LevaStore) => {
    return state.activeControls.filter((control: string) =>
      !state.disabledControls.has(control) && !state.hiddenControls.has(control)
    )
  },

  getVisibleControls: (state: LevaStore) => {
    return state.activeControls.filter((control: string) =>
      !state.hiddenControls.has(control)
    )
  },

  isControlEnabled: (controlPath: string) => (state: LevaStore) => {
    return !state.disabledControls.has(controlPath)
  },

  isControlVisible: (controlPath: string) => (state: LevaStore) => {
    return !state.hiddenControls.has(controlPath)
  },

  isFolderCollapsed: (folderPath: string) => (state: LevaStore) => {
    return state.collapsedFolders.has(folderPath)
  },

  isFolderExpanded: (folderPath: string) => (state: LevaStore) => {
    return state.expandedFolders.has(folderPath)
  },

  // Panel selectors
  getPanelSettings: (state: LevaStore) => ({
    visible: state.isVisible,
    collapsed: state.isCollapsed,
    position: state.panelPosition,
    width: state.panelWidth
  }),

  // Theme selectors
  getCurrentTheme: (state: LevaStore) => {
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
  getUIState: (configState: ConfigStore, validationState: ValidationState, levaState: LevaStore) => ({
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
  getSectionValidationStatus: (section: string) => (_configState: ConfigStore, validationState: ValidationState) => {
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
  getAppState: (configState: ConfigStore, validationState: ValidationState, levaState: LevaStore) => ({
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