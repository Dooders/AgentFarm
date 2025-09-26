import React from 'react'
import { folder, FolderSettings } from 'leva'
import styled from 'styled-components'

export interface LevaFolderProps {
  /** Display name for the folder */
  label: string
  /** Initial collapsed state */
  collapsed?: boolean
  /** Custom icon or emoji for the folder */
  icon?: string
  /** Folder description for tooltips */
  description?: string
  /** Child components or controls to render inside */
  children: React.ReactNode
  /** Callback when folder is toggled */
  onToggle?: (collapsed: boolean) => void
  /** Additional styling class */
  className?: string
}

// Styled wrapper for enhanced folder appearance
const FolderWrapper = styled.div`
  .leva-c-folder {
    --leva-colors-elevation1: var(--leva-elevation1, #1a1a1a) !important;
    --leva-colors-elevation2: var(--leva-elevation2, #2a2a2a) !important;
    --leva-colors-elevation3: var(--leva-elevation3, #3a3a3a) !important;
    --leva-colors-accent1: var(--leva-accent1, #666666) !important;
    --leva-colors-accent2: var(--leva-accent2, #888888) !important;
    --leva-colors-highlight1: var(--leva-highlight1, #ffffff) !important;

    border-radius: var(--leva-radii-md, 8px) !important;
    margin: var(--leva-space-sm, 8px) 0 !important;
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a) !important;
    background: var(--leva-colors-elevation1, #1a1a1a) !important;
    transition: all 0.2s ease !important;

    &:hover {
      border-color: var(--leva-colors-accent1, #666666) !important;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }
  }

  .leva-c-folder__title {
    font-family: var(--leva-fonts-sans, 'Albertus') !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--leva-colors-highlight1, #ffffff) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
  }

  .leva-c-folder__icon {
    color: var(--leva-colors-accent2, #888888) !important;
    margin-right: var(--leva-space-xs, 4px) !important;
  }

  .leva-c-folder__content {
    padding: var(--leva-space-sm, 8px) !important;
  }
`

// Enhanced folder settings with metadata support
export interface EnhancedFolderSettings extends FolderSettings {
  /** Icon to display next to folder name */
  icon?: string
  /** Description for tooltip */
  description?: string
  /** Custom color for the folder (greyscale only) */
  color?: string
  /** Whether to show the folder header border */
  showBorder?: boolean
  /** Custom spacing for the folder content */
  spacing?: number
}

/**
 * Enhanced Leva Folder wrapper component with custom styling and metadata support
 *
 * Features:
 * - Custom greyscale theme integration
 * - Icon support for different section types
 * - Tooltip descriptions
 * - Collapsible/expandable functionality
 * - Enhanced visual styling
 */
export const LevaFolder: React.FC<LevaFolderProps> = ({
  label,
  collapsed = false,
  icon,
  description,
  children,
  onToggle,
  className
}) => {
  // Create enhanced folder settings
  const folderSettings: EnhancedFolderSettings = {
    collapsed,
    render: (collapsed) => {
      // Custom render function for enhanced styling
      return (
        <FolderWrapper className={className}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              padding: '8px 12px',
              cursor: 'pointer',
              borderBottom: collapsed ? 'none' : '1px solid var(--leva-colors-elevation2, #2a2a2a)',
              transition: 'border-color 0.2s ease'
            }}
            onClick={() => onToggle?.(!collapsed)}
            title={description}
          >
            <span className="leva-c-folder__icon">
              {icon || (collapsed ? '▶' : '▼')}
            </span>
            <span className="leva-c-folder__title">
              {label}
            </span>
            {description && (
              <span
                style={{
                  marginLeft: 'auto',
                  fontSize: '10px',
                  color: 'var(--leva-colors-accent2, #888888)',
                  opacity: 0.7
                }}
                title={description}
              >
                ℹ
              </span>
            )}
          </div>
          {!collapsed && (
            <div className="leva-c-folder__content">
              {children}
            </div>
          )}
        </FolderWrapper>
      )
    }
  }

  // Return the Leva folder with enhanced settings
  return folder(label, children, folderSettings)
}

/**
 * Pre-configured folder types for common use cases
 */
export const createSectionFolder = (
  section: string,
  children: React.ReactNode,
  collapsed = false
): any => {
  const icons: Record<string, string> = {
    'Environment': '🌍',
    'Agent Behavior': '🤖',
    'Learning & AI': '🧠',
    'Visualization': '👁️',
    'Settings': '⚙️',
    'Parameters': '📊',
    'Controls': '🎛️',
    'Display': '📺',
    'Animation': '🎬',
    'Metrics': '📈'
  }

  const descriptions: Record<string, string> = {
    'Environment': 'World settings and simulation parameters',
    'Agent Behavior': 'Movement, gathering, combat, and sharing parameters',
    'Learning & AI': 'Learning rates, neural network settings, and AI parameters',
    'Visualization': 'Display settings, colors, and animation controls',
    'Settings': 'General configuration settings',
    'Parameters': 'Parameter controls and settings',
    'Controls': 'Control panel settings',
    'Display': 'Display and rendering options',
    'Animation': 'Animation timing and transition settings',
    'Metrics': 'Performance metrics and monitoring'
  }

  return folder(section, children, {
    collapsed,
    icon: icons[section],
    description: descriptions[section],
    render: undefined // Use default render for simple sections
  })
}

/**
 * Utility function to create a collapsible folder with metadata
 */
export const createMetadataFolder = (
  label: string,
  metadata: {
    description?: string
    icon?: string
    color?: string
    collapsed?: boolean
  },
  children: React.ReactNode
): any => {
  return folder(label, children, {
    collapsed: metadata.collapsed ?? false,
    icon: metadata.icon,
    description: metadata.description,
    color: metadata.color
  })
}