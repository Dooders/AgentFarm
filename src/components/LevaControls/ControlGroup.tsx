import React, { useCallback, useMemo } from 'react'
import styled from 'styled-components'
import { ControlGroup as ControlGroupType, ControlCategory } from './MetadataSystem'
import { LevaFolder } from './LevaFolder'

export interface ControlGroupProps {
  /** Group configuration */
  group: ControlGroupType
  /** Category information */
  category?: ControlCategory
  /** Child components to render */
  children: React.ReactNode
  /** Whether the group is collapsed by default */
  collapsed?: boolean
  /** Callback when group is toggled */
  onToggle?: (collapsed: boolean) => void
  /** Custom styling */
  className?: string
  /** Whether to show group metadata */
  showMetadata?: boolean
  /** Additional actions to show in group header */
  actions?: React.ReactNode
}

// Styled wrapper for control groups
const GroupWrapper = styled.div`
  position: relative;
  margin: var(--leva-space-sm, 8px) 0;
  border-radius: var(--leva-radii-md, 8px);
  overflow: hidden;

  .group-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--leva-space-sm, 8px) var(--leva-space-md, 16px);
    background: var(--leva-colors-elevation2, #2a2a2a);
    border-bottom: 1px solid var(--leva-colors-elevation3, #3a3a3a);
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;

    &:hover {
      background: var(--leva-colors-elevation3, #3a3a3a);
    }
  }

  .group-title {
    display: flex;
    align-items: center;
    gap: var(--leva-space-sm, 8px);
    font-family: var(--leva-fonts-sans, 'Albertus');
    font-size: 12px;
    font-weight: 600;
    color: var(--leva-colors-highlight1, #ffffff);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .group-icon {
    font-size: 14px;
    color: var(--leva-colors-accent2, #888888);
  }

  .group-description {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: var(--leva-colors-accent2, #888888);
    margin-top: 2px;
    line-height: 1.2;
  }

  .group-content {
    padding: var(--leva-space-md, 16px);
    background: var(--leva-colors-elevation1, #1a1a1a);
  }

  .group-actions {
    display: flex;
    align-items: center;
    gap: var(--leva-space-xs, 4px);
  }

  .group-action {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: var(--leva-radii-xs, 2px);
    background: var(--leva-colors-elevation3, #3a3a3a);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 10px;
    color: var(--leva-colors-accent2, #888888);

    &:hover {
      background: var(--leva-colors-accent1, #666666);
      color: var(--leva-colors-highlight1, #ffffff);
    }
  }

  .group-separator {
    height: 1px;
    background: var(--leva-colors-elevation2, #2a2a2a);
    margin: var(--leva-space-xs, 4px) 0;
  }

  .group-controls-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--leva-space-sm, 8px);
  }

  .group-controls-list {
    display: flex;
    flex-direction: column;
    gap: var(--leva-space-xs, 4px);
  }
`

// Group header component
const GroupHeader: React.FC<{
  group: ControlGroupType
  category?: ControlCategory
  collapsed: boolean
  onToggle: () => void
  showMetadata: boolean
  actions?: React.ReactNode
}> = ({ group, category, collapsed, onToggle, showMetadata, actions }) => {
  const handleToggle = useCallback(() => {
    onToggle()
  }, [onToggle])

  return (
    <div className="group-header" onClick={handleToggle}>
      <div className="group-title">
        <span className="group-icon">{group.icon || '📊'}</span>
        <span>{group.label}</span>
        {category && (
          <span
            style={{
              fontSize: '10px',
              color: 'var(--leva-colors-accent2, #888888)',
              fontWeight: 'normal'
            }}
          >
            in {category.label}
          </span>
        )}
      </div>

      <div className="group-actions">
        {actions}
        <div
          className="group-action"
          title={collapsed ? 'Expand group' : 'Collapse group'}
        >
          {collapsed ? '▶' : '▼'}
        </div>
      </div>
    </div>
  )
}

// Group content component
const GroupContent: React.FC<{
  group: ControlGroupType
  children: React.ReactNode
  layout?: 'grid' | 'list'
}> = ({ group, children, layout = 'list' }) => {
  const contentClass = layout === 'grid' ? 'group-controls-grid' : 'group-controls-list'

  return (
    <div className="group-content">
      {group.description && (
        <div className="group-description">{group.description}</div>
      )}

      <div className={contentClass}>
        {children}
      </div>
    </div>
  )
}

/**
 * ControlGroup component for organizing related parameters
 *
 * Features:
 * - Visual separation and organization
 * - Collapsible/expandable groups
 * - Custom icons and descriptions
 * - Grid or list layout options
 * - Integration with metadata system
 * - Consistent theming
 */
export const ControlGroup: React.FC<ControlGroupProps> = ({
  group,
  category,
  children,
  collapsed = false,
  onToggle,
  className,
  showMetadata = true,
  actions
}) => {
  const [isCollapsed, setIsCollapsed] = React.useState(collapsed || group.collapsed || false)

  const handleToggle = useCallback(() => {
    const newCollapsed = !isCollapsed
    setIsCollapsed(newCollapsed)
    onToggle?.(newCollapsed)
  }, [isCollapsed, onToggle])

  const groupStyle = useMemo(() => {
    if (group.color) {
      return {
        borderLeft: `4px solid ${group.color}`,
        '--group-color': group.color
      } as React.CSSProperties
    }
    return {}
  }, [group.color])

  return (
    <GroupWrapper className={className} style={groupStyle}>
      <GroupHeader
        group={group}
        category={category}
        collapsed={isCollapsed}
        onToggle={handleToggle}
        showMetadata={showMetadata}
        actions={actions}
      />

      {!isCollapsed && (
        <GroupContent group={group} layout="list">
          {children}
        </GroupContent>
      )}
    </GroupWrapper>
  )
}

/**
 * Control grouping utility functions
 */

// Create a control group
export const createControlGroup = (
  id: string,
  label: string,
  controls: string[],
  options: Partial<ControlGroupType> = {}
): ControlGroupType => ({
  id,
  label,
  controls,
  collapsed: false,
  priority: 0,
  ...options
})

// Create a control category
export const createControlCategory = (
  id: string,
  label: string,
  groups: string[],
  options: Partial<ControlCategory> = {}
): ControlCategory => ({
  id,
  label,
  groups,
  collapsed: false,
  ...options
})

// Group layout options
export type GroupLayout = 'list' | 'grid' | 'compact'

/**
 * Enhanced control group with layout options
 */
export const EnhancedControlGroup: React.FC<ControlGroupProps & {
  layout?: GroupLayout
}> = ({ layout = 'list', ...props }) => {
  const getLayoutClass = () => {
    switch (layout) {
      case 'grid':
        return 'group-controls-grid'
      case 'compact':
        return 'group-controls-compact'
      default:
        return 'group-controls-list'
    }
  }

  return (
    <ControlGroup {...props}>
      <div className={getLayoutClass()}>
        {props.children}
      </div>
    </ControlGroup>
  )
}

/**
 * Control group with automatic metadata integration
 */
export const MetadataControlGroup: React.FC<{
  groupId: string
  children: React.ReactNode
  fallbackGroup?: ControlGroupType
  fallbackCategory?: ControlCategory
}> = ({ groupId, children, fallbackGroup, fallbackCategory }) => {
  // This would typically use the metadata context
  // For now, use fallback values
  const group = fallbackGroup || {
    id: groupId,
    label: groupId,
    controls: [],
    collapsed: false
  }

  const category = fallbackCategory

  return (
    <ControlGroup
      group={group}
      category={category}
      showMetadata={true}
    >
      {children}
    </ControlGroup>
  )
}

/**
 * Hook for managing control group state
 */
export const useControlGroup = (groupId: string, initialCollapsed = false) => {
  const [collapsed, setCollapsed] = React.useState(initialCollapsed)

  const toggle = useCallback(() => {
    setCollapsed(prev => !prev)
  }, [])

  const expand = useCallback(() => {
    setCollapsed(false)
  }, [])

  const collapse = useCallback(() => {
    setCollapsed(true)
  }, [])

  return {
    collapsed,
    toggle,
    expand,
    collapse,
    setCollapsed
  }
}