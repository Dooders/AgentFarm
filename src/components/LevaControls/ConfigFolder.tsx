import React, { useState } from 'react'
import { ConfigFolderProps } from '@/types/leva'
import styled from 'styled-components'

const FolderContainer = styled.div`
  border: 1px solid var(--leva-colors-accent1);
  border-radius: var(--leva-radii-md);
  background: var(--leva-colors-elevation2);
  margin-bottom: var(--leva-space-md);
  overflow: hidden;
`

const FolderHeader = styled.div`
  display: flex;
  align-items: center;
  padding: var(--leva-space-sm) var(--leva-space-md);
  background: var(--leva-colors-elevation3);
  cursor: pointer;
  user-select: none;

  &:hover {
    background: var(--leva-colors-elevation1);
  }
`

const FolderTitle = styled.h3`
  font-size: 12px;
  font-weight: 600;
  color: var(--leva-colors-highlight2);
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

const ToggleIcon = styled.span<{ collapsed: boolean }>`
  margin-right: var(--leva-space-sm);
  font-size: 10px;
  color: var(--leva-colors-accent2);
  transform: ${props => props.collapsed ? 'rotate(-90deg)' : 'rotate(0deg)'};
  transition: transform 0.2s ease;
  display: inline-block;
`

const FolderContent = styled.div<{ collapsed: boolean }>`
  padding: ${props => props.collapsed ? '0' : 'var(--leva-space-md)'};
  max-height: ${props => props.collapsed ? '0' : 'none'};
  overflow: hidden;
  transition: all 0.2s ease;
  background: var(--leva-colors-elevation2);
`

export const ConfigFolder: React.FC<ConfigFolderProps> = ({
  label,
  collapsed = false,
  children,
  onToggle
}) => {
  const [isCollapsed, setIsCollapsed] = useState(collapsed)

  const handleToggle = () => {
    const newCollapsed = !isCollapsed
    setIsCollapsed(newCollapsed)
    onToggle?.()
  }

  return (
    <FolderContainer>
      <FolderHeader onClick={handleToggle}>
        <ToggleIcon collapsed={isCollapsed}>
          ▶
        </ToggleIcon>
        <FolderTitle>{label}</FolderTitle>
      </FolderHeader>
      <FolderContent collapsed={isCollapsed}>
        {children}
      </FolderContent>
    </FolderContainer>
  )
}