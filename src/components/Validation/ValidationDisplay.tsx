import React, { useMemo } from 'react'
import styled from 'styled-components'
import { useValidationStore } from '@/stores/validationStore'
import { ValidationError } from '@/types/validation'

interface ValidationDisplayProps {
  paths?: string[]
  prefixPaths?: string[]
  compact?: boolean
  showIcons?: boolean
  className?: string
  title?: string
}

const Container = styled.div`
  margin-top: 8px;
`

const Item = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 8px;
  margin-bottom: 4px;
  background: var(--background-primary);
  border: 1px solid var(--border-subtle);
  border-left-width: 3px;
  border-radius: 4px;
`

const ErrorItem = styled(Item)`
  border-left-color: var(--error-border, #dc2626);
`

const WarningItem = styled(Item)`
  border-left-color: var(--warning-border, #a16207);
`

const Path = styled.div`
  font-family: var(--font-mono, 'JetBrains Mono');
  font-size: 11px;
  color: var(--text-secondary);
`

const Message = styled.div`
  color: var(--text-primary);
  font-size: 12px;
  line-height: 1.3;
`

const Title = styled.div`
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 6px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
`

export const ValidationDisplay: React.FC<ValidationDisplayProps> = ({
  paths = [],
  prefixPaths = [],
  compact = true,
  showIcons = true,
  className,
  title
}) => {
  const { errors, warnings } = useValidationStore()

  const visibleErrors = useMemo(() => filterIssues(errors, paths, prefixPaths), [errors, paths, prefixPaths])
  const visibleWarnings = useMemo(() => filterIssues(warnings, paths, prefixPaths), [warnings, paths, prefixPaths])

  if (visibleErrors.length === 0 && visibleWarnings.length === 0) return null

  return (
    <Container className={className}>
      {title && <Title>{title}</Title>}
      {visibleErrors.map((err, idx) => (
        <ErrorItem key={`err-${idx}`}>
          {showIcons && <span aria-hidden>✖</span>}
          <div>
            {!compact && <Path>{err.path}</Path>}
            <Message>{err.message}</Message>
          </div>
        </ErrorItem>
      ))}
      {visibleWarnings.map((warn, idx) => (
        <WarningItem key={`warn-${idx}`}>
          {showIcons && <span aria-hidden>⚠</span>}
          <div>
            {!compact && <Path>{warn.path}</Path>}
            <Message>{warn.message}</Message>
          </div>
        </WarningItem>
      ))}
    </Container>
  )
}

function filterIssues(issues: ValidationError[], paths: string[], prefixPaths: string[]): ValidationError[] {
  if (paths.length === 0 && prefixPaths.length === 0) return []
  return issues.filter(issue => {
    if (paths.includes(issue.path)) return true
    return prefixPaths.some(prefix => issue.path === prefix || issue.path.startsWith(`${prefix}.`))
  })
}

