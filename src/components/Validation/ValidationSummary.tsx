import React, { useMemo } from 'react'
import styled from 'styled-components'
import { useValidationStore } from '@/stores/validationStore'

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`

const Row = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--background-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 10px 12px;
`

const Stat = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-primary);
  font-size: 12px;
`

const Dot = styled.span<{ color: string }>`
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${(p: { color: string }) => p.color};
`

const List = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`

const Issue = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 12px;
  color: var(--text-secondary);
`

interface ValidationSummaryProps {
  limit?: number
  onIssueClick?: (path: string) => void
}

export const ValidationSummary: React.FC<ValidationSummaryProps> = ({ limit = 8, onIssueClick }) => {
  const { errors, warnings, isValidating } = useValidationStore()

  const hasIssues = errors.length > 0 || warnings.length > 0

  const topIssues = useMemo(() => {
    const combined = [
      ...errors.map(e => ({ ...e, kind: 'error' as const })),
      ...warnings.map(w => ({ ...w, kind: 'warning' as const }))
    ]
    return combined.slice(0, limit)
  }, [errors, warnings, limit])

  return (
    <Container>
      <Row>
        <Stat>
          <Dot color={hasIssues ? 'var(--error-border, #dc2626)' : 'var(--success, #16a34a)'} />
          <span>{hasIssues ? 'Validation issues detected' : 'All parameters validated successfully'}</span>
        </Stat>
        <Stat>
          <Dot color={'var(--error-border, #dc2626)'} />
          <span>{errors.length} errors</span>
        </Stat>
        <Stat>
          <Dot color={'var(--warning-border, #a16207)'} />
          <span>{warnings.length} warnings</span>
        </Stat>
        {isValidating && <Stat><span>Validating…</span></Stat>}
      </Row>

      {hasIssues && (
        <List>
          {topIssues.map((issue, idx) => (
            <Issue key={idx} role="button" onClick={() => onIssueClick?.(issue.path)}>
              <span aria-hidden>{issue.kind === 'error' ? '✖' : '⚠'}</span>
              <span>
                <strong style={{ color: 'var(--text-primary)' }}>{issue.path}:</strong> {issue.message}
              </span>
            </Issue>
          ))}
        </List>
      )}
    </Container>
  )
}

