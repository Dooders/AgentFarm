// @ts-nocheck
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import styled from 'styled-components'
import { useConfigStore } from '@/stores/configStore'
import { useValidationStore } from '@/stores/validationStore'
import { configSelectors, validationSelectors } from '@/stores/selectors'
import { ipcService } from '@/services/ipcService'

const Bar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 6px 12px;
  border-top: 1px solid var(--border-subtle);
  background: var(--background-secondary);
  font-size: 12px;
  color: var(--text-secondary);
`

const Section = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
`

const Button = styled.button`
  padding: 4px 8px;
  font-size: 12px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  cursor: pointer;
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`

const Dot = styled.span<{ color: string }>`
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${(p: { color: string }) => p.color};
`

const FilePath = styled.span`
  max-width: 320px;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
  color: var(--text-secondary);
`

export const StatusBar: React.FC = () => {
  const isDirty = useConfigStore((s) => configSelectors.getIsDirty(s))
  const currentFilePath = useConfigStore((s) => configSelectors.getCurrentFilePath(s) as string | undefined)
  const lastSaveTime = useConfigStore((s) => configSelectors.getLastSaveTime(s) as number | undefined)
  const lastLoadTime = useConfigStore((s) => configSelectors.getLastLoadTime(s) as number | undefined)
  const validateConfig = useConfigStore((s) => s.validateConfig)

  const errorCount = useValidationStore((s) => validationSelectors.getErrorCount(s))
  const warningCount = useValidationStore((s) => validationSelectors.getWarningCount(s))
  const isValidating = useValidationStore((s) => validationSelectors.getIsValidating(s))
  const lastValidationTime = useValidationStore((s) => validationSelectors.getLastValidationTime(s))

  const [autoValidate, setAutoValidate] = useState<boolean>(false)
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected')

  // Initialize preferences and connection polling
  useEffect(() => {
    try {
      const pref = localStorage.getItem('ui:auto-validate')
      const enabled = pref === '1' || pref === 'true'
      setAutoValidate(enabled)
    } catch {}

    setConnectionStatus(ipcService.getConnectionStatus())
    const interval = setInterval(() => {
      setConnectionStatus(ipcService.getConnectionStatus())
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  // Auto validation loop
  useEffect(() => {
    if (!autoValidate) return
    let cancelled = false
    const interval = setInterval(async () => {
      if (cancelled) return
      try {
        await validateConfig()
      } catch {}
    }, 3000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [autoValidate, validateConfig])

  const onValidate = useCallback(async () => {
    try {
      await validateConfig()
    } catch (err) {
      console.error('Validate failed:', err)
    }
  }, [validateConfig])

  const toggleAuto = useCallback(() => {
    setAutoValidate((prev) => {
      const next = !prev
      try { localStorage.setItem('ui:auto-validate', next ? '1' : '0') } catch {}
      return next
    })
  }, [])

  const goToValidation = useCallback(() => {
    const el = document.getElementById('validation-content')
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      ;(el as HTMLElement).focus?.()
    }
  }, [])

  const hasIssues = useMemo(() => errorCount + warningCount > 0, [errorCount, warningCount])

  return (
    <Bar role="status" aria-label="Application status bar" aria-live="polite">
      <Section aria-label="Validation status">
        <span title={hasIssues ? 'Issues detected' : 'No issues'}>
          <Dot color={hasIssues ? 'var(--error-border, #dc2626)' : 'var(--success, #16a34a)'} />
        </span>
        <span>Errors: {errorCount}</span>
        <span>Warnings: {warningCount}</span>
        {isValidating ? <span>Validating…</span> : (
          <Button onClick={onValidate} aria-label="Validate configuration now">Validate</Button>
        )}
        <Button onClick={toggleAuto} aria-pressed={autoValidate} aria-label="Toggle auto validation">
          {autoValidate ? 'Auto On' : 'Auto Off'}
        </Button>
        <Button onClick={goToValidation} aria-label="View validation issues">View Issues</Button>
        {lastValidationTime ? (
          <span title={new Date(lastValidationTime).toLocaleString()}>Last Check: {new Date(lastValidationTime).toLocaleTimeString()}</span>
        ) : (
          <span>Last Check: —</span>
        )}
      </Section>

      <Section aria-label="Save and system status">
        <span>{isDirty ? '● Unsaved' : '✓ Saved'}</span>
        {currentFilePath && <FilePath title={currentFilePath}>{currentFilePath}</FilePath>}
        {lastSaveTime && <span>Last Save: {new Date(lastSaveTime).toLocaleTimeString()}</span>}
        {lastLoadTime && <span>Last Load: {new Date(lastLoadTime).toLocaleTimeString()}</span>}
        <span>Conn: {connectionStatus}</span>
      </Section>
    </Bar>
  )
}

export default StatusBar

