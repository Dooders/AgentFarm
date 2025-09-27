import React, { useCallback, useEffect, useRef, useState } from 'react'
import styled from 'styled-components'
import { useConfigStore } from '@/stores/configStore'
import type { ConfigStore } from '@/types/config'
import { configSelectors, useValidationStore, validationSelectors } from '@/stores/selectors'
import { ipcService } from '@/services/ipcService'
import { toYaml } from '@/utils/yaml'
// Search UI is integrated in Right Panel; toolbar provides jump + shortcut

const Bar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-subtle);
  background: var(--background-secondary);
`

const Section = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`

const Button = styled.button`
  padding: 6px 10px;
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

const Sep = styled.span`
  width: 1px;
  height: 20px;
  background: var(--border-subtle);
  display: inline-block;
`

const Status = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--text-secondary);
`

export const Toolbar: React.FC = () => {
  const config = useConfigStore(configSelectors.getConfig)
  const compareConfig = useConfigStore(configSelectors.getCompareConfig)
  const showComparison = useConfigStore(configSelectors.getShowComparison)
  const isDirty = useConfigStore(configSelectors.getIsDirty)
  const currentFilePath = useConfigStore(configSelectors.getCurrentFilePath) as string | undefined
  const lastSaveTime = useConfigStore(configSelectors.getLastSaveTime) as number | undefined
  const lastLoadTime = useConfigStore(configSelectors.getLastLoadTime) as number | undefined
  const loadConfig = useConfigStore((s: ConfigStore) => s.loadConfig)
  const openConfigFromContent = useConfigStore((s: ConfigStore) => s.openConfigFromContent)
  const saveConfig = useConfigStore((s: ConfigStore) => s.saveConfig)
  const exportConfigMeta = useConfigStore((s: ConfigStore) => s.exportConfig)
  const resetToDefaults = useConfigStore((s: ConfigStore) => s.resetToDefaults)
  const toggleComparison = useConfigStore((s: ConfigStore) => s.toggleComparison)
  const clearComparison = useConfigStore((s: ConfigStore) => s.clearComparison)
  const setComparison = useConfigStore((s: ConfigStore) => s.setComparison)
  const setComparisonPath = useConfigStore((s: ConfigStore) => s.setComparisonPath)
  const applyAll = useConfigStore((s: ConfigStore) => s.applyAllDifferencesFromComparison)

  const errorCount = useValidationStore((s) => validationSelectors.getErrorCount(s))
  const warningCount = useValidationStore((s) => validationSelectors.getWarningCount(s))

  const [isGrayscale, setIsGrayscale] = useState<boolean>(false)
  const filePickerRef = useRef<HTMLInputElement>(null)
  const comparePickerRef = useRef<HTMLInputElement>(null)

  // Initialize grayscale state from localStorage
  useEffect(() => {
    try {
      const pref = localStorage.getItem('ui:grayscale')
      const enabled = pref === '1' || pref === 'true'
      setIsGrayscale(enabled)
      document.body.classList.toggle('grayscale', enabled)
    } catch {}
  }, [])

  const toggleGrayscale = useCallback(() => {
    setIsGrayscale((prev) => {
      const next = !prev
      document.body.classList.toggle('grayscale', next)
      try { localStorage.setItem('ui:grayscale', next ? '1' : '0') } catch {}
      return next
    })
  }, [])

  const doOpen = useCallback((): void => { filePickerRef.current?.click() }, [])
  const doOpenCompare = useCallback((): void => { comparePickerRef.current?.click() }, [])

  const onFileChosen: React.ChangeEventHandler<HTMLInputElement> = async (e): Promise<void> => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      if (window?.electronAPI) {
        await loadConfig((file as any).path || file.name)
      } else {
        const text = await file.text()
        await openConfigFromContent(text, 'json')
      }
    } catch (err) {
      console.error('Open failed:', err)
    } finally {
      if (filePickerRef.current) filePickerRef.current.value = ''
    }
  }

  const onCompareChosen: React.ChangeEventHandler<HTMLInputElement> = async (e): Promise<void> => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      const parsed = JSON.parse(text)
      setComparison(parsed)
      setComparisonPath(file.name)
    } catch (err) {
      console.error('Compare load failed:', err)
      setComparison(null)
      setComparisonPath(undefined)
    } finally {
      if (comparePickerRef.current) comparePickerRef.current.value = ''
    }
  }

  const doSave = useCallback(async (): Promise<void> => {
    try {
      await saveConfig(currentFilePath)
    } catch (err) {
      console.error('Save failed:', err)
    }
  }, [saveConfig, currentFilePath])

  const doSaveAs = useCallback(async (): Promise<void> => {
    try {
      if (window?.electronAPI) {
        const result: any = await ipcService.invoke('dialog:save', {
          filters: [
            { name: 'Configuration Files', extensions: ['json', 'yaml', 'yml', 'toml'] },
            { name: 'All Files', extensions: ['*'] }
          ]
        })
        if (result && !result.canceled && result.filePath) {
          await saveConfig(result.filePath)
          return
        }
      }

      // Fallback for browser
      const json = JSON.stringify(config, null, 2)
      const blob = new Blob([json], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'config.json'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Save As failed:', err)
    }
  }, [config, saveConfig])

  const doExportYaml = useCallback(async (): Promise<void> => {
    try {
      await ipcService.exportConfig({ config, format: 'yaml', includeMetadata: false })
    } catch {
      const yaml = toYaml(config)
      const blob = new Blob([yaml], { type: 'text/yaml' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'config.yaml'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    }
  }, [config])

  const doExportJson = useCallback(async (): Promise<void> => {
    try {
      const meta = exportConfigMeta('json')
      const content = JSON.stringify(meta.config, null, 2)
      const blob = new Blob([content], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'config.json'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Export JSON failed:', err)
    }
  }, [exportConfigMeta])

  // Keyboard shortcuts
  useEffect(() => {
    const shortcutMap: Record<string, (e: KeyboardEvent) => void> = {
      'mod+o': (e) => { e.preventDefault(); doOpen() },
      'mod+s': (e) => { e.preventDefault(); doSave() },
      'mod+shift+s': (e) => { e.preventDefault(); doSaveAs() },
      'mod+g': (e) => { e.preventDefault(); toggleGrayscale() },
      'mod+y': (e) => { e.preventDefault(); doExportYaml() },
      'mod+f': (e) => { e.preventDefault(); const el = document.getElementById('toolbar-search'); if (el) (el as HTMLInputElement).focus() }
    }
    const onKey = (e: KeyboardEvent) => {
      const isMod = e.ctrlKey || e.metaKey
      if (!isMod) return
      const key = e.key.toLowerCase()
      const shortcut = `mod${e.shiftKey ? '+shift' : ''}+${key}`
      const handler = shortcutMap[shortcut]
      if (handler) handler(e)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [doOpen, doSave, doSaveAs, doExportYaml, toggleGrayscale])

  const [connectionStatus, setConnectionStatus] = useState<string>(() => ipcService.getConnectionStatus())

  useEffect(() => {
    const interval = setInterval(() => {
      setConnectionStatus(ipcService.getConnectionStatus())
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <Bar role="toolbar" aria-label="Application toolbar">
      <Section aria-label="File operations">
        <Button onClick={doOpen} aria-label="Open configuration (Ctrl/Cmd+O)">Open…</Button>
        <input ref={filePickerRef} type="file" accept="application/json,.json" style={{ display: 'none' }} onChange={onFileChosen} />
        <Button onClick={doSave} disabled={!isDirty} aria-label="Save (Ctrl/Cmd+S)">Save</Button>
        <Button onClick={doSaveAs} aria-label="Save As (Ctrl/Cmd+Shift+S)">Save As…</Button>
        <Sep />
        <Button onClick={doExportJson}>Export JSON</Button>
        <Button onClick={doExportYaml} aria-label="Export YAML (Ctrl/Cmd+Y)">Export YAML</Button>
      </Section>

      <Section aria-label="Comparison controls">
        <Button onClick={() => toggleComparison()}>{showComparison ? 'Hide Compare' : 'Show Compare'}</Button>
        <Button onClick={doOpenCompare}>Load Compare…</Button>
        <input ref={comparePickerRef} type="file" accept="application/json,.json" style={{ display: 'none' }} onChange={onCompareChosen} />
        <Button onClick={() => clearComparison()} disabled={!compareConfig}>Clear Compare</Button>
        <Button onClick={() => applyAll()} disabled={!compareConfig}>Apply All</Button>
      </Section>

      <Section aria-label="Application controls">
        <Button onClick={() => {
          const el = document.getElementById('presets-section')
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }}>Presets</Button>
        <Button onClick={toggleGrayscale} aria-pressed={isGrayscale} aria-label="Toggle grayscale (Ctrl/Cmd+G)">{isGrayscale ? 'Grayscale On' : 'Grayscale Off'}</Button>
        <Button onClick={() => resetToDefaults()}>Reset</Button>
      </Section>

      <Status aria-label="Status indicators">
        <span>{isDirty ? '● Unsaved' : '✓ Saved'}</span>
        <span>Errors: {errorCount}</span>
        <span>Warnings: {warningCount}</span>
        {currentFilePath && <span title={currentFilePath}>{currentFilePath}</span>}
        {lastSaveTime && <span>Last Save: {new Date(lastSaveTime).toLocaleTimeString()}</span>}
        {lastLoadTime && <span>Last Load: {new Date(lastLoadTime).toLocaleTimeString()}</span>}
        <span>Conn: {connectionStatus}</span>
      </Status>
    </Bar>
  )
}

export default Toolbar

