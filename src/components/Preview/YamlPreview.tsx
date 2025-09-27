/// <reference types="react" />
/// <reference types="react-dom" />
import React, { useMemo, useRef, useState } from 'react'
import styled from 'styled-components'
import { useConfigStore } from '@/stores/configStore'
import { configSelectors } from '@/stores/selectors'
import { ipcService } from '@/services/ipcService'

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--background-secondary);
  border-top: 1px solid var(--border-subtle);
`

const Header = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-subtle);
  background: var(--background-secondary);
`

const Title = styled.h3`
  margin: 0;
  font-size: 14px;
  color: var(--text-secondary);
  flex: 1;
`

const Toolbar = styled.div`
  display: flex;
  gap: 8px;
`

const Button = styled.button`
  padding: 6px 10px;
  font-size: 12px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  cursor: pointer;
`

const Content = styled.div<{ wrap: boolean; fontSize: number }>`
  flex: 1;
  overflow: auto;
  padding: 12px;
  background: var(--background-primary);
  pre {
    margin: 0;
    white-space: ${(
      p: { wrap: boolean; fontSize: number }
    ) => (p.wrap ? 'pre-wrap' : 'pre')};
    word-break: break-word;
    font-size: ${(
      p: { wrap: boolean; fontSize: number }
    ) => p.fontSize}px;
    line-height: 1.5;
  }
  code {
    color: var(--text-primary);
  }
  /* Greyscale syntax highlighting */
  .tok-key { color: var(--text-secondary); font-weight: 600; }
  .tok-str { color: #8aa0a8; }
  .tok-num { color: #a88a8a; }
  .tok-bool { color: #9aa88a; font-weight: 600; }
  .tok-null { color: #a0a0a0; font-style: italic; }
  .tok-dash { color: #b0b0b0; }
  .yaml-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 12px;
  }
  .yaml-diff-row { display: contents; }
  .yaml-diff-row .col {
    padding: 2px 4px;
    border-bottom: 1px dashed var(--border-subtle);
  }
  .yaml-diff-row.diff .col.left { background: rgba(200, 80, 80, 0.08); }
  .yaml-diff-row.diff .col.right { background: rgba(80, 200, 120, 0.08); }
`

type PreviewMode = 'yaml' | 'diff'

function toYaml(value: unknown, indent: number = 0): string {
  const pad = (n: number) => '  '.repeat(n)
  if (value === null || value === undefined) return 'null'
  const t = typeof value
  if (t !== 'object') {
    if (t === 'string') return JSON.stringify(value as string)
    return String(value)
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    return (value as unknown[])
      .map(v => `${pad(indent)}- ${toYaml(v, indent + 1)}`)
      .join('\n')
  }
  const obj = value as Record<string, unknown>
  const keys = Object.keys(obj)
  if (keys.length === 0) return '{}'
  return keys
    .map(k => {
      const v = obj[k]
      const isObj = v && typeof v === 'object'
      const rendered = toYaml(v as unknown, indent + 1)
      if (Array.isArray(v)) {
        if (v.length === 0) return `${pad(indent)}${k}: []`
        const arr = (v as unknown[])
          .map(item => `${pad(indent + 1)}- ${toYaml(item, indent + 2)}`)
          .join('\n')
        return `${pad(indent)}${k}:\n${arr}`
      }
      if (isObj) return `${pad(indent)}${k}:\n${rendered}`
      return `${pad(indent)}${k}: ${rendered}`
    })
    .join('\n')
}

function flatten(obj: any, prefix = ''): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  const isObj = obj && typeof obj === 'object' && !Array.isArray(obj)
  if (!isObj) {
    out[prefix || '(root)'] = obj
    return out
  }
  Object.keys(obj).sort().forEach(k => {
    const val = obj[k]
    const keyPath = prefix ? `${prefix}.${k}` : k
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      Object.assign(out, flatten(val, keyPath))
    } else {
      out[keyPath] = val
    }
  })
  return out
}

export const YamlPreview: React.FC = () => {
  const config = useConfigStore(configSelectors.getConfig)
  const compare = useConfigStore(configSelectors.getCompareConfig)
  const [mode, setMode] = useState<PreviewMode>('yaml')
  const [wrap, setWrap] = useState<boolean>(true)
  const [fontSize, setFontSize] = useState<number>(12)
  const [compact, setCompact] = useState<boolean>(false)
  const preRef = useRef<HTMLPreElement>(null)

  const yamlText = useMemo(() => toYaml(config), [config])

  function classifyScalar(value: string): string {
    if (value === 'null' || value === '~') return `<span class=\"tok-null\">${value}</span>`
    if (value === 'true' || value === 'false') return `<span class=\"tok-bool\">${value}</span>`
    if (/^[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/.test(value)) return `<span class=\"tok-num\">${value}</span>`
    if (/^".*"$/.test(value) || /^'.*'$/.test(value)) return `<span class=\"tok-str\">${value}</span>`
    return value
  }

  function highlightYaml(yaml: string): string {
    const lines = yaml.split('\n')
    const out: string[] = []
    for (const line of lines) {
      let m = line.match(/^(\s*)-\s*(.*)$/)
      if (m) {
        const [, indent, rest] = m
        if (rest.includes(':')) {
          const km = rest.match(/^([^:]+):(.*)$/)
          if (km) {
            const key = km[1].trimEnd()
            const val = km[2].trimStart()
            const valCls = val ? classifyScalar(val) : ''
            out.push(`${indent}<span class=\"tok-dash\">-</span> <span class=\"tok-key\">${key}</span>: ${valCls}`)
            continue
          }
        }
        out.push(`${indent}<span class=\"tok-dash\">-</span> ${classifyScalar(rest)}`)
        continue
      }
      m = line.match(/^(\s*)([^:]+):(\s*)(.*)$/)
      if (m) {
        const [, indent, key, sp, rest] = m
        const valCls = rest ? classifyScalar(rest) : ''
        out.push(`${indent}<span class=\"tok-key\">${key}</span>:${sp}${valCls}`)
        continue
      }
      out.push(line)
    }
    return out.join('\n')
  }

  const diffGridHtml = useMemo(() => {
    if (!compare) return ''
    const left = flatten(config)
    const right = flatten(compare)
    const keys = Array.from(new Set([...Object.keys(left), ...Object.keys(right)])).sort()
    const rows = keys.map(k => {
      const lv = left[k]
      const rv = right[k]
      const same = JSON.stringify(lv) === JSON.stringify(rv)
      const render = (v: unknown) => {
        if (v === undefined) return '—'
        if (v === null) return 'null'
        if (typeof v === 'object') return JSON.stringify(v)
        return String(v)
      }
      return `<div class="yaml-diff-row ${same ? 'same' : 'diff'}"><div class="col key">${k}</div><div class="col left">${render(lv)}</div><div class="col right">${render(rv)}</div></div>`
    }).join('')
    return `<div class="yaml-grid">${rows}</div>`
  }, [config, compare])

  function getDiffText(): string {
    if (!compare) return ''
    const left = flatten(config)
    const right = flatten(compare)
    const keys = Array.from(new Set([...Object.keys(left), ...Object.keys(right)])).sort()
    const render = (v: unknown) => {
      if (v === undefined) return '—'
      if (v === null) return 'null'
      if (typeof v === 'object') return JSON.stringify(v)
      return String(v)
    }
    const lines = keys.map(k => {
      const lv = left[k]
      const rv = right[k]
      const same = JSON.stringify(lv) === JSON.stringify(rv)
      return `${same ? ' ' : '*'} ${k} | ${render(lv)} | ${render(rv)}`
    })
    return ['Key | Left | Right', '--- | --- | ---', ...lines].join('\n')
  }

  const copyToClipboard = async () => {
    try {
      const text = mode === 'yaml' ? yamlText : getDiffText()
      await navigator.clipboard.writeText(text)
    } catch {
      // no-op
    }
  }

  const exportYaml = async () => {
    try {
      await ipcService.exportConfig({ config, format: 'yaml', includeMetadata: false })
    } catch {
      // Fallback: trigger download in web env
      const blob = new Blob([yamlText], { type: 'text/yaml' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'config.yaml'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <Container>
      <Header>
        <Title>{mode === 'yaml' ? 'YAML Preview' : 'YAML Diff'}</Title>
        <Toolbar>
          <Button onClick={() => setMode((m: PreviewMode) => (m === 'yaml' ? 'diff' : 'yaml'))} disabled={!compare && mode === 'diff'}>
            {mode === 'yaml' ? 'Show Diff' : 'Show YAML'}
          </Button>
          <Button onClick={() => setCompact((c: boolean) => !c)}>{compact ? 'Normal' : 'Compact'}</Button>
          <Button onClick={() => setWrap((w: boolean) => !w)}>{wrap ? 'No Wrap' : 'Wrap'}</Button>
          <Button onClick={() => setFontSize((s: number) => Math.max(10, s - 1))}>A-</Button>
          <Button onClick={() => setFontSize((s: number) => Math.min(20, s + 1))}>A+</Button>
          <Button onClick={copyToClipboard}>Copy</Button>
          <Button onClick={exportYaml}>Export</Button>
        </Toolbar>
      </Header>
      <Content wrap={compact ? false : wrap} fontSize={compact ? 11 : fontSize}>
        {mode === 'yaml' && (
          <pre ref={preRef} aria-label="YAML preview"><code dangerouslySetInnerHTML={{ __html: highlightYaml(yamlText) }} /></pre>
        )}
        {mode === 'diff' && compare && (
          <div dangerouslySetInnerHTML={{ __html: diffGridHtml }} />
        )}
      </Content>
    </Container>
  )
}

export default YamlPreview

