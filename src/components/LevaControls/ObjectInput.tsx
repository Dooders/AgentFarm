import React, { useMemo, useState } from 'react'
import { ConfigInputProps } from '../../types/leva'
import styled from 'styled-components'

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`

const Label = styled.label`
  font-size: 11px;
  font-weight: 500;
  color: var(--leva-colors-highlight2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

const ToggleButton = styled.button`
  height: 28px;
  padding: 0 8px;
  border: 1px solid var(--leva-colors-accent1);
  background: var(--leva-colors-elevation2);
  color: var(--leva-colors-highlight1);
  border-radius: var(--leva-radii-sm);
  font-size: 11px;
  font-family: var(--leva-fonts-sans);
`

const Editor = styled.textarea`
  width: 100%;
  min-height: 120px;
  padding: 8px;
  background: var(--leva-colors-elevation2);
  border: 1px solid var(--leva-colors-accent1);
  border-radius: var(--leva-radii-sm);
  color: var(--leva-colors-highlight1);
  font-family: var(--leva-fonts-mono);
  font-size: 11px;
  line-height: 1.4;

  &:focus {
    outline: none;
    border-color: var(--leva-colors-accent2);
    box-shadow: 0 0 0 1px var(--leva-colors-accent2);
  }
`

const Pre = styled.pre`
  margin: 0;
  padding: 8px;
  background: var(--leva-colors-elevation2);
  border: 1px solid var(--leva-colors-accent1);
  border-radius: var(--leva-radii-sm);
  color: var(--leva-colors-highlight1);
  font-family: var(--leva-fonts-mono);
  font-size: 11px;
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  /* Simple inline syntax highlight tones (greyscale accent) */
  .k { color: var(--leva-colors-accent3); } /* keys */
  .s { color: var(--leva-colors-highlight2); } /* strings */
  .n { color: var(--leva-colors-accent2); } /* numbers */
`

const ErrorText = styled.div`
  font-size: 10px;
  color: #ff6b6b;
  margin-top: 2px;
`

export const ObjectInput: React.FC<ConfigInputProps> = ({
  value,
  onChange,
  label,
  error,
  disabled = false
}) => {
  const [collapsed, setCollapsed] = useState(true)
  const [text, setText] = useState(() => {
    try {
      return JSON.stringify(value ?? {}, null, 2)
    } catch {
      return '{}'
    }
  })
  const [parseError, setParseError] = useState<string | null>(null)

  const highlightedLines = useMemo(() => {
    try {
      const json = JSON.stringify(value ?? {}, null, 2) || '{}'
      const toValueNode = (token: string) => {
        const str = token.trim()
        let m = str.match(/^"([^"]*)"(,?)$/)
        if (m) return (<><span className="s">"{m[1]}"</span>{m[2]}</>)
        m = str.match(/^(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(,?)$/)
        if (m) return (<><span className="n">{m[1]}</span>{m[2]}</>)
        m = str.match(/^(true|false|null)(,?)$/)
        if (m) return (<><span className="n">{m[1]}</span>{m[2]}</>)
        return token
      }
      return json.split('\n').map((line, i) => {
        const keyMatch = line.match(/^(\s*)"([^"]+)"\s*:\s*(.*)$/)
        if (keyMatch) {
          const [, indent, key, rest] = keyMatch
          return (
            <div key={i}>
              {indent}
              <span className="k">"{key}"</span>
              {": "}
              {toValueNode(rest)}
            </div>
          )
        }
        return (<div key={i}>{line}</div>)
      })
    } catch {
      return [<div key="0">{`{}`}</div>]
    }
  }, [value])

  const handleBlur = () => {
    try {
      const parsed = JSON.parse(text || '{}')
      setParseError(null)
      onChange(parsed)
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Invalid JSON'
      setParseError(message)
    }
  }

  return (
    <Container>
      <Header>
        {label && <Label>{label}</Label>}
        <ToggleButton type="button" onClick={() => setCollapsed(!collapsed)} disabled={disabled}>
          {collapsed ? 'Expand' : 'Collapse'}
        </ToggleButton>
      </Header>
      {collapsed ? (
        <Pre>
          {highlightedLines}
        </Pre>
      ) : (
        <Editor
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={handleBlur}
          disabled={disabled}
        />
      )}
      {(error || parseError) && <ErrorText>{parseError || error}</ErrorText>}
    </Container>
  )
}

