export function toYaml(value: unknown, indent: number = 0): string {
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

