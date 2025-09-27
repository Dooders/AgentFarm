/// <reference types="react" />
/// <reference types="react-dom" />
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

vi.mock('@/stores/configStore', () => {
  const state: any = { config: { width: 100, visualization: { canvas_width: 800 } }, compareConfig: null }
  const hook = (selector?: any) => (selector ? selector(state) : state)
  hook.__setState = (next: any) => { Object.assign(state, next) }
  return { useConfigStore: hook }
})

import { useConfigStore } from '@/stores/configStore'
import { YamlPreview } from './YamlPreview'

describe('YamlPreview', () => {
  it('renders live YAML and updates on config change', () => {
    render(<YamlPreview />)
    expect(screen.getByText('YAML Preview')).toBeTruthy()
    const pre = screen.getByLabelText('YAML preview')
    expect(pre.innerHTML).toContain('width')
    expect(pre.innerHTML).toContain('100')

    // Update mocked config and re-render
    ;(useConfigStore as any).__setState({ config: { width: 120, visualization: { canvas_width: 800 } } })
    render(<YamlPreview />)
    const pre2 = screen.getByLabelText('YAML preview')
    expect(pre2.innerHTML).toContain('120')
  })

  it('switches to diff mode when comparison exists', () => {
    ;(useConfigStore as any).__setState({ config: { width: 100 }, compareConfig: { width: 200 } })

    render(<YamlPreview />)
    const toggle = screen.getByText('Show Diff')
    fireEvent.click(toggle)
    // Diff grid should render
    expect(document.querySelector('.yaml-grid')).toBeTruthy()
  })
})

