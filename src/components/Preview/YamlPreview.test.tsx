import { describe, it, expect, vi } from 'vitest'
import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { YamlPreview } from './YamlPreview'
import * as selectors from '@/stores/selectors'

describe('YamlPreview', () => {
  it('renders live YAML and updates on config change', () => {
    const config = { width: 100, visualization: { canvas_width: 800 } } as any
    const getConfig = vi.spyOn(selectors, 'useConfigStore' as any).mockImplementation((sel: any) => {
      if (sel) {
        const state = {
          config,
          compareConfig: null
        }
        return sel(state)
      }
      return null
    })

    render(<YamlPreview />)
    expect(screen.getByText('YAML Preview')).toBeTruthy()
    const pre = screen.getByLabelText('YAML preview')
    expect(pre.innerHTML).toContain('width')
    expect(pre.innerHTML).toContain('100')

    // Update mocked config and re-render
    ;(config as any).width = 120
    render(<YamlPreview />)
    const pre2 = screen.getByLabelText('YAML preview')
    expect(pre2.innerHTML).toContain('120')

    getConfig.mockRestore()
  })

  it('switches to diff mode when comparison exists', () => {
    const state = {
      config: { width: 100 },
      compareConfig: { width: 200 }
    } as any
    const getState = vi.spyOn(selectors, 'useConfigStore' as any).mockImplementation((sel: any) => sel(state))

    render(<YamlPreview />)
    const toggle = screen.getByText('Show Diff')
    fireEvent.click(toggle)
    // Diff grid should render
    expect(document.querySelector('.yaml-grid')).toBeTruthy()
    getState.mockRestore()
  })
})

