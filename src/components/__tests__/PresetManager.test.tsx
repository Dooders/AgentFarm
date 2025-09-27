import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { PresetManager } from '@/components/ConfigExplorer/PresetManager'
import { useConfigStore } from '@/stores/configStore'

vi.mock('@/stores/configStore', async (orig) => {
  const mod = await orig()
  const state: any = {
    templates: [
      { name: 'Default Configuration', description: 'Standard simulation setup', category: 'system', baseConfig: {}, tags: [] },
      { name: 'Small World', description: 'Smaller grid and fewer agents', category: 'user', baseConfig: {}, tags: [] }
    ],
    listTemplates: vi.fn().mockResolvedValue(undefined),
    loadTemplate: vi.fn().mockResolvedValue(undefined),
    applyTemplatePartial: vi.fn().mockResolvedValue(undefined),
    saveTemplate: vi.fn().mockResolvedValue(undefined),
    deleteTemplate: vi.fn().mockResolvedValue(undefined),
  }
  return {
    ...mod,
    useConfigStore: (sel: any) => sel(state)
  }
})

describe('PresetManager', () => {
  it('renders and filters presets', async () => {
    render(<PresetManager />)
    expect(await screen.findByText('Default Configuration')).toBeInTheDocument()
    expect(screen.getByText('Small World')).toBeInTheDocument()

    const search = screen.getByPlaceholderText('Search presets...')
    fireEvent.change(search, { target: { value: 'small' } })
    expect(screen.queryByText('Default Configuration')).not.toBeInTheDocument()
    expect(screen.getByText('Small World')).toBeInTheDocument()
  })

  it('applies a preset', async () => {
    render(<PresetManager />)
    const apply = await screen.findAllByText('Apply')
    fireEvent.click(apply[0])
    await waitFor(() => {
      const sel = useConfigStore((s) => s)
      expect(sel.loadTemplate).toHaveBeenCalled()
    })
  })
})

