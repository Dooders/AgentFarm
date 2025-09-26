import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ConfigExplorer } from '../ConfigExplorer/ConfigExplorer'

describe('ConfigExplorer', () => {
  it('renders without crashing', () => {
    render(<ConfigExplorer />)
    expect(screen.getByText('Configuration Explorer')).toBeInTheDocument()
  })

  it('displays the main container', () => {
    const { container } = render(<ConfigExplorer />)
    expect(container.firstChild).toHaveClass('config-explorer')
  })

  it('contains configuration explorer content for left panel', () => {
    render(<ConfigExplorer />)
    expect(screen.getByText('Configuration Explorer')).toBeInTheDocument()
    expect(screen.getByText('Leva Controls')).toBeInTheDocument()
  })

  it('contains comparison content for right panel', () => {
    render(<ConfigExplorer />)
    expect(screen.getByText('Configuration Comparison')).toBeInTheDocument()
    expect(screen.getByText('Current Configuration')).toBeInTheDocument()
  })
})