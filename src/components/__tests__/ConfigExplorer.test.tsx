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

  it('contains placeholder content for left panel', () => {
    render(<ConfigExplorer />)
    expect(screen.getByText('Left panel content will be implemented in subsequent issues.')).toBeInTheDocument()
    expect(screen.getByText('Navigation tree')).toBeInTheDocument()
  })

  it('contains placeholder content for right panel', () => {
    render(<ConfigExplorer />)
    expect(screen.getByText('Comparison Panel')).toBeInTheDocument()
    expect(screen.getByText('Diff highlighting')).toBeInTheDocument()
  })
})