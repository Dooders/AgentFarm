import React from 'react'
import { render, screen } from '@/test/test-utils'
import { useValidationStore } from '@/stores/validationStore'
import { ValidationSummary } from '@/components/Validation/ValidationSummary'

describe('ValidationSummary', () => {
  beforeEach(() => {
    useValidationStore.setState({
      isValidating: false,
      errors: [],
      warnings: [],
      lastValidationTime: 0,
    } as any)
  })

  it('shows success when no issues', () => {
    render(<ValidationSummary />)
    expect(screen.getByText(/All parameters validated successfully/i)).toBeInTheDocument()
  })

  it('shows counts and top issues when present', () => {
    useValidationStore.setState({
      errors: [ { path: 'width', message: 'Invalid width', code: 'invalid_range' } ],
      warnings: [ { path: 'environment', message: 'Performance warning', code: 'performance_warning' } ]
    } as any)

    render(<ValidationSummary />)
    expect(screen.getByText(/1 errors/i)).toBeInTheDocument()
    expect(screen.getByText(/1 warnings/i)).toBeInTheDocument()
    expect(screen.getByText(/width:/i)).toBeInTheDocument()
    expect(screen.getByText(/Performance warning/i)).toBeInTheDocument()
  })
})

