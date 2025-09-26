import React from 'react'
import { describe, it, expect, beforeEach } from 'vitest'
import { render } from '../../test/test-utils'
import { screen } from '@testing-library/react'
import '@testing-library/jest-dom'
import { useValidationStore } from '@/stores/validationStore'
import { ValidationDisplay } from '@/components/Validation/ValidationDisplay'

describe('ValidationDisplay', () => {
  beforeEach(() => {
    // reset store
    useValidationStore.setState({
      isValidating: false,
      errors: [],
      warnings: [],
      lastValidationTime: 0,
    } as any)
  })

  it('renders field errors scoped by prefix', () => {
    useValidationStore.setState({
      errors: [
        { path: 'width', message: 'Invalid width', code: 'invalid_range' },
        { path: 'learning_rate', message: 'Too high', code: 'too_big' },
      ],
      warnings: []
    } as any)

    render(<ValidationDisplay prefixPaths={['width']} title="Env Issues" />)

    expect(screen.getByText('Env Issues')).toBeInTheDocument()
    expect(screen.getByText('Invalid width')).toBeInTheDocument()
    expect(screen.queryByText('Too high')).not.toBeInTheDocument()
  })

  it('renders warnings', () => {
    useValidationStore.setState({
      errors: [],
      warnings: [
        { path: 'environment', message: 'Large environment size may impact performance', code: 'performance_warning' }
      ]
    } as any)

    render(<ValidationDisplay prefixPaths={['environment']} />)

    expect(screen.getByText('Large environment size may impact performance')).toBeInTheDocument()
  })
})

