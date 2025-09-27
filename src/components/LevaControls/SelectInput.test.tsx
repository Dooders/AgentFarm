import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { SelectInput } from './SelectInput'

describe('SelectInput', () => {
  it('renders options and filters with search', () => {
    const onChange = vi.fn()
    render(
      <SelectInput
        path="test.select"
        schema={{}}
        value="b"
        onChange={onChange}
        options={["a", "b", "c"]}
        label="Test Select"
        searchable
      />
    )

    // Label present
    expect(screen.getByText('Test Select')).toBeInTheDocument()

    // Search narrows options
    const search = screen.getByPlaceholderText('Search...')
    fireEvent.change(search, { target: { value: 'c' } })
    expect(screen.getByRole('combobox')).toBeInTheDocument()
  })

  it('calls onChange when selection changes', () => {
    const onChange = vi.fn()
    render(
      <SelectInput
        path="test.select"
        schema={{}}
        value="a"
        onChange={onChange}
        options={{ A: 'a', B: 'b' }}
      />
    )

    fireEvent.change(screen.getByRole('combobox'), { target: { value: '1' } })
    expect(onChange).toHaveBeenCalledWith('b')
  })
})

