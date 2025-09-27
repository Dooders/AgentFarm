import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ArrayInput } from './ArrayInput'

describe('ArrayInput', () => {
  it('adds and removes items', () => {
    const onChange = vi.fn()
    render(
      <ArrayInput
        path="test.array"
        schema={{}}
        value={[1, 2]}
        onChange={onChange}
        label="Array"
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /add/i }))
    expect(onChange).toHaveBeenCalled()

    fireEvent.click(screen.getAllByRole('button', { name: /remove/i })[0])
    expect(onChange).toHaveBeenCalled()
  })
})

