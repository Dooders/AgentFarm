import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ObjectInput } from './ObjectInput'

describe('ObjectInput', () => {
  it('shows pretty JSON when collapsed and edits when expanded', () => {
    const onChange = vi.fn()
    render(
      <ObjectInput
        path="test.object"
        schema={{}}
        value={{ a: 1 }}
        onChange={onChange}
        label="Obj"
      />
    )

    // Expand editor
    fireEvent.click(screen.getByRole('button', { name: /expand/i }))
    const textarea = screen.getByRole('textbox')
    fireEvent.change(textarea, { target: { value: '{"a":2}' } })
    fireEvent.blur(textarea)
    expect(onChange).toHaveBeenCalledWith({ a: 2 })
  })
})

