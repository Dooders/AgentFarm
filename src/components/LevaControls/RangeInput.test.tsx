import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { RangeInput } from './RangeInput'

describe('RangeInput', () => {
  it('renders and updates start/end', () => {
    const onChange = vi.fn()
    render(
      <RangeInput
        path="test.range"
        schema={{}}
        value={[0, 10]}
        onChange={onChange}
        min={0}
        max={20}
        label="Range"
      />
    )

    const sliders = screen.getAllByRole('slider')
    fireEvent.change(sliders[0], { target: { value: '2' } })
    expect(onChange).toHaveBeenCalledWith([2, 10])
  })
})

