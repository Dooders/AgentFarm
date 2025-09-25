import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useValidationStore } from '../validationStore'
import { validationSelectors } from '../selectors'
import { ValidationError } from '@/types/validation'

describe('ValidationStore', () => {
  beforeEach(() => {
    // Reset the store before each test
    useValidationStore.getState().clearErrors()
    useValidationStore.getState().clearWarnings()
    useValidationStore.setState({
      isValidating: false,
      errors: [],
      warnings: [],
      lastValidationTime: 0
    })
  })

  it('initializes with empty validation state', () => {
    const { result } = renderHook(() => useValidationStore())

    expect(result.current.errors).toHaveLength(0)
    expect(result.current.warnings).toHaveLength(0)
    expect(result.current.isValidating).toBe(false)
    expect(result.current.lastValidationTime).toBe(0)
    expect(validationSelectors.getIsValid(result.current)).toBe(true)
    expect(validationSelectors.getHasErrors(result.current)).toBe(false)
    expect(validationSelectors.getHasWarnings(result.current)).toBe(false)
  })

  it('adds and removes errors correctly', () => {
    const { result } = renderHook(() => useValidationStore())

    const testError: ValidationError = {
      path: 'width',
      message: 'Width must be between 10 and 1000',
      code: 'invalid_range'
    }

    act(() => {
      result.current.addError(testError)
    })

    expect(result.current.errors).toHaveLength(1)
    expect(result.current.errors[0]).toEqual(testError)
    expect(validationSelectors.getHasErrors(result.current)).toBe(true)
    expect(validationSelectors.getIsValid(result.current)).toBe(false)
    expect(validationSelectors.getErrorCount(result.current)).toBe(1)

    act(() => {
      result.current.clearErrors()
    })

    expect(result.current.errors).toHaveLength(0)
    expect(validationSelectors.getHasErrors(result.current)).toBe(false)
    expect(validationSelectors.getIsValid(result.current)).toBe(true)
  })

  it('adds and removes warnings correctly', () => {
    const { result } = renderHook(() => useValidationStore())

    const testWarning: ValidationError = {
      path: 'learning_rate',
      message: 'Learning rate is very low, consider increasing',
      code: 'performance_warning'
    }

    act(() => {
      result.current.addWarning(testWarning)
    })

    expect(result.current.warnings).toHaveLength(1)
    expect(result.current.warnings[0]).toEqual(testWarning)
    expect(validationSelectors.getHasWarnings(result.current)).toBe(true)
    expect(validationSelectors.getWarningCount(result.current)).toBe(1)

    act(() => {
      result.current.clearWarnings()
    })

    expect(result.current.warnings).toHaveLength(0)
    expect(validationSelectors.getHasWarnings(result.current)).toBe(false)
  })

  it('sets validation result correctly', () => {
    const { result } = renderHook(() => useValidationStore())

    const errors: ValidationError[] = [
      { path: 'width', message: 'Invalid width', code: 'invalid_range' }
    ]
    const warnings: ValidationError[] = [
      { path: 'height', message: 'Height warning', code: 'performance' }
    ]

    act(() => {
      result.current.setValidationResult({
        success: false,
        errors,
        warnings
      })
    })

    expect(result.current.errors).toEqual(errors)
    expect(result.current.warnings).toEqual(warnings)
    expect(result.current.isValidating).toBe(false)
    expect(result.current.lastValidationTime).toBeGreaterThan(0)
  })

  it('handles field-specific error operations', () => {
    const { result } = renderHook(() => useValidationStore())

    const error1: ValidationError = { path: 'width', message: 'Error 1', code: 'error1' }
    const error2: ValidationError = { path: 'height', message: 'Error 2', code: 'error2' }

    act(() => {
      result.current.addError(error1)
      result.current.addError(error2)
    })

    expect(result.current.getFieldError('width')).toEqual(error1)
    expect(result.current.getFieldError('height')).toEqual(error2)
    expect(result.current.getFieldError('nonexistent')).toBeUndefined()

    expect(result.current.getFieldErrors('width')).toEqual([error1])
    expect(result.current.hasFieldError('width')).toBe(true)
    expect(result.current.hasFieldError('nonexistent')).toBe(false)

    act(() => {
      result.current.clearFieldErrors('width')
    })

    expect(result.current.getFieldError('width')).toBeUndefined()
    expect(result.current.getFieldError('height')).toEqual(error2)
  })

  it('validates field and handles success/error cases', async () => {
    const { result } = renderHook(() => useValidationStore())

    // Mock successful validation
    let validationResult: any
    await act(async () => {
      validationResult = await result.current.validateField('width', 100)
    })

    expect(validationResult?.success).toBe(true)
    expect(validationResult?.errors).toHaveLength(0)
    expect(result.current.isValidating).toBe(false)
  })

  it('updates validation time correctly', () => {
    const { result } = renderHook(() => useValidationStore())

    const beforeTime = Date.now()

    act(() => {
      result.current.updateLastValidationTime()
    })

    expect(result.current.lastValidationTime).toBeGreaterThanOrEqual(beforeTime)
  })
})