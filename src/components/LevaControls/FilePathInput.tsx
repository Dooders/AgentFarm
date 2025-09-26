import React, { useCallback, useEffect, useMemo, useState } from 'react'
import styled from 'styled-components'
import { ConfigInput, BaseInputProps, InputMetadata, ValidationRule } from './ConfigInput'
import { ValidationError } from '@/types/validation'

// File path input specific props
export interface FilePathProps extends Omit<BaseInputProps, 'value' | 'onChange' | 'placeholder'> {
  /** Current file path value */
  value: string | null
  /** Callback when value changes */
  onChange: (value: string) => void
  /** File selection mode */
  mode?: 'file' | 'directory' | 'save'
  /** File type filters (extensions without dots) */
  filters?: string[]
  /** Whether to show file browser button */
  showBrowser?: boolean
  /** Base directory for relative paths */
  baseDir?: string
  /** Whether to allow relative paths */
  allowRelative?: boolean
  /** Whether to validate file existence */
  validateExistence?: boolean
  /** Placeholder text */
  placeholder?: string
  /** Whether to use compact layout */
  compact?: boolean
}

// Styled components for FilePath input
const FilePathWrapper = styled.div`
  position: relative;
  margin: var(--leva-space-xs, 4px) 0;

  .filepath-container {
    display: flex;
    align-items: center;
    min-height: 28px;
    padding: 4px 8px;
    background: var(--leva-colors-elevation2, #2a2a2a);
    border-radius: var(--leva-radii-sm, 4px);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    transition: all 0.2s ease;
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');

    &:hover {
      border-color: var(--leva-colors-accent1, #666666);
    }

    &:focus-within {
      border-color: var(--leva-colors-accent2, #888888);
      box-shadow: 0 0 0 2px var(--leva-colors-accent1, #666666);
    }
  }

  .filepath-label {
    font-family: var(--leva-fonts-sans, 'Albertus');
    font-size: 11px;
    font-weight: 600;
    color: var(--leva-colors-highlight1, #ffffff);
    margin-right: var(--leva-space-sm, 8px);
    min-width: 120px;
    text-align: right;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .filepath-controls {
    display: flex;
    align-items: center;
    gap: var(--leva-space-sm, 8px);
    flex: 1;
  }

  .filepath-input {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 11px;
    color: var(--leva-colors-highlight1, #ffffff);
    background: var(--leva-colors-elevation3, #3a3a3a);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    border-radius: var(--leva-radii-xs, 2px);
    outline: none;
    padding: 2px 4px;
    min-height: 20px;
    transition: all 0.2s ease;
    flex: 1;

    &:hover {
      border-color: var(--leva-colors-accent1, #666666);
    }

    &:focus {
      border-color: var(--leva-colors-accent2, #888888);
      box-shadow: 0 0 0 1px var(--leva-colors-accent1, #666666);
    }

    &::placeholder {
      color: var(--leva-colors-accent2, #888888);
    }
  }

  .filepath-browser {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 20px;
    background: var(--leva-colors-elevation3, #3a3a3a);
    border: 1px solid var(--leva-colors-elevation2, #2a2a2a);
    border-radius: var(--leva-radii-xs, 2px);
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 12px;
    color: var(--leva-colors-accent2, #888888);

    &:hover {
      border-color: var(--leva-colors-accent1, #666666);
      background: var(--leva-colors-elevation1, #1a1a1a);
      color: var(--leva-colors-highlight1, #ffffff);
    }

    &:active {
      transform: scale(0.95);
    }
  }

  .filepath-info {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: var(--leva-colors-accent2, #888888);
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
    display: flex;
    align-items: center;
    gap: var(--leva-space-xs, 4px);
  }

  .filepath-status {
    font-size: 8px;
    padding: 1px 4px;
    border-radius: 2px;
    background: var(--leva-colors-elevation1, #1a1a1a);
  }

  .filepath-error {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: #ff6b6b;
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }

  .filepath-help {
    font-family: var(--leva-fonts-mono, 'JetBrains Mono');
    font-size: 9px;
    color: var(--leva-colors-accent2, #888888);
    margin-top: 2px;
    padding-left: 128px;
    line-height: 1.2;
  }
`

// File browser button component
const FileBrowserButton: React.FC<{
  mode: 'file' | 'directory' | 'save'
  filters?: string[]
  onSelect: (path: string) => void
  disabled?: boolean
}> = ({ mode, filters, onSelect, disabled = false }) => {
  const handleBrowse = useCallback(async () => {
    if (disabled) return

    try {
      // Use Electron API or browser file API
      if (window.electron) {
        const result = await window.electron.openFileDialog({
          mode,
          filters: filters?.map(ext => ({ name: ext.toUpperCase(), extensions: [ext] }))
        })

        if (result && result.filePaths && result.filePaths.length > 0) {
          onSelect(result.filePaths[0])
        }
      } else {
        // Fallback to browser file input
        const input = document.createElement('input')
        input.type = mode === 'directory' ? 'file' : 'file'
        input.webkitdirectory = mode === 'directory'

        if (filters && mode === 'file') {
          input.accept = filters.map(ext => `.${ext}`).join(',')
        }

        input.onchange = (e) => {
          const file = (e.target as HTMLInputElement).files?.[0]
          if (file) {
            onSelect(file.name) // Browser returns relative path
          }
        }

        input.click()
      }
    } catch (error) {
      console.error('File browser error:', error)
    }
  }, [mode, filters, onSelect, disabled])

  const getBrowserIcon = () => {
    switch (mode) {
      case 'directory': return '📁'
      case 'save': return '💾'
      default: return '📄'
    }
  }

  return (
    <div
      className="filepath-browser"
      onClick={handleBrowse}
      title={`Browse ${mode}`}
    >
      {getBrowserIcon()}
    </div>
  )
}

// File status indicator
const FileStatus: React.FC<{
  path: string
  mode: 'file' | 'directory' | 'save'
  validateExistence?: boolean
}> = ({ path, mode, validateExistence = false }) => {
  const [status, setStatus] = useState<'valid' | 'invalid' | 'checking' | 'none'>('none')

  useEffect(() => {
    const checkFileStatus = async () => {
      if (!validateExistence || !path) {
        setStatus('none')
        return
      }

      setStatus('checking')

      try {
        // Use Electron API or fs API to check file existence
        if (window.electron) {
          const exists = await window.electron.fileExists(path, mode === 'directory')
          setStatus(exists ? 'valid' : 'invalid')
        } else {
          // Browser fallback - we can't check file existence
          setStatus('valid')
        }
      } catch (error) {
        setStatus('invalid')
      }
    }

    checkFileStatus()
  }, [path, mode, validateExistence])

  if (!validateExistence || status === 'none') {
    return null
  }

  const getStatusInfo = () => {
    switch (status) {
      case 'valid': return { text: '✓', color: '#4ade80' }
      case 'invalid': return { text: '✗', color: '#ff6b6b' }
      case 'checking': return { text: '⟳', color: '#888888' }
      default: return { text: '?', color: '#888888' }
    }
  }

  const { text, color } = getStatusInfo()

  return (
    <span
      className="filepath-status"
      style={{ color }}
      title={status === 'checking' ? 'Checking file...' : `File ${status === 'valid' ? 'exists' : 'not found'}`}
    >
      {text}
    </span>
  )
}

/**
 * FilePathInput component for file/directory path selection
 *
 * Features:
 * - File browser integration (Electron/browser fallback)
 * - File type filtering
 * - Path validation
 * - Existence checking
 * - Relative/absolute path support
 * - Consistent theming
 */
export const FilePathInput: React.FC<FilePathProps> = ({
  path,
  label,
  value,
  onChange,
  disabled = false,
  error,
  help,
  metadata,
  mode = 'file',
  filters,
  showBrowser = true,
  baseDir,
  allowRelative = true,
  validateExistence = false,
  placeholder = 'Select file...',
  required = false,
  compact = false,
  className
}) => {
  // Handle path input change
  const handlePathChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value

    // Validate path format
    if (!allowRelative && !isAbsolutePath(inputValue)) {
      return
    }

    if (baseDir && inputValue.startsWith('./') && !inputValue.startsWith(baseDir)) {
      return
    }

    onChange(inputValue)
  }, [onChange, allowRelative, baseDir])

  // Handle file browser selection
  const handleFileSelect = useCallback((selectedPath: string) => {
    onChange(selectedPath)
  }, [onChange])

  // Validation rules
  const validationRules: ValidationRule[] = useMemo(() => [
    {
      name: 'valid_path_format',
      description: 'Path must be a valid file or directory path',
      validator: (val) => {
        if (!val || typeof val !== 'string') return false
        if (!allowRelative && !isAbsolutePath(val)) return false
        return true
      },
      errorMessage: 'Invalid path format',
      severity: 'error'
    },
    {
      name: 'file_exists',
      description: 'File or directory must exist',
      validator: async (val) => {
        if (!validateExistence || !val) return true

        try {
          if (window.electron) {
            return await window.electron.fileExists(val, mode === 'directory')
          }
          return true // Browser can't check file existence
        } catch {
          return false
        }
      },
      errorMessage: 'File or directory not found',
      severity: 'error'
    }
  ], [allowRelative, validateExistence, mode])

  // Enhanced metadata
  const enhancedMetadata: InputMetadata = useMemo(() => ({
    ...metadata,
    inputType: 'file',
    validationRules,
    tooltip: metadata?.tooltip || `Select ${mode} path${filters ? ` (${filters.join(', ')})` : ''}`
  }), [metadata, validationRules, mode, filters])

  // Error messages
  const errorMessages = useMemo(() => {
    if (!error) return []

    if (typeof error === 'string') {
      return [error]
    }

    if (Array.isArray(error)) {
      return error.map(e => typeof e === 'string' ? e : e.message || 'Validation error')
    }

    return []
  }, [error])

  // Path info display
  const pathInfo = useMemo(() => {
    if (!value) return null

    const isAbs = isAbsolutePath(value)
    const hasExt = value.includes('.')
    const extension = hasExt ? value.split('.').pop()?.toUpperCase() : null

    return { isAbs, hasExt, extension }
  }, [value])

  return (
    <FilePathWrapper className={className}>
      <div className="filepath-container">
        <label className="filepath-label">
          {label}
          {required && <span style={{ color: '#ff6b6b', marginLeft: '2px' }}>*</span>}
        </label>

        <div className="filepath-controls">
          <input
            type="text"
            className="filepath-input"
            value={value || ''}
            onChange={handlePathChange}
            disabled={disabled}
            placeholder={placeholder}
            title={enhancedMetadata.tooltip}
          />

          {showBrowser && (
            <FileBrowserButton
              mode={mode}
              filters={filters}
              onSelect={handleFileSelect}
              disabled={disabled}
            />
          )}
        </div>
      </div>

      <div className="filepath-info">
        {pathInfo && (
          <>
            <span>{pathInfo.isAbs ? 'Absolute' : 'Relative'}</span>
            {pathInfo.extension && (
              <span style={{ color: 'var(--leva-colors-accent1, #666666)' }}>
                .{pathInfo.extension}
              </span>
            )}
          </>
        )}
        <FileStatus
          path={value || ''}
          mode={mode}
          validateExistence={validateExistence}
        />
      </div>

      {help && (
        <div className="filepath-help" title={enhancedMetadata.tooltip}>
          {help}
        </div>
      )}

      {errorMessages.length > 0 && (
        <div className="filepath-error">
          {errorMessages.map((message, index) => (
            <div key={index}>• {message}</div>
          ))}
        </div>
      )}
    </FilePathWrapper>
  )
}

// Path validation functions
function isAbsolutePath(path: string): boolean {
  if (typeof window !== 'undefined' && window.electron) {
    // Electron can handle both Unix and Windows paths
    return path.startsWith('/') || /^[A-Za-z]:/.test(path)
  }

  // Browser environment - check for Unix absolute paths
  return path.startsWith('/')
}

// Extend window interface for Electron APIs
declare global {
  interface Window {
    electron?: {
      openFileDialog: (options: {
        mode: 'file' | 'directory' | 'save'
        filters?: { name: string; extensions: string[] }[]
      }) => Promise<{ filePaths: string[] }>
      fileExists: (path: string, isDirectory?: boolean) => Promise<boolean>
    }
  }
}

/**
 * Utility function to create FilePathInput with preset configuration
 */
export const createFilePathInput = (
  label: string,
  config: Partial<FilePathProps>
) => {
  return (props: FilePathProps) => (
    <FilePathInput
      label={label}
      {...config}
      {...props}
    />
  )
}

/**
 * Hook for managing file path state with validation
 */
export const useFilePathState = (
  initialValue: string = '',
  validator?: (value: string) => ValidationError[]
) => {
  const [value, setValue] = useState<string>(initialValue)
  const [errors, setErrors] = useState<ValidationError[]>([])

  const updateValue = useCallback((newValue: string) => {
    setValue(newValue)

    if (validator) {
      const validationErrors = validator(newValue)
      setErrors(validationErrors)
    }
  }, [validator])

  return { value, setValue: updateValue, errors }
}