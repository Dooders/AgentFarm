// Extended Leva types for our configuration system
export interface LevaConfigFolder {
  label: string
  children: LevaConfigInput[]
  collapsed?: boolean
  onToggle?: () => void
}

export interface LevaConfigInput {
  path: string
  value: any
  schema: any
  onChange: (value: any) => void
  error?: string
  label?: string
  settings?: LevaInputSettings
}

export interface LevaInputSettings {
  min?: number
  max?: number
  step?: number
  options?: string[] | { [key: string]: any }
  placeholder?: string
  disabled?: boolean
}

// Custom Leva component props
export interface ConfigInputProps {
  path: string
  value: any
  schema: any
  onChange: (value: any) => void
  error?: string
  label?: string
  settings?: LevaInputSettings
}

// Number input specific props
export interface NumberInputProps extends ConfigInputProps {
  min?: number
  max?: number
  step?: number
}

// Boolean input specific props
export interface BooleanInputProps extends ConfigInputProps {
  // No additional props needed for boolean inputs
}

// String input specific props
export interface StringInputProps extends ConfigInputProps {
  placeholder?: string
  maxLength?: number
}

// Select input specific props
export interface SelectInputProps extends ConfigInputProps {
  options: string[] | { [key: string]: any }
}

// Folder props for organizing controls
export interface ConfigFolderProps {
  label: string
  collapsed?: boolean
  children: React.ReactNode
  onToggle?: () => void
}