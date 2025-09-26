import { ConfigPath, ConfigSection, ConfigFieldMetadata, ConfigValueFormatter, ConfigValueTransformer } from './config'
import { ValidationError } from './validation'
import { ValidationResult } from './validation'

// =====================================================
// Extended Leva Types for Configuration System
// =====================================================

// Base configuration input interface
export interface BaseConfigInput {
  path: ConfigPath
  value: any
  onChange: (value: any) => void
  label?: string
  description?: string
  disabled?: boolean
  hidden?: boolean
  transient?: boolean
  render?: (get: () => any, set: (value: any) => void) => React.ReactNode
}

// Leva configuration folder interface
export interface LevaConfigFolder {
  label: string
  children: LevaConfigInput[]
  collapsed?: boolean
  onToggle?: () => void
  description?: string
  icon?: string
  color?: string
}

// Leva configuration input interface
export interface LevaConfigInput extends BaseConfigInput {
  schema: any
  error?: string
  warning?: string
  settings?: LevaInputSettings
  formatter?: ConfigValueFormatter
  transformer?: ConfigValueTransformer
  validation?: {
    rules?: string[]
    debounceMs?: number
  }
  metadata?: ConfigFieldMetadata
}

// Leva input settings interface
export interface LevaInputSettings {
  // Numeric settings
  min?: number
  max?: number
  step?: number

  // Selection settings
  options?: string[] | { [key: string]: any }

  // String settings
  placeholder?: string
  maxLength?: number

  // Boolean settings
  label?: string

  // Display settings
  disabled?: boolean
  readonly?: boolean
  hidden?: boolean

  // Custom settings
  [key: string]: any
}

// =====================================================
// Custom Leva Component Props
// =====================================================

// Base configuration input component props
export interface ConfigInputProps extends BaseConfigInput {
  schema: any
  error?: string
  warning?: string
  settings?: LevaInputSettings
  formatter?: ConfigValueFormatter
  transformer?: ConfigValueTransformer
  validationResult?: ValidationResult
  metadata?: ConfigFieldMetadata
  onValidation?: (result: ValidationResult) => void
}

// Number input specific props
export interface NumberInputProps extends ConfigInputProps {
  min?: number
  max?: number
  step?: number
  precision?: number
  suffix?: string
  prefix?: string
  slider?: boolean
  logScale?: boolean
}

// Boolean input specific props
export interface BooleanInputProps extends ConfigInputProps {
  trueLabel?: string
  falseLabel?: string
  toggle?: boolean
  switch?: boolean
}

// String input specific props
export interface StringInputProps extends ConfigInputProps {
  placeholder?: string
  maxLength?: number
  multiline?: boolean
  rows?: number
  pattern?: RegExp
  inputType?: 'text' | 'password' | 'email' | 'url' | 'search' | 'tel'
}

// Select input specific props
export interface SelectInputProps extends ConfigInputProps {
  options: string[] | { [key: string]: any }
  multiple?: boolean
  searchable?: boolean
  clearable?: boolean
  placeholder?: string
}

// Color input specific props
export interface ColorInputProps extends ConfigInputProps {
  format?: 'hex' | 'rgb' | 'hsl' | 'hsv'
  alpha?: boolean
  picker?: 'sketch' | 'chrome' | 'compact' | 'wheel'
}

// Vector input specific props
export interface VectorInputProps extends ConfigInputProps {
  dimensions: number
  min?: number
  max?: number
  step?: number
  precision?: number
  suffix?: string
  labels?: string[]
}

// File input specific props
export interface FileInputProps extends ConfigInputProps {
  accept?: string
  multiple?: boolean
  directory?: boolean
  maxSize?: number
  onFileSelect?: (files: FileList) => void
}

// Image input specific props
export interface ImageInputProps extends ConfigInputProps {
  maxWidth?: number
  maxHeight?: number
  quality?: number
  format?: 'jpeg' | 'png' | 'webp'
  onImageSelect?: (image: HTMLImageElement) => void
}

// Range input specific props
export interface RangeInputProps extends ConfigInputProps {
  min: number
  max: number
  step?: number
  showValue?: boolean
  showRange?: boolean
  formatValue?: (value: number) => string
}

// Folder props for organizing controls
export interface ConfigFolderProps {
  label: string
  description?: string
  collapsed?: boolean
  children: React.ReactNode
  onToggle?: () => void
  icon?: string
  color?: string
  indent?: number
  expandable?: boolean
}

// Panel props for organizing sections
export interface ConfigPanelProps {
  title: string
  description?: string
  children: React.ReactNode
  collapsible?: boolean
  collapsed?: boolean
  onToggle?: () => void
  icon?: string
  actions?: React.ReactNode
  className?: string
}

// =====================================================
// Layout Component Props
// =====================================================

// Resizable panels props
export interface ResizablePanelsProps {
  leftPanel: React.ReactNode
  rightPanel: React.ReactNode
  leftPanelWidth: number
  rightPanelWidth: number
  onResize: (leftWidth: number, rightWidth: number) => void
  minLeftWidth?: number
  minRightWidth?: number
  maxLeftWidth?: number
  maxRightWidth?: number
  orientation?: 'horizontal' | 'vertical'
  className?: string
  disabled?: boolean
}

// Left panel props
export interface LeftPanelProps {
  sections: ConfigSection[]
  selectedSection: ConfigSection
  expandedSections: Set<ConfigSection>
  onSectionSelect: (section: ConfigSection) => void
  onSectionToggle: (section: ConfigSection) => void
  onTemplateLoad?: (templateName: string) => void
  className?: string
  width?: number
}

// Right panel props
export interface RightPanelProps {
  selectedSection: ConfigSection
  comparisonMode?: boolean
  compareConfig?: any
  onComparisonToggle?: () => void
  className?: string
  width?: number
}

// =====================================================
// Control Component Props
// =====================================================

// Configuration explorer props
export interface ConfigExplorerProps {
  config: any
  onChange: (path: ConfigPath, value: any) => void
  validationErrors?: ValidationError[]
  sections?: ConfigSection[]
  expandedSections?: Set<ConfigSection>
  comparisonMode?: boolean
  compareConfig?: any
  onComparisonToggle?: () => void
  onSectionToggle?: (section: ConfigSection) => void
  onTemplateLoad?: (templateName: string) => void
  className?: string
}

// Leva controls props
export interface LevaControlsProps {
  config: any
  onChange: (path: ConfigPath, value: any) => void
  schema?: any
  validationErrors?: ValidationError[]
  settings?: Record<string, LevaInputSettings>
  folders?: LevaConfigFolder[]
  collapsed?: boolean
  hidden?: boolean
  theme?: any
  className?: string
}

// =====================================================
// Form Component Props
// =====================================================

// Configuration form props
export interface ConfigFormProps {
  config: any
  onChange: (updates: Record<ConfigPath, any>) => void
  onSubmit?: (config: any) => void
  onReset?: () => void
  validationErrors?: ValidationError[]
  submitButtonText?: string
  resetButtonText?: string
  disabled?: boolean
  loading?: boolean
  className?: string
}

// Field group props
export interface FieldGroupProps {
  title: string
  description?: string
  children: React.ReactNode
  collapsible?: boolean
  collapsed?: boolean
  onToggle?: () => void
  error?: string
  className?: string
}

// =====================================================
// Display Component Props
// =====================================================

// Configuration summary props
export interface ConfigSummaryProps {
  config: any
  sections?: ConfigSection[]
  showComparison?: boolean
  compareConfig?: any
  onFieldClick?: (path: ConfigPath) => void
  className?: string
}

// Value display props
export interface ValueDisplayProps {
  value: any
  path: ConfigPath
  formatter?: ConfigValueFormatter
  showRaw?: boolean
  className?: string
}

// Validation display props
export interface ValidationDisplayProps {
  errors: ValidationError[]
  warnings: ValidationError[]
  path?: ConfigPath
  compact?: boolean
  showIcons?: boolean
  className?: string
}

// =====================================================
// Utility Component Props
// =====================================================

// Loading indicator props
export interface LoadingIndicatorProps {
  size?: 'small' | 'medium' | 'large'
  color?: string
  message?: string
  overlay?: boolean
  className?: string
}

// Error boundary props
export interface ErrorBoundaryProps {
  children: React.ReactNode
  fallback?: React.ComponentType<{ error: Error; resetError: () => void }>
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
  className?: string
}

// Tooltip props
export interface TooltipProps {
  content: React.ReactNode
  children: React.ReactNode
  placement?: 'top' | 'right' | 'bottom' | 'left'
  delay?: number
  className?: string
}

// Modal props
export interface ModalProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  children: React.ReactNode
  size?: 'small' | 'medium' | 'large' | 'fullscreen'
  closeOnOverlayClick?: boolean
  closeOnEscape?: boolean
  className?: string
}

// =====================================================
// Advanced Component Props
// =====================================================

// Configuration history props
export interface ConfigHistoryProps {
  history: Array<{ id: string; config: any; timestamp: number; action: string }>
  currentIndex: number
  onSelect: (index: number) => void
  onUndo?: () => void
  onRedo?: () => void
  onClear?: () => void
  className?: string
}

// Template manager props
export interface TemplateManagerProps {
  templates: Array<{ name: string; description: string; category: string }>
  onLoad: (templateName: string) => void
  onSave?: (template: { name: string; description: string; category: string }) => void
  onDelete?: (templateName: string) => void
  categories?: string[]
  className?: string
}

// Export/Import props
export interface ConfigExportImportProps {
  config: any
  onImport: (config: any) => void
  formats?: Array<'json' | 'yaml' | 'xml'>
  exportFileName?: string
  className?: string
}