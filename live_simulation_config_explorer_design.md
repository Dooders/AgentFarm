# Live Simulation Config Explorer - Design Specification

## Overview

This document outlines the design and implementation plan for a **Live Simulation Config Explorer** built with React + TypeScript + Leva. The application provides a dual-side editing interface for exploring and editing simulation configurations with live binding, validation, and professional greyscale theming.

## Project Context

The existing codebase contains a sophisticated simulation framework with complex configuration structures. This new explorer will replace the existing vanilla JavaScript implementation with a modern React-based solution that maintains the same functionality while providing enhanced UX and developer experience.

## Core Requirements

### Technology Stack
- **React 18+** with TypeScript
- **Leva** for control panels and live binding
- **Electron** for desktop application
- **Zustand** for state management
- **Zod** for schema validation
- **Vite** for build tooling

### Design Philosophy
- **Single Responsibility Principle**: Each component handles one concern
- **Composition over Inheritance**: Flexible component composition
- **Clean Architecture**: Clear separation of concerns
- **Minimalist Design**: Compact, intuitive interface
- **Professional Aesthetic**: Greyscale theme with cinematic feel

## Architecture Design

### High-Level Structure

```
src/
├── components/
│   ├── ConfigExplorer/
│   │   ├── ConfigExplorer.tsx          # Main container
│   │   ├── LeftPanel.tsx               # Leva controls panel
│   │   ├── RightPanel.tsx              # Preview/validation panel
│   │   ├── DualPanelLayout.tsx         # Split view layout
│   │   └── ComparisonView.tsx          # Side-by-side comparison
│   ├── Controls/
│   │   ├── LevaFolder.tsx              # Custom Leva folder wrapper
│   │   ├── ConfigInput.tsx             # Type-specific input components
│   │   └── ValidationDisplay.tsx       # Error/warning display
│   ├── Layout/
│   │   ├── ResizablePanels.tsx         # Split panel implementation
│   │   ├── Toolbar.tsx                 # Main toolbar
│   │   └── StatusBar.tsx               # Save/validation status
│   └── UI/
│       ├── ThemeProvider.tsx           # Greyscale theme provider
│       ├── Typography.tsx              # Font specifications
│       └── FocusRing.tsx              # High-contrast focus states
├── stores/
│   ├── configStore.ts                  # Zustand state management
│   ├── levaStore.ts                    # Leva integration store
│   └── validationStore.ts              # Validation state
├── services/
│   ├── configService.ts                # Config file operations
│   ├── validationService.ts            # Zod schema validation
│   ├── ipcService.ts                   # Electron IPC communication
│   └── exportService.ts                # Config export functionality
├── types/
│   ├── config.ts                       # TypeScript config definitions
│   ├── validation.ts                   # Validation types
│   └── leva.ts                         # Leva-specific types
├── hooks/
│   ├── useConfig.ts                    # Config state hooks
│   ├── useLeva.ts                      # Leva integration hooks
│   └── useValidation.ts                # Validation hooks
└── styles/
    ├── theme.css                       # Greyscale theme
    ├── leva-overrides.css              # Leva customizations
    └── layout.css                      # Layout styles
```

### State Management (Zustand)

```typescript
interface ConfigStore {
  // Primary config state
  config: SimulationConfig
  originalConfig: SimulationConfig
  isDirty: boolean

  // Comparison state
  compareConfig: SimulationConfig | null
  showComparison: boolean

  // UI state
  selectedSection: string
  expandedFolders: Set<string>
  validationErrors: ValidationError[]

  // Actions
  updateConfig: (path: string, value: any) => void
  loadConfig: (filePath: string) => Promise<void>
  saveConfig: (filePath?: string) => Promise<void>
  setComparison: (config: SimulationConfig | null) => void
  toggleSection: (section: string) => void
  validateConfig: () => void
}
```

## Leva Integration Design

### Folder Structure Mapping

Based on the simulation config structure, Leva folders will be organized as:

#### Environment
- **World Settings**: width, height, position discretization, interpolation
- **Population**: agent counts, ratios, resource levels
- **Resource Management**: regeneration, limits, consumption rates

#### Agent Behavior
- **Movement Parameters**: target update frequency, memory, learning rates
- **Gathering Parameters**: efficiency, costs, rewards, penalties
- **Combat Parameters**: attack/defense mechanics, health system
- **Sharing Parameters**: cooperation, altruism, social interactions

#### Learning & AI
- **General Learning**: learning rates, epsilon decay, memory settings
- **Module-Specific Learning**: individual parameter sets for each behavior module

#### Visualization
- **Display Settings**: canvas size, colors, scaling
- **Animation Settings**: frame limits, delays
- **Metrics Display**: color schemes, font settings

### Leva Control Specifications

#### Control Heights & Density
- **Height**: 28px (compact design requirement)
- **Spacing**: Reduced padding for minimal layout
- **Borders**: Subtle 1px borders with muted contrast

#### Typography
- **Labels**: Albertus (John Carpenter font) - 12px
- **Numbers/Metrics**: JetBrains Mono - 11px
- **Focus States**: High-contrast monochrome rings, no neon

### Custom Leva Components

```typescript
// Custom folder component for section grouping
interface LevaConfigFolderProps {
  label: string
  children: React.ReactNode
  collapsed?: boolean
  onToggle?: () => void
}

// Type-specific input components
interface ConfigInputProps {
  path: string
  value: any
  schema: ZodType
  onChange: (value: any) => void
  error?: string
}

// Number input with JetBrains Mono
const NumberInput = ({ value, onChange, min, max, step }) => (
  <input
    type="number"
    value={value}
    onChange={(e) => onChange(Number(e.target.value))}
    style={{ fontFamily: 'JetBrains Mono', fontSize: '11px', height: '28px' }}
    min={min}
    max={max}
    step={step}
  />
)

// Boolean input with compact styling
const BooleanInput = ({ value, onChange }) => (
  <input
    type="checkbox"
    checked={value}
    onChange={(e) => onChange(e.target.checked)}
    style={{ height: '28px', width: '28px' }}
  />
)
```

## Dual Side Editing Design

### Layout Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Toolbar                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌───────────────────────────────────┐  │
│  │   Section List  │  │      Primary Config Panel         │  │
│  │   (Navigation)  │  │  (Leva Controls + Validation)     │  │
│  │                 │  │                                   │  │
│  │  • Environment  │  │  ┌─────────┐ ┌─────────────────┐   │  │
│  │  • Agent Behavior│  │  │ Controls │ │   YAML Preview  │   │  │
│  │  • Learning     │  │  └─────────┘ └─────────────────┘   │  │
│  │  • Visualization│  │                                   │  │
│  │  • ...          │  │                                   │  │
│  └─────────────────┘  └───────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────┐  ┌───────────────────────────────────┐  │
│  │   Comparison    │  │      Secondary Config Panel       │  │
│  │   Config Panel  │  │  (Read-only comparison view)      │  │
│  │                 │  │                                   │  │
│  │  (Collapsible)  │  │                                   │  │
│  └─────────────────┘  └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Panel Components

#### Primary Panel (Left Side)
- **Navigation Tree**: Collapsible sections with icons
- **Leva Controls**: Live-editable parameters
- **Validation Display**: Real-time error feedback
- **YAML Preview**: Live YAML representation

#### Comparison Panel (Right Side)
- **Toggleable**: Can be shown/hidden
- **Read-only**: Displays comparison config
- **Diff Highlighting**: Visual differences from primary
- **Copy Controls**: One-click field copying

#### Resizable Layout
- **Vertical Split**: Between navigation and main content
- **Horizontal Split**: Between controls and preview
- **Draggable Handles**: Smooth resize interactions

## Theme Design

### Color Palette

```css
/* Greyscale Professional Palette */
:root {
  /* Base Colors */
  --slate-50: #f8fafc;
  --slate-100: #f1f5f9;
  --slate-200: #e2e8f0;
  --slate-300: #cbd5e1;
  --slate-400: #94a3b8;
  --slate-500: #64748b;
  --slate-600: #475569;
  --slate-700: #334155;
  --slate-800: #1e293b;
  --slate-900: #0f172a;

  /* Stone Variants for Depth */
  --stone-50: #fafaf9;
  --stone-100: #f5f5f4;
  --stone-200: #e7e5e4;
  --stone-300: #d6d3d1;
  --stone-400: #a8a29e;
  --stone-500: #78716c;
  --stone-600: #57534e;
  --stone-700: #44403c;
  --stone-800: #292524;
  --stone-900: #1c1917;

  /* Semantic Colors */
  --background-primary: var(--slate-50);
  --background-secondary: var(--slate-100);
  --background-tertiary: var(--stone-100);
  --border-subtle: var(--slate-200);
  --border-medium: var(--slate-300);
  --text-primary: var(--slate-900);
  --text-secondary: var(--slate-600);
  --text-muted: var(--slate-500);

  /* Focus & Interaction */
  --focus-ring: var(--slate-800);
  --focus-ring-hover: var(--slate-900);
  --accent-primary: var(--slate-700);
  --accent-hover: var(--slate-800);

  /* Validation States */
  --error-bg: var(--stone-100);
  --error-border: var(--stone-400);
  --error-text: var(--stone-800);
  --warning-bg: var(--slate-100);
  --warning-border: var(--slate-400);
  --warning-text: var(--slate-700);
  --success-bg: var(--slate-100);
  --success-border: var(--slate-400);
  --success-text: var(--slate-600);
}
```

### Typography System

```css
/* Typography Hierarchy */
:root {
  /* Font Families */
  --font-label: 'Albertus', serif;
  --font-mono: 'JetBrains Mono', monospace;
  --font-ui: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;

  /* Font Sizes */
  --text-xs: 11px;
  --text-sm: 12px;
  --text-base: 14px;
  --text-lg: 16px;
  --text-xl: 18px;

  /* Line Heights */
  --leading-tight: 1.25;
  --leading-normal: 1.5;
  --leading-relaxed: 1.75;
}

/* Specific Typography Classes */
.leva-c-control__label {
  font-family: var(--font-label);
  font-size: var(--text-sm);
  color: var(--text-primary);
}

.leva-c-input__input {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

.leva-c-button {
  font-family: var(--font-label);
  font-size: var(--text-sm);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  background: var(--background-secondary);
}

.leva-c-button:hover {
  background: var(--background-tertiary);
  border-color: var(--border-medium);
}

/* Focus States */
.leva-c-input__input:focus,
.leva-c-button:focus,
.leva-c-select__select:focus {
  outline: 2px solid var(--focus-ring);
  outline-offset: 1px;
  border-color: var(--focus-ring-hover);
}
```

## Validation System

### Zod Schema Structure

```typescript
// Main configuration schema
const SimulationConfigSchema = z.object({
  // Environment settings
  width: z.number().min(10).max(1000),
  height: z.number().min(10).max(1000),
  position_discretization_method: z.enum(['floor', 'round', 'ceil']),
  use_bilinear_interpolation: z.boolean(),

  // Agent settings
  system_agents: z.number().int().min(0).max(1000),
  independent_agents: z.number().int().min(0).max(1000),
  control_agents: z.number().int().min(0).max(1000),
  agent_type_ratios: z.object({
    SystemAgent: z.number().min(0).max(1),
    IndependentAgent: z.number().min(0).max(1),
    ControlAgent: z.number().min(0).max(1)
  }).refine(data => {
    const sum = Object.values(data).reduce((a, b) => a + b, 0)
    return Math.abs(sum - 1.0) < 0.001
  }, 'Agent type ratios must sum to 1.0'),

  // Learning parameters with validation
  learning_rate: z.number().min(0.0001).max(1.0),
  epsilon_start: z.number().min(0).max(1),
  epsilon_min: z.number().min(0).max(1),
  epsilon_decay: z.number().min(0.9).max(0.999),

  // Nested object validation
  agent_parameters: z.object({
    SystemAgent: AgentParameterSchema,
    IndependentAgent: AgentParameterSchema,
    ControlAgent: AgentParameterSchema
  }),

  // Visualization with nested validation
  visualization: VisualizationConfigSchema,

  // Complex module parameter validation
  gather_parameters: ModuleParameterSchema,
  share_parameters: ModuleParameterSchema,
  move_parameters: ModuleParameterSchema,
  attack_parameters: ModuleParameterSchema
})

// Individual module parameter schema
const ModuleParameterSchema = z.object({
  target_update_freq: z.number().int().min(1),
  memory_size: z.number().int().min(100),
  learning_rate: z.number().min(0.0001).max(1.0),
  gamma: z.number().min(0.1).max(0.999),
  epsilon_start: z.number().min(0).max(1),
  epsilon_min: z.number().min(0).max(1),
  epsilon_decay: z.number().min(0.9).max(0.999),
  dqn_hidden_size: z.number().int().min(8).max(256),
  batch_size: z.number().int().min(1).max(512),
  tau: z.number().min(0.001).max(0.1),
  // Module-specific parameters with custom validation
  success_reward: z.number().min(-1).max(10),
  failure_penalty: z.number().min(-10).max(1),
  base_cost: z.number().min(-5).max(5)
})
```

### Validation Integration

- **Real-time Validation**: Zod schemas validate on every change
- **Contextual Errors**: Field-specific error messages
- **Visual Feedback**: Color-coded validation states
- **Schema-driven UI**: Input types and constraints from schema
- **Cross-field Validation**: Complex business logic validation

## Implementation Phases

### Phase 1: Core Architecture
1. Set up React + TypeScript + Vite project structure
2. Implement Zustand stores for state management
3. Create basic Leva integration with custom controls
4. Implement Zod schema validation system
5. Build basic dual-panel layout structure

### Phase 2: Leva Controls & Theme
1. Implement all Leva folder structures and controls
2. Apply greyscale theme with custom typography
3. Create type-specific input components
4. Implement validation display components
5. Add focus states and accessibility features

### Phase 3: Dual Side Editing
1. Implement comparison panel functionality
2. Add diff highlighting and copy controls
3. Create resizable panel layout
4. Implement YAML preview with live updates
5. Add toolbar and status bar components

### Phase 4: Advanced Features
1. File operations (load, save, export)
2. Preset system for configuration templates
3. Search and filtering capabilities
4. Validation rule customization
5. Performance optimization and caching

### Phase 5: Integration & Polish
1. Electron integration with IPC communication
2. Testing and validation
3. Performance optimization
4. Documentation and user guide
5. Migration from existing implementation

## Success Criteria

### Functional Requirements
- ✅ All simulation config parameters editable via Leva
- ✅ Live validation with Zod schemas
- ✅ Dual-side comparison and editing
- ✅ YAML preview with real-time updates
- ✅ File operations (load/save/export)
- ✅ Professional greyscale theming applied
- ✅ Compact 28px control heights
- ✅ Albertus/ JetBrains Mono typography

### Technical Requirements
- ✅ TypeScript with full type safety
- ✅ Zustand state management
- ✅ React 18+ with modern patterns
- ✅ Leva custom controls and integration
- ✅ Zod schema validation
- ✅ Electron IPC communication
- ✅ Responsive resizable layout
- ✅ Accessibility compliance

### Design Requirements
- ✅ Greyscale professional theme
- ✅ Cinematic, understated aesthetic
- ✅ High-contrast monochrome focus states
- ✅ Compact, minimal layout
- ✅ Intuitive folder-based organization
- ✅ Clean typography hierarchy

## Risk Assessment

### Technical Risks
- **Leva Customization Complexity**: May require deep customization of Leva internals
- **Schema Validation Performance**: Large schemas may impact real-time validation
- **State Management Complexity**: Dual-panel state synchronization challenges
- **TypeScript Integration**: Complex type definitions for dynamic config structures

### Mitigation Strategies
- **Modular Architecture**: Clear separation of concerns for easier debugging
- **Performance Optimization**: Memoization and selective validation
- **Fallback Patterns**: Graceful degradation for edge cases
- **Comprehensive Testing**: Unit tests for validation and state management
- **Documentation**: Clear architecture documentation for maintenance

## Conclusion

This design specification provides a comprehensive blueprint for building a professional-grade simulation configuration explorer. The combination of React + TypeScript + Leva with the specified greyscale theme and dual-side editing capabilities will create a powerful yet intuitive interface for simulation configuration management.

The phased implementation approach ensures manageable development while maintaining the high-quality standards required for a professional desktop application.