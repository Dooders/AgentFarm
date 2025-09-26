# Leva Controls Integration

This directory contains the implementation of the basic Leva integration for the Live Simulation Config Explorer.

## Components

### LevaControls
The main Leva controls component that integrates with the Zustand stores and provides the control panel interface.

**Features:**
- Real-time synchronization with configuration state
- Custom theme configuration
- Panel visibility and collapse controls
- Value binding between Leva controls and Zustand config store

### NumberInput
Custom number input component with min/max validation and step controls.

**Props:**
- `value`: Current number value
- `onChange`: Callback for value changes
- `min`: Minimum allowed value (optional)
- `max`: Maximum allowed value (optional)
- `step`: Step increment (default: 1)
- `label`: Display label
- `error`: Error message (optional)
- `disabled`: Whether the input is disabled

### BooleanInput
Custom boolean input component (checkbox) with proper styling.

**Props:**
- `value`: Current boolean value
- `onChange`: Callback for value changes
- `label`: Display label
- `error`: Error message (optional)
- `disabled`: Whether the input is disabled

### StringInput
Custom string input component with max length validation.

**Props:**
- `value`: Current string value
- `onChange`: Callback for value changes
- `placeholder`: Placeholder text (optional)
- `maxLength`: Maximum character length (optional)
- `label`: Display label
- `error`: Error message (optional)
- `disabled`: Whether the input is disabled

### ConfigFolder
Organizes controls into collapsible folders for better UX.

**Props:**
- `label`: Folder display name
- `collapsed`: Initial collapsed state (default: false)
- `children`: Child components to render inside
- `onToggle`: Callback when folder is toggled

## Integration

The Leva controls are integrated with:
- **Zustand Config Store**: Real-time value synchronization
- **Zustand Leva Store**: Panel state management
- **Custom Theme System**: Consistent styling with the app

## Theme Configuration

Custom theme variables are defined in `src/styles/leva-theme.css`:
- Custom colors for elevation levels
- Typography using JetBrains Mono and Albertus fonts
- Consistent spacing and border radius
- Dark/light theme support

## Usage

```tsx
import { LevaControls } from '@/components/LevaControls'
import { useLevaStore } from '@/stores/levaStore'

function MyComponent() {
  const levaStore = useLevaStore()

  return (
    <div>
      <LevaControls />
      <button onClick={() => levaStore.setPanelVisible(!levaStore.isVisible)}>
        Toggle Leva Panel
      </button>
    </div>
  )
}
```

## State Management

The integration uses two main Zustand stores:

1. **Config Store**: Manages the simulation configuration state
2. **Leva Store**: Manages panel visibility, theme, and folder states

Values are synchronized bidirectionally:
- Changes in Leva controls update the config store
- Changes in config store update Leva controls
- Panel state is persisted across sessions