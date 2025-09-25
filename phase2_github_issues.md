# Phase 2: Leva Controls & Theme - GitHub Issues

## Overview
Phase 2 focuses on implementing the complete Leva control system with custom theming, validation display, and professional greyscale styling. This phase builds upon the core architecture to create the actual user interface for configuration editing.

## Epic: Leva Controls & Professional Theme
**Labels:** phase-2, leva, theming, epic

### Issue #9: Implement Complete Leva Folder Structure
**Labels:** phase-2, leva, ui
**Priority:** High
**Effort:** 4 points

**Description:**
Create the hierarchical Leva folder structure mapping all simulation configuration sections as specified in the design document.

**Tasks:**
- Implement Environment folder with sub-folders:
  - World Settings (width, height, position discretization, interpolation)
  - Population (agent counts, ratios, resource levels)
  - Resource Management (regeneration, limits, consumption rates)
- Create Agent Behavior folder with sub-folders:
  - Movement Parameters (target update frequency, memory, learning rates)
  - Gathering Parameters (efficiency, costs, rewards, penalties)
  - Combat Parameters (attack/defense mechanics, health system)
  - Sharing Parameters (cooperation, altruism, social interactions)
- Add Learning & AI folder with sub-folders:
  - General Learning (learning rates, epsilon decay, memory settings)
  - Module-Specific Learning (individual parameter sets for each behavior module)
- Implement Visualization folder with sub-folders:
  - Display Settings (canvas size, colors, scaling)
  - Animation Settings (frame limits, delays)
  - Metrics Display (color schemes, font settings)

**Acceptance Criteria:**
- ✅ All configuration sections are organized in logical Leva folders
- ✅ Folder hierarchy matches design specification
- ✅ Folders can be collapsed/expanded
- ✅ Configuration values are properly bound to folders
- ✅ No missing parameters or orphaned controls

**Dependencies:** Phase 1 completion
**Estimated Time:** 3-4 days

---

### Issue #10: Apply Professional Greyscale Theme
**Labels:** phase-2, theming, design
**Priority:** High
**Effort:** 3 points

**Description:**
Implement the complete greyscale professional theme with custom typography, colors, and styling as specified in the design document.

**Tasks:**
- Implement CSS custom properties for color palette:
  - Slate colors (50-900) for primary UI elements
  - Stone colors (50-900) for depth and accents
  - Semantic colors for backgrounds, borders, text
- Apply Albertus font for all labels and UI text (12px)
- Apply JetBrains Mono font for numbers and metrics (11px)
- Set control heights to exactly 28px (compact design requirement)
- Implement subtle borders and muted contrast
- Create theme provider component for global theme management
- Add CSS overrides for Leva's default styling
- Implement high-contrast monochrome focus rings (no neon colors)

**Acceptance Criteria:**
- ✅ Greyscale color palette is fully applied
- ✅ Typography uses Albertus for labels, JetBrains Mono for numbers
- ✅ All controls are exactly 28px in height
- ✅ Focus states use high-contrast monochrome rings
- ✅ Overall aesthetic is professional and cinematic
- ✅ No bright or colorful elements remain

**Dependencies:** Phase 1 completion
**Estimated Time:** 2-3 days

---

### Issue #11: Create Type-Specific Input Components
**Labels:** phase-2, components, validation
**Priority:** High
**Effort:** 4 points

**Description:**
Build custom input components for different data types with proper validation, styling, and behavior matching the design specifications.

**Tasks:**
- Create `NumberInput.tsx` component:
  - JetBrains Mono font styling
  - Min/max/step validation
  - Proper number parsing and formatting
  - Scientific notation support for small numbers
- Implement `BooleanInput.tsx` component:
  - Compact checkbox styling (28px height)
  - Clear visual states
  - Keyboard accessibility
- Build `SelectInput.tsx` component:
  - Enum value dropdowns
  - Proper option rendering
  - Search/filter functionality for large enums
- Create `ObjectInput.tsx` component:
  - JSON editor with syntax highlighting
  - Collapsible object display
  - Validation error display
- Implement `ArrayInput.tsx` component:
  - Dynamic array editing
  - Add/remove item controls
  - Type-specific array element inputs
- Add `RangeInput.tsx` component:
  - Slider controls for numeric ranges
  - Dual handle range selection
  - Real-time value updates

**Acceptance Criteria:**
- ✅ All input types render correctly with proper styling
- ✅ Validation errors are displayed appropriately
- ✅ Typography matches design specifications
- ✅ Controls are keyboard accessible
- ✅ Components handle edge cases gracefully
- ✅ Performance is acceptable with many controls

**Dependencies:** Phase 1 completion
**Estimated Time:** 3-4 days

---

### Issue #12: Implement Validation Display System
**Labels:** phase-2, validation, ui
**Priority:** High
**Effort:** 3 points

**Description:**
Create a comprehensive validation display system that shows real-time errors, warnings, and validation feedback to users.

**Tasks:**
- Implement `ValidationDisplay.tsx` component:
  - Field-specific error messages
  - Warning indicators for non-critical issues
  - Success confirmations for valid fields
- Create validation state management:
  - Real-time validation on field changes
  - Debounced validation to avoid performance issues
  - Cross-field validation for dependent parameters
- Build validation error formatting:
  - User-friendly error messages
  - Technical details for advanced users
  - Contextual help and suggestions
- Add visual validation indicators:
  - Color-coded error states (subtle greyscale)
  - Field highlighting for errors
  - Inline validation messages
- Implement validation summary panel:
  - Overview of all validation issues
  - Quick navigation to problematic fields
  - Validation statistics and counts

**Acceptance Criteria:**
- ✅ Real-time validation works smoothly
- ✅ Error messages are clear and helpful
- ✅ Visual indicators use appropriate greyscale colors
- ✅ Validation doesn't impact performance
- ✅ Users can easily identify and fix issues
- ✅ Cross-field validation works correctly

**Dependencies:** Phase 1 completion (Zod validation)
**Estimated Time:** 2-3 days

---

### Issue #13: Add Focus States and Accessibility Features
**Labels:** phase-2, accessibility, ux
**Priority:** Medium
**Effort:** 2 points

**Description:**
Implement comprehensive accessibility features including keyboard navigation, focus management, and ARIA compliance.

**Tasks:**
- Implement high-contrast focus rings:
  - Monochrome focus indicators (no neon)
  - Proper focus ring sizing and positioning
  - Focus trapping in modal dialogs
- Add keyboard navigation:
  - Arrow key navigation through folder structure
  - Tab order through all interactive elements
  - Enter/Space activation for buttons and folders
  - Escape to close expanded sections
- Create ARIA compliance:
  - Proper ARIA roles and labels
  - Screen reader announcements for changes
  - Live regions for dynamic content updates
- Implement skip navigation links:
  - Skip to main content functionality
  - Skip to validation errors
  - Skip to comparison panel
- Add color contrast compliance:
  - Ensure all text meets WCAG AA standards
  - High contrast mode support
  - Focus indicators visible without color alone

**Acceptance Criteria:**
- ✅ All interactive elements have proper focus indicators
- ✅ Keyboard navigation works throughout the interface
- ✅ Screen readers can navigate and understand the interface
- ✅ ARIA labels and roles are properly implemented
- ✅ Color contrast meets accessibility standards
- ✅ No accessibility-related console warnings

**Dependencies:** Phase 2 core issues
**Estimated Time:** 1-2 days

---

### Issue #14: Create Leva Custom Controls Integration
**Labels:** phase-2, leva, components
**Priority:** Medium
**Effort:** 3 points

**Description:**
Build custom Leva controls that integrate seamlessly with the existing control system while providing enhanced functionality.

**Tasks:**
- Create `LevaFolder.tsx` wrapper component:
  - Custom folder styling with greyscale theme
  - Collapsible/expandable functionality
  - Icon support for different section types
- Implement `ConfigInput.tsx` base component:
  - Unified interface for all input types
  - Consistent error handling and display
  - Theme integration
- Build specialized control components:
  - `Vector2Input.tsx` for coordinate inputs
  - `ColorInput.tsx` for color parameters (greyscale compatible)
  - `FilePathInput.tsx` for file/directory paths
  - `PercentageInput.tsx` for ratio/percentage values
- Add control metadata system:
  - Help text and tooltips
  - Units and formatting hints
  - Validation rule descriptions
- Implement control grouping:
  - Related parameter grouping
  - Visual separation of control groups
  - Consistent spacing and layout

**Acceptance Criteria:**
- ✅ Custom controls integrate with Leva seamlessly
- ✅ All controls follow the greyscale theme
- ✅ Enhanced functionality works correctly
- ✅ Metadata and help systems are functional
- ✅ Performance remains smooth with many controls
- ✅ Controls are reusable and maintainable

**Dependencies:** Phase 1 completion
**Estimated Time:** 2-3 days

---

## Phase 2 Summary

**Total Issues:** 6
**Total Effort Points:** 19
**Estimated Timeline:** 2-3 weeks
**Critical Path:** Issues #9 → #10 → #11 → #12

### Success Criteria for Phase 2 Completion:
- ✅ Complete Leva folder structure implemented
- ✅ Professional greyscale theme fully applied
- ✅ All type-specific input components working
- ✅ Real-time validation display system operational
- ✅ Accessibility features implemented and tested
- ✅ Custom Leva controls integrated and functional

### Dependencies Between Issues:
- Issues #9, #10, #11, #12, #14 can be worked in parallel
- Issue #13 (accessibility) depends on completion of other Phase 2 issues
- All issues depend on Phase 1 completion

### Labels Reference:
- `phase-2`: All issues in this phase
- `leva`: Leva control system issues (#9, #11, #14)
- `theming`: Theme and styling issues (#10)
- `validation`: Validation display issues (#12)
- `accessibility`: Accessibility features (#13)
- `components`: Component development (#11, #14)
- `design`: Visual design implementation (#10)
- `ux`: User experience improvements (#13)

This phase transforms the basic architecture into a fully functional, professionally-styled configuration interface that users can interact with effectively.