## Config Explorer UI/UX Visual Refinement Report

### Scope and goals
- Refine alignment, layout, spacing, and visual style across the Config Explorer.
- Keep the grayscale aesthetic; improve consistency, legibility, and affordances.
- Minimize churn: prefer tokenization and component-level tweaks over large rewrites.

### Quick wins (high impact, low risk)
- **Unify spacing scale**: Adopt 4/8/12/16/24px tokens and apply to panel headers, sections, buttons, and stats to remove subtle misalignments.
- **Standardize border radii**: Use radius tokens (2/4/6/8px). Prefer 6px for most cards/sections to soften without appearing bubbly.
- **Normalize toolbar/status bar height**: Make both a consistent 40px–44px with identical vertical paddings and baselines.
- **Keyboard and focusable resizers**: Give split handles `role="separator"`, `aria-orientation`, and keyboard resizing with Arrow keys; add a visible focus ring.
- **Disable panel width transitions while dragging**: Remove width/height transitions during drag to avoid jitter and lag.
- **Harmonize surfaces**: Avoid triple layering in the right panel by alternating backgrounds deliberately (e.g., header: secondary, content area: primary, cards: secondary), not primary→secondary→primary in one stack.
- **Replace grayscale filter with a grayscale theme**: Prefer CSS variables for monochrome over `filter: grayscale(1)` to preserve crisp focus/accents.
- **Typography consistency**: Use a reliable, available label font (e.g., Cinzel) instead of unavailable Albertus. Ensure the same families/weights for headings, labels, and mono values.
- **Consolidate button styles**: Merge Toolbar `Button` and RightPanel `ActionButton` into one shared style with size variants; align padding/height and focus states.
- **Improve long value handling**: Let config values wrap with ellipsis-on-hover tooltips; avoid permanent truncation in tight layouts.

### Design tokens to introduce (CSS Custom Properties)
Add these to `src/styles/index.css` under `:root` and use everywhere (styled-components can keep using CSS vars):

```css
:root {
  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;

  /* Radii */
  --radius-xs: 2px;
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;

  /* Elevation */
  --shadow-sm: 0 1px 2px rgba(0,0,0,.06);
  --shadow-md: 0 2px 6px rgba(0,0,0,.08);
}
```

Then apply:
- Toolbar/StatusBar paddings to `var(--space-2)` vertical, `var(--space-3)` horizontal.
- Panel headers/sections to `var(--space-4)`.
- Card/section radius to `var(--radius-md)`.

### Layout and alignment
- **DualPanelLayout**: The root uses `100vh/100vw`; ensure the fixed `connection-status` bar does not overlap content. Add top offset/margin when the banner is present.
- **ResizablePanels**:
  - Add a `dragging` class to the container while dragging; remove width/height transition in that state to prevent jitter.
  - Increase handle hit area to 16px with an invisible padding, while keeping the visible bar at 8px.
  - Add double-click on the handle to reset to persisted defaults (50/50 or last saved preset).
  - Provide keyboard support and a clear focus state.

Example CSS refinements for the handle in `src/styles/index.css`:

```css
.resizable-panels.dragging .panel { transition: none; }
.split-handle { position: relative; }
.split-handle::before { /* extend hit area */
  content: '';
  position: absolute;
  top: 0; bottom: 0;
  left: -4px; right: -4px; /* 8px visible + 8px extra = 16px hit */
}
.split-handle:focus {
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}
```

Add ARIA to the handle component in `src/components/Layout/ResizablePanels.tsx`:

```tsx
// Add to the handle element
role="separator"
aria-orientation={direction === 'horizontal' ? 'vertical' : 'horizontal'}
tabIndex={0}
onKeyDown={(e) => {
  const step = e.shiftKey ? 5 : 1; // percentage points
  if (direction === 'horizontal') {
    if (e.key === 'ArrowLeft') { /* decrease left, increase right */ }
    if (e.key === 'ArrowRight') { /* increase left, decrease right */ }
  } else {
    if (e.key === 'ArrowUp') { /* adjust sizes */ }
    if (e.key === 'ArrowDown') { /* adjust sizes */ }
  }
}}
onDoubleClick={() => persistSizes([50, 50])}
```

### Surface layering (right panel)
- Current stack alternates backgrounds frequently:
  - `RightPanelContainer`: primary
  - `ContentArea`: secondary
  - `ContentSection`: primary
  - Nested cards: secondary
- Prefer a simpler two-surface pattern for clarity:
  - Container and large scroll areas: `--background-primary`.
  - Cards/sections: `--background-secondary` with `--radius-md`, `--shadow-sm`.
  - Headers: stay secondary but drop borders in favor of subtle shadow to reduce visual noise.

Update in `src/components/ConfigExplorer/RightPanel.tsx` styles:
- `ContentArea` background: primary.
- `ContentSection` background: secondary; add `box-shadow: var(--shadow-sm)`.

### Toolbar and StatusBar
- **Consistency**: Align vertical rhythm by matching paddings, gaps, and height. Use the same tokenized `Button` style.
- **Grouping**: Keep separators only between functional groups. On narrow widths, wrap as a single line with an overflow menu rather than multi-line wrapping.
- **Typography**: Ensure the same font family/size across Toolbar/StatusBar; prefer sans 12–13px with 600 weight for emphasis.

Files: `src/components/Layout/Toolbar.tsx`, `src/components/Layout/StatusBar.tsx`.

### Left panel (navigation and controls)
- Reduce nested borders: where `ControlGroup` is inside `ConfigFolder`, avoid an extra border on the folder if the group already has one.
- Increase scroll affordance by adding a very subtle gradient at the top/bottom of the scroll area to hint overflow.
- Convert inline style blocks (e.g., placeholder text boxes) to styled components for consistency and theming.

File: `src/components/ConfigExplorer/LeftPanel.tsx`.

### Comparison and diff presentation
- Use a CSS grid template with fixed gutters to align `Current → Comparison` values perfectly.
- Allow multi-line values with preserved monospace and a max-height with internal scroll for long JSON/text values, not global truncation.
- Add a subtle background tint per variant (added/removed/changed) to reinforce the left border color.

File: `src/components/ConfigExplorer/ComparisonPanel.tsx`.

### Typography
- Replace `Albertus` with `Cinzel` (already imported) for labels to avoid missing font fallbacks; document the choice.
- Consider a modern system UI font (e.g., Inter) via import for general UI if desired, but not required.
- Keep JetBrains Mono for values; avoid using 11px below 100% DPI—prefer 12px for readability.

Files: `src/styles/index.css`, `src/styles/leva-theme.css`.

### Grayscale and high contrast
- Replace `body.grayscale { filter: grayscale(1); }` with a variable-driven monochrome palette toggle to retain crisp focus rings and avoid blurring text.
- Keep the filter as a fallback only. Switch on/off by toggling a `data-mode="grayscale"` attribute on `body` and adjusting color variables accordingly.

Example toggle:

```css
body[data-mode="grayscale"] {
  --accent-primary: var(--slate-700);
  --accent-hover: var(--slate-800);
  /* optionally reduce chroma further by mapping accent/text to slate/stone */
}
```

### Interaction states
- Ensure hovering over actionable text/buttons slightly increases contrast (`--accent-hover`) and applies a subtle elevation for tactile feel.
- Keep focus rings consistent and visible; ensure no component overrides them with too-strong box-shadows.

### Implementation checklist (by file)
- `src/styles/index.css`
  - Add spacing, radius, and shadow tokens; remove grayscale filter for the new attribute-driven approach.
  - Normalize paddings/gaps for `.skip-link`, panels, and handles using tokens.
  - Add `.resizable-panels.dragging` rule and extended handle hit area.
- `src/styles/leva-theme.css`
  - Align sizing/typography with tokens; ensure label font uses `Cinzel` consistently.
- `src/components/Layout/ResizablePanels.tsx`
  - Add `dragging` while actively resizing; add ARIA, keyboard handlers, and double-click reset.
- `src/components/Layout/Toolbar.tsx` and `StatusBar.tsx`
  - Share a `Button` style (or utility) and align paddings/heights.
  - Consider an overflow menu for narrow widths.
- `src/components/ConfigExplorer/RightPanel.tsx`
  - Swap `ContentArea` to primary background; `ContentSection` to secondary with shadow.
  - Allow multi-line values in diff; add variant tints.
- `src/components/ConfigExplorer/LeftPanel.tsx`
  - Reduce nested borders; add scroll affordance gradients.
- `src/components/UI/ThemeProvider.tsx`
  - Toggle `data-mode="grayscale"` instead of applying a `filter` class.

### Reference notes from codebase

```19:29:src/components/Layout/DualPanelLayout.tsx
    <div className="dual-panel-layout" data-testid="dual-panel-layout" style={{ height: '100vh', width: '100vw', display: 'flex', flexDirection: 'column' }}>
      <div style={{ flex: '0 0 auto' }}>
        <Toolbar />
```

```221:239:src/components/Layout/ResizablePanels.tsx
            <div
              className="split-handle resize-handle"
              data-testid="panel-resizer"
              onMouseDown={onMouseDown(i)}
              onTouchStart={onTouchStart(i)}
              style={{
                [direction === 'horizontal' ? 'width' : 'height']: `${gutterSize}px`,
                [direction === 'horizontal' ? 'height' : 'width']: '100%',
                cursor: direction === 'horizontal' ? 'col-resize' : 'row-resize',
                backgroundColor: 'var(--border-subtle)',
                borderLeft: direction === 'horizontal' ? '1px solid var(--border-medium)' : undefined,
                borderRight: direction === 'horizontal' ? '1px solid var(--border-medium)' : undefined,
                borderTop: direction === 'vertical' ? '1px solid var(--border-medium)' : undefined,
                borderBottom: direction === 'vertical' ? '1px solid var(--border-medium)' : undefined,
                position: 'relative'
              }}
            >
```

```23:41:src/components/ConfigExplorer/RightPanel.tsx
const ContentArea = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  background: var(--background-secondary);
`
```

```28:36:src/styles/leva-theme.css
[data-theme="custom"] {
  --leva-elevation1: #1a1a1a;
  --leva-elevation2: #2a2a2a;
  --leva-elevation3: #3a3a3a;
  /* ... */
}
```

### Risk and validation
- All changes are token-driven; component styles continue to use CSS variables, minimizing refactors.
- Accessibility improves with focusable resizers and consistent focus rings.
- Performance improves (no transition during drag; reduced layout thrash).
- Visual deltas are controlled and reversible behind tokens.

### Suggested acceptance criteria
- Spacing/radius/shadow tokens present and used in Toolbar, StatusBar, RightPanel `ContentSection`, and resizer handles.
- Resizers are keyboard accessible and display proper focus outlines.
- Toolbar and StatusBar heights are consistent; baseline-aligned content.
- Right panel layering simplified; cards have subtle shadow and consistent radius.
- Grayscale mode uses variables; focus ring and accent colors remain crisp.

