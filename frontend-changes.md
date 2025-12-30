# Frontend Changes: Theme Toggle Button

## Overview
Added a dark/light theme toggle button to the Course Materials Assistant UI, positioned in the top-right corner of the page.

## Files Modified

### 1. `frontend/index.html`
- Added a theme toggle button element before the main container
- Button includes both sun and moon SVG icons (Feather Icons style)
- Includes proper accessibility attributes (`aria-label`, `title`, `type="button"`)

### 2. `frontend/style.css`
- Added light theme CSS variables under `[data-theme="light"]` selector
- Added `--code-bg` variable for consistent code block backgrounds
- Added `.theme-toggle` button styles with:
  - Fixed positioning (top-right corner)
  - Smooth hover/active transitions
  - Focus ring for keyboard navigation
  - Icon visibility toggling based on theme
  - Hover rotation animation on icons
  - Responsive adjustments for mobile

### 3. `frontend/script.js`
- Added `themeToggle` DOM element reference
- Added theme initialization on page load (`initializeTheme()`)
- Added `toggleTheme()` function for click handling
- Added `applyTheme()` function to apply theme and update aria-label
- Theme preference persisted in `localStorage`

## Features
- **Icon-based design**: Sun icon for light mode, moon icon for dark mode
- **Smooth transitions**: 0.3s ease transitions on all theme-affected elements
- **Accessible**: Keyboard navigable with focus styles, dynamic aria-label
- **Persistent**: Theme preference saved to localStorage
- **Responsive**: Adapts size and position on mobile devices

## Theme Colors

| Variable | Dark Theme | Light Theme |
|----------|------------|-------------|
| `--background` | `#0f172a` | `#f8fafc` |
| `--surface` | `#1e293b` | `#ffffff` |
| `--surface-hover` | `#334155` | `#f1f5f9` |
| `--text-primary` | `#f1f5f9` | `#1e293b` |
| `--text-secondary` | `#94a3b8` | `#64748b` |
| `--border-color` | `#334155` | `#e2e8f0` |
| `--assistant-message` | `#374151` | `#f1f5f9` |
| `--shadow` | `rgba(0,0,0,0.3)` | `rgba(0,0,0,0.1)` |
| `--welcome-bg` | `#1e3a5f` | `#eff6ff` |
| `--code-bg` | `rgba(0,0,0,0.2)` | `rgba(0,0,0,0.05)` |

## Light Theme Design Decisions

### Accessibility Standards
- **Text contrast**: Dark text (`#1e293b`) on light background (`#f8fafc`) meets WCAG AA standards
- **Secondary text**: `#64748b` provides sufficient contrast for secondary content
- **Focus indicators**: Blue focus ring (`rgba(37, 99, 235, 0.2)`) remains visible in both themes

### Color Palette (Tailwind Slate)
The light theme uses Tailwind CSS Slate color palette for consistency:
- Background: `slate-50` (`#f8fafc`)
- Surface: `white` (`#ffffff`)
- Surface hover: `slate-100` (`#f1f5f9`)
- Text primary: `slate-800` (`#1e293b`)
- Text secondary: `slate-500` (`#64748b`)
- Border: `slate-200` (`#e2e8f0`)

### Visual Adjustments
- Reduced shadow intensity (0.1 vs 0.3 opacity) for softer appearance
- Lighter code block backgrounds for readability
- Welcome message uses light blue tint (`#eff6ff`) instead of dark blue

## JavaScript Functionality

### Theme Toggle Implementation (`script.js`)

```javascript
// Initialize theme on page load
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    applyTheme(savedTheme);
}

// Toggle between themes
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
    localStorage.setItem('theme', newTheme);
}

// Apply theme to document
function applyTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        themeToggle.setAttribute('aria-label', 'Switch to dark theme');
    } else {
        document.documentElement.removeAttribute('data-theme');
        themeToggle.setAttribute('aria-label', 'Switch to light theme');
    }
}
```

### Key Features
- **Click handler**: `toggleTheme()` switches between dark and light
- **Persistence**: Theme saved to `localStorage` and restored on page load
- **Accessibility**: `aria-label` updated dynamically to reflect current action

## Smooth Transitions

### CSS Transition Rules (`style.css`)

Added transition rules to all theme-affected elements:

```css
body,
.container,
.sidebar,
.chat-main,
.chat-container,
.chat-messages,
.chat-input-container,
.message-content,
.stat-item,
.suggested-item,
#chatInput,
#sendButton,
.stats-header,
.suggested-header,
.new-chat-btn,
.sources-collapsible,
.course-title-item {
    transition: background-color 0.3s ease,
                color 0.3s ease,
                border-color 0.3s ease,
                box-shadow 0.3s ease;
}
```

### Transition Properties
- **Duration**: 0.3 seconds
- **Timing**: `ease` for smooth acceleration/deceleration
- **Properties animated**:
  - `background-color` - backgrounds fade smoothly
  - `color` - text colors transition
  - `border-color` - borders blend
  - `box-shadow` - shadows adjust softly

## Implementation Details

### CSS Custom Properties Architecture

The theme system uses CSS custom properties (variables) defined in `:root` for dark theme (default) and overridden via `[data-theme="light"]` selector:

```css
/* Dark theme (default) */
:root {
    --background: #0f172a;
    --surface: #1e293b;
    --text-primary: #f1f5f9;
    /* ... more variables */
}

/* Light theme override */
[data-theme="light"] {
    --background: #f8fafc;
    --surface: #ffffff;
    --text-primary: #1e293b;
    /* ... more variables */
}
```

### data-theme Attribute

The theme is controlled by setting `data-theme="light"` on the `<html>` element (`document.documentElement`):

```javascript
// Light theme
document.documentElement.setAttribute('data-theme', 'light');

// Dark theme (remove attribute to use :root defaults)
document.documentElement.removeAttribute('data-theme');
```

### Elements Using Theme Variables

All existing elements use CSS variables and work in both themes:

| Element | Variables Used |
|---------|----------------|
| `body` | `--background`, `--text-primary` |
| `.sidebar` | `--surface`, `--border-color` |
| `.chat-messages` | `--background` |
| `.message-content` | `--surface`, `--text-primary` |
| `.message.user` | `--user-message` |
| `#chatInput` | `--surface`, `--border-color`, `--text-primary` |
| `#sendButton` | `--primary-color`, `--primary-hover` |
| `.stat-item` | `--background`, `--border-color` |
| `.suggested-item` | `--background`, `--border-color`, `--text-primary` |
| `.welcome-message` | `--surface`, `--border-color` |
| `code`, `pre` | `--code-bg` |

### Visual Hierarchy Preservation

The design language is maintained across themes:
- **Primary actions**: Blue (`--primary-color`) stays consistent
- **User messages**: Always blue background for clear distinction
- **Assistant messages**: Adaptive surface color for readability
- **Interactive elements**: Consistent hover/focus states
- **Typography scale**: Unchanged between themes
- **Spacing and layout**: Identical in both themes
