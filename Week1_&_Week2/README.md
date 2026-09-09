# Folder 1: Web Development Suite (Weeks 1 & 2)

This folder contains two rich, responsive web applications built with HTML5, Vanilla CSS (dark glassmorphism, micro-animations), and Vanilla JavaScript.

## Projects Included

### 1. Week 1: AuraLib — Modern Library Management System
- **Path**: `week1_library_management/`
- **Features**:
  - Full CRUD inventory management with `localStorage` persistence.
  - Interactive book borrowing and return lifecycle with borrower tracking and due dates.
  - Live search (`/` shortcut), multi-genre filtering, and dynamic sorting.
  - Curated seed database with 8 seminal AI, CS, and Math textbooks.
  - Activity history drawer with chronological event logs.
  - JSON data export for backup.
- **Run**:
  Open `week1_library_management/index.html` in any modern browser, or run:
  ```powershell
  python -m http.server 8080 --directory week1_library_management
  ```

### 2. Week 2: AuraBoard — Interactive Studio Whiteboard
- **Path**: `week2_whiteboard/`
- **Features**:
  - High-performance HTML5 Canvas engine with retina display scaling.
  - Freehand brush with smooth quadratic bezier interpolation.
  - Geometric primitives: Lines, Arrows, Rectangles, Circles/Ellipses with fill/stroke options.
  - Draggable, persistent Sticky Notes with rich text notes.
  - Dynamic Undo & Redo history buffer (shortcuts: `Ctrl+Z`, `Ctrl+Y`).
  - Color palette, custom color picker, and dynamic stroke width slider.
  - High-res PNG image export and JSON project persistence.
- **Run**:
  Open `week2_whiteboard/index.html` in any modern browser, or run:
  ```powershell
  python -m http.server 8081 --directory week2_whiteboard
  ```
