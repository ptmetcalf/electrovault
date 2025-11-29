# Photo Brain Frontend

React + TypeScript + Vite UI for the Photo Brain operator console. The UI is intentionally lightweight (no extra UI dependencies) and uses a small set of design tokens to keep the experience polished and consistent for agentic workflows.

## Run the UI

```bash
cd frontend
npm install
npm run dev        # or npm run build for production
```

The API is assumed to run on port 8000 (see `photo_brain.api.http_api`), which matches the UI defaults.

## Design system (no external UI kit)

- **Tokens**: Defined in `src/index.css` (`--bg`, `--surface`, `--border`, `--text`, `--accent`, radii, shadows, fonts). Adjust tokens first before touching component styles.
- **Primitives** (in `src/App.css`):
  - `panel`: raised surfaces with blur + border; use for sidebars or tool panes.
  - `card`: grid items for photos/results; `thumb` for media frames.
  - `pill` / `badge`: small status/filters; `badge.active` for selected labels.
  - Buttons: default = primary gradient; `button.secondary` for ghost/quiet actions. Keep button text short and verbs.
  - Inputs: default text fields/textarea share focus ring + border. Reuse existing markup to inherit styling.
  - Layout: `.grid` = 2-column shell (filters left, workspace right). `.detail` = selected photo drawer with `.image-frame` and `.face-box` overlays.
- **States**: `.error` (soft red), `.empty` (muted), `.stat` (compact KPIs), `.pager` for pagination rows.

## UI conventions

- Keep interactions predictable: use existing `button`/`secondary` styles, reuse `badge` for chips, and prefer `panel`/`card` over ad-hoc containers.
- Dark mode only; honor the tokens instead of hard-coded colors. Avoid inline styles unless dynamic positioning (e.g., face boxes).
- Maintain responsive behavior (`@media` in `App.css`): the grid collapses to a single column under 1100px; detail view stacks under 900px.
- Accessibility: focusable elements already get focus styling; preserve `type="button"` on non-submit buttons.
- Data wiring: this UI talks directly to the Photo Brain API (search, photos, faces, events). Keep new calls in `App.tsx` colocated with their UI section and avoid introducing new deps unless justified in `package.json`.

## Extending the UI

1) Start with tokens — extend or tweak `src/index.css` rather than inlining new colors.
2) Reuse existing primitives (`panel`, `card`, `badge`, `stat`) and spacing helpers (`row`, `stack`, `flex-between`).
3) Add new variants by extending the CSS classes instead of per-element overrides to keep consistency.
4) Update this README when adding components, tokens, or layout patterns so future changes stay aligned.
