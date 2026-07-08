# Crossproduct demo: fix some of the billboard labels

**Status:** proposed — awaiting Bill's specifics (he'll say what needs fixing)
**Created:** 2026-07-08

## Context

The billboard labels + menubar landed 2026-07-08
(tasks/port-mvp-billboard-labels.md) and Bill confirmed the demo "looks
pretty good" on his display. Some of the **label content/behavior needs
fixing** — Bill will supply the specifics later; this doc holds the current
state so the fixes are easy to point at.

## Current label inventory (src/crossproduct/crossproduct.py)

All strings must stay amsmath-safe (texExpToPng renders
`\documentclass{standalone}` + amsmath only — `\lVert…\rVert`, no `\norm`,
no `\triangleq`).

- **Axis tips** (`x`, `y`, `z`) — shown while Draw Natural Basis is on and
  the ground hasn't been removed; placed at 1.18 × unit along each axis in
  the math-coordinate frame (`coords_M`).
- **`a_label()`** by step: `\vec{a}` → (rotate_z) `\vec{a}\,'` →
  (rotate_y … undo_rotate_x) `\vec{a}\,''` → (undo_rotate_y) `\vec{a}\,'` →
  (undo_rotate_z onward) `\vec{a}`.
- **`b_label()`** by step: `\vec{b}` → (rotate_z) `\vec{f}_a^{zx}(\vec{b})`
  → (rotate_y) `\vec{b}\,''` → (rotate_x … rotate_to_z) `\vec{b}\,'''` →
  (undo_rotate_x) `\vec{b}\,''` → (undo_rotate_y) `\vec{f}_a^{zx}(\vec{b})`
  → (undo_rotate_z onward) `\vec{b}`.
- **`c_label()`** (the derived vector) by step:
  (show_triangle, project_onto_y) `c = \lVert\vec{b}\rVert\sin(\theta)` →
  (rotate_to_z) `c` → (undo steps) `\vec{f}(\vec{b})` →
  (scale_by_mag_a, show_plane) `\vec{a}\times\vec{b}`.
- Vector-tip labels sit at 1.08 × the vector, in each vector's actual draw
  frame (`model_M` / `vec3_label_M`); default size `height_px = 44`.
- The **menu text** (`"vec a" + a_extra_text()` primes in the Vectors menu,
  Highlight a/b labels) uses the old prime-suffix scheme, which for `a`
  overshoots the proof (it shows `'''` at rotate_x+, but `a` only ever earns
  two primes) — a known candidate if the fix list touches menu labels too.

## To fill in (Bill)

- [ ] Which labels are wrong, and what they should say / when they should
      show.
