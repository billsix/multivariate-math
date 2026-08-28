# Remove the vestigial static-image texture path; fix the axis early-return

**Status:** proposed — needs go-ahead. Created 2026-08-27 (William Emerison Six <billsix@gmail.com>).
**Priority:** 5
**Difficulty:** 2

## Goal

The crossproduct demo's math labels are now produced at runtime by `texExpToPng`
(`tasks/reference/crossproduct-demo-architecture.md`), which superseded an older static-image path that
is still in the tree but never called. Remove (or explicitly mark) that dead code, and fix a suspect
early return the review surfaced.

## Plan

- [ ] **Vestigial static-image path** — `generate_texture` (`src/crossproduct/crossproduct.py:468`) and
      `do_draw_image` (`src/crossproduct/renderer.py:482`) are defined but never called in the frame loop
      (superseded by the `_labels.py` `texExpToPng` billboards). Remove them + the unused
      `src/crossproduct/images/*.png`, or mark them dead with a comment if kept intentionally.
- [ ] **`vertices_of_axis` early return** — `src/crossproduct/renderer.py:371` returns after building a
      single axis arrow's vertices. Confirm whether it should build the full set; fix if it's truncating.
- [ ] Run the demo (or the label graceful-degradation path) to confirm no regression.

## Notes

Grounded by `tasks/reference/crossproduct-demo-architecture.md`. Label *content* fixes are a separate
task (`tasks/crossproduct-label-fixes.md`).

## Open questions

1. Is the static-image path kept for a reason (a fallback when `texExpToPng` is absent — though
   `_labels.py` already no-ops gracefully)? If not, delete; if yes, wire it as the documented fallback.
