# multivariate-math — crossproduct demo architecture & LaTeX-label pipeline

**Reference document** — how the crossproduct OpenGL demo is structured, and (the non-obvious part) how
its math labels are produced by shelling out to `texExpToPng`. This repo's first reference doc. Not a
task; update in place. Created 2026-08-27 (William Emerison Six <billsix@gmail.com>) from a direct read
(load-bearing anchors verified).

## Stack & load-bearing import order

- `src/crossproduct/crossproduct.py` imports **glfw before imgui_bundle on purpose** (`:29-32`, with the
  window/context setup at `:33-77`) — order matters for the GL context.
- Matrix stack: `src/crossproduct/pyMatrixStack.py` (`MatrixStack` enum at `:29`) — the model/view/
  projection stack the demo pushes/pops.

## Renderer & the VAO hazard

- `src/crossproduct/renderer.py` uses **one VAO per shader program**, created at program lifetime
  (`renderer.py:82-104`). This is the hazard the label module must work around (below): it must save and
  restore the caller's bound VAO / array buffer so it doesn't clobber the renderer's.

## The label pipeline (the non-obvious subsystem) — `src/crossproduct/_labels.py`

Math labels are **generated at runtime by `texExpToPng`** (the C wrapper around `latex` + `dvipng`),
loaded as RGBA textures, and drawn as camera-facing billboards:

- **Resolution + graceful degradation (the module's whole point):** `TEXEXP = shutil.which("texExpToPng")`
  (`_labels.py:50`) — resolved once; if `texExpToPng` is absent, the module **no-ops** rather than
  crashing (`:28-30,58`).
- **Render + cache:** `subprocess.run(...)` (`_labels.py:134`) invokes `texExpToPng`, keyed by a
  **sha1 of `latex|dpi|fg`** (`:129`) so each expression renders once.
- **Upload:** PIL decode → RGBA + vertical flip → `GL_RGBA8` texture (`_labels.py:157-161`).
- **VAO save/restore:** the draw path **saves and restores the caller's VAO + array-buffer binding**
  (`_labels.py:185-208, :241`) to avoid `GL_INVALID_OPERATION` against the renderer's one-VAO-per-program
  model.
- **Billboard:** `billboard.vert` sizes the quad to a constant screen-pixel size in both ortho and
  perspective.

## State machine & vendored dependency

- A 12-step state machine drives the demo: `crossproduct.py:237` (`StepNumber`), `:603`
  (`STEP_NEXT_LABEL`), per-frame draw block `:1245-1267`.
- **`texExpToPng` is vendored / pinned at image-build** (`Dockerfile`, the texExpToPng git-clone block,
  ~`:74`) — the label pipeline depends on that binary being on `PATH` in the container.

## Dead / suspect code (grounds the follow-on)

- **Vestigial static-image texture path:** `generate_texture` (`crossproduct.py:468`) and `do_draw_image`
  (`renderer.py:482`) + the `images/*.png` assets are defined but **never called in the frame loop** —
  superseded by the runtime `texExpToPng` billboards.
- **Suspect early return:** `vertices_of_axis()` (`renderer.py`) `return`s after building a **single axis
  arrow's** vertices (`renderer.py:371`) — confirm whether it should build the full set of arrows; as
  written it emits one.

## Follow-on

`tasks/remove-vestigial-static-image-path.md` — remove (or explicitly mark dead) the static-image path
and unused `images/*.png`, and confirm/fix the `vertices_of_axis` early return. (What the labels *say* is
a separate content question parked in `tasks/crossproduct-label-fixes.md` — out of scope here.)
