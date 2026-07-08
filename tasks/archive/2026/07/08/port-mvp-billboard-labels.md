# Port mvp's TeX billboard labels into the crossproduct demo

**Status:** COMPLETE 2026-07-08 — implemented, gated (image build, all 14
label strings render, ty at 45 diagnostics vs 46 baseline, ruff clean), and
**display-verified by Bill** ("looks pretty good"). Follow-ups spun out:
label-content tweaks → tasks/crossproduct-label-fixes.md; the two mvp
backports (Pillow Transpose enum, VAO save/restore) →
modelviewprojection tasks/backport-labels-fixes-from-mvm.md.
**Created:** 2026-07-08

## Implementation notes (2026-07-08)

- **`src/crossproduct/_labels.py`** — ported from mvp with two fixes worth
  BACKPORTING to mvp's copy: `Image.Transpose.FLIP_TOP_BOTTOM` instead of the
  module-level `Image.FLIP_TOP_BOTTOM` (removed in Pillow 10 — mvp's copy
  would crash on a modern Pillow when it next regenerates a texture), and a
  `TEXEXP is None` guard in `_generate` (ty cleanliness). Billboard shader
  pair copied verbatim.
- **Menubar** (File / Animation / Camera / Vectors / Highlight / View) per
  the spec below; the floating "Cross Product" window is gone. Keyboard:
  Space = next step, R = restart, P = autoplay, F11 = fullscreen, Esc = quit;
  shortcuts respect `want_capture_keyboard`, and orbit/scroll respect
  `want_capture_mouse` (required for menubar usability).
- **The step machine**: `STEP_NEXT_LABEL`/`REL_FLAG` dicts + a single
  "Next: <action>" menu item (disabled until the current stage's animation
  completes). Transition side effects moved verbatim into
  `process_pre_step_transitions()` / `process_post_step_transitions()`,
  called at the exact points in the frame where the old in-window buttons
  executed — position matters because the Show Triangle transition reads the
  model matrix AFTER the frame's rotation blocks. One deliberate state
  change vs the old code: the final action now sets
  `step_number = show_plane` (the old code left it at `scale_by_mag_a` and
  only set `do_remove_ground`; all ratio/gate expressions verified
  equivalent).
- **Labels**: proof-derived, per-step (`a_label()`/`b_label()`/`c_label()`
  `match` functions — x/y/z at axis tips, `\vec{a}` → `\vec{a}'` →
  `\vec{a}''`, `\vec{b}` → `\vec{f}_a^{zx}(\vec{b})` → `\vec{b}''` →
  `\vec{b}'''`, and on the derived vector
  `c = \lVert\vec{b}\rVert\sin(\theta)` → `c` → `\vec{f}(\vec{b})` →
  `\vec{a}\times\vec{b}`), walking back down through the undo steps. Tip
  positions use the matrices captured at each vector's actual draw frame
  (`coords_M` / `model_M` / `vec3_label_M`).
- Behaviour notes: vector edits and Swap now work mid-derivation (they
  restart the derivation, preserving vectors/camera — the old UI only
  exposed them at the beginning step); the relative-coordinate x'/y'/z'
  highlight toggles are always visible in the Highlight menu (previously
  step-conditional); auto-rotate-camera now advances every frame (previously
  only while the Camera header was expanded — a quirk of UI/scene
  interleaving in the old window).

## Goal

Annotate the crossproduct demo's vectors/axes with LaTeX labels rendered by
`texExpToPng`, drawn as camera-facing billboards — porting mvp's **rendering
infrastructure only**. Two constraints from Bill (2026-07-08):

- **Label CONTENT comes from `proofs/crossproduct.tex`**, not from mvp's
  label table — mvp's version "did a really bad job at determining the latex
  to show". The demo animates exactly the proof's derivation, so each stage
  shows the proof's own notation (table below).
- **No Cayley graphs.** mvp's `mathdemos/crossproduct.py` restructured the
  demo around mvp's Cayley-scene abstraction — do NOT port any of that. This
  repo's demo keeps its own procedural structure; only `_labels.py` (the
  billboard renderer) and the per-step label calls come over.

## DONE (2026-07-08)

- **Dockerfile builds texExpToPng from a SHA-pinned clone** (Bill's call —
  no vendored copy in this repo):
  `git clone https://github.com/billsix/tex-expression-to-png.git` +
  `git checkout fbbd9a3fefa48ab86136ca4fba9861553289c5ee` (upstream HEAD as
  of 2026-07-08; bump deliberately) + meson setup/compile/install →
  `/usr/local/bin/texExpToPng`. apt adds `dvipng`, `texlive-latex-extra`
  (ships `standalone.cls` — the tool renders `\documentclass{standalone}` +
  amsmath via `latex` then `dvipng`), `meson`, `ninja-build`, `pkg-config`,
  `libglib2.0-dev`.
  - Noted 2026-07-08: the local `/foo/opt/texExpToPng` working repo's origin
    is the Pi git server and its HEAD (`ebea794d`) differs from the GitHub
    mirror's pinned `fbbd9a3f` — if the mirror is behind, push and bump the
    pin.
- Gate: in-container `texExpToPng --exp ... --fg "rgb 1 1 1" --bg Transparent`
  renders a PNG headlessly (see session summary for result).

## Remaining work (needs go-ahead)

1. **Port `_labels.py`** from mvp `src/modelviewprojection/mathdemos/_labels.py`
   into `src/crossproduct/` (plus its billboard shader pair). It is
   self-contained: `LabelRenderer(shader_dir, dpi=600, fg="rgb 1 1 1")`,
   `begin(view_M, proj_M, viewport)` / `draw(latex, center_world)` / `end()` /
   `cleanup()`. GL 3.3 core + pyMatrixStack — same stack as this demo.
   **Graceful degradation contract:** `shutil.which("texExpToPng")` absent →
   `available = False`, every call no-ops, demo runs unchanged.
   Strip mvp-isms (its `modelviewprojection.*` imports → local imports;
   nothing Cayley-related is in `_labels.py` itself).
2. **Per-step label calls** in the demo loop (drawn last, over the scene, at
   slightly-extended vector tips — mvp's ~line-1109 block shows the begin/
   draw/end shape).
3. **Menubar UI, mvp-style (Bill, 2026-07-08)** — replace the floating
   "Cross Product" window (collapsing headers + inline step buttons) with a
   main menubar like mvp's `mathdemos/crossproduct.py` `menubar()`
   (~lines 1189–1310), **without any Cayley machinery**. What to copy:
   - **Menu layout:** `File` (Quit / Esc) · `Animation` ("Next: <step name>"
     action, Restart / R, AutoPlay / P with checkmark, "Seconds / step"
     `slider_float` inline in the menu, and a step-conditional "Draw Relative
     Coordinates" toggle) · `Camera` (Auto-Rotate checkmark, Camera Radius
     slider when not ortho, View Down X / −Y / Z actions) · `Vectors`
     (`input_float3` for a and b inline — edits restart the derivation —
     plus Swap and Highlight a/b) · `Highlight` (x y z x' y' z' checkmark
     toggles) · `View` (Fullscreen / F11, Draw Natural Basis).
   - **Local helpers to write in mvm** (small, no mvp imports):
     `menu_action(label, key, action, *, selected=False)` — a 5-line
     `imgui.menu_item` wrapper (cayley_gl.py:508, copy verbatim);
     a `_STEP_NEXT_LABEL: dict[StepNumber, str | None]` mapping each stage to
     the "Next:" action name (replaces the per-step buttons scattered through
     the old Time section — "Rotate Z", "Undo Rotate X", … become one menu
     item that always shows the current step's successor); a
     `_REL_FLAG: dict[StepNumber, str]` for which draw-relative-coordinates
     flag the current step exposes; optionally `WindowState` +
     `toggle_fullscreen` (cayley_gl.py:495/517, ~15 lines, saves/restores
     windowed geometry).
   - **No `cayley_gl.run_loop`:** mvm's existing while-loop stays; call
     `menubar()` between `imgui.new_frame()` and `imgui.render()` where the
     `imgui.begin("Cross Product")` window used to be.
   - **Keyboard shortcuts** shown in the menu right column and handled in the
     existing `on_key`: Esc (already), Space = next step, R = restart,
     P = autoplay, F11 = fullscreen.
   - Optional niceties seen in mvp's frame loop, Bill's call (behaviour
     tweaks): wall-clock `dt`-based animation (clamped, fps-independent)
     instead of the fixed `1/60` increment; skip orbit/scroll input when
     `imgui.get_io().want_capture_mouse` (cursor over a menu).

## Label table — derived from `proofs/crossproduct.tex`

Proof defs: `k \triangleq \sqrt{a_x^2+a_y^2}`,
`c \triangleq \sqrt{{b''_y}^2+{b''_z}^2}`. The demo's `StepNumber` stages are
exactly the proof's derivation steps:

| StepNumber | a-tip label | b-tip (or result) label |
|---|---|---|
| beginning | `\vec{a}` | `\vec{b}` |
| rotate_z (proof: $\vec{f}_a^{zx}$, a onto the xz plane) | `\vec{a}\,' = \vec{f}_a^{zx}(\vec{a})` | `\vec{f}_a^{zx}(\vec{b})` |
| rotate_y (proof: $\vec{f}_{a'}^{x}$, a onto the x axis) | `\vec{a}\,'' = \begin{bmatrix}\norm{\vec{a}}\\0\\0\end{bmatrix}` | `\vec{b}\,'' = (\vec{f}_{a'}^{x}\circ\vec{f}_a^{zx})(\vec{b})` |
| rotate_x (proof: $\vec{f}_{b''}^{xy}$, b'' onto the xy plane) | (unchanged) | `\vec{b}\,''' = (\vec{f}_{b''}^{xy}\circ\ldots)(\vec{b})` |
| show_triangle | — | `c = \norm{\vec{b}}\sin\theta` on the perpendicular leg |
| project_onto_y (proof: $\vec{f}_{b'''}^{y}$) | — | `\vec{f}_{b'''}^{y}(\vec{b}\,''') = \begin{bmatrix}0\\c\\0\end{bmatrix}` |
| rotate_to_z (proof: $\vec{f}_y^{z}$, 90° in the yz plane) | — | `\begin{bmatrix}0\\0\\c\end{bmatrix}` |
| undo_rotate_x (proof: $(\vec{f}_{b''}^{xy})^{-1}$) | — | `(\vec{f}_{b''}^{xy})^{-1}(\ldots)` |
| undo_rotate_y (proof: $(\vec{f}_{a'}^{x})^{-1}$) | — | `(\vec{f}_{a'}^{x})^{-1}(\ldots)` |
| undo_rotate_z (proof: $(\vec{f}_a^{zx})^{-1}$) | back to `\vec{a}` | `\vec{f}(\vec{b}) = \frac{1}{\norm{\vec{a}}}\begin{bmatrix}a_yb_z-a_zb_y\\a_zb_x-a_xb_z\\a_xb_y-a_yb_x\end{bmatrix}` |
| scale_by_mag_a | — | `\norm{\vec{a}}\,\vec{f}(\vec{b}) = \vec{a}\times\vec{b}` |
| show_plane | — | `\vec{a}\times\vec{b}` |

Notes: exact strings/abbreviations are Bill's call (full column vectors may be
too wide for a billboard — the composed-function names like
`(\vec{f}_{a'}^{x}\circ\vec{f}_a^{zx})(\vec{b})` are the compact fallback,
and they are literally the proof's notation). Axis-tip labels `x y z` (the
proof speaks in axes, not $e_i$). The intermediate-label TEXT advances with
the step exactly as the primes advance in the proof.

**LaTeX dialect caveat:** texExpToPng's baked-in template is
`\documentclass{standalone}` + `\usepackage{amsmath}` ONLY, while the proof's
preamble also loads `commath`/`amssymb`. So the table's `\norm{\vec{a}}` must
be written `\lVert\vec{a}\rVert` (amsmath) in the actual label strings, and
`\triangleq` avoided (use `=`), unless we extend the tool's preamble — which
would be a change to the vendored C source and must then be replicated to the
external texExpToPng repo and mvp's copy per the vendoring contract.

## Verification

- Done for the texExpToPng layer: image builds; headless in-container render.
- For the port: py_compile + ruff/ty on changed files; **on-display check is
  Bill's** (labels render, face camera, text matches the proof per step).
