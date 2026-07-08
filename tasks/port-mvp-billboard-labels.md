# Port mvp's TeX billboard labels into the crossproduct demo

**Status:** in progress — texExpToPng built in the image from a SHA-pinned
clone (2026-07-08); the label renderer port + demo integration are proposed
and need go-ahead
**Created:** 2026-07-08

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
   draw/end shape and is the only part of mvp's demo worth reading).

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
