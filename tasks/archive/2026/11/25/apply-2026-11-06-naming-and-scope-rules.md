# Apply the 2026-11-06 naming and scope rules across multivariate-math

**Done:** 2026-11-25 (work commit; the archive move follows). Source: the maintainer (William Emerison Six <billsix@gmail.com>, 2026-11-06), who asked for the survey in geometricalgebra and then "add a similar task to every other python project in opt"; run on 2026-11-25 as the fourth repository of six.

## What this did

Five rules entered the shared Python standard on 2026-11-06 (github.com/billsix/runClaudeInContainer, `tasks/reference/python-coding-standard.md`: "Inline a value used once, and name it in a comment instead of a binding" and "What earns an extraction"): inline a binding used once; a binding used twice does not automatically earn a name; declare a name at the point of use; nest a helper with one caller inside that caller; shadow deliberately when nesting. The task surveyed the hand-written Python under `src/` (`crossproduct/`, `plotting/`, `geometric-algebra/`; `proofs/` is LaTeX and the vendored Emacs tree is out) and refactored without changing behaviour.

## Decisions

- **Survey first, then discretion; a helper a test imports stays at module level** (William Emerison Six <billsix@gmail.com>, 2026-11-24, for the mvp task; applied to all six repositories). This repository has no tests.
- **Survey:** `tasks/adhoc/apply-2026-11-06-naming-and-scope-rules/discover.py` wrote 103 leads in tracked files (51 `ONCE`, 33 `TWICE`, 12 `FAR`, 7 `ONE_CALLER`), plus 4 more in an untracked `.ipynb_checkpoints` copy that was ignored.
- **Left alone, whole file:** `src/crossproduct/pyMatrixStack.py` (the maintainer's own copy of the github.com/billsix/pyMatrixStack library; its one `ONE_CALLER` lead, `set_current_matrix`, is public API of that library).
- **Inlined (rule 1):** `image` and `inverted_image` in `load_and_flip_image` (its comment above names the result); `key` in `_Labels._generate` (a comment names the short hash that names the cache file); `data` in `_load_texture` (`img.tobytes()` moved into the `glTexImage2D` call, with a comment); `clip` in `_Labels.draw` (a comment names the clip-space `w`); `parallel`, `perp` and `rotated_parallel` in `rotate` in `geometric-algebra/crossproduct.py` (a comment names the two parts; the evaluation order of the two pure calls changed, the value did not).
- **Nested (rule 4), by `nest_one_caller.py`:** `a_extra_text`, `can_advance`, `menu_action`, `swap_vectors`, `restart_derivation`, `view_down` and `toggle` into `menubar` in `crossproduct.py`. A second run found nothing. Nested helpers keep their parameter names; `menubar` is called every frame, so the seven definitions are rebuilt each frame (a few closures at the frame rate; no measurement was made).
- **Left alone, with the reason:**
  - The GL handle and uniform-location blocks in `renderer.py` (`mvpMatrixLoc`, `colorLoc`, `thicknessLoc`, `viewportLoc`, `numberOfVertices`, `vao`, `vbo`, `ebo`, `position`): each is read in several branches (`xy`, `yz`, `zx`), so a declaration at one use would be repeated at the others; the `FAR` flag is the distance to the first branch.
  - Data tables (`corners` in `_Labels.draw`, `vertices` and `indices` in `do_draw_image`, the arrow vertex list in `do_draw_vector`).
  - `magnitude` in `do_draw_vector` (read inside the nested `vertices_of_arrow`, declared right above it).
  - `sw` in `restart_derivation` (`restart()` rebinds `g.swap` between the save and the restore, so inlining would change the result).
  - `vs`, `fs` in `_compile` (the file handle closes before `compileProgram`); `stride` in `_Labels.__init__` (declared above the loop it feeds); tuple-unpack targets (`clicked`, `ca`, `cb`, `tex_id`, `iw`, `ih`: a tuple cannot be inlined one name at a time); the math names `b1`, `b2`, `b3`, `b_doubleprime_*` in `process_pre_step_transitions` (each is a name from the derivation the demo teaches).

## Result

- Applied: the inlines and the nesting above. The program's behaviour is unchanged by construction; it was not run (see Verification).
- **Found, not changed:** the repository's `make format` gate was already red on `master` (`ty` reports 54 diagnostics and `ruff check` 58 findings, which `tasks/ty-diagnostics-format-gate.md` tracks), and its `ruff check . --fix` step rewrote 42 of them on a run, across six files, three of which this task never meant to touch (`renderer.py`, `plotting.py`, `twod.py`). Those auto-fixes were reverted, not committed; the three files this task changes carry only its own edits and the formatter's re-wrap of the nested code.

## Verification (containerized, 2026-11-25)

- `ruff format --line-length=88` on the changed files: no unrelated reformatting beyond the nested blocks. `ruff check . --statistics` and `ty check` give identical results before and after my changes (58 findings, 54 diagnostics, no `E501`, the same two `F821`), so no new undefined name or type error.
- **Not exercised:** the programs themselves. `crossproduct.py` is a module-level GL program that opens a window, and the image has no Xvfb, so no render probe could run. The menubar (File, Animation, Vectors, Highlight, View menus, now closures inside `menubar`) and the label billboards need a look on a display.

## Owed

The maintainer runs `crossproduct.py` once on a display and clicks through the menus (Animation: Next, Restart, AutoPlay, View Down; Vectors; Highlight; View) and checks that the TeX labels still draw.
