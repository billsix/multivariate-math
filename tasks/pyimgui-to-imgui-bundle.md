# Migrate from pyimgui (billsix fork) to imgui-bundle

**Status:** implemented 2026-07-07 (+ EGL fix 2026-07-08) — image + static gates pass; GUI needs Bill's on-display re-test
**Created:** 2026-07-07

## Follow-up fix (2026-07-08): "Attempt to retrieve context when no valid context"

Bill's first in-container run crashed in `GlfwRenderer(window)` →
`OpenGL.error.Error: Attempt to retrieve context when no valid context`.
Diagnosis: NOT missing drivers — the image already carries the full GL stack
(libEGL, libOpenGL/glvnd, libgl1-mesa-dri + mesa-libgallium DRI drivers,
libwayland-egl1, libxkbcommon0, even mesa-vulkan-drivers; verified with dpkg
in the built image), and GLFW had already created the window+context. The
real cause: under Wayland GLFW makes an **EGL** context, but Debian's
PyOpenGL defaults to the **GLX** platform (verified in-image: default
`GLXPlatform`), so `contextdata.getContext()` finds nothing. Fedora patches
PyOpenGL to auto-select EGL when `XDG_SESSION_TYPE=wayland` — that patch is
why mvp works without doing anything; Debian has no such patch. Fix:
`-e PYOPENGL_PLATFORM=egl` added to `WAYLAND_FLAGS_FOR_CONTAINER` (verified
in-image: loads `EGLPlatform`, `GetCurrentContext` resolves). Note if the
demo is ever run on a Debian *host* under Wayland, the same
`PYOPENGL_PLATFORM=egl` export is needed there too.

## Why

The Dockerfile built imgui from `https://github.com/billsix/pyimgui.git`,
which is now 404 (fork deleted/private), so `make image` could not build.
mvp already migrated to **imgui-bundle** (`pthom/imgui_bundle`) in commit
`bbb24cf6` ("updated imgui calls to new library") and uses it in all demos —
Bill called it "the better version". This task ports multivariate-math the
same way.

## What changed

- **`src/crossproduct/crossproduct.py`** — API migration, mapping taken from
  mvp's `bbb24cf6` diff:
  - `import imgui` / `from imgui.integrations.glfw import GlfwRenderer` →
    `from imgui_bundle import imgui` /
    `from imgui_bundle.python_backends.glfw_backend import GlfwRenderer`
    (PyPI `glfw` must import *before* imgui_bundle — see
    pthom/imgui_bundle#321; comment in the file).
  - `imgui.FIRST_USE_EVER` → `imgui.Cond_.first_use_ever`;
    `set_next_window_size(w, h, cond)` → `set_next_window_size(ImVec2(w, h), cond)`;
    `set_next_window_position(...)` → `set_next_window_pos(ImVec2(...), ...)`.
  - `show, _ = collapsing_header(x, flags=TREE_NODE_DEFAULT_OPEN)` →
    `set_next_item_open(True, imgui.Cond_.once)` + `show = collapsing_header(x)`
    (returns plain bool now; mvp's default-open idiom).
  - `input_float3(label=…, value0=…, value1=…, value2=…)` →
    `input_float3(label, [x, y, z])` (returns `(changed, [x, y, z])`).
  - `checkbox(label=…, state=…)` → positional `checkbox(label, value)` (9 sites).
  - Unchanged (same signature in imgui_bundle): `begin/end`, `button`, `text`,
    `same_line`, `label_text`, `slider_float`, `input_float`, `new_frame`,
    `render`, `get_draw_data`, `create_context`, `GlfwRenderer`,
    `impl.process_inputs()`.
- **`pyproject.toml`** — `imgui` → `imgui-bundle`; removed duplicate `"glfw"` entry.
- **`Dockerfile`** — the pyimgui git-clone/submodule/pip RUN replaced with
  `pip install 'imgui-bundle[glfw]'` into the venv (preinstalled so the
  ephemeral container's `pip install -e .` doesn't re-download each run).
- **`Makefile`** — display flags brought up to mvp's pattern:
  `XDG_SESSION_TYPE=wayland`, `PYGLFW_LIBRARY=/usr/lib/x86_64-linux-gnu/libglfw.so.3`
  (Debian path for the dual X11+Wayland system libglfw3; PyPI glfw's
  Wayland-only build lacks `glfwGetX11Window` which imgui_bundle needs),
  `DRI_DEVICE` passthrough (`--device /dev/dri` when present). Also fixed the
  `pdfs` target's dangling `$(USE_X)` → `$(X_FLAGS_FOR_CONTAINER)`, and
  earlier the `image: image` self-dependency typo.

## Verification

- `python3 -m py_compile` on crossproduct.py: pass.
- `make image` (nested podman): **pass** — `imgui_bundle-1.92.801` cp313
  manylinux wheel installs on trixie/py3.13; the `[glfw]` extra resolves
  against the system `python3-pyglfw` (pyGLFW 2.8.0) via the
  system-site-packages venv.
- In-container import smoke test (`import glfw; from imgui_bundle import
  imgui; from imgui_bundle.python_backends.glfw_backend import
  GlfwRenderer`): **pass**.
- ruff + ty (`format.sh` semantics): **no diagnostics on any changed line**;
  file passes `ruff format --check` and `ruff check --select F,E9`. ty/ruff
  do report pre-existing issues elsewhere (lowercase `any` dataclass
  annotations in this file's untouched lines, `renderer.py` unused var,
  notebook-style `src/geometric-algebra/crossproduct.py`).
- **NOT verified: the actual GUI.** On-screen GL can't run in the nested
  sandbox (no display); Bill needs to `make shell` and run
  `python src/crossproduct/crossproduct.py` on the host to confirm the
  panel renders and the controls work (checkboxes, input_float3 vectors,
  collapsing headers default-open, camera slider).

## Known remaining drift (pre-existing, untouched)

- `FILES_TO_MOUNT` mounts `./entrypoint/.bashrc`, which does not exist in the
  repo (no `.bashrc` under `entrypoint/` or `entrypoint/dotfiles/`).
- The `pdfs` target mounts `./entrypoint/pdfs.sh`, which also does not exist.
- (Correction while verifying: `apt`'s `python3-pyglfw` **is** pyGLFW 2.8.0 —
  it provides the `glfw` module the demo imports, and pip's `imgui-bundle[glfw]`
  extra resolves against it via the system-site-packages venv. It stays.)
