# Install ALL Python packages into the venv at image-build time

**Status:** proposed — needs go-ahead
**Created:** 2026-07-08

## Goal

Every Python dependency is installed **during `make image`**, into **`/venv`**,
so the ephemeral container starts instantly and offline: `shell.sh` /
`jupyter.sh`'s runtime `pip install -e .` should only register the project
itself (no downloads), like mvp's `--no-deps` editable install.

## Current state (why this is needed)

The Dockerfile bakes only `pip`/`setuptools`, `imgui-bundle[glfw]`, and `ty`
into the venv. Everything else in `pyproject.toml` — numpy, matplotlib,
sympy, galgebra, pyMatrixStack, glfw, PyOpenGL, ruff, the jupyter stack… —
downloads **at container start** via `pip install -e .` in shell.sh /
jupyter.sh, every single run (the container is `--rm` ephemeral). Meanwhile a
second, overlapping set comes from **apt** (`python3-opengl`,
`python3-pyglfw`, `jupyter`, `jupyterlab`, `python3-jupytext`,
`python3-jupyter-server-mathjax`) and leaks into the venv through
`--system-site-packages`. Two package managers, split ownership, slow starts.

## Proposed design

1. **`requirements.txt` as the single dependency list** (mvp's pattern):
   move the `dependencies` array out of `pyproject.toml`, set
   `dynamic = ["dependencies"]` + `[tool.setuptools.dynamic]` pointing at
   `requirements.txt`, so `pip install -e .` and the Dockerfile read the same
   list.
2. **Dockerfile**: `COPY requirements.txt /requirements.txt` and
   `pip install -r /requirements.txt` into `/venv` (one RUN, after the venv
   is created). The source tree is NOT copied — it stays a runtime bind
   mount, exactly as today.
3. **All Python via pip, apt only for native libs.** Drop the apt Python
   packages (`python3-opengl`, `python3-pyglfw`, `jupyter`, `jupyterlab`,
   `python3-jupytext`, `python3-jupyter-server-mathjax`) and add their pip
   equivalents to requirements.txt (`PyOpenGL`, `glfw`, `jupyter`,
   `jupyterlab`, `jupytext`, `jupyter-server-mathjax` — most are already
   there). apt keeps only non-Python things: `libglfw3`, mesa, texlive,
   emacs, tmux, gcc/g++ (build deps for any sdist), fonts-mathjax.
   - The venv can then drop `--system-site-packages` entirely → fully
     isolated, reproducible venv. (If anything turns out to genuinely need a
     distro Python package, keep `--system-site-packages` and document why.)
4. **shell.sh / jupyter.sh**: change `python3 -m pip install -e .` to
   `python3 -m pip install --no-deps -e .` so startup can't re-resolve or
   download anything (mvp's shell.sh idiom).

## Caveats / decisions to confirm

- **PyPI `glfw` + system libglfw**: fine — the Makefile already pins
  `PYGLFW_LIBRARY=/usr/lib/x86_64-linux-gnu/libglfw.so.3` (the dual
  X11+Wayland build), which pip's glfw respects. Keep libglfw3 in apt.
- **PyPI `PyOpenGL` vs Debian's**: identical for our purposes — neither has
  Fedora's Wayland auto-EGL patch, and the Makefile already exports
  `PYOPENGL_PLATFORM=egl` (see tasks/pyimgui-to-imgui-bundle.md), so nothing
  regresses.
- **Image size/time**: the jupyter stack via pip adds to the image instead of
  apt — roughly a wash overall, and the dnf/apt cache-mount + pip cache keep
  rebuilds cheap. (Optionally add a pip cache mount:
  `RUN --mount=type=cache,target=/root/.cache/pip pip install -r …`.)
- **`ty` and `imgui-bundle`** move into requirements.txt too — one list, no
  special-case RUNs left (the imgui-bundle RUN added 2026-07-08 gets folded
  in).
- Consider pinning `imgui-bundle==1.92.801` (the version verified working;
  mvp's task doc suggests the same pin after hitting bundled-glfw drift).

## Verification

- `make image` then `make shell`: startup must do **zero** downloads
  (`pip install --no-deps -e .` completes offline, e.g. test with
  `--network none`), then `python -c "import numpy, matplotlib, sympy,
  galgebra, glfw, OpenGL, imgui_bundle"` inside.
- `make jupyter` still serves; `make format` still finds ruff + ty.
