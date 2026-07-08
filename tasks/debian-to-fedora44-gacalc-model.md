# Migrate the container from Debian trixie to Fedora 44, on the gacalc model

**Status:** implemented 2026-07-08 — gate results in the "Implementation notes" section below
**Created:** 2026-07-08

## Implementation notes (2026-07-08)

**Round 2 fix — venv shadowing (Bill's on-display run caught it):** the first
build used `uv pip install ".[extras]"`, and **uv ignores system-site-packages
when resolving** (unlike pip), so it installed PyPI copies of numpy/
matplotlib/pillow/**pyopengl 3.1.10** into the venv — shadowing Fedora's
patched python3-pyopengl. Combined with Fedora's bundled `OpenGL_accelerate`
(which resolves EGL at import time) and no EGL runtime lib in the image, the
demo crashed at `import OpenGL.GL` with
`'NoneType' object has no attribute 'eglGetCurrentContext'`. Fixes: the
dependency-resolving install now uses **pip** (`python -m pip install
--no-build-isolation ".[dev,notebooks,jupyter]"`; python3-pip added to dnf);
uv remains only for the runtime `--no-deps` editable installs;
**mesa-libEGL/mesa-libGL/mesa-dri-drivers installed unconditionally**; gate 2
now imports `OpenGL.GL` headless under wayland env and asserts OpenGL/numpy
resolve from /usr, not /venv. Also added this round: gacalc's `.extrabashrc`
ported (prompt + format-on-exit trap; Dockerfile sources it) and build
artifacts gitignored (output/, proofs/*.{aux,log,pdf}, __pycache__/,
src/*.egg-info/, imgui.ini, *.tar).

Landed as specified, with these concrete choices:

- **pyproject.toml**: runtime `dependencies` = galgebra, glfw, imgui-bundle,
  matplotlib, numpy, pillow, pyMatrixStack, PyOpenGL, sympy; extras
  `notebooks` (jupytext), `jupyter` (jupyterlab/jupyter/jupyter-lsp/
  jupyter-ydoc/jupyterlab-mathjax3/mathjax), `dev` (autoflake8, removestar,
  ruff). autoflake8/removestar kept from the old flat list.
- **Dockerfile**: gacalc shape — dnf upgrade layer; toolchain layer (dnf
  python3-numpy/matplotlib/pillow/pyopengl/sympy/setuptools/wheel + ruff/ty/
  uv/tmux/git/gcc/meson/ninja-build/pkgconfig/glib2-devel/glfw) with
  `USE_EMACS` (installs emacs + runs install-melpa-packages.el at build —
  mvm has no vendored elpa, unlike gacalc) and `USE_SPYDER` gates, both
  defaulting 0 (Makefile also 0, matching gacalc rather than the template's
  usual 1); venv `--system-site-packages`; gacalc's verified nbconvert-PDF
  XeLaTeX block verbatim + this repo's four TeX additions (anyfontsize,
  commath [proofs], dvipng + standalone [texExpToPng]); COPY project files +
  `uv pip install --no-build-isolation ".[dev,notebooks,jupyter]"`;
  SHA-pinned texExpToPng clone unchanged (`fbbd9a3f…`).
- **Makefile**: gacalc shape (conditional TMUX/GITCONFIG/GNUPG mounts,
  containerized `format`, `image-export`/`image-import`, help) + mvm
  keepers: `clean`, `pdfs`, output/ mount, X/Wayland/DRI blocks with
  `PYGLFW_LIBRARY=/usr/lib64/libglfw.so.3` and `PYOPENGL_PLATFORM=egl`
  (redundant on Fedora's patched python3-pyopengl; kept as documentation).
  The old broken `entrypoint/.bashrc` mount is gone.
- **entrypoint/**: entrypoint.sh/shell.sh/jupyter.sh/format.sh moved to the
  gacalc idiom (venv activate + `uv pip install --no-deps --no-index
  --no-build-isolation -e .`); **pdfs.sh created** (was referenced by the
  Makefile but never existed) — runs `make -C proofs` and copies PDFs to
  /output. `.gitignore` gained `*.tar` for image-export.
- Old `USE_JUPYTER` ARG dropped (was declared but never referenced; jupyter
  now comes unconditionally from the pip extras).

## Goal (Bill, 2026-07-08)

Rebuild multivariate-math's `Dockerfile` and `Makefile` on **Fedora 44**,
modeled directly on **geometricalgebra** (`/foo/opt/geometricalgebra`). History:
this container predates Bill's jupyter work; Debian was chosen because its TeX
in jupyter looked nice. gacalc has since solved that properly on Fedora — the
key was **which packages come from dnf versus PyPI** — so multivariate-math
should adopt the same shape. This also folds the repo back into the standard
Fedora-44 container template (it's currently the only Debian holdout).

## The gacalc model (what "same way" means concretely)

**Dockerfile** (`/foo/opt/geometricalgebra/Dockerfile`):

1. `FROM registry.fedoraproject.org/fedora:44`; dnf cache-mount idiom
   (`--mount=type=cache,target=/var/cache/libdnf5` + `/var/lib/dnf`,
   `keepcache=True`); `dnf upgrade -y` first.
2. **dnf installs the toolchain + heavy/native Python**: `python3`,
   `python3-setuptools`, `python3-wheel`, `python3-sympy`, `python3-pandas`,
   `python3-pytest`, `ruff`, `ty`, `uv`, `tmux`, `which`; flag-gated emacs
   and spyder blocks (`USE_EMACS`/`USE_SPYDER`, Makefile default 1 /
   Dockerfile default 0 per the template).
3. **`python3 -m venv --system-site-packages /venv`**, then **`uv pip
   install`** for everything Pythonic that isn't dnf-worthy.
4. **pyproject `[project.optional-dependencies]` is the single source of
   truth**: gacalc defines `notebooks` (matplotlib, ipython,
   matplotlib-inline, pandas, jupytext), `jupyter` (jupyterlab, jupyter,
   jupyter-lsp, jupyterlab-mathjax3), `dev` (build, twine, ruff) extras and
   the Dockerfile does `uv pip install --no-build-isolation
   ".[dev,notebooks,jupyter]"`. **The JupyterLab stack comes from PyPI, not
   dnf** — this is the dnf-vs-PyPI split Bill figured out.
5. **The nice-TeX-in-jupyter answer — nbconvert "Export to PDF"**: a
   dedicated dnf layer, verified end-to-end against a math-heavy notebook
   (`jupyter nbconvert --to pdf --execute`): `pandoc`, `texlive-xetex`,
   `texlive-collection-fontsrecommended`, `texlive-collection-latexrecommended`,
   plus the template's named deps: `texlive-adjustbox tcolorbox collectbox
   ucs titling enumitem rsfs jknapltx upquote ulem soul eurosym pgf environ
   trimspaces parskip`. Copy this block verbatim.
6. COPY build-relevant project files late (layer caching); runtime bind
   mount overlays them.

**Makefile** (`/foo/opt/geometricalgebra/Makefile`): `.DEFAULT_GOAL := help`;
`readlink -f` conditional mounts (TMUX/GITCONFIG/GNUPG); `FILES_TO_MOUNT`;
`EXPOSE_PORT = -p 8888:8888`; X + Wayland flag blocks; `image` passing
`--build-arg`s; `shell` / `jupyter` / containerized `format` / `help`;
`image-export`/`image-import` pair.

## multivariate-math-specific mapping

- **pyproject.toml**: restructure `dependencies` into runtime deps +
  `notebooks`/`jupyter`/`dev` extras, gacalc-style. mvm's GL/imgui deps
  (`glfw`, `imgui-bundle`, `PyOpenGL`, `pyMatrixStack`, `galgebra`,
  `pillow`) become the runtime `dependencies` (or a `demos` extra).
- **Debian→Fedora package renames** in the texExpToPng build stanza (keep
  the SHA-pinned clone, added 2026-07-08): `libglib2.0-dev`→`glib2-devel`,
  `ninja-build`→`ninja-build`(same name on Fedora)/`ninja`, `pkg-config`→
  `pkgconfig`, `dvipng`→`texlive-dvipng`, `texlive-latex-extra` (for
  standalone.cls)→`texlive-standalone`; `git` must be in the dnf list.
- **GL/Wayland flags** (keep, with Fedora paths): `PYGLFW_LIBRARY` becomes
  `/usr/lib64/libglfw.so.3` (mvp's value; dnf `glfw` is the dual build).
  **PyOpenGL: install via dnf `python3-pyopengl`** like mvp — Fedora's
  package carries the auto-select-EGL-under-Wayland patch, which makes the
  Makefile's `PYOPENGL_PLATFORM=egl` redundant (keep it anyway — harmless,
  and it documents the requirement for non-Fedora hosts).
- **entrypoint scripts**: shell.sh/jupyter.sh switch to gacalc's idiom —
  venv activate + `uv pip install` editable (`--no-deps` so container start
  does zero downloads); jupyter.sh keeps the mvm ipykernel registration.
- Keep the mvm-only targets (`pdfs`) and the apt→dnf conversion removes the
  2026-07-08 apt keep-cache block (superseded by the dnf keepcache idiom).

## Supersedes / interacts

- **Absorbs `tasks/python-deps-at-image-build.md`** — that task proposed the
  mvp-style `requirements.txt` mechanism; Bill's chosen model is gacalc's
  pyproject-extras + uv instead. Its goal (all deps installed at image build,
  into the venv, fast offline container start) is achieved by this task.
  Archive it when this lands (its survey of the current apt/pip split is
  still useful reading).
- `tasks/pyimgui-to-imgui-bundle.md`: nothing regresses — imgui-bundle
  1.92.801 has fedora-compatible manylinux cp3xx wheels (mvp uses it on
  Fedora 44 / py3.14 already); consider the `==1.92.801` pin here.
- `tasks/port-mvp-billboard-labels.md`: texExpToPng build moves to Fedora
  packages (mvp's own Dockerfile is the reference — it builds the same tool
  on Fedora 44 with `texlive-standalone`/`texlive-dvipng`).

## Gates

- `make image` with default flags.
- `make shell`: `python -c "import numpy, matplotlib, sympy, galgebra, glfw,
  OpenGL, imgui_bundle"` with **zero downloads** at start.
- `make jupyter` serves; **`jupyter nbconvert --to pdf --execute` on a
  math-heavy notebook** renders (the whole point of the migration — gacalc's
  verified check).
- In-container `texExpToPng --exp '$\lVert\vec{a}\rVert$' --size 600 --fg
  "rgb 1 1 1" --bg Transparent -o /tmp/x.png` still renders.
- crossproduct demo static gate (py_compile + ruff/ty); on-display run is
  Bill's (Wayland flags with the Fedora libglfw path).
