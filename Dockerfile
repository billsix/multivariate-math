FROM registry.fedoraproject.org/fedora:44

ARG USE_SPYDER=0
ARG USE_EMACS=0

RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
    echo "keepcache=True" >> /etc/dnf/dnf.conf && \
    dnf upgrade -y

COPY entrypoint/dotfiles/ /root/

# System-package installation lives in per-group scripts (entrypoint/0N-install-*.sh),
# host-runnable with no container runtime; the scripts take no options -- WHICH optional
# groups run is decided here by the ARG `if` blocks. base + notebook-tex are always
# installed; emacs/spyder are flag-gated. The dnf cache mount + keepcache stay in the
# Dockerfile (build plumbing); `dnf upgrade` ran in the earlier layer above.
COPY entrypoint/01-install-base.sh \
     entrypoint/02-install-emacs.sh \
     entrypoint/03-install-spyder.sh \
     entrypoint/04-install-notebook-tex.sh /usr/local/bin/

# Toolchain + the heavy/native Python packages from dnf (the venv below is
# --system-site-packages, so uv sees these as satisfied and doesn't re-download
# them from PyPI).  python3-pyopengl in particular SHOULD come from dnf: Fedora
# patches PyOpenGL to auto-select the EGL platform under Wayland, which is how
# the GL demos find the context GLFW creates (the Makefile's PYOPENGL_PLATFORM
# comment has the full story).  gcc/meson/ninja/pkgconfig/glib2-devel build
# texExpToPng below; git clones it.  libwayland-cursor/libwayland-egl/
# libxkbcommon are dlopen'd by GLFW's Wayland backend at window-open time --
# the glfw rpm doesn't Require them, and only a real display exercises them
# (found by running the demo on-screen, 2026-07-08).
RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
    /usr/local/bin/01-install-base.sh && \
    if [ "$USE_EMACS" = "1" ];  then /usr/local/bin/02-install-emacs.sh;  fi && \
    if [ "$USE_SPYDER" = "1" ]; then /usr/local/bin/03-install-spyder.sh; fi && \
    echo "/usr/local/bin/jupyter.sh # JupyterLab on http://127.0.0.1:8888/lab" >> ~/.bash_history && \
    echo "source ~/.extrabashrc" >> ~/.bashrc && \
    python3 -m venv --system-site-packages /venv/

# Notebook "Export to PDF": nbconvert's PDF path renders the notebook through
# pandoc -> XeLaTeX, so the image needs pandoc plus a XeLaTeX toolchain with the
# packages nbconvert's default LaTeX template pulls in.  This set was verified
# end to end in geometricalgebra against a math-heavy notebook
# (`jupyter nbconvert --to pdf --execute`) -- copied from its Dockerfile.
# The last four packages are this repo's own TeX consumers beyond nbconvert:
# the pdflatex proofs (proofs/*.tex use commath on top of the recommended
# collections) and texExpToPng (\documentclass{standalone} + amsmath rendered
# via latex + dvipng; anyfontsize for the DPI scaling).
RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
    /usr/local/bin/04-install-notebook-tex.sh

# Install the package + ALL its optional extras from pyproject's own
# [project.optional-dependencies] -- the single source of truth (no
# requirements.txt, no hardcoded package list).  At runtime `make shell`'s bind
# mount overlays /mvm with the live host tree, so this copy only feeds this
# build step.
COPY pyproject.toml setup.py LICENSE /mvm/
COPY src /mvm/src
# pip, NOT uv, for this dependency-RESOLVING install: pip treats the dnf
# packages visible through the --system-site-packages venv as already
# satisfied, while uv resolves in isolation and installs PyPI copies into the
# venv -- which shadow the system ones on sys.path.  That must not happen to
# python3-pyopengl (Fedora's carries the auto-EGL-under-Wayland patch the GL
# demos rely on).  The runtime editable installs (shell.sh/jupyter.sh) keep
# using uv: with --no-deps there is nothing to resolve.
RUN export VIRTUAL_ENV_DISABLE_PROMPT=1 && source /venv/bin/activate && \
    cd /mvm && python -m pip install --no-build-isolation ".[dev,notebooks,jupyter]" && \
    jupytext-config set-default-viewer python && \
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# texExpToPng: renders LaTeX expressions to PNG (latex + dvipng); used by the
# crossproduct demo's billboard labels at runtime (the demo no-ops without it).
# Pinned to the upstream HEAD as of 2026-07-08 so image builds stay
# reproducible even if upstream moves; bump the SHA deliberately.
RUN git clone https://github.com/billsix/tex-expression-to-png.git /tmp/tex_exp_to_png && \
    cd /tmp/tex_exp_to_png && \
    git checkout fbbd9a3fefa48ab86136ca4fba9861553289c5ee && \
    meson setup builddir && \
    meson compile -C builddir && \
    meson install -C builddir && \
    rm -rf /tmp/tex_exp_to_png


ENTRYPOINT ["/entrypoint.sh"]
