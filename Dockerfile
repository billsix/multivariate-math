FROM docker.io/debian:trixie

ARG USE_JUPYTER=1
ARG USE_SPYDER=1


# Keep downloaded .debs (the apt analog of dnf's keepcache=True) so the
# --mount=type=cache mounts below accumulate packages across builds.
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install necessary packages for OpenGL
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt update -y && \
    apt install -y \
    python3 \
    python3-dev \
    python3-pip \
    libglfw3 \
    python3-opengl \
    python3-pyglfw \
    gcc \
    g++ \
    mesa-va-drivers \
    mesa-vdpau-drivers \
    texlive-latex-base texlive-latex-recommended texlive-science  \
    texlive-latex-extra \
    dvipng \
    meson \
    ninja-build \
    pkg-config \
    libglib2.0-dev \
    emacs \
    tmux \
    which \
    python3-venv

RUN echo FOO && python3 -m venv /venv --system-site-packages  && \
    . /venv/bin/activate && \
    python -m pip install --upgrade pip setuptools



COPY entrypoint/dotfiles/ /root/

RUN echo "/usr/local/bin/jupyter.sh # JupyterLab on http://127.0.0.1:8888/lab" >> ~/.bash_history

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt update -y && apt install -y git


# imgui-bundle preinstalled in the venv so the ephemeral container's
# `pip install -e .` (shell.sh/jupyter.sh) doesn't re-download it every run.
# [glfw] also pins the PyPI glfw binding the demos import.
RUN export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
       . /venv/bin/activate && \
        python3 -m pip install 'imgui-bundle[glfw]' --root-user-action=ignore

RUN export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
       . /venv/bin/activate && \
        python3 -m pip install ty --root-user-action=ignore


RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt update -y && \
    apt install -y jupyter \
             jupyterlab \
             python3-jupytext \
             fonts-mathjax \
             python3-jupyter-server-mathjax






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
