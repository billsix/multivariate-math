#!/usr/bin/env bash
#
# 01-install-base.sh -- the always-needed Fedora packages: the toolchain plus the
# heavy/native Python packages that come from dnf (the venv is --system-site-packages,
# so uv/pip see these as already satisfied and don't re-download from PyPI).
#
# One package group per script, no options: WHICH optional groups also get installed is
# decided by the Dockerfile's ARG `if` blocks (or by a human choosing which scripts to
# run). Same packages during `podman build`, on a bare Fedora host, or in a guest with
# no container runtime. `dnf upgrade` stays in the Dockerfile here (it runs in its own
# earlier layer, before the dotfiles COPY), so this script is purely `dnf install`.
set -uo pipefail

if ! command -v dnf >/dev/null 2>&1; then
    echo "01-install-base.sh: needs 'dnf' (this installs Fedora packages), not found." >&2
    echo "Run on a Fedora host/guest, or inside the project's Fedora-based image." >&2
    exit 1
fi

dnf install -y \
    gcc \
    git \
    glfw \
    glib2-devel \
    libwayland-cursor \
    libwayland-egl \
    libxkbcommon \
    mesa-dri-drivers \
    mesa-libEGL \
    mesa-libGL \
    meson \
    ninja-build \
    pkgconfig \
    python3 \
    python3-matplotlib \
    python3-numpy \
    python3-pillow \
    python3-pip \
    python3-pyopengl \
    python3-setuptools \
    python3-sympy \
    python3-wheel \
    ruff \
    tmux \
    ty \
    uv \
    which
