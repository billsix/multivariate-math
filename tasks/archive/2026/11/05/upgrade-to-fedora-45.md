# Upgrade the container base from Fedora 44 to Fedora 45 (done)

Done 2026-11-05 (William Emerison Six <billsix@gmail.com>). Part of a fleet-wide
Fedora 44 → 45 bump.

## What changed

- `Dockerfile`: `registry.fedoraproject.org/fedora:44` → `fedora:45`.

No source changes were needed. `make image` is green nested on the 45 base —
the full image built: the Python package + `[dev,notebooks,jupyter]` extras
installed cleanly under **Python 3.15**, the Jupyter+XeLaTeX toolchain, and the
embedded texExpToPng meson/clang build (pinned SHA `fbbd9a3`). Image
`localhost/multivariate-math:latest`.

The feared Python-3.15 wheel-availability gap did not materialize: the
`--system-site-packages` venv picks up numpy/glfw/pyopengl/etc. from dnf
`python3-*`, and the remaining PyPI deps resolved under 3.15 with no fallback
build failures. The texExpToPng C build was clean under LLVM 23.
