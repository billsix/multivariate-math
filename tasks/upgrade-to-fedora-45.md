# Upgrade the container base from Fedora 44 to Fedora 45

**Status:** in progress
**Priority:** 3
**Difficulty:** 5

## BLUF

Bump `Dockerfile` from `registry.fedoraproject.org/fedora:44` to `fedora:45` and
get `make image` green nested. The image installs the Python package + its
`[dev,notebooks,jupyter]` extras, a Jupyter+XeLaTeX toolchain, and builds
texExpToPng from source. "Done" = the image builds on the 45 base.

## Context (read first)

Part of a fleet-wide Fedora 44 → 45 bump (William Emerison Six
<billsix@gmail.com>, 2026-11-05). Fedora 45 ships **Python 3.15** (was 3.14 on
44), GCC 16.2, clang/LLVM 23. Main risk: the PyPI dependency set resolving under
**Python 3.15** — a package without a 3.15 wheel falls back to a source build
that may fail (the venv is `--system-site-packages`, so numpy/scipy/pyopengl
come from dnf, which reduces but doesn't remove the risk for the `[dev,notebooks,
jupyter]` extras). Secondary: the embedded texExpToPng meson/clang build under
LLVM 23 (pinned SHA `fbbd9a3`).

- Build gate: `make image` nested (`PODMAN_RUN_FLAGS` auto-adds
  `--cgroups=disabled`).

## Plan

1. `Dockerfile`: `fedora:44` → `fedora:45`.
2. `make image` nested; on a pip resolve/build failure, identify the offending
   package and research its 3.15 status (prefer the dnf/system copy or a minimal
   version bump; log the decision here).
3. Re-run until green; stage by path; archive.

## Notes / findings

(appended as the build runs)
