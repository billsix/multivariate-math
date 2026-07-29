# Suppress JupyterLab's "official Jupyter news" notification prompt

**Status:** DONE 2026-07-29 — implemented and gate-verified; ready to archive

Outcome: `make image` PASSED; in the built image the plugin lists under
"Disabled extensions", `/venv/etc/jupyter/labconfig/page_config.json` has the
entry, and the jupytext default-viewer setting survived (regression check).
One better-than-planned detail: this JupyterLab version **auto-locks** the
plugin when disabling at sys-prefix level (`lockedExtensions` written
alongside `disabledExtensions`), so no separate `lock` step was needed —
users cannot re-enable it from the UI. Remaining human check: `make jupyter`
in a fresh browser profile shows no news prompt.

## Goal

`make jupyter` should come up without the "Would you like to receive
official Jupyter news?" toast. That prompt is JupyterLab's built-in
announcements plugin (`@jupyterlab/apputils-extension:announcements`)
phoning the news feed; disabling the plugin removes both the prompt and
the fetch.

## The change

In the `Dockerfile`, append to the same venv-activated RUN that already
runs `jupytext-config set-default-viewer python` (the
`python -m pip install --no-build-isolation ".[dev,notebooks,jupyter]"`
layer):

```dockerfile
    jupytext-config set-default-viewer python && \
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements"
```

## Verified in the built image (2026-07-29, current `localhost/multivariate-math`)

- With `/venv` active, the command writes
  `/venv/etc/jupyter/labconfig/page_config.json`:
  `{"disabledExtensions": {"@jupyterlab/apputils-extension:announcements": true}}`
  — sys-prefix level, so it must run with the venv activated (which the
  target RUN already does). Nothing in the Makefile mounts over `/venv`,
  so the baked config survives to `make jupyter`.
- `jupyter labextension list` then shows the plugin as disabled.
- The command prints a warning that since JupyterLab 4.1 a user can
  re-enable disabled plugins in the UI unless they are locked
  (`jupyter labextension lock`), and exits 0. For this single-user
  teaching container the lock is unnecessary — a user re-enabling it is
  making a deliberate choice; skip the lock unless that proves annoying.

## Verification

1. `make image`.
2. In the built image (venv active): `jupyter labextension list` shows
   `@jupyterlab/apputils-extension:announcements` disabled, and
   `/venv/etc/jupyter/labconfig/page_config.json` has the entry above.
3. Real check: `make jupyter`, open http://127.0.0.1:8888/lab in a fresh
   browser profile/private window (the prompt is also suppressed by a
   browser-side "answered" flag, so an already-used profile can false-pass)
   — no news prompt should appear.
