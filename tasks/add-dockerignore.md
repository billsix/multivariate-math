# Add a `.dockerignore` (stop baking the vendored elpa/ into the image)

**Status:** proposed — needs go-ahead. Created 2026-11-08 (William Emerison Six <billsix@gmail.com>).
**Priority:** 3
**Difficulty:** 1

## BLUF

The `Dockerfile` runs `COPY entrypoint/dotfiles/ /root/` with no `.dockerignore`, so the 23 MB vendored
`entrypoint/dotfiles/.emacs.d/elpa` is baked into the image, and `make shell`'s `ELPA_MOUNT` then hides
it. That `COPY` is the second layer, so every `make update-emacs-packages` also invalidates every layer
below it. `COPY src /mvm/src` picks up the host's `__pycache__/` and `src/*.egg-info/` as well. Add a
`.dockerignore` like github.com/billsix/geometricalgebra's. Done = the file exists, `make image`
succeeds, and the built image has no `/root/.emacs.d/elpa` and no `__pycache__` under `/mvm/src`.

## Context

- Dockerfile contract, `.dockerignore` bullet: the personal overlay (github.com/billsix/dotfiles
  `.ai-coding-conventions.personal.md`). Emacs side: the `emacs-container-contract.md` reference doc in
  github.com/billsix/runClaudeInContainer (`tasks/reference/`).
- Model: geometricalgebra `.dockerignore` (excludes `entrypoint/dotfiles/.emacs.d/elpa`, `.git`,
  `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`, `build/`, `dist/`).
- Host-side untracked files found on 2026-11-08 (`git status --ignored`): `.ruff_cache/`, `output/*.pdf`,
  `proofs/*.aux|log|pdf`, `__pycache__/`.

## Plan

1. Add `.dockerignore`: the elpa line (with geometricalgebra's comment), `.git`, `__pycache__/`,
   `*.py[cod]`, `.ruff_cache/`, `.pytest_cache/`, `*.egg-info/`, `output/`, `proofs/*.aux`,
   `proofs/*.log`, `proofs/*.pdf`, `*.tar`. None of these are named by a `COPY`.
2. `make image`, then
   `podman run --rm --entrypoint /bin/bash multivariate-math -c 'ls /root/.emacs.d; find /mvm/src -name __pycache__'`
   shows no `elpa` and no `__pycache__`.
3. `make shell`: Emacs still loads its packages (from the mount).
