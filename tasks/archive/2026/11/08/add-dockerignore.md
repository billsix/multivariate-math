# Add a `.dockerignore` (stop baking the vendored elpa/ into the image)

Done 2026-11-08 (William Emerison Six <billsix@gmail.com> asked for it), on branch `add-dockerignore`
(merged).

## What changed

The new `.dockerignore` excludes `entrypoint/dotfiles/.emacs.d/elpa` (the 23 MB vendored package tree
that `COPY entrypoint/dotfiles/ /root/` was baking, and that `make shell`'s `ELPA_MOUNT` hides),
`.git`, `**/__pycache__/`, `**/*.py[cod]`, `**/.pytest_cache/`, `**/.ruff_cache/`, `**/*.egg-info/`,
`output/`, `proofs/*.aux|log|pdf` and `*.tar`.

The first version used root-only patterns (`__pycache__/`, `*.egg-info/`); `.dockerignore` patterns
are anchored at the context root, unlike `.gitignore`, so the second commit added `**/`. The trap is
in the Dockerfile contract's `.dockerignore` bullet (personal overlay, github.com/billsix/dotfiles
`.ai-coding-conventions.personal.md`).

Because the dotfiles `COPY` is the second layer, the vendored `elpa/` used to invalidate every later
layer on each `make update-emacs-packages`; it no longer does.

## Verification (2026-11-08, nested podman, default `make image` flags)

- `make image` succeeded.
- In the image: no `/root/.emacs.d/elpa`; no `__pycache__` under `/mvm`; `pip list` shows
  `multivariatemath 0.0.1`. The one `src/*.egg-info` is created by the build's own `pip install -e .`
  (timestamped at build time), not copied from the host.
