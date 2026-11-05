# Emacs Python LSP: install pyright from npm so lsp-pyright attaches offline — done-record

**Completed:** 2026-11-05. William Emerison Six <billsix@gmail.com> gave the go-ahead ("go on
multivariate-math"). The setup follows github.com/billsix/geometricalgebra's
(`tasks/reference/emacs-lsp-setup.md` there), which the maintainer confirmed on 2026-11-05 works as wanted.

## What was found

- `init.el` was already configured for lsp-pyright, and `lsp-pyright` was vendored, but the image never
  installed pyright. lsp-mode therefore fell back to its built-in `ty-ls` client, plus ruff. A headless
  probe on 2026-11-04 showed `PROBE servers: (ruff ty-ls)`.
- Installing pyright alone was not enough. The vendored lsp-mode (2026-07-02) registers `ty-ls` as an
  **add-on**, so ty ran alongside pyright, because `ty` is on PATH from dnf. The result was duplicated
  completions, and signature help returned nothing. gacalc's newer lsp-mode (2026-10-09) dropped the
  add-on flag.
- `ty check` (in `format.sh`) was type-checking the vendored elpy under elpa. That was 249 of its 348
  diagnostics.

## What changed

- **`entrypoint/02-install-emacs.sh`** installs `emacs`, `nodejs` and `npm`, then runs
  `npm install -g pyright`, accumulating the exit status. The mode stays `100755`.
- **`elpa/` refreshed** with `make update-emacs-packages`. lsp-mode moved from 20260702.527 to
  20261020.1728, in which `ty-ls` is no longer an add-on.
- **`pyproject.toml`:** `[tool.ty.src] exclude = ["entrypoint"]`, matching the existing ruff exclude.
  The backlog note in `tasks/ty-diagnostics-format-gate.md` was updated with the new count.
- No `init.el` change was needed.

## Verification (2026-11-05)

- `make image USE_EMACS=1` exited 0. The image has `/usr/local/bin/pyright-langserver`, pyright 1.1.414.
- **Offline headless probe** with the real `init.el`, under `--network=none`, with the repo mounted
  `:ro` and elpa `:O`:
  - lsp-mode attached `(pyright ruff)`.
  - Completion of `math.sq` returned `isqrt` and `sqrt`.
  - Signature help inside `math.isclose(` returned the full signature.
- **`make format USE_EMACS=1`** still fails, on the pre-existing `ty` backlog: 99 diagnostics, all in
  `src/`, tracked in `tasks/ty-diagnostics-format-gate.md`.
  - Ruff's run reformatted four `src/` files that were already unformatted. Those edits were reverted,
    since they are not part of this change.

## Note

- `USE_EMACS` defaults to `0` in this repo's `Makefile`, unlike the template's `1`. Pass
  `make image USE_EMACS=1` to get Emacs and pyright. This was left as it was.
