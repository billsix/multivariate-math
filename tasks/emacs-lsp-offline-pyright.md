# Emacs Python LSP: install pyright from npm so lsp-pyright attaches offline

**Status:** proposed — needs go-ahead (William Emerison Six <billsix@gmail.com>, 2026-11-05)

**Priority:** 3
**Difficulty:** 2

## BLUF

Emacs here is configured for pyright, but the image never installs pyright, so lsp-mode quietly
falls back to `ty server` (ty-ls). Completion and parameter hints are weaker as a result. The fix is
the one that worked in github.com/billsix/geometricalgebra on 2026-11-04: install the npm pyright in
the Emacs package-group script. Done means that offline, with `--network=none`, a headless Emacs
attaches pyright and ruff and answers both completion and signature help.

## Context

- **What is configured.** `entrypoint/dotfiles/.emacs.d/init.el` matches gacalc's: `lsp-mode`, then a
  `use-package lsp-pyright` block with `typechecking-mode "basic"`, then the root pinned to `/mvm/`.
  `lsp-pyright` is vendored in `entrypoint/dotfiles/.emacs.d/elpa/`, so no elisp change is needed.
- **What is installed.** `entrypoint/02-install-emacs.sh` runs only `dnf install -y emacs`. Nothing in
  the image provides `pyright-langserver`. `01-install-base.sh` dnf-installs `ty`.
- **What happens.** lsp-mode picks the highest-priority client whose server binary exists. lsp-pyright
  has priority 2 but no binary, so the client that wins is lsp-mode's built-in `ty-ls` (priority -1),
  plus ruff as an add-on. Measured on 2026-11-04 with a headless probe of this repo's image: `PROBE
  servers: (ruff ty-ls)`.
- **How gacalc fixed it.** Read `tasks/reference/emacs-lsp-setup.md` and
  `tasks/archive/2026/11/04/dockerfile-lean-layers-first.md` in github.com/billsix/geometricalgebra.
  Until the maintainer merges it, that work is on the `emacs-python-lsp-investigation` branch. The rules
  it settled:
  - Install pyright with **npm** (`npm install -g pyright` puts `pyright-langserver` in
    `/usr/local/bin`). The npm package is self-contained.
  - **Never use the pip `pyright` package.** It is only a launcher: on first use in each fresh
    container it downloads Node and the real server into `~/.cache`, so it fails offline.
  - lsp-mode shows parameter hints when you type `(` or `,`, and on demand with
    `M-x lsp-signature-activate`.

## Plan

1. **`entrypoint/02-install-emacs.sh`:** install `nodejs npm` alongside `emacs`, then
   `npm install -g pyright`. That makes the script two commands, so accumulate their exit statuses
   (`s=0; … || s=1; …; exit $s`), as github.com/billsix/towersofhanoi `python/entrypoint/04-install-emacs.sh`
   already does. Update the header comment to name the LSP backend.
   - The npm install belongs in the Emacs script, not the base one, because only Emacs uses pyright
     here. gacalc put it in its base script only because its `make check-notebooks` also runs pyright.
   - Restore `+x` after rewriting the script (`git add --chmod=+x`).
2. **`make image`**, with `USE_EMACS=1`, the Makefile default.
3. **Verify offline.** Run a headless Emacs that loads the real `init.el` in the image, using the probe
   pattern from gacalc's history (`tasks/adhoc/emacs-python-lsp-investigation/`, removed when that task
   was archived):
   - Mount the repo `:ro` and the elpa tree `:O` (overlay), so the probe writes nothing back to the host.
   - Run under `--network=none`.
   - Open a `.py` file and require `lsp-mode` to report servers `(pyright ruff)`.
   - Check that completion after a module attribute returns members.
   - Check that `lsp-signature-activate` inside a call returns a signature.
4. **Run the gates:** `make format` (ruff + ty) and the tests. Nothing in `src/` should change.
5. **Record the setup** here in a short `tasks/reference/` doc, or as a `CLAUDE.md` line, whichever the
   maintainer prefers, in the same staging.

## Open questions

None. The maintainer confirmed the gacalc setup works as wanted on 2026-11-05.
