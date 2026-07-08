# Add a `make update-emacs-packages` target (vendored-elpa model)

**Status:** COMPLETE 2026-07-08 — implemented and gated: `make image
USE_EMACS=1` builds with **zero** MELPA contact (grep of the full build log);
the refresh flow populated `entrypoint/dotfiles/.emacs.d/elpa/` (22 MB, 1129
files, elc/eln stripped, staged via `git add -A -f` for Bill to commit).
Delivered per plan: ELPA_MOUNT conditional (elpa-only, `:U,z`) wired into
shell+jupyter, gacalc-shaped `update-emacs-packages` target, Dockerfile
USE_EMACS block reduced to `dnf install emacs`, `.gitignore` +=
`*.elc`/`*.eln`. Remaining check that's Bill's: `make shell USE_EMACS=1` and
confirm emacs loads the vendored packages interactively.
**Created:** 2026-07-08

## Goal (Bill, 2026-07-08)

Give multivariate-math the same MELPA-refresh workflow as
**modelviewprojection** and **geometricalgebra**: a dedicated
`make update-emacs-packages` target that refreshes a **vendored, committed
elpa tree**, and — critically — **MELPA is never touched during a normal
`make image`**.

## Current state (what changes)

mvm has `entrypoint/dotfiles/.emacs.d/` with only the hand-written config
(`init.el`, `helm.el`, `preferences.el`, `install-melpa-packages.el`) — **no
vendored `elpa/` tree**. The 2026-07-08 Fedora Dockerfile's `USE_EMACS=1`
block runs `emacs --batch --load install-melpa-packages.el` **at image-build
time** (the old texExpToPng pattern): needs network during the build, re-downloads
on every emacs-enabled rebuild, and bakes the packages into the image rather
than the repo. That build-time install is exactly what this task removes.

## Plan (model: geometricalgebra's Makefile target, mvp's elpa-only mount)

1. **Vendor the elpa tree**: create `entrypoint/dotfiles/.emacs.d/elpa/`
   (first populated by running the new target once) and **commit it** — the
   vendored tree is intentional, like gacalc's (whose CLAUDE.md marks it
   off-limits to tooling; add the same note to this repo's docs if desired).
2. **Makefile**:
   - `ELPA_MOUNT` conditional (gacalc-style): `USE_EMACS=1` bind-mounts
     **only** `entrypoint/dotfiles/.emacs.d/elpa` → `/root/.emacs.d/elpa:U,z`
     (mvp mounts only elpa/ too — keeps any image-side `.emacs.d` content
     intact; `:U` chowns for the container user).
   - Pass `$(ELPA_MOUNT)` in `shell` (and `image`? gacalc passes it to build
     as well — decide; mvm's build no longer needs it once the MELPA step
     leaves the Dockerfile).
   - **`update-emacs-packages` target** (copy gacalc's shape verbatim):
     `$(MAKE) image USE_EMACS=1`; then a `podman run --rm` with the elpa dir
     mounted `:U,z` and `install-melpa-packages.el` mounted read-only, that
     wipes the tree (`find /root/.emacs.d/elpa -mindepth 1 -delete`) and runs
     `emacs --batch --load install-melpa-packages.el`; then host-side strip
     `*.elc`/`*.eln` (regenerated, machine-specific) and `git add -A -f` the
     tree so it's ready to commit. `## `-documented, `.PHONY`. Needs network.
3. **Dockerfile**: in the `USE_EMACS` block, keep `dnf install emacs` (and
   the bashrc alias) but **delete** the `emacs --batch --load
   install-melpa-packages.el` build-time step — a normal image build must
   not reach for MELPA, per Bill.
4. **.gitignore**: add `*.elc` and `*.eln` (so the strip step's exclusions
   hold; `git add -A -f` in the target overrides them for anything MELPA
   ships precompiled, matching gacalc).

## Verification

- `make image` (USE_EMACS=0 and =1) completes **with no MELPA/network
  package fetches** beyond dnf/pip.
- `make update-emacs-packages` populates `entrypoint/dotfiles/.emacs.d/elpa/`
  and leaves it staged; a following `make shell USE_EMACS=1` gets a working
  emacs with the vendored packages (Bill's interactive check).
- Nested-sandbox note: the target's `podman run` needs `--cgroups=disabled`
  to run nested; normally Bill runs this on the host, so don't add the flag
  to the Makefile — the standing add-run-revert arrangement covers testing it
  from the sandbox.

## References

- `geometricalgebra/Makefile` — `update-emacs-packages` + `ELPA_MOUNT`
  (elpa-only, `:U,z`), and its CLAUDE.md's "vendored tree is intentional and
  off-limits" note.
- `modelviewprojection/Makefile` — same target; mounts only `elpa/` to
  preserve the image's build-time tree-sitter grammar (a concern mvm doesn't
  have, but the narrower mount is still the right shape).
- `tasks/archive/2026/06/07/emacs-package-install-strategy.md` in
  geometricalgebra — the full rationale for vendoring.
