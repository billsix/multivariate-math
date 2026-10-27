# Push unit tests into doctests and wire them into `make test`

**Status:** proposed — needs go-ahead. **Thin:** this project currently has **no test suite and no
`make test`** (see Context) — the investigation may conclude there is little to migrate and the
real work is deciding whether to add doctests + a test gate at all.
**Priority:** 8
**Difficulty:** 3
**Created:** 2026-10-27 (William Emerison Six <billsix@gmail.com>)
**Updated:** 2026-10-27 (William Emerison Six <billsix@gmail.com>)

## BLUF

Investigate pushing unit tests into **doctests** for multivariate-math and wiring them into a
`make test` gate. "Done" (investigation) = a decision, recorded here, on whether this project
warrants doctests + a test target, and if yes, the library functions' examples live in docstrings
and `make test` runs them with `--doctest-modules`.

**Reality check first:** unlike gacalc/mvp/hanoi, this repo has **no `test_*.py` files, no pytest
config, and no `test` target in the Makefile** (targets: `all`/`image`/`shell`/`shell-exec`/
`jupyter`/`format`/`pdfs`/`update-emacs-packages`/`image-export`). So there is nothing to *migrate*
yet — the task is really "should the library modules grow doctests, and should there be a test
gate." Keep it low priority until the maintainer confirms it's worth it.

## Context (cold start)

- Repo: github.com/billsix/multivariate-math — a Python + notebooks/proofs project. Source under
  `src/` (`geometric-algebra/`, `crossproduct/`, `plotting/`), a Sphinx/`pdfs` proofs pipeline,
  container-template Makefile.
- No tests, no doctest usage in `src/` as of 2026-10-27.

## Plan

1. Decide with the maintainer whether multivariate-math should carry doctests/tests at all, or
   whether it's a demo/notebook project where this isn't worth the upkeep (if so, close this task).
2. If yes: add `--doctest-modules` pytest config + a `test: image` target to the Makefile
   (container-template shape, mirroring `shell`'s env), and add short deterministic `>>>` examples
   to the library functions (`crossproduct`, the `geometric-algebra` modules) that double as docs.
3. Green gate.

## Notes

Part of a cross-repo "doctests by default" initiative. The shared convention + lessons now live in
**`~/.claude/reference/doctests-by-default.md`** (canonical: github.com/billsix/runClaudeInContainer;
read its `## Python` section first).

**Required on completion** (if this project grows doctests at all — see the thinness caveat above):
append anything new it taught to `doctests-by-default.md` (its `## Python` section, or the
language-agnostic core if general) and re-sync the byte-identical runCrush mirror
(`runCrushInContainer tasks/reference/doctests-by-default.md`).
