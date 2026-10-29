# Parameterize bare generic type annotations

**Status:** done (2026-10-29). **Priority:** 5  **Difficulty:** 1

## BLUF

Three annotations in `src/crossproduct/` use a bare `dict` where the type
parameters are knowable from the surrounding code; a bare generic silently
degrades its parameters to implicit `Any` (per the shared Python standard,
"Parameterize generic types — never a bare generic in an annotation"). Give each
its real parameters. "Done" = the three sites parameterized and `make format`
(ruff + `ty`) green.

## Context (cold-start)

Found by an AST scan for annotations whose value is a generic container
(`dict`/`list`/…) left unsubscripted. The maintainer's rule: prefer `Any` as an
explicit parameter over no parameter at all — but here each site's exact
parameters are clear from the literal it annotates, so no `Any` is needed.

## The sites and their fixes

- `src/crossproduct/crossproduct.py` `STEP_NEXT_LABEL` — keys are `StepNumber`,
  values are label strings except the terminal step which is `None`:
  `dict[StepNumber, str | None]`.
- `src/crossproduct/crossproduct.py` `REL_FLAG` — `StepNumber` → flag-name string,
  every value a `str`: `dict[StepNumber, str]`.
- `src/crossproduct/_labels.py` `self._tex_cache` — the inline comment already
  documents it (`latex -> (tex_id, w, h) | None`): `dict[str, tuple[int, int, int]
  | None]`. (`_texture_for` assigns either `None` or `self._load_texture(png)`.)

## Verification

`make format` (containerized ruff + `ty`); no runtime behaviour changes — these are
annotation-only.
