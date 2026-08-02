# Triage the 302 ty diagnostics now failing the format gate

**Status:** proposed — needs go-ahead (found 2026-07-29; NOT fixed that night —
out of scope of the gate-honesty fix that surfaced it)

**Priority:** 4
**Difficulty:** 6

## How this surfaced

On 2026-07-29 `entrypoint/format.sh` was fixed to **propagate every step's
failure** (before, the script's exit was the last command's alone — the flaw
found in gacalc; see the global-CLAUDE.md section "A multi-step check script
must propagate every step's failure"). Two masked failures immediately became
visible in this repo:

1. **`ruff check` had been failing on the vendored Emacs tree** on every run —
   FIXED the same night: `[tool.ruff] extend-exclude = ["entrypoint"]` added to
   `pyproject.toml` (same mechanism/rationale as gacalc's).
2. **`ty check` reports 302 diagnostics** — real, pre-existing, and NOT fixed.
   `make format` is honestly RED until they're triaged. That is this task.

## The 302, by class (from the 2026-07-29 `make format` log)

| count | class |
|---|---|
| 219 | `unresolved-attribute` |
| 35 | `invalid-type-form` |
| 12 | `invalid-assignment` |
| 8 | `invalid-argument-type` |
| 7 | `unresolved-import` |
| 5 | `missing-argument` |
| 4 | `not-subscriptable` |
| 3 + 3 | `call-non-callable` + `possibly-missing-submodule` |
| ≤2 each | `unresolved-reference`, `not-iterable`, `unsupported-operator`, `invalid-return-type` |

Notes for the triage: several diagnostics reference **galgebra** ("Consider
explicitly importing `galgebra.mv`") — mvm imports the upstream galgebra
library, whose loose typing likely drives a chunk of the `unresolved-attribute`
mass. The instrumentation-driven method applies: categorize by class × file,
fix by class, watch the count drop; a `ty.toml`/pyproject override scoped to
galgebra-facing modules may be the right call for what upstream typing can't
support (document the reason at the suppression site).

## Also pending review: the formatter's own diff (reverted 2026-07-29)

The same gate run applied ruff fixes to 6 files under `src/` — including an
**import reordering** in `src/crossproduct/crossproduct.py` (`from _labels
import LabelRenderer` hoisted into the sorted import block) and an
`import OpenGL.GL as GL` → `from OpenGL import GL` rewrite. These were
REVERTED that night rather than staged: mvp's conventions document that import
order around glfw/OpenGL/imgui can be **semantically load-bearing**, and the
GL demos couldn't be eyeball-verified at that hour. The next `make format`
run will re-apply them deterministically — review them then (run the
crossproduct demo before trusting the reorder), or pin the intended order
with an isort-suppression comment if the order IS required.

## Interim state

`make format` exits nonzero (honest). Ruff passes clean on the repo's own
source only in the sense that its findings are auto-fixable -- see above. Nothing in this repo's
own gate conventions says format must gate other work — but be aware it is red
until this task is done.
