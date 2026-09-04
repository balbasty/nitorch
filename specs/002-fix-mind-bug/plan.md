# Implementation Plan: Fix mind=True Crash

**Branch**: `002-fix-mind-bug` | **Date**: 2026-09-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-fix-mind-bug/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

`make_image(dat, mind=True)` currently crashes with `AttributeError: module
'nitorch.core.bounds' has no attribute 'zero_'`, because `make_image`'s
default boundary condition (`bound='zero'`) is forwarded through
`spatial.rmind` to `nitorch.core.utils.roll`, which looks up a `zero_`
index-remapping function on `nitorch.core.bounds` that was never
implemented — even though `'zero'` is documented as one of the module's
seven canonical boundary conditions and 6 of the other 7 already have a
working implementation. The fix is additive-only: implement `zero`/`zero_`
in `bounds.py` matching the existing bound-function contract used by
`replicate`/`dct1`/`dct2`/`dst1`/`dst2`. No other file needs to change —
`utils.roll`'s alias normalization already correctly resolves `'zero'`,
`'zeros'`, `'constant'`, and `BoundType.zero` to the string `'zero'`.

## Technical Context

**Language/Version**: Python (matches nitorch's existing floor, `>= 3.6` per
`setup.cfg`); the fix uses only `torch` tensor operations already used
throughout `bounds.py`.

**Primary Dependencies**: None new — `torch`, already required.

**Storage**: N/A.

**Testing**: `pytest`, matching nitorch's existing test suite
(`nitorch/tests/`).

**Target Platform**: Same as the rest of nitorch — CPU or CUDA GPU;
`bounds.zero`/`zero_` operate on plain tensors/ints with no
device-specific logic, consistent with every other function in the file.

**Project Type**: Library (single project) — this is a two-function
addition to an existing internal utility module.

**Performance Goals**: No new performance target; the added functions are
O(n) index/mask operations, matching the cost profile of the existing
`replicate`/`replicate_` functions they mirror.

**Constraints**: Zero behavioral change for any boundary condition other
than `'zero'`/its aliases (FR-003); no new dependency; no change to any
public function's signature (spec Assumptions).

**Scale/Scope**: Two new functions in one file (`nitorch/core/bounds.py`).
Exactly two existing call sites are affected end-to-end
(`nitorch/spatial/_finite_differences.py`'s `mind()` and `rmind()`, per
research.md §2's exhaustive `grep` of `utils.roll` callers).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` v1.0.0:

| Principle | Gate | Status |
|---|---|---|
| I. Code Quality | Fix the actual defect with the smallest correct change, following existing conventions. | **PASS** — `zero`/`zero_` mirror the exact existing `<name>`/`<name>_` contract every other bound function in the file already implements (research.md §3); no new pattern, no workaround-at-the-call-site. |
| II. Atomic & Regular Commits (NON-NEGOTIABLE) | Procedural — enforced during implementation (tasks.md). | **N/A at plan stage.** |
| III. Testing Discipline | A bug fix MUST include a test that fails without the fix and passes with it. | **PASS** — research.md §4 specifies both a direct regression test (the originally-reported `make_image(dat, mind=True)` crash) and a focused unit test on the new bound functions themselves. |

No violations requiring justification — Complexity Tracking table below is
empty.

**Post-Phase-1 re-check**: No Phase 1 design artifacts beyond `research.md`
and `quickstart.md` were needed (no new data entities, no new external
interface — see Project Structure below); nothing introduced since the
initial check changes this. Gate status unchanged: **PASS**.

## Project Structure

### Documentation (this feature)

```text
specs/002-fix-mind-bug/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

`data-model.md` and `contracts/` are intentionally omitted: this fix
introduces no new data entities and no new external interface (CLI flag,
public function signature, or API) — it restores previously-broken,
already-documented behavior of an existing internal function pair.

### Source Code (repository root)

```text
nitorch/
├── core/
│   ├── bounds.py                  # MODIFIED: add zero(i, n) and zero_(i, n)
│   └── utils.py                   # MODIFIED: fix meshgrid-mixing bug in roll()
│                                   # and _pad_bound(), found during implementation
│                                   # (research.md §3b)
└── tests/
    └── test_bounds_zero.py        # NEW: unit test on zero/zero_, and a
                                    # regression test for make_image(mind=True)
```

**Not touched**: `nitorch/core/utils.py` (`roll`'s alias-resolution logic
is already correct), `nitorch/spatial/_finite_differences.py` (`mind`/
`rmind` already correctly forward `bound`), `nitorch/tools/registration/
pairwise_makeobj.py` (`make_image`'s `mind=` wiring is already correct —
this was never a `make_image`-level bug).

**Structure Decision**: Single-project (library) layout — this is a
minimal, additive-only fix within nitorch's existing package layout. It
touches exactly one existing file (`bounds.py`) and adds one new test
file. No new package, module, or public interface is introduced.

## Complexity Tracking

*No Constitution Check violations — table intentionally empty.*
