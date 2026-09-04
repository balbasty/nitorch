# Tasks: Fix mind=True Crash

**Input**: Design documents from `/specs/002-fix-mind-bug/`

**Prerequisites**: plan.md, spec.md, research.md, quickstart.md

**Tests**: Included — the project constitution (`.specify/memory/constitution.md`
Principle III) requires a bug fix to include a test that fails before the fix and
passes after.

**Organization**: This feature has a single user story (US1), and the fix itself is a
small, additive-only change (two new functions in one existing file — see plan.md).
Setup and Foundational phases are intentionally omitted: there is no new
package/dependency to initialize and no shared infrastructure that other work would
block on beyond the fix itself.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1)
- Exact file paths are included in every task description

---

## Phase 1: User Story 1 - Use MIND-based feature registration without crashing (Priority: P1) 🎯 MVP

**Goal**: `make_image(dat, mind=True)` (and any other truthy `mind=` value), using
default settings, completes successfully and returns MIND feature maps instead of
raising `AttributeError: module 'nitorch.core.bounds' has no attribute 'zero_'`.

**Independent Test**: Build an image pyramid with MIND features enabled on a
representative 3D volume using default settings, and confirm it completes successfully
(spec.md US1 Acceptance Scenario 1).

### Tests for User Story 1 ⚠️

> Write these tests FIRST, ensure they FAIL before implementation (T004).

- [X] T001 [P] [US1] Unit test for the new `bounds.zero`/`bounds.zero_` functions in `nitorch/tests/test_bounds_zero.py`: given a tensor of indices spanning before/at/after the valid range `[0, n)`, confirm the returned index is clamped into `[0, n-1]` (safe for gathering) and the returned multiplier is exactly `1` where the original index was in-bounds and `0` where it was out-of-bounds (research.md §3, quickstart.md Scenario 3).
- [X] T002 [P] [US1] Regression test in `nitorch/tests/test_bounds_zero.py`: `make_image(dat, mind=True)` (default `bound='zero'`) completes successfully, returns an `ImagePyramid` whose level `.dat` holds MIND feature maps, and whose `.preview` holds the original intensities (spec.md FR-001, US1 Acceptance Scenario 1; quickstart.md Scenario 1). This is the exact originally-reported crash.
- [X] T003 [P] [US1] Regression test in `nitorch/tests/test_bounds_zero.py`: `make_image(dat, mind=True, bound='dct2')` (an already-working boundary condition) produces identical output before and after the fix (spec.md FR-003, SC-002, US1 Acceptance Scenario 2; quickstart.md Scenario 2).

### Implementation for User Story 1

- [X] T004 [US1] Implement `zero(i, n)` and `zero_(i, n)` in `nitorch/core/bounds.py`, mirroring the exact `(index, multiplier)` contract already used by `replicate`/`replicate_` in the same file: clamp `i` into `[0, n-1]`; return multiplier `1` where the original (unclamped) `i` was in `[0, n)` and `0` elsewhere (research.md §3). **Extended during implementation** (research.md §3b): this alone was not sufficient — it exposed a second, pre-existing latent bug in `nitorch/core/utils.py`'s `roll` and `_pad_bound`, where a mix of a real per-index multiplier tensor (now produced by `zero_`) and an untouched dimension's scalar `1` initializer crashes `torch.meshgrid`. Fixed both by wrapping remaining scalars as 1-element tensors before the meshgrid call. Depends on T001-T003 (written and failing first).
- [X] T005 [US1] Run T001-T003, confirm they now pass.

**Checkpoint**: User Story 1 is fully functional and independently testable — this is
the entire feature (single-story MVP).

---

## Phase 2: Polish & Cross-Cutting Concerns

**Purpose**: Final regression validation.

- [ ] T006 Run the full existing nitorch test suite (`nitorch/tests/`, `nitorch/io/tests/`) and confirm zero new failures (spec.md SC-003, quickstart.md Scenario 4).

---

## Dependencies & Execution Order

- T001, T002, T003 have no dependencies on each other or on T004/T005 — all three can
  run in parallel (different test functions in the same new file, no shared state).
- T004 depends on T001-T003 existing and failing first (TDD, constitution Principle III).
- T005 depends on T004.
- T006 depends on T005 (Phase 1 complete).

## Parallel Example

```bash
# Launch all three tests together:
Task: "Unit test for bounds.zero/zero_ in nitorch/tests/test_bounds_zero.py"
Task: "Regression test make_image(mind=True) in nitorch/tests/test_bounds_zero.py"
Task: "Regression test make_image(mind=True, bound='dct2') unaffected in nitorch/tests/test_bounds_zero.py"
```

## Implementation Strategy

1. Write T001-T003, confirm they fail against the current (broken) `bounds.py`.
2. Implement T004 (the two new functions).
3. Run T005 to confirm all three tests now pass.
4. Run T006 (full suite) as the final regression gate.
5. Commit per the constitution's atomic-commit discipline (Principle II) — the tests
   and the fix may be a single commit given their tight coupling and small size, or two
   commits (tests, then fix); either is atomic and reviewable. Never bundle unrelated
   changes into the same commit, and never amend.

## Notes

- [P] tasks touch the same new file but different, independent test functions with no
  shared state — safe to write/run in parallel.
- Verify T001-T003 fail before implementing T004 (constitution Principle III).
- This is a small enough fix that Setup/Foundational phases were intentionally omitted
  (see Organization note above) rather than padded with vacuous tasks.
