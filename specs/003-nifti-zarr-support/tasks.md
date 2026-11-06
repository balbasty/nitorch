# Tasks: NIfTI-Zarr Volume Support

**Input**: Design documents from `/specs/003-nifti-zarr-support/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/niftizarr-api.md, quickstart.md

**Tests**: Included — the project constitution (`.specify/memory/constitution.md`
Principle III, "Testing Discipline") requires every new feature to ship with
automated tests.

**Organization**: Tasks are grouped by user story (US1/US2/US3, from spec.md). A new
`NiftiZarrArray` backend registers into nitorch's existing `MappedArray`/`reader_classes`
system (no new dispatch mechanism); `ImagePyramid`'s existing level-construction loop
gains a native-level-fetch capability check (no new registration mechanism either).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths are included in every task description

## Path Conventions

Single project (library) — paths are real repository paths per plan.md's Project
Structure, not placeholders.

---

## Phase 1: Setup

**Purpose**: Create the new package skeleton and dependency extra this feature adds.

- [X] T001 Create the `nitorch/io/volumes/zarr/` package skeleton: `__init__.py`, `array.py`, `metadata.py`, and `ome_header.py` as empty/stub modules.
- [X] T002 [P] Add a new `zarr` extra (`zarr`, `dask`) to `setup.cfg`'s (repo root) `[options.extras_require]`, and include it in the aggregate `io`/`all` extras, mirroring the existing `nibabel`/`tiff` extras (research.md §4).

**Checkpoint**: Package structure and dependency extra exist; no existing extra modified.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: A constructible `NiftiZarrArray` that correctly recognizes a store and
exposes its header metadata — embedded or OME-derived — that every user story builds on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 Implement store-opening and metadata detection in `nitorch/io/volumes/zarr/array.py`: open the Zarr group/array at a given path and parse its root/group-level attributes to locate (a) an embedded NIfTI header attribute if present, and (b) OME multiscale metadata (`multiscales`/`ome` key) if present (research.md §1).
- [X] T004 Implement `NiftiZarrArray.possible_extensions()` (`('.zarr',)`) and `.sniff()` in `array.py`, using T003: recognized if either an embedded NIfTI header or OME multiscale metadata is present (FR-002, research.md §1). Depends on T003.
- [X] T005 Register `NiftiZarrArray`: append it to `reader_classes` at the end of `array.py` (mirroring `babel/array.py`/`tiff/array.py`'s own `reader_classes.append(...)` pattern exactly); add `zarr`/`dask` availability checks to `nitorch/io/optionals.py` (mirroring the existing `nibabel`/`tifffile` checks); add a conditional `from .zarr import NiftiZarrArray` to `nitorch/io/volumes/__init__.py`, gated on `optionals.zarr and optionals.dask` (mirrors the existing `babel`/`tiff` conditional imports) (research.md §1).
- [X] T006 [P] Implement embedded-NIfTI-header metadata population in `nitorch/io/volumes/zarr/metadata.py`, reusing `babel/metadata.py::header_to_metadata` on the header embedded in the store (FR-003, research.md §2).
- [X] T007 [P] Implement the OME-Zarr-derived header fallback in `nitorch/io/volumes/zarr/ome_header.py`: per-axis scale/translation from `coordinateTransformations` to an affine (unit-converted to mm/s), OME axis names mapped to the standard shape ordering, and `Nifti1Header`/`Nifti2Header` selection based on whether any dimension exceeds 2^15 — mirroring `nifti-zarr-py`'s `default_nifti_header()`/`_ome2affine()` exactly (FR-009, research.md §3).
- [X] T008 Wire T006/T007 into `NiftiZarrArray.__init__`/`affine`/`voxel_size`/`dtype` in `array.py`: use the embedded header (T006) when present, otherwise the OME-derived header (T007) (FR-003, FR-009). Depends on T003, T006, T007.
- [X] T009 Implement `FailedReadError` raising in `NiftiZarrArray.__init__` (`array.py`) when neither an embedded NIfTI header nor recognizable OME multiscale metadata is found (FR-005, data-model.md validation rules). Depends on T003.
- [X] T010 Implement `.data()`/`.fdata()` on `NiftiZarrArray` in `array.py`, eagerly computing the store's zarr array so it fulfills the existing `MappedArray` contract and is usable anywhere any other backend already is (FR-006). Depends on T003.

**Checkpoint**: `NiftiZarrArray` can be constructed from a valid nifti-zarr or plain OME-Zarr store and exposes correct header metadata; an unrecognizable path raises the expected error.

---

## Phase 3: User Story 1 - Load a nifti-zarr volume through the existing loading interface (Priority: P1) 🎯 MVP

**Goal**: `nitorch.io.map()`/`load()` recognizes and loads both nifti-zarr stores
(embedded header) and plain OME-Zarr stores (derived header), exposing correct header
metadata, with a clear error for anything else.

**Independent Test**: Point `map()` at a valid nifti-zarr store and confirm it is
recognized, loaded, and exposes the same orientation/voxel-size metadata a NIfTI file
of the same content would (spec.md US1 Acceptance Scenario 1).

### Tests for User Story 1 ⚠️

> Write these tests FIRST — they should pass immediately given Foundational is already
> complete, but confirm the end-to-end `map()`/`load()` path, not just `NiftiZarrArray`
> in isolation.

- [X] T011 [P] [US1] Test that `nitorch.io.map()`/`load()` on a nifti-zarr store (embedded header) produces header metadata byte-for-byte equivalent to loading the same content's plain NIfTI file, in `nitorch/io/tests/test_niftizarr.py` (SC-001, SC-003, contracts §1, quickstart Scenario 1).
- [X] T012 [P] [US1] Test that `map()` on a plain OME-Zarr store (no embedded header) succeeds, with derived affine/voxel-size matching the store's own OME-Zarr metadata exactly, in `nitorch/io/tests/test_niftizarr.py` (FR-009, SC-006, quickstart Scenario 3c).
- [X] T013 [P] [US1] Test that `map()` on a path that is neither a nifti-zarr store nor a recognizable OME-Zarr store raises nitorch's existing no-matching-reader error, in `nitorch/io/tests/test_niftizarr.py` (FR-005, SC-004, quickstart Scenario 3).
- [X] T014 [P] [US1] Regression test that loading existing NIfTI/MGH/TIFF files through `map()`/`load()` is unchanged, in `nitorch/io/tests/test_niftizarr.py` (FR-006).

### Implementation for User Story 1

- [X] T015 [US1] Run T011-T014, confirm they pass, and manually walk through quickstart.md Scenarios 1, 3, and 3c.

**Checkpoint**: User Story 1 is fully functional and independently testable — this is the MVP.

---

## Phase 4: User Story 2 - Access nifti-zarr data without loading the whole volume into memory (Priority: P2)

**Goal**: The store's array data is available as a lazily evaluated, chunk-addressable
array, so a sub-region read only touches the overlapping on-disk chunks.

**Independent Test**: Load a nifti-zarr store larger than a small threshold, read a
small sub-region, and confirm only the corresponding chunk(s) were read (spec.md US2
Acceptance Scenario 1).

### Tests for User Story 2 ⚠️

- [X] T016 [P] [US2] Test that `.as_dask()` returns an uncomputed `dask.array.Array`, in `nitorch/io/tests/test_niftizarr.py` (contracts §2).
- [X] T017 [P] [US2] Test that slicing the dask array before `.compute()` only reads the overlapping chunks (verifiable via a chunk-access counter/mock), succeeding for a store whose full size would exceed a small memory budget, in `nitorch/io/tests/test_niftizarr.py` (FR-004, SC-002, quickstart Scenario 2).
- [X] T018 [P] [US2] Test that `.data()`/`.fdata()` still behave correctly once implemented via computing the dask array, in `nitorch/io/tests/test_niftizarr.py` (FR-006 regression guard for this format).

### Implementation for User Story 2

- [X] T019 [US2] Implement `.as_dask()` on `NiftiZarrArray` in `array.py`, wrapping the store's zarr array as a `dask.array.Array` via `dask.array.from_zarr` (FR-004, research.md §4). Depends on Foundational T003.
- [X] T020 [US2] Update `.data()`/`.fdata()` (T010) to compute the dask array from T019 instead of duplicating the zarr-opening logic, keeping a single source of truth for array access. Depends on T019.
- [X] T021 [US2] Run T016-T018, confirm they pass, and manually walk through quickstart.md Scenario 2.

**Checkpoint**: User Story 2 is independently testable.

---

## Phase 5: User Story 3 - Reuse a nifti-zarr store's native pyramid during registration (Priority: P3)

**Goal**: `ImagePyramid` (and thus `nitorch register`'s `-l/--levels`) uses a
multiscale nifti-zarr store's own native resolution data instead of re-deriving it by
downsampling, falling back to downsampling only beyond what the store natively provides.

**Independent Test**: Register using a multiscale nifti-zarr store with a specific set
of resolution levels requested, and confirm the data used at each level matches the
store's own native data for that level (spec.md US3 Acceptance Scenario 1).

### Tests for User Story 3 ⚠️

- [X] T022 [P] [US3] Test that a multiscale `NiftiZarrArray`'s level-fetch capability returns a new `NiftiZarrArray` bound to the requested native level's array path, sharing the same header, in `nitorch/io/tests/test_niftizarr.py` (FR-007, contracts §3, quickstart Scenario 4).
- [X] T023 [P] [US3] Test that `ImagePyramid` built from a multiscale nifti-zarr source uses the store's native data — compared directly against the store's own per-level arrays — for every requested level the store provides, rather than downsampling, in `nitorch/tests/test_niftizarr_registration.py` (FR-008, SC-005, quickstart Scenario 5).
- [X] T024 [P] [US3] Test that `ImagePyramid` falls back to nitorch's existing downsampling, applied from the store's coarsest native level, for any requested level beyond what the store natively provides, in `nitorch/tests/test_niftizarr_registration.py` (FR-008 fallback clause).
- [X] T025 [P] [US3] Regression test that `ImagePyramid` built from a non-nifti-zarr source (e.g. plain NIfTI) still downsamples exactly as before, in `nitorch/tests/test_niftizarr_registration.py` (FR-006 regression guard).

### Implementation for User Story 3

- [X] T026 [US3] Parse OME-Zarr multiscale metadata (level array paths + scale factors) in `array.py`/`ome_header.py` and implement the level-fetch capability on `NiftiZarrArray`: number of native levels, plus a method constructing a `NiftiZarrArray` bound to a specific level's array path and sharing the same header (FR-007, research.md §5). Depends on Foundational T003, T007.
- [X] T027 [US3] Add the native-level-fetch capability check to `ImagePyramid`'s level-construction loop in `nitorch/tools/registration/objects.py`: when the source exposes T026's capability, fetch each requested native level directly instead of downsampling; fall back to the existing downsampling from the coarsest native level for levels beyond what's available. Use a capability (`hasattr`-style) check, not an `isinstance` check against `NiftiZarrArray` (FR-008, research.md §6). Depends on T026.
- [X] T028 [US3] Run T022-T025, confirm they pass, and manually walk through quickstart.md Scenario 5.

**Checkpoint**: All three user stories are independently functional and tested.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and final regression validation across all stories.

- [X] T029 [P] Add/verify docstrings for `NiftiZarrArray`, `.as_dask()`, and the level-fetch method in `nitorch/io/volumes/zarr/array.py`, describing shapes/dtypes/behavior per Constitution Principle I.
- [X] T030 [P] Document the new `zarr` extra alongside the existing `nibabel`/`tiff` extras documentation (README or equivalent install docs).
- [X] T031 Walk through all of quickstart.md's scenarios (1, 2, 3, 3b, 3c, 4, 5, 6) end-to-end as a final combined validation.
- [X] T032 Run the full existing nitorch test suite (`nitorch/tests/`, `nitorch/io/tests/`) and confirm zero new failures, as the final check for FR-006. Result: 1242 passed, 2 failed (`test_babel.py::test_nifti`, `test_babel.py::test_mgh` — pre-existing, unrelated to this feature: nibabel API drift and a network-dependent fixture 404, confirmed via `git stash` earlier in this session), 0 new regressions. All 13 new nifti-zarr tests pass.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T001). BLOCKS all user stories.
- **User Stories (Phase 3-5)**: All depend on Foundational (T003-T010) completion. Once
  Foundational is done, US1/US2/US3 can proceed in parallel or in priority order.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational. No dependency on US2/US3 — its
  tests exercise capability Foundational already delivers.
- **User Story 2 (P2)**: Can start after Foundational. Independent of US1/US3 (adds
  `.as_dask()`, doesn't change recognition or header logic).
- **User Story 3 (P3)**: Can start after Foundational. Its `array.py` work (T026)
  depends only on Foundational, not on US1/US2's own tasks; its `ImagePyramid` work
  (T027) is independent of the IO-layer stories entirely.

### Within Each User Story

- Tests (T011-T014, T016-T018, T022-T025) MUST be written and confirmed passing/failing
  appropriately before/alongside their corresponding implementation tasks.
- Within Foundational: T004 depends on T003; T008 depends on T003/T006/T007; T009
  depends on T003.
- Within US2: T020 depends on T019.
- Within US3: T027 depends on T026.

### Parallel Opportunities

- T002 (Setup) can run in parallel with T001.
- T006 and T007 (Foundational) can run in parallel — different files, both depend only
  on T003.
- All Phase 3 test tasks (T011-T014) can run in parallel — different test functions in
  the same file, no shared state.
- All Phase 4 test tasks (T016-T018) can run in parallel.
- All Phase 5 test tasks (T022-T025) can run in parallel — T022 is in a different file
  from T023-T025.
- T029 and T030 (Polish) can run in parallel.
- Once Foundational completes, US1/US2/US3 can be staffed in parallel: US1 only adds
  tests, US2 touches `array.py`'s array-access methods, US3 touches `array.py`'s
  multiscale parsing plus `objects.py` — largely disjoint from US2's changes within
  `array.py` (different methods) and entirely disjoint from `objects.py`.

---

## Parallel Example: Foundational

```bash
# Launch the two independent header-derivation tasks together (after T003):
Task: "Embedded-NIfTI-header metadata population in nitorch/io/volumes/zarr/metadata.py"
Task: "OME-Zarr-derived header fallback in nitorch/io/volumes/zarr/ome_header.py"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002).
2. Complete Phase 2: Foundational (T003-T010) — CRITICAL, blocks all stories.
3. Complete Phase 3: User Story 1 (T011-T015).
4. **STOP and VALIDATE**: quickstart.md Scenarios 1, 3, 3c.
5. This is a usable MVP: nifti-zarr and plain OME-Zarr stores load through the existing interface.

### Incremental Delivery

1. Setup + Foundational → a constructible, correctly-recognizing `NiftiZarrArray`.
2. Add User Story 1 → validate → MVP delivered.
3. Add User Story 2 → validate → lazy/chunked access available.
4. Add User Story 3 → validate → registration reuses native pyramids.
5. Polish (Phase 6) → final documentation + full regression pass.

### Commit Discipline

Per the project constitution (Principle II, NON-NEGOTIABLE): commit after each task or
small logical group, never amend, never bundle an entire phase into one commit. A
reasonable grouping: one commit per task for implementation tasks (T001-T010,
T019-T020, T026-T027), and one commit per test file addition/extension for test tasks
grouped by story (T011-T014 together, T016-T018 together, T022 together, T023-T025
together).

---

## Notes

- [P] tasks touch different files (or independent functions/methods with no shared
  state) with no dependency on an incomplete task.
- [Story] labels map every user-story-phase task to spec.md's US1/US2/US3 for
  traceability.
- `NiftiZarrArray` introduces **no new dispatch mechanism** (reuses
  `possible_extensions()`/`sniff()`/`reader_classes`) and `ImagePyramid`'s native-level
  reuse introduces **no new registration mechanism** (a capability check inside the
  existing level-construction loop) — both intentional (research.md §1, §6), and covered
  by the FR-006 regression tests (T014, T025).
- Verify each story's tests reflect the intended behavior before/alongside implementing
  that story (constitution Principle III).
- Commit after each task or small logical group (constitution Principle II) — never
  amend, never bundle an entire phase into one commit.
- Stop at any checkpoint to validate a story independently before continuing.
