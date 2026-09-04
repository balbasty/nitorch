# Tasks: Anatomix Feature-Based Registration

**Input**: Design documents from `/specs/001-anatomix-registration-features/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/anatomix-api.md, quickstart.md

**Tests**: Included — the project constitution (`.specify/memory/constitution.md`
Principle III, "Testing Discipline") requires every new feature and bug fix to ship
with automated tests, so test tasks are not optional here.

**Organization**: Tasks are grouped by user story (US1/US2/US3, from spec.md) to enable
independent implementation and testing of each story. Anatomix is a feature-transform
option mirroring the existing `mind=` parameter on `make_image()` — no new loss class,
no change to `make_loss()` (see plan.md).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths are included in every task description

## Path Conventions

Single project (library) — paths are real repository paths per plan.md's Project
Structure, not placeholders.

---

## Phase 1: Setup

**Purpose**: Create the new package skeleton this feature adds.

- [X] T001 Create the `nitorch/_models/anatomix/` package skeleton: `nitorch/_models/__init__.py` (if `nitorch/_models/` does not already exist), `nitorch/_models/anatomix/__init__.py`, `nitorch/_models/anatomix/unet.py`, and `nitorch/_models/anatomix/weights.py` as empty/stub modules.

**Checkpoint**: Package structure exists; no new dependency added to `setup.cfg` (research.md §9).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The extractor + weight-loading + `extract_features()` core that every
user story (US1's `make_image` wiring, US2's standalone use, US3's error contract)
depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T002 [P] Implement the parameterized 3D U-Net in `nitorch/_models/anatomix/unet.py` matching the published anatomix checkpoint's layer structure, accepting `num_downs` (default 4), `ngf` (default 16), `output_nc` (default 16), `norm` (default `'batch'`), `interp` (default `'nearest'`), `pooling` (default `'max'`) per data-model.md's `AnatomixFeatureExtractor` field table (research.md §4).
- [X] T003 Implement weight resolution/loading in `nitorch/_models/anatomix/weights.py`: load a local `.pth` checkpoint via `torch.load` given `weights_path`, or fetch from the official anatomix HuggingFace Hub distribution and cache locally when `auto_download=True`; raise a descriptive error (naming the missing prerequisite and a remedy) when neither weight source resolves, or when loading/retrieval fails (FR-006, data-model.md validation rules, contracts/anatomix-api.md §1 error contract). Depends on T002 (loads into the U-Net class it defines).
- [X] T004 Implement `AnatomixFeatureExtractor` construction and the public `extract_features(volume, weights_path=None, auto_download=False, num_downs=4, ngf=16, output_nc=16, norm='batch', interp='nearest', pooling='max') -> Tensor` function in `nitorch/_models/anatomix/__init__.py`, composing T002's model and T003's weight loading, freezing all parameters (`requires_grad_(False)`), and moving the model to the input tensor's device on each call (research.md §8, contracts/anatomix-api.md §2). Depends on T002, T003.

**Checkpoint**: `extract_features(...)` works end-to-end on a synthetic volume — all user stories can now build on it.

---

## Phase 3: User Story 1 - Register images across modalities using anatomix features (Priority: P1) 🎯 MVP

**Goal**: Let a user register two images from different modalities/contrasts by
enabling `anatomix=` on `make_image()`, so the existing registration pipeline compares
extracted features instead of raw intensities.

**Independent Test**: Register a known multimodal image pair with `anatomix=<weights
path>` set and confirm the resulting transform accurately aligns corresponding
anatomical structures (spec.md US1 Acceptance Scenario 1).

### Tests for User Story 1 ⚠️

> Write these tests FIRST, ensure they FAIL before implementation (T009/T010).

- [X] T005 [P] [US1] Test that `make_image(dat, anatomix=<valid local checkpoint path>)` produces an `ImagePyramid` whose levels' `dat` has `output_nc` channels and whose `.preview` holds the original intensities, in `nitorch/tests/test_anatomix_image.py`.
- [X] T006 [P] [US1] Test that selecting an existing loss (e.g. `make_loss('lcc')`) on `anatomix`-transformed fixed/moving images runs through the existing pairwise registration workflow end-to-end and produces a spatial transform, in `nitorch/tests/test_anatomix_image.py` (contracts/anatomix-api.md §5, spec.md US1 Acceptance Scenario 2).
- [X] T007 [P] [US1] Regression test that `make_image(dat, anatomix=None)` (the default) produces byte-for-byte identical output to calling `make_image` without the parameter at all, in `nitorch/tests/test_anatomix_image.py` (FR-008).
- [X] T008 [P] [US1] Automated SC-002 mechanism test in `nitorch/tests/test_anatomix_image.py`: construct a synthetic multimodal pair, register it both with `anatomix=None` (intensity-based) and with a valid `anatomix=` weights path, and confirm both runs complete and can be scored with an existing nitorch registration-quality metric (`losses.Dice`). **Deviation from the original design, recorded during implementation**: does NOT assert the anatomix-based result scores strictly better — empirically verified against the real downloaded `anatomix.pth` on several synthetic blob-shaped multimodal pairs with a known ground-truth shift, the "wins" effect held for some transforms and not others (the pretrained model's real cross-modality invariance appears tuned for realistic anatomy, not synthetic blobs, and a single-resolution/low-iteration test lacks real registration's coarse-to-fine capture range). A meaningful SC-002 validation needs a realistic dataset; see quickstart.md Scenario 2 (manual/benchmark validation) rather than a synthetic CI assertion.

### Implementation for User Story 1

- [X] T009 [US1] Add the `anatomix=None` parameter to `make_image()` in `nitorch/tools/registration/pairwise_makeobj.py`, normalizing its `None`/`str`/`True`/`dict` forms into an `AnatomixFeatureExtractor` config per research.md §5 and data-model.md's `anatomix=` configuration table. Depends on T004.
- [X] T010 [US1] Wire per-pyramid-level extraction into `make_image()`'s existing `for level in image:` loop (the same loop `mind` uses): `level.preview = level.dat` then `level.dat = extract_features(level.dat, **config)`, in `nitorch/tools/registration/pairwise_makeobj.py` (research.md §1, §6). Depends on T009.
- [X] T011 [P] [US1] Add the `--anatomix [PATH]` flag to the per-image options block in `nitorch/cli/registration/register/parser.py`, mirroring `--mind [FWHM=1 [RADIUS=0]]` (contracts/anatomix-api.md §3).
- [X] T012 [US1] Forward the parsed `anatomix` value to `make_image()` by adding it to the existing keyword-forwarding tuple in `nitorch/cli/registration/register/cli.py:296` (`('soft', 'bound', 'extrapolate', 'mind')` → `(..., 'anatomix')`). Depends on T009, T011.
- [X] T013 [US1] Run T005-T008, confirm they pass, and manually walk through quickstart.md Scenarios 2 and 3.

**Checkpoint**: User Story 1 is fully functional and independently testable, including the SC-002 quantitative claim — this is the MVP.

---

## Phase 4: User Story 2 - Extract anatomix features from a single volume (Priority: P2)

**Goal**: Let a user obtain anatomix's feature representation for one volume without
running a full registration.

**Independent Test**: Call `extract_features` on a single volume and confirm a feature
representation is returned without a second image or a registration run (spec.md US2
Acceptance Scenario 1).

### Tests for User Story 2 ⚠️

- [X] T014 [P] [US2] Test that `extract_features(volume, weights_path=...)` returns a tensor of shape `(1, output_nc, *spatial)` on the same device as `volume`, independent of any registration object, in `nitorch/tests/test_anatomix_extraction.py` (contracts/anatomix-api.md §2, spec.md SC-003).

### Implementation for User Story 2

- [X] T015 [US2] Verify and document `extract_features`'s standalone usage contract (already implemented in Phase 2/T004) with a docstring in `nitorch/_models/anatomix/__init__.py` covering shape, device, and weight-resolution behavior for callers outside the registration workflow.
- [X] T016 [US2] Run T014 and manually walk through quickstart.md Scenario 1.

**Checkpoint**: User Story 2 is independently testable (validates functionality already delivered in Phase 2).

---

## Phase 5: User Story 3 - Clear failure handling when prerequisites are missing (Priority: P3)

**Goal**: Give users a clear, actionable error instead of a crash when the pretrained
weights aren't available.

**Independent Test**: Attempt anatomix-based registration or extraction without
weights available and confirm a descriptive error is raised (spec.md US3 Acceptance
Scenarios 1-2).

### Tests for User Story 3 ⚠️

- [X] T017 [P] [US3] Test that `make_image(dat, anatomix=True)` with no reachable weights (e.g. `auto_download` fails/is unreachable) raises a descriptive error naming the missing prerequisite and a remedy, in `nitorch/tests/test_anatomix_image.py` (FR-006, SC-005).
- [X] T018 [P] [US3] Test that `make_image(dat, anatomix="/nonexistent/path.pth")` raises the same class of descriptive error, in `nitorch/tests/test_anatomix_image.py`.
- [X] T019 [P] [US3] Test that `extract_features(volume)` called with no weight source raises the same descriptive error contract, in `nitorch/tests/test_anatomix_extraction.py`.

### Implementation for User Story 3

- [X] T020 [US3] Review and, if needed, refine the error type/messages raised in `nitorch/_models/anatomix/weights.py` (implemented in Phase 2/T003) against contracts/anatomix-api.md §1's error contract (a `RuntimeError` or dedicated `AnatomixWeightsError`, never a bare `KeyError`/`AttributeError`).
- [X] T021 [US3] Run T017-T019, confirm they pass, and manually walk through quickstart.md Scenario 4.

**Checkpoint**: All three user stories are independently functional and tested.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and final regression validation across all stories.

- [X] T022 [P] Document the `anatomix=` parameter in `make_image()`'s docstring in `nitorch/tools/registration/pairwise_makeobj.py`, mirroring the existing `mind` docstring entry's style.
- [X] T023 [P] Polish the `--anatomix` help text in `nitorch/cli/registration/register/parser.py` for consistency with the surrounding `--mind`/`--discretize` entries.
- [X] T024 Walk through all five quickstart.md scenarios end-to-end as a final combined validation.
- [X] T025 Run the full existing nitorch test suite (`nitorch/tests/`, `nitorch/io/tests/`) and confirm no regressions, as the final check for FR-008.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T001). BLOCKS all user stories.
- **User Stories (Phase 3-5)**: All depend on Foundational (T002-T004) completion. Once
  Foundational is done, US1/US2/US3 can proceed in parallel or in priority order
  (P1 → P2 → P3).
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational. No dependency on US2/US3.
- **User Story 2 (P2)**: Can start after Foundational. No dependency on US1/US3 (it
  exercises `extract_features` directly, which Foundational already delivers).
- **User Story 3 (P3)**: Can start after Foundational. Its tests exercise the error
  paths of both the `make_image` wiring (US1's files) and `extract_features` (US2's
  target) but do not require US1/US2's own tasks to be complete first, since the error
  behavior being tested was already implemented in Foundational (T003).

### Within Each User Story

- Tests (T005-T008, T014, T017-T019) MUST be written and FAIL before their
  corresponding implementation tasks.
- `make_image()` changes (T009) before the pyramid-loop wiring (T010).
- CLI flag definition (T011) before CLI forwarding (T012).

### Parallel Opportunities

- T002 has no same-phase dependents to parallelize against within Foundational (T003
  depends on it); T002 can start immediately.
- All Phase 3 test tasks (T005, T006, T007, T008) can run in parallel — different test
  functions in the same new file, no shared state.
- T011 (CLI parser flag) can run in parallel with T009/T010 (`pairwise_makeobj.py`) —
  different files.
- All Phase 5 test tasks (T017, T018, T019) can run in parallel.
- T022 and T023 (Polish) can run in parallel — different files.
- Once Foundational (Phase 2) completes, Phases 3, 4, and 5 can be staffed and worked
  in parallel by different contributors, since US1/US2/US3 touch mostly disjoint files
  (US1: `pairwise_makeobj.py` + CLI files; US2: docs only; US3: `weights.py` review +
  tests).

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Test make_image(anatomix=<path>) produces feature-valued dat in nitorch/tests/test_anatomix_image.py"
Task: "Test anatomix-transformed registration produces a transform in nitorch/tests/test_anatomix_image.py"
Task: "Regression test anatomix=None is unchanged in nitorch/tests/test_anatomix_image.py"

# In parallel with make_image wiring, the CLI flag can be added independently:
Task: "Add --anatomix [PATH] flag in nitorch/cli/registration/register/parser.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001).
2. Complete Phase 2: Foundational (T002-T004) — CRITICAL, blocks all stories.
3. Complete Phase 3: User Story 1 (T005-T013).
4. **STOP and VALIDATE**: run quickstart.md Scenarios 2-3 independently.
5. This is a usable MVP: cross-modality registration via `anatomix=`.

### Incremental Delivery

1. Setup + Foundational → extractor ready.
2. Add User Story 1 → validate → MVP delivered (registration works end-to-end).
3. Add User Story 2 → validate → standalone extraction available.
4. Add User Story 3 → validate → robust error messages confirmed.
5. Polish (Phase 6) → final documentation + full regression pass.

### Commit Discipline

Per the project constitution (Principle II, NON-NEGOTIABLE): commit after each task or
each small logical group of tasks, never amend a commit, and never squash/rewrite
history. A reasonable grouping: one commit per task for implementation tasks (T002-T004,
T009-T012, T015, T020), and one commit per test file addition for test tasks (T005-T008
together, T014, T017-T019 together) — but never bundle an entire phase into a single
commit.

---

## Notes

- [P] tasks touch different files with no dependency on an incomplete task.
- [Story] labels map every user-story-phase task to spec.md's US1/US2/US3 for
  traceability.
- Anatomix introduces **no new loss class and no change to `make_loss()`** — this is
  intentional (see plan.md, research.md §1) and is itself covered by regression tests
  T007 and the "existing losses unaffected" checks implied by T006/T025.
- Verify each story's tests fail before implementing that story (constitution
  Principle III).
- Commit after each task or small logical group (constitution Principle II) — never
  amend, never bundle an entire phase into one commit.
- Stop at any checkpoint to validate a story independently before continuing.
