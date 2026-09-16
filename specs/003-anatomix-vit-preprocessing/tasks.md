# Tasks: Anatomix 3D ViT Preprocessing

**Input**: Design documents from `/specs/003-anatomix-vit-preprocessing/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/anatomix-vit-api.md, quickstart.md

**Tests**: Included — the project constitution (`.specify/memory/constitution.md`
Principle III, "Testing Discipline") requires every new feature to ship with
automated tests.

**Organization**: Tasks are grouped by user story (US1/US2/US3, from spec.md).
The new ViT extractor is vendored alongside the existing anatomix U-Net in
`nitorch/_models/anatomix/` (no new package); registration-pipeline
integration extends the existing `mind=`/`anatomix=` mechanism in
`pairwise_makeobj.py` and the CLI parser/cli — no new registration entry
point or dispatch mechanism.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths are included in every task description

---

## Phase 1: Setup

**Purpose**: Resolve the one genuinely unknown fact blocking all implementation — the ViT's real architecture parameters.

- [ ] T001 Download the real `anatomix-dev-vit` checkpoint via
      `nitorch._models.anatomix.weights.resolve_weights_path(auto_download=True, variant='anatomix-dev-vit')`
      (reused unmodified, per `research.md` §1) and inspect its `state_dict`
      keys/tensor shapes to determine `AnatomixViT`'s real patch size,
      embedding dimension, depth, number of attention heads, and
      `output_nc`. Append findings to `specs/003-anatomix-vit-preprocessing/research.md`
      §4 (replacing the "deferred" note with the actual values).

**Checkpoint**: Architecture parameters known; `AnatomixViT` can now be implemented with real values instead of placeholders.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core components every user story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T002 [P] Implement `AnatomixViT` (nn.Module) in
      `nitorch/_models/anatomix/vit.py`, using the architecture parameters
      determined in T001 (FR-002). Docstring describing input/output
      shapes and dtypes (Constitution Principle I). Depends on T001.
- [ ] T003 [P] Implement `SlidingWindowRunner` in
      `nitorch/_models/anatomix/sliding_window.py`: pads to exactly 128³
      (replicate padding, crop back after) when every spatial axis of the
      input is ≤ 128; otherwise tiles into overlapping 128³ windows and
      reassembles via blended overlap (FR-002) — per `research.md` §2 and
      `data-model.md`. Docstring per Constitution Principle I.
- [ ] T004 Extend `nitorch/_models/anatomix/weights.py`: confirm
      `resolve_weights_path(variant='anatomix-dev-vit')` resolves correctly
      against the real checkpoint from T001 (per `research.md` §1, expected
      to work unmodified) (FR-005); add a ViT-specific state-dict loader
      (mirroring `load_state_dict_into`/`_remap_flat_sequential_state_dict`'s
      pattern, adapted to the ViT's own key structure) for cases where a
      direct `load_state_dict` fails. Depends on T001, T002.
- [ ] T005 Add `AnatomixViTFeatureExtractor` and `VIT_ARCHITECTURE_DEFAULTS`
      to `nitorch/_models/anatomix/__init__.py` (FR-005): wires
      `AnatomixViT` + `SlidingWindowRunner` + `weights.py` weight
      resolution + the `research.md` §3 per-voxel feature normalization
      together, mirroring `AnatomixFeatureExtractor`'s existing structure
      (lazy build-on-first-use, frozen weights). Depends on T002, T003, T004.
- [ ] T006 Generalize `_normalize_anatomix_config` in
      `nitorch/tools/registration/pairwise_makeobj.py` into a shared helper
      usable for both the existing `anatomix=` and the new `anatomix_vit=`
      (same `None`/`False`/`True`/`str`/`dict` normalization contract).
      Regression guard: existing `anatomix=` behavior MUST be unchanged
      (FR-003) — covered by T007.

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 - Register using ViT-based anatomix features (Priority: P1) 🎯 MVP

**Goal**: A user can enable the anatomix 3D ViT feature extractor through the
same entry points (`make_image()`, `nitorch register`) already used for
MIND/U-Net anatomix, on an input of any shape, with existing
mind/anatomix-only behavior completely unaffected when it's not enabled.

**Independent Test**: Run `nitorch register ... --anatomix-vit ...` (or the
equivalent `make_image(dat, anatomix_vit=True)` call) on a real single
-channel volume and confirm registration completes and produces a
transform, exactly as it does today with `--anatomix`.

### Tests for User Story 1 ⚠️

> Write these tests FIRST; confirm they fail before implementing T010-T012

- [ ] T007 [P] [US1] Regression test in `nitorch/tests/test_anatomix_image.py`:
      `make_image()` calls that do not pass `anatomix_vit` produce
      byte-identical output to before this feature (FR-003). Fails before
      T010 (parameter doesn't exist yet → `TypeError`), passes after.
- [ ] T008 [P] [US1] Standalone extractor test in new
      `nitorch/tests/test_anatomix_vit_extraction.py`: a synthetic fixture
      checkpoint (mirroring `test_anatomix_extraction.py`'s
      `fake_checkpoint` pattern, matching `AnatomixViT`'s real shapes from
      T002) run through `AnatomixViTFeatureExtractor` on a `(1, 1,
      *spatial)` input produces `(1, output_nc, *spatial)` output.
- [ ] T009 [P] [US1] `make_image()` integration test in
      `nitorch/tests/test_anatomix_image.py`: `anatomix_vit=True`/dict
      transforms `level.dat` into ViT features and keeps `level.preview`
      as the raw input, mirroring
      `test_make_image_anatomix_swaps_dat_keeps_preview`. Also covers
      FR-002's "at every pyramid level" clause: call with
      `pyramid=[0, 1, 2]` and assert every level's `.dat` was transformed
      into ViT features (not just level 0).

### Implementation for User Story 1

- [ ] T010 [US1] Add `anatomix_vit=` parameter to `make_image()` in
      `nitorch/tools/registration/pairwise_makeobj.py` (FR-001), using the
      generalized config normalization (T006) and
      `AnatomixViTFeatureExtractor` (T005), extending the existing
      mind/anatomix feature-concatenation block. Depends on T005, T006,
      and T007-T009 (tests exist and fail first).
- [ ] T011 [US1] Add `--anatomix-vit [PATH]` option to the `file` group in
      `nitorch/cli/registration/register/parser.py` (FR-001), mirroring
      the existing `--anatomix` option's `nargs`/`convert`/`action`
      exactly (bare flag = auto-download, string = local path).
- [ ] T012 [US1] Thread `anatomix_vit` through
      `nitorch/cli/registration/register/cli.py`'s `build_losses` (and any
      other `anatomix`-forwarding call site) the same way `anatomix`
      already is (FR-001). Depends on T010, T011.
- [ ] T013 [US1] CLI end-to-end test: `nitorch register ... --anatomix-vit
      ...` completes and writes a transform file, mirroring
      `quickstart.md` Scenario 6. Add alongside existing registration CLI
      tests (or extend `nitorch/tests/test_anatomix_image.py` with a
      `run()`-based end-to-end case, matching
      `test_make_image_anatomix_registration_end_to_end`'s pattern).
- [ ] T014 [US1] Run T007-T009 and T013, confirm they pass; manually walk
      through `quickstart.md` Scenarios 1 and 6.

**Checkpoint**: User Story 1 is fully functional and independently testable

---

## Phase 4: User Story 2 - Combine ViT with MIND and/or U-Net anatomix (Priority: P2)

**Goal**: A user can enable `anatomix_vit` together with `mind` and/or
`anatomix` in the same run, with all enabled feature sets concatenated in a
fixed (mind → anatomix → anatomix_vit) channel order.

**Independent Test**: Enable MIND, U-Net anatomix, and ViT anatomix
together on the same pair of images; confirm the resulting feature
representation is the exact channel-wise concatenation of each
individually-enabled output.

### Tests for User Story 2 ⚠️

- [ ] T015 [P] [US2] Test in `nitorch/tests/test_anatomix_image.py`
      (FR-004): `anatomix` (U-Net) + `anatomix_vit` together produce
      channel-wise-concatenated output equal to `[anatomix_only.dat;
      anatomix_vit_only.dat]` — incidentally also exercises the
      differing-output-channel-count edge case (spec.md Edge Cases),
      since the U-Net's `output_nc` and the ViT's `output_nc` are not
      expected to match. Mirrors
      `test_make_image_mind_and_anatomix_concatenate_channels`.
- [ ] T016 [P] [US2] Test (FR-004): `mind` + `anatomix` + `anatomix_vit`
      all three together concatenate without error, in the fixed mind →
      anatomix → anatomix_vit order.

### Implementation for User Story 2

- [ ] T017 [US2] Verify/adjust the concatenation ordering in T010's
      implementation for the 3-way case (mind, anatomix, anatomix_vit all
      enabled) — should already follow from T010's general design; this
      task confirms via T015-T016 and fixes ordering if needed. Depends on
      T010, T015, T016.
- [ ] T018 [US2] Run T015-T016, confirm they pass; manually walk through
      `quickstart.md` Scenario 2.

**Checkpoint**: User Stories 1 AND 2 both work independently

---

## Phase 5: User Story 3 - Frictionless access to ViT weights (Priority: P3)

**Goal**: A user with no local ViT checkpoint gets automatic download +
caching, or a clear actionable error if that's not possible — mirroring the
existing U-Net's weight-resolution UX exactly.

**Independent Test**: Enable `anatomix_vit` with no local weights path;
confirm the checkpoint is fetched once, cached, and reused on a second
call; confirm a clear `AnatomixWeightsError` (not a generic crash) when
download is not requested and nothing is cached, and also when a supplied
checkpoint file is corrupt or architecturally mismatched.

### Tests for User Story 3 ⚠️

- [ ] T019 [P] [US3] Test in `nitorch/tests/test_anatomix_image.py` (or
      `test_anatomix_vit_extraction.py`) (FR-005): `anatomix_vit=True`
      with no local path resolves via
      `resolve_weights_path(auto_download=True, variant='anatomix-dev-vit')`
      and caches the result; a second call reuses the cached file without
      re-downloading (mock the network call, mirroring
      `test_make_image_anatomix_true_download_failure_raises_descriptive_error`'s
      `monkeypatch` pattern).
- [ ] T020 [P] [US3] Test (FR-006): no local path + `auto_download=False`
      (and nothing cached) raises `AnatomixWeightsError` with an
      actionable message.
- [ ] T021 [P] [US3] Test (FR-006, spec.md Edge Cases — "checkpoint file
      that does not match the expected architecture"): supplying a local
      `weights_path` that exists but is corrupt or whose `state_dict`
      shapes don't match `AnatomixViT` raises `AnatomixWeightsError` with
      an actionable message, mirroring the existing U-Net's equivalent
      test (`nitorch/tests/test_anatomix_extraction.py` /
      `load_state_dict_into`'s own error-path tests). Add to
      `nitorch/tests/test_anatomix_vit_extraction.py`.

### Implementation for User Story 3

- [ ] T022 [US3] Verify T004/T005's weight resolution already satisfies
      T019-T021 unmodified (per `research.md` §1, the existing
      `resolve_weights_path` is expected to need no changes for the new
      `variant` value); fix if any gap is found.
- [ ] T023 [US3] Run T019-T021, confirm they pass; manually walk through
      `quickstart.md` Scenario 3.

**Checkpoint**: All user stories independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Shape-handling edge cases, documentation, and final regression validation

- [ ] T024 [P] Shape-handling tests in `nitorch/tests/test_anatomix_vit_extraction.py`
      (FR-002): an input with every axis ≤ 128 (pad path) and an input
      with some axis > 128 (sliding-window path) both return features of
      the same spatial shape as the input, with no sharp discontinuity at
      tile-blend borders in the tiled case — mirrors `quickstart.md`
      Scenario 4.
- [ ] T025 [P] Multi-channel-input error test (FR-007) in
      `nitorch/tests/test_anatomix_vit_extraction.py`, mirroring
      `test_make_image_anatomix_rejects_multichannel_input` —
      `quickstart.md` Scenario 5.
- [ ] T026 [P] Degenerate-shape error test (FR-007) in
      `nitorch/tests/test_anatomix_vit_extraction.py`: a spatial shape
      `SlidingWindowRunner` cannot handle even after padding/tiling (e.g.,
      a zero-size dimension) raises a clear error rather than crashing
      inside the pad/tile logic.
- [ ] T027 [P] Add/verify docstrings for `AnatomixViT`, `SlidingWindowRunner`,
      and `AnatomixViTFeatureExtractor` describing shapes/dtypes/behavior
      per Constitution Principle I.
- [ ] T028 [P] Update `nitorch register -h 3` help text in
      `nitorch/cli/registration/register/parser.py` documenting how
      `--anatomix-vit` differs from `--anatomix` (U-Net vs. ViT, both
      usable together) (FR-008).
- [ ] T029 Walk through all of `quickstart.md`'s scenarios (1-7) end-to-end
      as a final combined validation.
- [ ] T030 Run the full existing nitorch test suite
      (`nitorch/tests/`, `nitorch/io/tests/`) and confirm zero new
      failures, as the final check for FR-003.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Phase 1 (T001's architecture
  parameters). BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Phase 2. No dependency on US2/US3.
- **User Story 2 (Phase 4)**: Depends on Phase 2 and on US1's T010
  (extends the concatenation block US1 introduces) — not independently
  implementable before US1, but independently *testable* once both exist.
- **User Story 3 (Phase 5)**: Depends on Phase 2 (T004/T005's weight
  resolution) — independent of US1/US2's registration-pipeline wiring,
  could be implemented in parallel with US1/US2 if staffed separately.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### Parallel Opportunities

- T002 and T003 (different files, no shared dependency) can run in parallel
  once T001 completes.
- All test tasks marked [P] within a phase touch independent
  test-function-level concerns and can be written in parallel, though
  several land in the same file (`test_anatomix_image.py`) so should be
  merged carefully rather than edited concurrently by literal parallel
  processes.
- US3 (Phase 5) can proceed in parallel with US1 (Phase 3) / US2 (Phase 4)
  once Phase 2 is complete, since it only depends on the weight-resolution
  layer (T004/T005), not the registration-pipeline wiring (T010-T012).

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (T001) and Phase 2 (T002-T006).
2. Complete Phase 3 (US1: T007-T014).
3. **STOP and VALIDATE**: `nitorch register --anatomix-vit ...` works
   end-to-end; existing `mind=`/`anatomix=`-only behavior is unchanged.

### Incremental Delivery

1. Setup + Foundational → architecture known, extractor buildable.
2. User Story 1 → ViT usable standalone (MVP).
3. User Story 2 → ViT combinable with MIND/U-Net anatomix.
4. User Story 3 → frictionless weight download/caching confirmed, including
   the corrupt/mismatched-checkpoint error path.
5. Polish → shape-handling edge cases (including degenerate shapes), docs,
   full regression pass.
