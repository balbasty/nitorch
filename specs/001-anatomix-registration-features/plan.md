# Implementation Plan: Anatomix Feature-Based Registration

**Branch**: `001-anatomix-registration-features` | **Date**: 2026-09-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-anatomix-registration-features/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Add anatomix (a pretrained, modality-agnostic 3D U-Net feature extractor) as a new,
opt-in image **feature transform** for nitorch's existing pairwise registration
workflow, so that images from different imaging modalities/contrasts can be registered
by comparing extracted features instead of raw intensities. Technical approach: mirror
nitorch's existing `mind=` feature-transform parameter — add an `anatomix=` parameter
to `make_image()` that, per pyramid level, replaces `dat` with a 16-channel anatomix
feature map (keeping the original intensities in `.preview`). No new loss class and no
change to `make_loss()` or any existing `OptimizationLoss` subclass — whichever
existing loss the user already selects (default recommendation: `'lcc'`, per
`/speckit-clarify`) runs unchanged on the feature-valued data. A minimal,
dependency-light 3D U-Net matching the published anatomix checkpoint is vendored inside
nitorch; weights default to a user-supplied local path (no implicit network access),
with automatic download from the official distribution as an explicit opt-in. Only
anatomix's feature-extraction/model-loading surface is adopted — its own reference
registration script's separate ConvexAdam-based optimizer is out of scope, since
nitorch's existing optimizer stack already fills that role.

## Technical Context

**Language/Version**: Python (matches nitorch's existing floor, `>= 3.6` per
`setup.cfg`); the vendored U-Net and weight loading use only standard `torch` APIs, so
no language/version floor change is required for the rest of nitorch.

**Primary Dependencies**: `torch` (already required by nitorch, `>= 1.4`). No new
hard dependency is added to `install_requires` (research.md §9). Optional opt-in weight
download uses a plain HTTP request, matching the existing `wget`/`appdirs` optional
pattern already used for the `data` extra.

**Storage**: N/A (a local filesystem path to a `.pth` checkpoint file, not a database).

**Testing**: `pytest`, matching nitorch's existing test suite (`nitorch/tests/`,
`nitorch/io/tests/`).

**Target Platform**: Same as the rest of nitorch — Linux/macOS/Windows, CPU or CUDA GPU
(including HPC cluster environments without outbound network access, which motivated the
weight-loading default in `research.md` §3).

**Project Type**: Library (single project) — nitorch is a Python library with a CLI
entry point (`nitorch` console script); this feature adds a new small internal model
module and one new parameter on an existing function (`make_image`), plus matching CLI
flag wiring.

**Performance Goals**: No new performance target beyond the existing registration
pipeline's; feature extraction runs once per image, per pyramid level, at
image-preparation time (before the optimization loop starts), on whichever device
(CPU/GPU) the rest of the pipeline already uses (spec Assumptions; research.md §6, §8).

**Constraints**: No implicit network access by default (weights must be supplied
locally unless the user opts into download — `/speckit-clarify` decision); no new
required dependency; existing loss selections and every other `make_image` parameter
must remain byte-for-byte behaviorally unchanged when `anatomix=None` (FR-008).

**Scale/Scope**: One new `make_image()` parameter (`anatomix=`) plus a standalone
feature-extraction utility, scoped to nitorch's existing pairwise (two-image)
registration workflow. Out of scope: anatomix's own ConvexAdam-based optimizer
(research.md §2), fine-tuning, segmentation use cases, and 2D inputs (spec
Assumptions).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` v1.0.0:

| Principle | Gate | Status |
|---|---|---|
| I. Code Quality | New code must follow existing conventions rather than introduce a competing style. | **PASS** — `anatomix=` mirrors the existing `mind=` parameter on `make_image()` exactly (same per-level `preview`/`dat`-swap pattern); no new dispatch mechanism, no new loss abstraction. |
| II. Atomic & Regular Commits (NON-NEGOTIABLE) | Procedural — enforced during implementation (tasks.md), not a design-time gate. | **N/A at plan stage** — task breakdown (`/speckit-tasks`) will decompose the work into atomic, independently committable units. |
| III. Testing Discipline | Every new feature must ship with automated tests; every requirement must be testable. | **PASS** — `contracts/anatomix-api.md` §5 defines the required test contract (valid-weights success path, missing-weights error path, extraction shape/device check, `anatomix=None` regression check, CLI flag forwarding), directly traceable to spec FR-001–FR-008. |

No violations requiring justification — Complexity Tracking table below is empty.

**Post-Phase-1 re-check**: The Phase 1 design (data-model.md, contracts/,
quickstart.md) reuses an existing extension point (`make_image`'s `mind=`-style
feature-transform pattern) and adds zero new cross-cutting abstractions — it is, if
anything, simpler than the pre-Phase-1 design (no new `OptimizationLoss` subclass, no
`make_loss()` change, no autograd-through-network at optimization time). Gate status
unchanged: **PASS**.

## Project Structure

### Documentation (this feature)

```text
specs/001-anatomix-registration-features/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   └── anatomix-api.md
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
nitorch/
├── _models/
│   └── anatomix/                     # NEW: vendored, dependency-light model definition
│       ├── __init__.py               # public extract_features(), architecture defaults
│       ├── unet.py                   # parameterized 3D U-Net matching the checkpoint
│       └── weights.py                # local-path loading + opt-in HF Hub download/cache
├── tools/registration/
│   └── pairwise_makeobj.py           # MODIFIED: make_image() gains `anatomix=` param,
│                                      # mirroring the existing `mind=` wiring
├── cli/registration/register/
│   ├── parser.py                     # MODIFIED: add `--anatomix [PATH]` flag (mirrors --mind)
│   └── cli.py                        # MODIFIED: forward `anatomix` alongside
│                                      # ('soft', 'bound', 'extrapolate', 'mind')
└── tests/
    ├── test_anatomix_extraction.py   # NEW: Scenario 1 / US2 + error-path coverage
    └── test_anatomix_image.py        # NEW: Scenario 2/3/4/5 / US1, US3 + FR-008 regression
```

**Not touched**: `nitorch/tools/registration/losses/` (no new loss class),
`nitorch/tools/registration/pairwise_makeobj.py::make_loss()` (no new dispatch key),
`nitorch/tools/registration/objects.py` (`ImagePyramid`'s existing `dat`/`preview`
fields already support this without modification), `setup.cfg` `install_requires` (no
new hard dependency).

**Structure Decision**: Single-project (library) layout — this feature is additive
within nitorch's existing package layout. It introduces one new internal package
(`nitorch/_models/anatomix/`, underscore-prefixed as an implementation detail not part
of the public API) for the vendored architecture/weights, and touches exactly one
existing function (`make_image()`) plus its two CLI-forwarding files — mirroring
precisely how the existing `mind` feature is already wired end-to-end. No new
top-level package, CLI subcommand, loss class, or build target is introduced.

## Complexity Tracking

*No Constitution Check violations — table intentionally empty.*
