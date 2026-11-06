# Implementation Plan: NIfTI-Zarr Volume Support

**Branch**: `003-nifti-zarr-support` | **Date**: 2026-09-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-nifti-zarr-support/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Adds a new `NiftiZarrArray` backend to nitorch's existing `MappedArray` volume I/O
system, so nifti-zarr stores (NIfTI header + chunked Zarr array) load through the exact
same format-agnostic `nitorch.io.map()`/`load()` entry points every other format already
uses. The array is exposed as a lazily evaluated `dask.array.Array` (new, additive
`.as_dask()` method) so partial reads only touch the overlapping on-disk chunks, while
`.data()`/`.fdata()` keep fulfilling the existing eager `MappedArray` contract
unchanged. Per the `/speckit-clarify` decision (multiscale-options.md, "Option A"), a
multiscale store's finest level is the default and every other native level is
individually fetchable by explicit index — the same primitive that lets
`ImagePyramid`'s existing level-construction loop reuse a store's native pyramid levels
directly during registration ("Integration Point 1"), falling back to nitorch's
existing downsampling for any level beyond what the store natively provides. Plain
OME-Zarr stores with no embedded NIfTI header are also accepted, handled the same way
the reference `nifti-zarr-py` implementation does: header metadata is derived from the
store's own OME-Zarr metadata (`coordinateTransformations`/`axes`/`units`) instead of
requiring an embedded NIfTI header.

## Technical Context

**Language/Version**: Python (matches nitorch's existing floor, `>= 3.6` per
`setup.cfg`); no language-level change required.

**Primary Dependencies**: `zarr` and `dask` (new), added as a new optional extra
(mirroring the existing `nibabel`/`tiff` extras in `setup.cfg`), not a hard dependency —
consistent with the project's existing minimal-core/optional-extras convention
(research.md §4).

**Storage**: N/A beyond the nifti-zarr store itself (a local-filesystem Zarr directory
store, per spec.md's Assumptions — no remote/cloud store support in this feature).

**Testing**: `pytest`, matching nitorch's existing test suite (`nitorch/io/tests/`,
`nitorch/tests/`).

**Target Platform**: Same as the rest of nitorch — Linux/macOS/Windows; no
device-specific logic (this is CPU-side I/O, not a tensor computation on a specific
backend).

**Project Type**: Library (single project) — this feature adds one new backend package
under nitorch's existing `nitorch/io/volumes/` layout, plus an extension to the existing
`ImagePyramid` class in `nitorch/tools/registration/`.

**Performance Goals**: Reading a sub-region of a nifti-zarr store must only touch the
on-disk chunks overlapping that sub-region (FR-004/SC-002) — no numeric latency target
beyond "does not require reading the full store."

**Constraints**: No implicit network access (local-filesystem stores only, per spec.md
Assumptions); no change to any existing format's behavior through the same loading
entry points (FR-006); no change to `ImagePyramid`'s public shape, only to how each
level's data is obtained internally (data-model.md).

**Scale/Scope**: One new backend package (`nitorch/io/volumes/zarr/`), one
extension point inside `ImagePyramid`'s existing level-construction loop, and a new
optional dependency extra. Out of scope (spec.md Assumptions): remote/cloud stores,
writing/saving to nifti-zarr, and any resolution-level auto-selection beyond explicit
index-based fetch (Option A).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` v1.0.0:

| Principle | Gate | Status |
|---|---|---|
| I. Code Quality | New code follows existing conventions; smallest correct change. | **PASS** — `NiftiZarrArray` follows the exact `MappedArray` subclass + `reader_classes` registration pattern the `babel`/`tiff` backends already use (research.md §1); the registration-pyramid integration reuses `ImagePyramid`'s existing level-construction loop rather than introducing a parallel mechanism (research.md §6). |
| II. Atomic & Regular Commits (NON-NEGOTIABLE) | Procedural — enforced during implementation (tasks.md). | **N/A at plan stage.** |
| III. Testing Discipline | Every new feature ships with automated tests; every requirement testable. | **PASS** — `contracts/niftizarr-api.md` §5 defines the required test contract (metadata equivalence, invalid-store error path, OME-Zarr-derived-header verification, lazy-chunk-read verification, level-fetch, registration native-level reuse, and an explicit FR-006 regression guard for every existing format), directly traceable to FR-001–FR-009. |

No violations requiring justification — Complexity Tracking table below is empty.

**Post-Phase-1 re-check**: The Phase 1 design (data-model.md, contracts/,
quickstart.md) adds one new backend class conforming to the existing `MappedArray`
interface, two additive (not replacing) capabilities on it, and one internal extension
point inside an existing class (`ImagePyramid`) — no new cross-cutting abstraction.
Gate status unchanged: **PASS**.

## Project Structure

### Documentation (this feature)

```text
specs/003-nifti-zarr-support/
├── plan.md                    # This file (/speckit-plan command output)
├── research.md                # Phase 0 output (/speckit-plan command)
├── data-model.md               # Phase 1 output (/speckit-plan command)
├── quickstart.md               # Phase 1 output (/speckit-plan command)
├── contracts/                  # Phase 1 output (/speckit-plan command)
│   └── niftizarr-api.md
├── multiscale-options.md       # Pre-plan discussion doc (this feature's clarify phase)
└── tasks.md                    # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
nitorch/
├── io/volumes/
│   ├── zarr/                         # NEW: nifti-zarr backend package
│   │   ├── __init__.py               # registers NiftiZarrArray into reader_classes
│   │   ├── array.py                  # NiftiZarrArray (MappedArray subclass)
│   │   ├── metadata.py               # embedded-NIfTI-header <-> metadata conversion
│   │   │                              # (reuses babel/metadata.py::header_to_metadata)
│   │   └── ome_header.py             # NEW: derives header metadata from OME-Zarr
│   │                                  # coordinateTransformations/axes/units when no
│   │                                  # embedded NIfTI header is present (research.md §3)
│   └── __init__.py                   # MODIFIED: conditional `from .zarr import NiftiZarrArray`
│                                      # gated on optionals.zarr/optionals.dask (mirrors
│                                      # the existing babel/tiff conditional imports)
├── io/
│   └── optionals.py                  # MODIFIED: add zarr/dask availability checks
│                                      # (mirrors the existing nibabel/tifffile checks)
├── tools/registration/
│   └── objects.py                    # MODIFIED: ImagePyramid's level-construction loop
│                                      # gains the native-level-fetch check (research.md §6)
├── io/tests/
│   └── test_niftizarr.py             # NEW: IO-layer contract tests (US1, US2; data-model.md, contracts/)
└── tests/
    └── test_niftizarr_registration.py  # NEW: ImagePyramid/registration integration tests (US3)
```

**Not touched**: `nitorch/io/volumes/mapping.py` (`MappedArray`'s shared base contract
is unchanged — the new capabilities are additive methods on `NiftiZarrArray` itself, not
new abstract requirements every backend must implement), `nitorch/io/volumes/babel/`,
`nitorch/io/volumes/tiff/` (existing backends untouched, per FR-006),
`nitorch/tools/registration/pairwise_makeobj.py` and `pairwise_pyramid.py` (no CLI flag
or argument changes — Integration Point 1 is entirely internal to `ImagePyramid`,
per contracts/niftizarr-api.md §4), `setup.cfg`'s existing extras (only a new extra is
added, none modified).

**Structure Decision**: Single-project (library) layout. This feature is additive within
nitorch's existing `io/volumes/` package layout (one new backend package, following the
exact structure of `babel`/`tiff`) plus one small, well-contained extension inside
`ImagePyramid`'s existing level-construction loop. No new top-level package, CLI flag,
or public API surface beyond the two additive `NiftiZarrArray` methods.

## Complexity Tracking

*No Constitution Check violations — table intentionally empty.*
