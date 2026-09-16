# Feature Specification: NIfTI-Zarr Volume Support

**Feature Branch**: `003-nifti-zarr-support`

**Created**: 2026-09-08

**Status**: Draft

**Input**: User description: "add support for nifti-zarr
(https://github.com/neuroscales/nifti-zarr-py) by extending the
nitorch/io/volumes to read these with a nifti header and an array as a dask
array"

## Clarifications

### Session 2026-09-11

- Q: Should a partially-written nifti-zarr store (interrupted conversion, or a chunk file its own manifest references is missing) be caught when the store is loaded, or is it acceptable for it to only surface later, when a user actually reads the specific missing chunk? → A: Load-time validation covers only structural/header validity (is this a recognizable nifti-zarr store at all); a missing chunk is not specially detected or reported at all — matching the reference `nifti-zarr-py` implementation's own behavior (`zarr2nii`/`dask.array.from_zarr`, which does no eager chunk-existence check). **Correction, verified empirically during implementation**: Zarr's own default behavior for a missing chunk is to silently return the array's fill value (typically zero) rather than raising an error at all — this is standard, intentional Zarr behavior (sparse arrays), not specific to nitorch. So the realistic consequence of a partially-written store is silently zero-filled data at the missing region, not an exception of any kind. Accepted as-is (Option A's rationale — avoiding the cost of eagerly validating every chunk's existence — implies accepting Zarr's native behavior here too, whatever it turns out to be).
- Q: How should a plain OME-Zarr store (version 0.4 or 0.5) that has no embedded NIfTI header be handled? → A: Handled the same way `nifti-zarr-py` itself handles it: still loadable, with equivalent header metadata (affine, voxel size, shape) *derived* from the store's own OME-Zarr metadata (`coordinateTransformations` scale/translation per axis, `axes` names/units) rather than requiring an embedded NIfTI header. Only a Zarr store that is neither a nifti-zarr store nor a recognizable OME-Zarr store (no OME multiscale metadata at all) is rejected as unloadable — mirroring `nifti-zarr-py`'s own `default_nifti_header()`/`_ome2affine()` fallback and its one error condition ("this is a Zarr group but not an OME-Zarr").

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load a nifti-zarr volume through the existing loading interface (Priority: P1)

As a user of nitorch's volume I/O, I want to load a nifti-zarr store the same
way I already load a NIfTI or MGH file, so that I can work with large,
chunked neuroimaging datasets without learning a new loading interface or
converting the data to a different format first.

**Why this priority**: This is the core value of the feature — without it,
nifti-zarr data cannot be used in nitorch at all. Every other capability
depends on a nifti-zarr store first being recognized and loadable.

**Independent Test**: Can be fully tested by pointing nitorch's existing
volume-loading entry point at a valid nifti-zarr store and confirming it is
recognized, loaded, and exposes the same orientation/voxel-size metadata a
NIfTI file of the same content would.

**Acceptance Scenarios**:

1. **Given** a valid nifti-zarr store on the local filesystem, **When** a
   user loads it through nitorch's existing volume-loading interface,
   **Then** the resulting object exposes the same header-derived metadata
   (orientation matrix, voxel size, data type) that loading the equivalent
   plain NIfTI file would expose.
2. **Given** a file path that is not a valid nifti-zarr store (or does not
   exist), **When** a user attempts to load it as one, **Then** nitorch
   reports a clear error rather than crashing or silently returning
   incorrect data, consistent with how other unsupported/invalid inputs are
   already handled.

---

### User Story 2 - Access nifti-zarr data without loading the whole volume into memory (Priority: P2)

As a user working with a large nifti-zarr dataset, I want the underlying
array to be accessed lazily and in chunks, so that I can read and process
data that would not fit entirely in memory, and only pay the cost of reading
the parts I actually use.

**Why this priority**: This is the specific advantage nifti-zarr offers over
plain NIfTI (chunked, lazy access), and is explicitly called out in the
request. It is secondary to User Story 1 because a user must first be able
to load the store at all before this access pattern matters.

**Independent Test**: Can be fully tested by loading a nifti-zarr store
larger than a small threshold, reading a small sub-region of it, and
confirming that only the corresponding chunk(s) were read from storage
rather than the entire array.

**Acceptance Scenarios**:

1. **Given** a loaded nifti-zarr store, **When** a user requests a small
   spatial sub-region of the data, **Then** only the on-disk chunks
   overlapping that sub-region are read.
2. **Given** a loaded nifti-zarr store, **When** a user requests the array
   representation of the full data, **Then** they receive a chunked, lazily
   evaluated array they can further slice, compute on, or materialize
   on-demand, rather than an array that was already fully materialized in
   memory when it was requested.

---

### User Story 3 - Reuse a nifti-zarr store's native pyramid during registration (Priority: P3)

As a user registering two images with nitorch, where one or both are
nifti-zarr stores containing a precomputed multiscale pyramid, I want
selecting registration resolution levels to reuse the store's own
already-built levels, so that I get the accuracy and speed benefit of a
purpose-built pyramid instead of nitorch re-deriving a coarser version from
the finest level every time.

**Why this priority**: This is a concrete payoff of nifti-zarr's multiscale
structure, but it only matters once a store can be loaded (User Story 1)
and its levels can be individually accessed (User Story 2's chunked-access
capability, extended here to non-finest levels) — a coarse-to-fine
capability nitorch's registration tooling already has a notion of.

**Independent Test**: Can be fully tested by registering using a
multiscale nifti-zarr store with a specific set of resolution levels
requested, and confirming that the data used at each requested level
matches the store's own native data for that level, for every level the
store natively provides.

**Acceptance Scenarios**:

1. **Given** a multiscale nifti-zarr store and a registration resolution
   level that the store natively provides, **When** a user selects that
   level for registration, **Then** the store's own native data for that
   level is used, rather than data derived by downsampling the finest
   level.
2. **Given** a multiscale nifti-zarr store and a registration resolution
   level beyond what the store natively provides, **When** a user selects
   that level for registration, **Then** nitorch derives it by downsampling
   further from the store's coarsest native level, consistent with how
   registration resolution levels are already derived for other formats.

---

### Edge Cases

- What happens when a nifti-zarr store's NIfTI header metadata is missing,
  incomplete, or inconsistent with its array's shape/dtype? (Resolved for
  the "missing entirely" case: FR-009 — a plain OME-Zarr store with no
  embedded NIfTI header gets an equivalent header derived from its own
  OME-Zarr metadata instead. An embedded header that is present but
  internally inconsistent with the array falls under FR-005's general
  "not structurally recognizable" error, at the implementer's discretion
  for how strictly to validate consistency.)
- What happens when a nifti-zarr store contains multiple resolution levels
  (an OME-Zarr-style multiscale pyramid)? (Resolved: FR-007/FR-008, User
  Story 3 — the finest level is the default, others are fetchable by index,
  and registration reuses native levels where available.)
- What happens when a nifti-zarr store is only partially written (e.g. an
  interrupted conversion), or a chunk file referenced by its metadata is
  missing? (Resolved: FR-005/SC-004 — this is only guaranteed to be caught
  at load time if it makes the store structurally unrecognizable. A missing
  chunk in an otherwise-valid store is not detected or reported at all:
  Zarr's own default behavior silently returns the fill value, typically
  zero, for that region rather than raising an error.)
- What happens when a user attempts to write/save to a nifti-zarr store
  rather than only reading one? (Resolved: relies on existing `MappedArray`
  framework behavior — any backend that doesn't implement `set_data()`/
  `set_metadata()`/`save_new()` already raises `FailedWriteError` by
  default; no new FR needed since writing is out of scope, per Assumptions.)
- What happens when the same file path could plausibly be matched by more
  than one registered volume reader (format ambiguity)? (Resolved: relies
  on `nitorch.io.map()`'s existing multi-candidate dispatch — it already
  tries each matching reader in order and falls through on failure; no new
  FR needed.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow a user to load a nifti-zarr store through
  nitorch's existing, format-agnostic volume-loading interface, without
  requiring a separate, format-specific loading call.
- **FR-002**: System MUST correctly recognize a valid nifti-zarr store, or a
  plain OME-Zarr store carrying its own multiscale metadata, as loadable (as
  opposed to a plain NIfTI file or a Zarr store with no recognizable
  spatial-imaging metadata at all) when presented to the loading interface.
- **FR-003**: System MUST expose header metadata (orientation matrix, voxel
  size, data type, and other metadata already exposed for other supported
  volume formats) after loading: read directly from the store's embedded
  NIfTI header when present, or otherwise derived from the store's OME-Zarr
  metadata per FR-009.
- **FR-004**: System MUST expose the store's array data as a chunked, lazily
  evaluated array, such that requesting a sub-region reads only the
  overlapping on-disk chunks rather than the entire array.
- **FR-005**: System MUST report a clear, actionable error when a path is
  not a structurally recognizable nifti-zarr *or* OME-Zarr store (missing,
  not a Zarr store, or a Zarr store with no OME multiscale metadata and no
  embedded NIfTI header) at load time, rather than crashing or returning
  incorrect data. A store that loads successfully but has one or more
  missing chunks is explicitly not covered by this requirement: per Zarr's
  own default behavior, a missing chunk silently reads as the array's fill
  value (typically zero) rather than raising any error (Clarifications,
  Session 2026-09-11).
- **FR-006**: System MUST leave existing behavior for already-supported
  volume formats (NIfTI, MGH, TIFF, etc.) unchanged.
- **FR-007**: System MUST allow a user to explicitly fetch a specific
  resolution level of a multiscale nifti-zarr store by index, in addition
  to the finest level being accessible by default (FR-003/FR-004).
- **FR-008**: When registering images with nitorch's registration tooling,
  System MUST use a multiscale nifti-zarr store's native resolution data
  (via FR-007) for any requested registration level the store natively
  provides, instead of deriving that level by downsampling the finest
  level. For any requested level beyond what the store natively provides,
  System MUST fall back to nitorch's existing downsampling, applied from
  the store's coarsest native level.
- **FR-009**: When a store has no embedded NIfTI header, System MUST derive
  equivalent header metadata (orientation matrix, voxel size, shape) from
  the store's own OME-Zarr metadata (per-axis scale and translation, axis
  names, and units), using the same derivation approach as the reference
  `nifti-zarr-py` implementation, rather than requiring every loadable
  store to carry an embedded NIfTI header (Clarifications, Session
  2026-09-11).

### Key Entities

- **NIfTI-Zarr Store**: A dataset stored in the Zarr chunked-array format
  that embeds NIfTI header metadata (orientation, voxel size, and related
  imaging parameters) alongside its chunked array data; may reside on the
  local filesystem.
- **Volume Header Metadata**: The imaging parameters (orientation matrix,
  voxel size, data type) already associated with every loaded volume in
  nitorch, regardless of source format; for a nifti-zarr store this comes
  from its embedded NIfTI header, and for a plain OME-Zarr store (no
  embedded NIfTI header) it is derived from the store's own OME-Zarr
  metadata instead (FR-009).
- **Chunked Array**: The lazily evaluated, chunk-addressable representation
  of a nifti-zarr store's data, which can be partially read without loading
  the entire dataset into memory.
- **Resolution Level**: One entry in a multiscale nifti-zarr store's
  precomputed pyramid; the finest level is accessible by default, and any
  other level is individually fetchable by index (FR-007).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can load a nifti-zarr store and read its header
  metadata using the exact same call they already use for other supported
  volume formats, with no format-specific code required.
- **SC-002**: Reading a small sub-region of a nifti-zarr store whose full
  size exceeds available memory completes successfully, without requiring
  the full store to be read.
- **SC-003**: For a nifti-zarr store produced from a given NIfTI file, the
  orientation matrix, voxel size, and data type read back from the
  nifti-zarr store match those of the original NIfTI file exactly.
- **SC-004**: Attempting to load a path that is not a structurally
  recognizable nifti-zarr store produces a descriptive error in 100% of
  attempts, rather than a crash or silently incorrect data. (A store that
  loads successfully but has one or more missing chunks is exempt from this
  guarantee: Zarr's own default behavior silently returns the fill value —
  typically zero — for a missing chunk, with no error at all; see
  Clarifications, Session 2026-09-11.)
- **SC-005**: Registering with a specific set of resolution levels against a
  multiscale nifti-zarr store uses the store's own native data for every
  requested level it natively provides — verifiable by the data at that
  level matching the store's native data exactly, rather than a downsampled
  approximation of the finest level.
- **SC-006**: For a plain OME-Zarr store with no embedded NIfTI header, the
  derived orientation matrix and voxel size read back match what the
  store's own OME-Zarr metadata (coordinate transformations and axis units)
  specifies, exactly.

## Assumptions

- Only local-filesystem nifti-zarr stores are in scope; remote/cloud store
  access (e.g. object storage such as S3) is out of scope for this feature,
  consistent with this project's existing preference for no implicit
  network access.
- Only reading is in scope; writing/saving a nitorch volume out to a
  nifti-zarr store is out of scope for this feature.
- **Decided** (see [multiscale-options.md](./multiscale-options.md) for the
  full discussion): a multiscale nifti-zarr store's resolution levels are
  exposed as the finest level by default through the existing loading
  interface (FR-003/FR-004), with any other level individually fetchable by
  explicit index (FR-007) — "Option A" in that discussion — rather than all
  levels being returned together as a single collection. This primitive is
  also what lets nitorch's registration tooling reuse a store's native
  pyramid levels directly (FR-008/User Story 3, "Integration Point 1" in
  that discussion) without a mismatched interface shape.
- Loading a nifti-zarr store is offered as an additional, automatically
  recognized format alongside existing supported formats, rather than
  requiring the user to specify the format explicitly, consistent with how
  existing formats are already auto-detected.
- Plain OME-Zarr stores (no embedded NIfTI header) are in scope for reading
  too, with derived header metadata (FR-009), matching the OME-Zarr
  versions the reference `nifti-zarr-py` implementation itself supports
  (0.4 and 0.5); this is a natural extension of the same loading path, not
  a separate feature, since a nifti-zarr store *is* an OME-Zarr store with
  an additional embedded NIfTI header.
