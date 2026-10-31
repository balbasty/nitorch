# Phase 0 Research: NIfTI-Zarr Volume Support

## 1. Format recognition and dispatch (FR-001, FR-002, FR-005)

- **Finding**: `nitorch.io.loadsave.map()` (`nitorch/io/loadsave.py`) dispatches to a
  registered `MappedFile`/`MappedArray` subclass purely via two classmethods every
  backend already implements: `possible_extensions()` (candidate filter, matched against
  `os.path.splitext(file_like)` — already unwraps a `.gz` suffix the same way, so a
  compound extension like `.nii.zarr` is a precedented shape) and `sniff(file_like)`
  (a cheap, non-authoritative check; the authoritative check is simply whether
  constructing the class raises `FailedReadError`). Critically, `os.path.splitext` is
  purely string-based, so it works identically whether `file_like` is a regular file or
  (as a Zarr store on the local filesystem always is) a directory.
- **Decision**: Register a new `MappedArray` subclass (`NiftiZarrArray`) into
  `nitorch/io/volumes/readers.py::reader_classes`, following the exact registration
  pattern already used by the `babel` (NIfTI/MGH) and `tiff` backends. Implement
  `possible_extensions() -> ('.zarr',)` and `sniff()` to open the store's root metadata
  and confirm it carries *either* the nifti-zarr spec's embedded NIfTI-header attribute
  *or* recognizable OME-Zarr multiscale metadata (§3 below), distinguishing it from a
  plain/generic Zarr array with neither (satisfies FR-002). Constructing the class
  raises the existing `FailedReadError` convention on anything with neither, which
  `map()` already treats as "try the next candidate" — and surfaces as nitorch's
  existing "no reader could load this file" error when no backend matches, satisfying
  FR-005 without new error-handling machinery.
- **Alternatives considered**: A bespoke, format-specific loading function outside the
  `MappedArray` system (rejected — violates FR-001's requirement to use the existing,
  format-agnostic interface, and duplicates dispatch logic that already exists).

## 2. Header metadata (FR-003)

- **Finding**: Every existing backend exposes orientation/voxel-size/dtype through the
  same `MappedArray` properties (`affine`, `voxel_size`, `dtype`, ...), populated in each
  backend's constructor from that format's native header representation (see
  `nitorch/io/volumes/babel/metadata.py::header_to_metadata` for the NIfTI/MGH analog).
  A nifti-zarr store embeds a NIfTI header verbatim (per the nifti-zarr spec) alongside
  its chunked array — the same header structure `babel`'s metadata conversion already
  knows how to read.
- **Decision**: Reuse the existing NIfTI header-to-metadata conversion
  (`babel/metadata.py::header_to_metadata`) on the header embedded in the nifti-zarr
  store, rather than re-deriving orientation/voxel-size from scratch. This directly
  satisfies SC-003 (byte-for-byte equivalent metadata to the same content's plain NIfTI
  form) because it is the same conversion code path. When no embedded header is
  present, see §3 for the fallback.

## 3. Plain OME-Zarr stores with no embedded NIfTI header (FR-009)

- **Finding**: Reading `nifti-zarr-py`'s `_zarr2nii.py` directly: when a store is a Zarr
  group and either isn't recognized as a nifti-zarr store or has no `'nifti'` attribute,
  it does *not* error — it calls `default_nifti_header(inp0, ome)` to synthesize an
  equivalent header from the store's own OME-Zarr metadata instead:
  - **Affine**: `_ome2affine()` reads per-axis scale and translation from
    `ome[0]["datasets"][level]["coordinateTransformations"]`, converting spatial units to
    millimeters and temporal units to seconds per each axis's declared `unit`.
  - **Shape**: OME axis names (`x`, `y`, `z`, `c`, `t`, from `ome[0]["axes"]`) are mapped
    onto the standard NIfTI axis ordering.
  - **Header class**: `Nifti2Header` is used instead of `Nifti1Header` when any array
    dimension exceeds 2^15 (NIfTI-1's dimension field is too narrow otherwise).
  - **The one error condition**: a `ValueError` ("this is a Zarr group but not an
    OME-Zarr") is raised only when the store has *neither* OME multiscale metadata *nor*
    plain numeric level-index keys — i.e. a Zarr array/group with no recognizable
    spatial-imaging structure of any kind.
- **Decision**: `NiftiZarrArray` mirrors this exactly: on construction, if no embedded
  NIfTI header attribute is present, derive equivalent header metadata from the store's
  own OME-Zarr `coordinateTransformations`/`axes`/`units` metadata using the same
  affine-construction and header-class-selection logic, rather than requiring every
  loadable store to carry an embedded NIfTI header. `sniff()`/construction only raises
  `FailedReadError` (FR-005) when neither an embedded NIfTI header nor recognizable OME
  multiscale metadata is present — matching `nifti-zarr-py`'s own single error
  condition.
- **Rationale**: A nifti-zarr store *is* an OME-Zarr store, with an additional embedded
  NIfTI header — treating "has an embedded NIfTI header" as an optional refinement
  rather than a hard requirement for loadability is both what the reference
  implementation already does and the smallest correct extension of FR-002/FR-003
  (no new backend, no new dispatch mechanism — just a fallback branch in the same
  header-construction code path).
- **Alternatives considered**: Requiring an embedded NIfTI header and rejecting plain
  OME-Zarr stores (rejected — diverges from the reference implementation's own behavior
  for no clear benefit, and would reject a large class of otherwise-perfectly-readable
  neuroimaging-relevant Zarr stores); re-deriving the header via a from-scratch spatial
  convention instead of mirroring `_ome2affine()`'s exact derivation (rejected — risks
  producing an affine that doesn't match what `nifti-zarr-py` itself would derive from
  the same store, breaking SC-006's exactness guarantee).

## 4. Lazy, chunked array access (FR-004)

- **Finding**: No existing nitorch backend returns a lazily evaluated array — every
  current `.data()` implementation eagerly materializes into memory (`mapping.py`'s own
  docstring: "Load the array **in memory**"). Zarr stores are natively chunked, and
  `dask.array` is the standard, well-supported way to wrap a Zarr array as a lazily
  evaluated, chunk-addressable array without reimplementing chunked reads by hand
  (`dask.array.from_zarr`, or the array object zarr itself returns, wrapped as needed).
- **Decision**: `NiftiZarrArray` wraps its zarr array as a `dask.array.Array`. To avoid
  changing the shared `MappedArray.data()` contract (eager, in-memory) for every other
  format, `NiftiZarrArray` exposes the lazy array through a new, additive method
  (e.g. `.as_dask()`), while `.data()`/`.fdata()` still fulfill the existing eager
  contract by computing the dask array when called (so `NiftiZarrArray` remains a
  fully conforming `MappedArray` usable anywhere any other backend is, per FR-006).
  Indexing into the dask array before computing (standard dask slicing) is what
  realizes FR-004/SC-002 (only the overlapping chunks are read for a sub-region).
- **New dependency**: `zarr` and `dask` are added as a new optional extra (matching the
  existing `nibabel`/`tiff` extras in `setup.cfg`, e.g. a new `zarr` extra, included in
  the aggregate `io`/`all` extras) rather than a hard dependency — consistent with the
  project's existing minimal-core / optional-extras convention.
- **Alternatives considered**: Eagerly loading the full array in the constructor
  (rejected — defeats FR-004/User Story 2 entirely, the explicit reason to prefer
  nifti-zarr over plain NIfTI); hand-rolling chunked reads directly against the Zarr
  store without dask (rejected — reimplements what dask already does correctly, and
  dask arrays are what User Story 2's "further slice, compute on, or materialize"
  language calls for).

## 5. Fetching a specific resolution level (FR-007) — "Option A"

- **Finding**: `/speckit-clarify`'s multiscale-options.md decided that a multiscale
  nifti-zarr store exposes its finest level by default through the existing loading
  interface, with every other level individually fetchable by explicit index, rather
  than all levels being returned together as one collection (Option A over Option B).
- **Decision**: A multiscale nifti-zarr store's OME-Zarr multiscale metadata (listing
  each level's array path and scale factor) is read once at construction time.
  `NiftiZarrArray` exposes the number of available native levels and a method to
  construct a new `NiftiZarrArray` bound to a specific level's array path (reusing the
  same header/metadata, since all levels share one NIfTI header), leaving the default,
  no-argument construction bound to the finest level. This keeps every other format's
  `MappedArray` contract (and `ImagePyramid`'s default single-array construction path)
  completely unchanged (FR-006).

## 6. Reusing native levels in registration (FR-008, User Story 3) — "Integration Point 1"

- **Finding**: `nitorch register`'s `-l/--levels` option
  (`nitorch/cli/registration/register/parser.py`) flows into
  `pairwise_pyramid.pyramid_levels()`, which computes per-level target voxel
  sizes/shapes from the base image's voxel size, and then into
  `pairwise_makeobj.make_image(..., pyramid=levels, pyramid_method=...)`, which
  constructs an `ImagePyramid` that downsamples the finest loaded array once per
  requested level (`ImagePyramid._build_pyramid`, in
  `nitorch/tools/registration/objects.py`). `ImagePyramid` already builds itself
  level-by-level in a loop over the requested indices — the natural seam for
  substituting "fetch native level `i`" for "downsample to level `i`".
- **Decision**: In `ImagePyramid`'s level-construction loop, when the source `dat` is a
  `MappedArray` exposing FR-007's level-fetch capability, fetch each requested level
  that the store natively provides directly (via FR-007) instead of downsampling,
  wrapping it as an `Image` exactly as `_build_pyramid` already does for a
  self-downsampled level. For any requested level beyond what the store natively
  provides, fall back to nitorch's existing downsampling, applied from the store's
  coarsest native level (per FR-008's stated fallback rule) — i.e. the existing
  `_build_pyramid` codepath is reused unchanged as the "extend further" step, only its
  starting point changes (coarsest native level instead of the finest loaded array).
- **Rationale for the detection mechanism**: A capability check (e.g.
  `hasattr(source, '<level-fetch method>')`) rather than an `isinstance` check against a
  concrete `NiftiZarrArray` class keeps `ImagePyramid` decoupled from any specific
  backend — any future format with native multiresolution data could support the same
  capability without `ImagePyramid` needing to know about it by name, matching how
  `MappedArray` backends are already discovered generically (research.md §1).
- **Alternatives considered**: Reimplementing level-selection at the
  `pyramid_levels()`/CLI layer instead of inside `ImagePyramid` (rejected — per the
  `/speckit-clarify` discussion, this duplicates logic per format and doesn't benefit
  any other `ImagePyramid` caller besides the CLI; the chosen approach is a single,
  reusable integration point, consistent with research.md's existing preference in this
  project for fixing/extending shared mechanisms rather than one call site at a time,
  established during the `002-fix-mind-bug` work).
