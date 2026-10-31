# Phase 1 Data Model: NIfTI-Zarr Volume Support

Derived from the spec's Key Entities section, made concrete against nitorch's existing
`MappedArray` architecture (`nitorch/io/volumes/mapping.py`) and the `babel` backend it
already mirrors (`nitorch/io/volumes/babel/`).

## NiftiZarrArray (new `MappedArray` subclass)

Lives in a new `nitorch/io/volumes/zarr/` package, mirroring the existing
`babel`/`tiff` backend packages.

| Aspect | Value |
|---|---|
| Base class | `nitorch.io.volumes.mapping.MappedArray` |
| Registration | Appended to `nitorch/io/volumes/readers.py::reader_classes`, per the existing pattern |
| `possible_extensions()` | `('.zarr',)` |
| `sniff()` | Confirms the store's root metadata carries either the nifti-zarr spec's embedded NIfTI-header attribute or recognizable OME-Zarr multiscale metadata (distinguishes from a plain/generic Zarr array with neither) |
| `affine`, `voxel_size`, `dtype`, ... | Populated from the embedded NIfTI header via the existing `babel/metadata.py::header_to_metadata` conversion when present (research.md §2); otherwise **derived** from the store's own OME-Zarr metadata (`coordinateTransformations`/`axes`/`units`), mirroring `nifti-zarr-py`'s `default_nifti_header()`/`_ome2affine()` (research.md §3, FR-009) |
| `.data()` / `.fdata()` | Existing eager `MappedArray` contract — fulfilled by computing the underlying dask array (FR-006: other formats' behavior, and this format's own base contract, are both preserved) |
| `.as_dask()` (new, additive) | Returns the lazily evaluated `dask.array.Array` backing this level's data, per FR-004 |
| Level-fetch capability (new, additive) | Given a multiscale store, exposes the number of native levels and a way to construct a `NiftiZarrArray` bound to a specific level's array path, reusing the same header (FR-007); absent/inapplicable for a single-scale store |

**Validation rules**:
- Constructing a `NiftiZarrArray` on a path that is not a valid nifti-zarr store MUST
  raise the existing `MappedArray.FailedReadError` convention (FR-005), which
  `nitorch.io.map()` already treats as "try the next candidate" / surfaces as nitorch's
  existing no-matching-reader error when nothing matches.
- Requesting a level index beyond what a store natively provides is not this entity's
  concern — see `ImagePyramid`'s fallback behavior below (FR-008).

## Resolution Level (conceptual, not a new class)

Not a new persistent type — one entry in a multiscale nifti-zarr store's OME-Zarr
metadata (an array path + scale factor). Represented purely by constructing a
`NiftiZarrArray` bound to that level's array path (see above); there is no separate
"Resolution Level" object in the implementation.

## `ImagePyramid` (existing entity, extended)

`nitorch/tools/registration/objects.py::ImagePyramid` is not a new type — this feature
extends its existing level-construction loop (`_build_pyramid`) to check, per requested
level, whether the source exposes the level-fetch capability above:

```text
for each requested level i:
    if source has native level i (via the level-fetch capability):
        level_i = wrap(source.<level-fetch>(i))       # NiftiZarrArray's own data, unmodified
    else:
        level_i = downsample(coarsest_native_or_finest_loaded, ...)   # existing behavior, unchanged
```

No new fields are added to `ImagePyramid` itself; the change is in how each level's data
is obtained, not in the object's shape.

## Relationships

- `NiftiZarrArray` **is a** `MappedArray` (FR-001, FR-006): usable anywhere any other
  backend already is, through the same `nitorch.io.map()`/`load()` entry points.
- `NiftiZarrArray` (finest level, default) **shares its header** with every other
  `NiftiZarrArray` constructed from the same store at a different level (FR-007): the
  embedded NIfTI header is read once and reused, not re-parsed per level.
- `ImagePyramid` **consumes** `NiftiZarrArray`'s level-fetch capability when available
  (FR-008, User Story 3), falling back to its own existing downsampling otherwise —
  `ImagePyramid` never depends on `NiftiZarrArray` by name (research.md §6).
