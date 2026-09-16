# Quickstart: Validating NIfTI-Zarr Volume Support

See `contracts/niftizarr-api.md` for exact API shapes and `data-model.md` for entity
details.

## Prerequisites

- nitorch installed with the new optional extra providing `zarr`/`dask` (research.md
  §3) — exact extra name is a `tasks.md`/implementation detail.
- A nifti-zarr store to test against, e.g. produced from an existing NIfTI file with
  the upstream `nii2zarr` converter (https://github.com/neuroscales/nifti-zarr-py).

## Scenario 1 — Load through the existing interface (US1)

```python
from nitorch.io import map

vol = map('scan.nii.zarr')
print(vol.affine, vol.voxel_size, vol.dtype)
```

**Expected outcome**: metadata matches loading `scan.nii` (the same content's plain
NIfTI form) exactly (FR-001-003, SC-001, SC-003).

## Scenario 2 — Lazy, chunked access (US2)

```python
lazy = vol.as_dask()
sub = lazy[64:128, 64:128, :].compute()
```

**Expected outcome**: only the chunks overlapping the requested sub-region are read;
this completes even for a store much larger than available memory (FR-004, SC-002).

## Scenario 3 — A structurally invalid store produces a clear error at load time (FR-005)

```python
from nitorch.io import map

try:
    map('not_a_store.nii.zarr')   # missing, not a Zarr store, or no embedded NIfTI header
except Exception as e:
    print(e)  # nitorch's existing "no reader could load this file" error
```

**Note** (Clarifications, Session 2026-09-11): this guarantee covers structural
recognizability only. A store that loads successfully but has one or more
missing chunks is *not* required to fail at all — see Scenario 3b.

## Scenario 3b — A missing chunk silently reads as zero, not an error

```python
vol = map('interrupted_conversion.nii.zarr')  # succeeds: structurally valid
lazy = vol.as_dask()
region = lazy[...].compute()   # touches the missing chunk's region
# no exception is raised: per Zarr's own default behavior, the missing
# chunk's region reads as the array's fill value (typically zero) --
# verified empirically during implementation. nitorch does not override
# this default (consistent with Option A: no extra chunk-existence
# validation cost, whatever Zarr's own native behavior turns out to be).
```

## Scenario 3c — Plain OME-Zarr store with no embedded NIfTI header (FR-009)

```python
from nitorch.io import map

vol = map('plain.ome.zarr')   # valid OME-Zarr, no embedded NIfTI header
print(vol.affine, vol.voxel_size)  # derived from coordinateTransformations/axes/units
```

**Expected outcome**: loads successfully (does not require an embedded NIfTI
header); the derived affine/voxel-size matches what the store's own OME-Zarr
metadata specifies, exactly (SC-006) — the same derivation the reference
`nifti-zarr-py` implementation itself uses (Clarifications, Session
2026-09-11). A Zarr store with neither an embedded NIfTI header nor
recognizable OME-Zarr metadata still fails per Scenario 3.

## Scenario 4 — Fetch a specific resolution level (FR-007)

```python
vol = map('multiscale.nii.zarr')      # finest level by default
coarser = vol.<level-fetch>(1)         # exact method name: see tasks.md
assert coarser.affine is not None       # shares the same header
```

## Scenario 5 — Registration reuses native levels (US3, FR-008, SC-005)

```python
from nitorch.tools.registration.pairwise_makeobj import make_image, make_loss
from nitorch.tools.registration.objects import Similarity
from nitorch.tools.registration.pairwise_run import run

fixed = make_image('fixed.nii.zarr', pyramid=range(3))
moving = make_image('moving.nii.zarr', pyramid=range(3))
# each pyramid level's data should match the store's own native level's data,
# for every level the store natively provides
sim = Similarity(make_loss('lcc'), moving[0], fixed[0])
run(sim, ...)
```

**Expected outcome**: `nitorch register -l 0:3 ...` (or the equivalent Python call)
against a multiscale nifti-zarr input uses the store's native data at each level it
provides, verified by comparing pyramid level data directly against the store's own
per-level arrays (not just checking that registration "ran without error").

## Scenario 6 — Existing formats and behavior are unaffected (regression, FR-006)

Run the existing test suite and confirm loading NIfTI/MGH/TIFF files, and registering
with them, is unchanged.
