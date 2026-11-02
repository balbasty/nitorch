---
icon: octicons/move-24
---

# Orientation & resampling

These commands deal with where an image sits in space and how finely it is sampled. A volume is an array of voxels plus an *orientation matrix* that maps voxel indices to world (RAS) coordinates. The commands of this group read or change one or both of these, and differ in how much of the image they touch.

## Three levels of change

**Header only: [`orient`](orient.md).** Overwrites the orientation matrix and leaves the voxel array exactly as it is. It is meant for files whose header is wrong; after the command, the image sits somewhere else in world space.

**Storage order: [`reorient`](reorient.md).** Permutes and flips the voxel array to a requested layout (`RAS`, `LAS`...) and rewrites the orientation matrix to compensate. The image does not move in world space; the same voxels are simply stored in a different order. No value is interpolated.

**Resampling: [`resize`](resize.md) and [`reslice`](reslice.md).** Compute new voxel values on a new grid, by interpolation. `resize` changes the resolution of an image on a grid derived from its own, by a factor, a voxel size or a shape. `reslice` is the general tool: it resamples onto the grid of a target image or a chosen voxel size, through an optional chain of transformations.

The choice between `orient` and `reorient` is the one that most often causes confusion, so keep this rule in mind: if the header is **wrong**, use `orient`; if the header is **right** and you only want a different storage order, use `reorient`.

## Transformation utilities

Registration produces transformations: affine matrices, displacement fields, and velocity fields. Three commands handle them directly:

- [`compose`](compose.md) merges a chain of transformations, with optional inversions, into a single matrix or displacement field.
- [`vexp`](vexp.md) exponentiates a velocity field into a displacement or transformation field.
- [`meanspace`](meanspace.md) computes an average oriented voxel space from a set of images, which makes a convenient common target for `reslice`.

`reslice` and `compose` share one convention for writing chains of transformations, explained in [how transformations are chained](reslice.md#how-transformations-are-chained). The general conventions for transformations and units are in the [CLI conventions](../../conventions.md#spatial-transformations).

## Commands

| Command                      | Summary                                                                                                           | Touches                       |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| [`orient`](orient.md)        | Overwrite the orientation matrix of a volume, leaving the data untouched.                                          | Header only                   |
| [`reorient`](reorient.md)    | Permute and flip the data to a given layout and update the orientation matrix so that world positions are kept.    | Data layout and header        |
| [`resize`](resize.md)        | Resample a volume by a zoom factor, to a voxel size or to a shape.                                                 | Voxel values and header       |
| [`reslice`](reslice.md)      | Resample volumes (or streamlines) onto a target space through a chain of affine, displacement and velocity transformations. | Voxel values and header       |
| [`compose`](compose.md)      | Compose and invert transformations into one matrix or one displacement field.                                      | Transformation files only     |
| [`vexp`](vexp.md)            | Exponentiate a stationary velocity field by scaling and squaring.                                                  | Transformation files only     |
| [`meanspace`](meanspace.md)  | Compute an average oriented voxel space from a set of images, and optionally reslice them onto it.                 | Creates an empty reference image |
