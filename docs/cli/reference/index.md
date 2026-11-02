---
icon: octicons/list-unordered-24
---

# Reference

This section has one page per `nitorch` command, grouped by purpose. Each page
says what the command is for, gives its synopsis, lists every argument and
option with its default, explains the behaviour that is not obvious from the
option names, and ends with examples. Rules that all commands share, such as
how options are parsed, which file formats are supported and how output files
are named, are described once in [Common conventions](../conventions.md), and
the pages link to it where it matters.

## Images

Tools to inspect, convert and cut up volumes, to filter them, and to compare
them. See the [overview of the group](images/index.md).

| Command                                    | Summary                                                              |
| ------------------------------------------ | -------------------------------------------------------------------- |
| [`info`](images/info.md)                   | Print information about a volume: shape, orientation, metadata and, optionally, intensity statistics. |
| [`convert`](images/convert.md)             | Convert volumes between file formats and data types.                 |
| [`crop`](images/crop.md)                   | Crop a volume to a region of a given size and position.              |
| [`pad`](images/pad.md)                     | Pad a volume with a chosen boundary condition.                       |
| [`chunk`](images/chunk.md)                 | Split a volume into chunks of a given size along one dimension.      |
| [`unstack`](images/unstack.md)             | Split a volume into one file per slice or channel along one dimension. |
| [`extract_patches`](images/extract_patches.md) | Split a volume into a grid of patches.                           |
| [`pool`](images/pool.md)                   | Pool a volume over sliding windows (mean, sum, minimum, maximum, ...). |
| [`smooth`](images/smooth.md)               | Smooth a volume with a Gaussian, triangular or rectangular kernel.   |
| [`inpaint`](images/inpaint.md)             | Fill in missing data.                                                |
| [`metrics`](images/metrics.md)             | Compute overlap and similarity metrics between predicted and reference images. |

## Orientation and resampling

Tools to change how a volume sits in space, either by editing its orientation
matrix or by resampling its data. Several of them read and write spatial
transformations; see [Spatial transformations](../conventions.md#spatial-transformations).
See the [overview of the group](spatial/index.md).

| Command                                    | Summary                                                              |
| ------------------------------------------ | -------------------------------------------------------------------- |
| [`orient`](spatial/orient.md)              | Overwrite the orientation matrix of a volume. Use it when the matrix in the file is wrong. |
| [`reorient`](spatial/reorient.md)          | Reorder the data of a volume to a given layout while preserving its position in world space. |
| [`resize`](spatial/resize.md)              | Resample a volume to a different voxel size or shape.                |
| [`reslice`](spatial/reslice.md)            | Resample volumes (or streamlines) into another space, through linear, displacement or velocity transformations. |
| [`compose`](spatial/compose.md)            | Compose several spatial transformations into one.                    |
| [`vexp`](spatial/vexp.md)                  | Exponentiate a stationary velocity field into a displacement or transformation field. |
| [`meanspace`](spatial/meanspace.md)        | Compute the average oriented voxel space of a series of images.      |

## Registration

Tools to align images, pairwise or in groups. See the
[overview of the group](registration/index.md).

| Command                                    | Summary                                                              |
| ------------------------------------------ | -------------------------------------------------------------------- |
| [`register`](registration/register.md)     | Pairwise affine and non-linear registration, with a choice of similarity losses and optimisers. |
| [`simplereg`](registration/simplereg.md)   | Pairwise affine and non-linear registration through a simpler, more classical syntax. A wrapper around `register`. |
| [`affreg`](registration/affreg.md)         | Pairwise affine registration. A wrapper around `register`.           |
| [`optinorm`](registration/optinorm.md)     | Groupwise normalisation of several scans of the same subject to their mean space. |
| [`affopt`](registration/affopt.md)         | Compute optimal template-to-subject affine matrices from subject-to-subject pairs. |
| [`vopt`](registration/vopt.md)             | Compute optimal template-to-subject velocity fields from subject-to-subject pairs. |
| [`autoreg`](registration/autoreg.md)       | Registration with automatic differentiation and a flag-based syntax for losses, transformations and optimisers. |

## Quantitative MRI

Tools to estimate quantitative maps and to correct distortions in multi-echo
and echo-planar data. See the [overview of the group](qmri/index.md).

| Command                                    | Summary                                                              |
| ------------------------------------------ | -------------------------------------------------------------------- |
| [`estatics`](qmri/estatics.md)             | Fit a non-linear exponential decay to multi-echo data, with optional MEETUP distortion correction. |
| [`greeq`](qmri/greeq.md)                   | Fit T1, T2\* and, optionally, MTsat maps from multi-echo, multi-flip-angle data. |
| [`epic`](qmri/epic.md)                     | Jointly denoise bipolar multi-echo data and correct its geometric distortions with a fieldmap. |
| [`topup`](qmri/topup.md)                   | Estimate a one-dimensional distortion field from reverse-polarity images, and apply it. |

## Denoising and segmentation

See the [overview of the group](denoising/index.md).

| Command                                    | Summary                                                              |
| ------------------------------------------ | -------------------------------------------------------------------- |
| [`denoise`](denoising/denoise.md)          | Total-variation denoising of multi-channel images.                   |
| [`denoise_mri`](denoising/denoise_mri.md)  | Total-variation denoising of magnetic resonance images, alone or jointly across sequences. |
| [`uniseg`](denoising/uniseg.md)            | Tissue segmentation with a reimplementation of SPM12's Unified Segmentation. |

## Visualisation

See the [overview of the group](viewer/index.md).

| Command                                    | Summary                                                              |
| ------------------------------------------ | -------------------------------------------------------------------- |
| [`view`](viewer/view.md)                   | Interactive viewer for volumetric images.                            |

`register2` is a hidden alias of [`register`](registration/register.md). It
works, and it appears in the list that `nitorch` prints, but it is not
documented separately.
