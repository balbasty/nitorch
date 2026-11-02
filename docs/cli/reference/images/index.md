---
icon: octicons/image-24
---

# Images

These commands work on volumes as arrays of voxels: they report what a file
contains, change its format, cut it into smaller pieces or add to it, filter
it, and compare it with another image. None of them deals with registration or
with resampling onto another grid; for that, see
[Orientation & resampling](../spatial/index.md) and the registration commands.

The commands fall into four families. [`info`](info.md) and
[`convert`](convert.md) look at and rewrite files. [`crop`](crop.md),
[`pad`](pad.md), [`chunk`](chunk.md), [`unstack`](unstack.md) and
[`extract_patches`](extract_patches.md) change the field of view or split a
volume into pieces, and update the orientation matrix of every result so that
it stays aligned with the input in world space. [`pool`](pool.md),
[`smooth`](smooth.md) and [`inpaint`](inpaint.md) compute new voxel values
from the old ones. [`metrics`](metrics.md) scores a prediction against a
reference and writes the result to a JSON file.

Unless a page says otherwise, a command that writes a modified copy of a volume
puts it next to the input, under a name built from a template that you can
change with `-o`; see [Output filenames](../../conventions.md#output-filenames).
The exception is [`convert`](convert.md), whose default output name is that of
the input, so that it replaces the file.

## Commands

| Command                                    | Summary                                                                                              |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| [`info`](info.md)                          | Print the shape, orientation and header metadata of volumes, and optionally their intensity statistics. |
| [`convert`](convert.md)                    | Convert volumes between file formats and data types, and override header fields.                     |
| [`crop`](crop.md)                          | Crop a volume to a box of a given size and position, to a reference image or to a bounding box.      |
| [`pad`](pad.md)                            | Add voxels around a volume, filled according to a boundary condition.                                |
| [`chunk`](chunk.md)                        | Split a volume into consecutive chunks of a given thickness along one dimension.                     |
| [`unstack`](unstack.md)                    | Split a volume into one file per slice or volume along one dimension.                                |
| [`extract_patches`](extract_patches.md)    | Cut a volume into a regular grid of patches and save each patch as its own file.                     |
| [`pool`](pool.md)                          | Downsample a volume by reducing each window of voxels to one value (mean, sum, minimum, maximum, ...). |
| [`smooth`](smooth.md)                      | Blur a volume with a Gaussian, box or triangular filter.                                             |
| [`inpaint`](inpaint.md)                    | Fill in missing voxels of a volume from the values that surround them.                               |
| [`metrics`](metrics.md)                    | Compare a predicted image with a reference and write overlap and similarity scores to a JSON file.   |
