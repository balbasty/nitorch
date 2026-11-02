---
icon: octicons/terminal-24
---

# pad

Add voxels around a volume, for example to give it a margin before filtering,
to make its shape fit a network, or to restore the field of view that
[`crop`](crop.md) removed. The orientation matrix of the result is updated, so
the padded volume stays aligned with the original in world space.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch pad FILE... [-s SIZE... [SPACE]] [-b BOUND] [-k FILE]
                        [-o FILE...] [-t [FILE...]]
    ```

## Description

`pad` enlarges the array of a volume by a number of voxels on each side of each
spatial axis, and fills the new voxels according to a boundary condition. The
original voxels are copied unchanged, and the voxel size does not change. The
output has the data type, the format and the other header fields of the input;
only its shape and its orientation matrix differ.

You tell the command how much to add either explicitly, with `-s`, or by
naming a reference image with `-k`, in which case it pads the input until it
has the shape of the reference, centred where the reference is. A third way to
give the amounts is a transformation file written by an earlier crop or pad,
with `-t` (see [Transformation files](#transformation-files)). `-s` and `-k`
cannot be combined, and one of `-s`, `-k` and `-t` is required.

The command prints the padding that it applies, for example:

```text
Padding volume with shape [20, 24, 16] by [(2, 2), (2, 2), (2, 2)] voxels
```

Each pair is the number of voxels added before and after the data along one
stored axis. By default, the output is written to `{dir}{sep}{base}.pad{ext}`,
next to the input. The placeholders `{dir}`, `{sep}`, `{base}` and `{ext}` are
explained in [Output filenames](../../conventions.md#output-filenames).

When you give several input files, the same padding is applied to each of
them. `-o` and `-t` take one value per input file, and when you give fewer
values than files, the last one is reused.

### Padding amounts

`-s` takes a list with one entry per spatial axis. An entry is either a single
number, which is used on both sides of the axis, or a pair
`(lower, upper)` that gives different amounts before and after the data.
If you give fewer entries than there are axes, the last entry is repeated, so
`-s 10` pads every axis by 10 voxels on both sides. Amounts in voxels are
truncated to whole numbers.

A pair can be written as one word or split over several words; quote it so
that your shell leaves the brackets alone:

```shell
nitorch pad T1w.nii.gz -s 10 10 '(0,16)'
```

pads the first two axes by 10 voxels on each side and adds 16 voxels after the
data on the third axis. Square brackets work as well, and `'(0,' '16)'` or
`'(0, 16)'` are read the same way. A pair whose closing bracket is missing, or
that does not hold exactly two numbers, is rejected with an error.

Followed by the keyword `ras`, the amounts are read in millimetres and along
the world axes (x, y, z) rather than the stored ones. The command maps each
amount to the stored axis that points along it, swaps the lower and upper
values on axes that are flipped, divides by the voxel size and rounds up to a
whole number of voxels.

Negative amounts remove voxels. They are only dependable with the boundary
conditions `replicate`, `dct2` and `dft`; with the default constant fill, a
negative amount can fail with a message about tensor sizes. To remove voxels,
prefer [`crop`](crop.md).

### Boundary conditions

`-b` chooses the values given to the new voxels.

| Value         | New voxels                                                                   |
| ------------- | ---------------------------------------------------------------------------- |
| `zero`        | Zero. This is the default.                                                   |
| a number      | That constant, for example `-b 1000` or `-b -1`.                            |
| `replicate`   | The value of the nearest voxel at the edge of the image.                     |
| `dct2`        | The image mirrored about its edge, with the edge voxel repeated.             |
| `dft`         | The image wrapped around, as if it were periodic.                            |

The other boundary names and aliases used throughout NITorch are accepted as
well: `dct` (an alias of `dct2`), `dct1`, `dst1`, `dst2`, and the SciPy and
PyTorch spellings such as `reflect`, `mirror`, `wrap`, `nearest`, `border` or
`constant`. Any other word that is not a number is rejected with an error.

The padded values are written in the data type of the input, with no checks, so
a constant that does not fit in that type wraps around or is truncated.

### Using a reference image

With `-k`, the padding is computed per axis from the shape of the reference and
the position of its centre relative to the input. The result has the shape of
the reference when the two share the same voxel grid, and the input sits in it
at the position that it has in world space with respect to the reference. This
is the natural way to undo a crop:

```shell
nitorch crop T1w.nii.gz -s 8 6 4 -c 10 12 8 -o patch.nii.gz
nitorch pad patch.nii.gz -k T1w.nii.gz -o patch_full.nii.gz
```

`patch_full.nii.gz` then has the shape and the orientation matrix of
`T1w.nii.gz`, with zeros outside the region of the patch.

The command does not resample. The input and the reference must have the same
layout (axis order and flips); otherwise it stops with an error that names the
two layouts. If the reference is smaller than the input, or if its centre is
offset in a way that requires a negative amount on one side, the padding
amounts are negative there, with the limitations described above.

### How the orientation matrix changes

Adding `lower` voxels before the data moves the voxel that was at index 0 to
index `lower`. The orientation matrix is multiplied on the right by a
translation of `-lower` voxels along each axis, so the origin moves outwards
while the directions and the voxel size stay the same. Every original voxel
keeps its world position. For example, padding a 2 mm volume whose origin is at
`(0, 0, 0)` mm by 2 voxels on each side gives an origin at `(-4, -4, -4)` mm.

### Transformation files

With `-t`, the padding is recorded in, or read from, a linear transformation
file in the `.lta` format (see
[Spatial transformations](../../conventions.md#spatial-transformations)).

*Writing.* If you gave `-s` or `-k`, the file is written. It holds an identity
matrix between world spaces, because padding does not move anything in world
space, and stores the two volume geometries (shape and orientation matrix)
that carry the information: the input as source, the padded volume as
destination. If you give `-t` without a file name, the file is named
`{dir}{sep}{base}.pad.lta`.

*Reading.* If you give `-t FILE` and neither `-s` nor `-k`, the file is read,
and the input is padded so that its shape matches the destination geometry of
the file. A file name is required in this case. This lets you apply the padding
of one image to others.

## Arguments

| Name   | Description                                          |
| ------ | ---------------------------------------------------- |
| `FILE` | One or more input volumes. Any format NITorch reads, see [File formats](../../conventions.md#file-formats). |

## Options

| Option               | Value               | Description                                                                              | Default                      |
| -------------------- | ------------------- | ---------------------------------------------------------------------------------------- | ---------------------------- |
| `-s`, `--size`       | `SIZE... [SPACE]`   | Amount of padding per axis: a number, or a pair `(lower, upper)`. The last entry is repeated for missing axes. [See above](#padding-amounts). | none, required unless `-k` or `-t` is given |
| `-b`, `--bound`      | `BOUND`             | Boundary condition: `zero`, a number, `replicate`, `dct2`, `dft` or another bound name. [See above](#boundary-conditions). | `zero`                       |
| `-k`, `--like`       | `FILE`              | Reference volume whose shape and position define the padding. [See above](#using-a-reference-image). | none                         |
| `-o`, `--output`     | `FILE...`           | Output name template, one per input. Placeholders: `{dir}`, `{sep}`, `{base}`, `{ext}`.  | `{dir}{sep}{base}.pad{ext}`  |
| `-t`, `--transform`  | `[FILE...]`         | Name of a `.lta` file to write or read, one per input. [See above](#transformation-files). | none; `{dir}{sep}{base}.pad.lta` if given with no value |
| `-h`, `--help`       |                     | Print the built-in help and exit.                                                        |                              |

## Examples

Add a margin of 10 voxels on every side of a volume. The result is written to
`T1w.pad.nii.gz`:

```shell
nitorch pad T1w.nii.gz -s 10
```

Add 16 voxels after the data along the third axis only, repeating the last
slice:

```shell
nitorch pad T1w.nii.gz -s 0 0 '(0,16)' -b replicate
```

Pad a mask with a margin of 5 voxels, wrapping the image around instead of
filling with zeros, and choose the output name:

```shell
nitorch pad sub-01_mask.nii.gz -s 5 -b dft -o sub-01_mask_padded.nii.gz
```

Restore the field of view of a cropped image, using the original image as the
reference, and record the padding in a transformation file:

```shell
nitorch pad sub-01_T1w.crop.nii.gz -k sub-01_T1w.nii.gz \
    -o sub-01_T1w.restored.nii.gz -t sub-01_T1w.restore.lta
```

Apply the same padding to another image:

```shell
nitorch pad sub-01_FLAIR.crop.nii.gz -t sub-01_T1w.restore.lta \
    -o sub-01_FLAIR.restored.nii.gz
```

## See also

- [`crop`](crop.md) removes voxels from around a volume.
- [`resize`](../spatial/resize.md) and [`reslice`](../spatial/reslice.md)
  change the field of view by resampling.
