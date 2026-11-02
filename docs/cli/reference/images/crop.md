---
icon: octicons/terminal-24
---

# crop

Cut a box of voxels out of a volume, for instance to remove empty background,
to extract a region of interest, or to give several images the same field of
view. The orientation matrix of the result is updated, so the cropped volume
stays aligned with the original in world space.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch crop FILE... [-s SIZE... [SPACE]] [-c CENTER... [SPACE]] [-k FILE]
                         [-b [THRESH]] [-o FILE...] [-t [FILE...]]
    ```

## Description

`crop` keeps a rectangular block of voxels and discards the rest. It does not
resample anything: the voxels that remain are the original ones, with their
original values, and the voxel size is unchanged. What the command changes in
the output is the shape of the array and the orientation matrix, whose
translation is moved to the first voxel that was kept (see
[How the orientation matrix changes](#how-the-orientation-matrix-changes)).
The data type, the format and every other header field are copied from the
input.

A crop is described by a **size** and a **centre**. You can give them
directly with `-s` and `-c`, or let the command derive them from a reference
image (`-k`), from the bounding box of the non-zero voxels (`-b`), or from a
transformation file written by an earlier crop (`-t`). The command prints the
voxel range it keeps, for example:

```text
Cropping patch [6:14, 8:16, 4:12] from volume with shape [20, 24, 16]
```

The ranges are in the usual half-open convention, `first:last`, along each of
the stored axes. The output is written to `{dir}{sep}{base}.crop{ext}` by
default, that is, next to the input with `.crop` added to its name. The
placeholders `{dir}`, `{sep}`, `{base}` and `{ext}` are explained in
[Output filenames](../../conventions.md#output-filenames).

When you give several input files, the same crop is applied to each of them.
The options `-o` and `-t` take one value per input file, and when you give
fewer values than files, the last one is reused.

### Choosing the region

**Size.** `-s` takes up to three numbers, one per spatial axis, optionally
followed by `vox` (the default) or `ras`. If you give fewer numbers than there
are axes, the last one is repeated, so `-s 64` means `-s 64 64 64`. With `vox`,
the numbers are counts of voxels along the stored axes. With `ras`, they are
lengths in millimetres along the world axes R, A and S, whatever the layout of
the file; the command converts them to voxels axis by axis, using the voxel
size, and rounds up. Sizes are read as numbers but truncated to whole numbers
before they are used, which for `ras` means whole millimetres: `-s 2.5 ras` on
a 2 mm volume keeps one voxel, not two.

**Centre.** `-c` takes the same kind of list, followed by `vox` or `ras`. With
`vox`, it is a position in voxel coordinates, where the first voxel is at
index 0 and the centre of voxel `k` is at coordinate `k`, so fractional values
are allowed. With `ras`, it is a point in world space, which the command
converts to voxel coordinates with the inverse of the orientation matrix. If
you do not give a centre, the centre of the field of view is used, which is
the coordinate `(shape - 1) / 2` along each axis.

**The box.** The first voxel kept along an axis is `centre - (size - 1) / 2`,
rounded to an integer (halves round to the nearest even number), and the box
extends over `size` voxels from there. The box is **clipped to the field of
view**: `crop` never adds voxels, so a box that sticks out of the image gives
a smaller result than requested. To enlarge a volume, use [`pad`](pad.md).

**Reference image, `-k`.** The size is the shape of the reference, and the
centre is the position of the reference's centre in the voxel grid of the
input. This is a way to cut the same region out of images that were
acquired in the same space, or to reproduce a crop made on another image. The
command does not resample, so if the reference has a different voxel size from
the input, you still obtain a box with the shape of the reference. If the
reference is stored with a different layout (axis order or flips) from the
input, the input is first reoriented to the layout of the reference, and the
output is stored in that layout.

**Bounding box, `-b`.** The size and centre are those of the smallest box that
contains every voxel whose value is strictly greater than `THRESH` (0 if you
give no value). For volumes with more than three dimensions, a voxel counts if
the condition holds in any of the extra channels.

**Combining them.** The built-in help says that only one of `-s`, `-k` and
`-b` may be used, but the command does not check this. When several are given,
an explicit `-s` or `-c` always takes precedence over the size or the centre
that would be derived from `-k`, `-b` or `-t`, and `-k` takes precedence over
`-b`.

!!! warning "Current limitations of `-k` and `-b`"
    In the present version, `-k` on its own stops with the error *At least one
    of size/like/transform must be provided*: it only supplies the centre, so
    you have to give `-s` as well (for example the shape of the reference).
    `-b` stops with the error *'numpy.ndarray' object has no attribute
    'numpy'* when the input is a file, after printing the voxel range. In
    addition, the box it computes is shifted by one voxel along an axis whose
    first non-zero index is odd.

### How the orientation matrix changes

The voxel at index `first` in the input becomes the voxel at index 0 in the
output. The orientation matrix is therefore multiplied on the right by a
translation of `first` voxels: the directions and voxel size stay the same,
and the translation column moves to the world position of the first voxel
kept. A voxel that has not been removed keeps exactly the same position in
world space, so the output and the input can be overlaid. For example, cropping
a 2 mm volume whose origin is at `(-10, -20, -30)` mm from voxel `(6, 8, 4)`
gives an output whose origin is `(-10 + 6 * 2, -20 + 8 * 2, -30 + 4 * 2)`,
that is `(2, -4, -22)` mm.

### Transformation files

With `-t`, the crop is recorded in, or read from, a linear transformation file
in the `.lta` format (see
[Spatial transformations](../../conventions.md#spatial-transformations)).

*Writing.* The file holds an identity matrix between world spaces, because
cropping does not move anything in world space. The crop itself is encoded in
the two volume geometries that the file stores: the source is the input (shape
and orientation matrix), the destination is the cropped volume. If you give
`-t` without a file name, the file is named `{dir}{sep}{base}.crop.lta`.

*Reading.* If you give `-t FILE` and none of `-s`, `-k` or `-b`, the file is
read instead: the shape and position of its destination geometry give the size
and the centre of the crop in the input. Use it to apply to another image the
crop that you made earlier, which also works if that image has a different
size, as long as the two share the same orientation layout. A file name is
required in this case.

!!! note "When is `-t` an input, and when an output?"
    The built-in help says that `-t` names an output when `-s`, `-c` or `-k`
    is given. In the code, `-t` is written only when `-k` or `-b` selects the
    region (and `-b` currently fails, see above). In every other case it is
    read. In particular `-s 64 64 64 -t crop.lta` tries to read `crop.lta`
    and fails if it does not exist. To record a crop, derive it from a reference
    with `-k` and `-s`.

## Arguments

| Name   | Description                                          |
| ------ | ---------------------------------------------------- |
| `FILE` | One or more input volumes. Any format NITorch reads, see [File formats](../../conventions.md#file-formats). |

## Options

| Option                  | Value                 | Description                                                                                  | Default                       |
| ----------------------- | --------------------- | -------------------------------------------------------------------------------------------- | ----------------------------- |
| `-s`, `--size`          | `SIZE... [SPACE]`     | Size of the cropped region, in voxels (`vox`) or in millimetres along R, A, S (`ras`). The last value is repeated for missing axes. | none, required unless `-k`, `-b` or `-t` provides it |
| `-c`, `--center`        | `CENTER... [SPACE]`   | Centre of the cropped region, as voxel coordinates (`vox`) or world coordinates in millimetres (`ras`). | centre of the field of view   |
| `-k`, `--like`          | `FILE`                | Reference volume whose shape and position define the crop. [See above](#choosing-the-region). | none                          |
| `-b`, `--bounding-box`  | `[THRESH]`            | Crop to the bounding box of the voxels greater than `THRESH`. [See above](#choosing-the-region). | off; `THRESH` is 0 if omitted |
| `-o`, `--output`        | `FILE...`             | Output name template, one per input. Placeholders: `{dir}`, `{sep}`, `{base}`, `{ext}`.      | `{dir}{sep}{base}.crop{ext}`  |
| `-t`, `--transform`     | `[FILE...]`           | Name of a `.lta` file to read or write, one per input. [See above](#transformation-files).   | none; `{dir}{sep}{base}.crop.lta` if given with no value |
| `-h`, `--help`          |                       | Print the built-in help and exit.                                                            |                               |

`SPACE` is `vox` or `ras`. The two keywords are independent: `-s` and `-c` each
have their own, and each defaults to `vox`.

## Examples

Crop a block of 64 voxels from the centre of the field of view. The result is
written to `T1w.crop.nii.gz`:

```shell
nitorch crop T1w.nii.gz -s 64 64 64
```

Crop a 40 x 40 x 20 mm block around a known point in world space, given in
millimetres:

```shell
nitorch crop sub-01_T1w.nii.gz -s 40 40 20 ras -c 12.4 -6.4 3.2 ras
```

Crop two co-registered images to the same box, writing the results in a
different directory:

```shell
nitorch crop sub-01_T1w.nii.gz sub-01_FLAIR.nii.gz -s 96 96 64 -c 100 110 90 \
    -o 'roi{sep}{base}{ext}'
```

Extract a region and keep it as a reference for other images, then cut the
same region out of an image that lives in the same space. The reference
provides the centre, and `-s` the size; the transformation file records the
crop:

```shell
nitorch crop sub-01_T1w.nii.gz -s 96 96 64 -c 100 110 90 -o T1w_roi.nii.gz
nitorch crop sub-01_FLAIR.nii.gz -k T1w_roi.nii.gz -s 96 96 64 \
    -o FLAIR_roi.nii.gz -t FLAIR_roi.lta
```

Apply a crop that has been recorded in a transformation file to another
volume:

```shell
nitorch crop sub-01_T2w.nii.gz -t FLAIR_roi.lta -o T2w_roi.nii.gz
```

## See also

- [`pad`](pad.md) is the inverse operation: it adds voxels around a volume.
- [`chunk`](chunk.md) and [`extract_patches`](extract_patches.md) cut a volume
  into several pieces.
- [`reslice`](../spatial/reslice.md) resamples a volume onto another grid, when
  a plain crop is not enough.
