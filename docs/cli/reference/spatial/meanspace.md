---
icon: octicons/terminal-24
---

# meanspace

Compute an average oriented voxel space from a set of images, write it as an empty "mock" image, and optionally reslice every input onto it. Use it to put a group of images, which may differ in field of view, voxel size and orientation, on one common grid.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch meanspace FILE... [-o FILE] [-r [FILE]] [-v VAL... [UNIT]]
                              [-p VAL... [UNIT]] [-l LAYOUT]
    ```

## Description

An *oriented voxel space* is a regular grid of voxels placed in world space: a shape (the number of voxels along each axis) together with an orientation matrix (where the grid sits, which way its axes point and how large its voxels are). Every image defines one. `meanspace` reads only the headers of its inputs, never their voxel data, and builds a single space that is representative of all of them and large enough to contain all of them.

The space is built in four steps:

1. **Orientation.** The orientation matrices of the inputs are first brought to a common `RAS` layout, so that matrices that describe the same orientation with different storage orders can be compared. Their mean is then computed as a *log-Euclidean mean* (the average of the matrix logarithms, mapped back with the exponential), which averages rotations, scalings and translations sensibly, instead of averaging matrix entries. Spurious shears are removed, so that the result is a rotation and translation combined with a voxel size. Finally, the matrix is converted to the layout that most inputs have (the *majority layout*).
2. **Voxel size.** By default, the voxel size is the one of the mean matrix: the average voxel size of the inputs. You can set another with `-v`, either in millimetres or as a percentage of that average.
3. **Field of view.** The corners of all input fields of view are projected into the new space. The space is the bounding box of all of them, so that no input is cropped. `-p` enlarges (or, with negative values, shrinks) the box.
4. **Shape.** The box is rounded to a whole number of voxels. A tolerance of 1% of a voxel is used, so that rounding errors do not add an extra slice.

### What it is used for

The mean space is the natural target when you need to compare, average or jointly analyse images that were not acquired on the same grid, for example before template building or group statistics, or to resample all the images of a study to one resolution without favouring any subject. The output of `meanspace` is a header and a shape, so the command writes them as an image filled with zeros. This image can be used as the target of [`reslice`](reslice.md) (`-t`); `meanspace` can also do this resampling for you with `-r`.

The shape and the orientation matrix of the mean space are printed to the terminal.

### Output files

- The **mean-space image** (`-o`) is a volume of zeros of floating-point type, with the shape and orientation matrix computed above. It is written in the format given by its extension, which is that of the first input by default. If you set `-o` to `no`, `false` or `0`, the image is written to a temporary file instead, which is only useful together with `-r`.
- With `-r`, each input is also **resliced** onto the mean space with the default settings of [`reslice`](reslice.md), that is, linear interpolation and zeros outside the field of view of the input. For other interpolation orders or for label maps, run `reslice` yourself with the mean-space image as the target.

## Arguments

| Name   | Description                                                                          |
| ------ | ------------------------------------------------------------------------------------ |
| `FILE` | One or more input images, of any format NITorch reads (see [file formats](../../conventions.md#file-formats)). Only the shape and the orientation matrix are used. |

## Options

| Option                  | Value             | Description                                                                                                                                       | Default                                  |
| ----------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `-o`, `--output`        | `FILE`            | Path of the mock mean-space image. `{dir}`, `{base}` and `{ext}` refer to the first input (`{dir}` is its absolute directory).                    | `{dir}/meanspace{ext}`                   |
| `-r`, `--resliced`      | `[FILE]`          | Also reslice every input onto the mean space. The optional value is a [template](../../conventions.md#output-filenames) with `{dir}`, `{base}` and `{ext}`, taken from each input. If the flag is given without a value, the default template is used. | off; when given alone: `{dir}/{base}.meanspace{ext}` |
| `-v`, `--voxel-size`    | `VAL... [UNIT]`   | Voxel size: one value for all axes or one per axis, followed by an optional unit. With `%` (or `pct`), values are percentages of the average voxel size. Without a unit, or with any unit other than `%`, they are millimetres. | `100 %`                                  |
| `-p`, `--pad`           | `VAL... [UNIT]`   | Amount by which the field of view is enlarged, one value or one per axis. With `%`, the field of view grows by this percentage of its size along the axis, split evenly between both ends. Negative values crop. | `0 %`                                    |
| `-l`, `--layout`        | `LAYOUT`          | Intended to set the layout of the mean space (`RAS`, `LPS`...). **Currently has no effect**: the majority layout is always used.                   | majority layout of the inputs            |
| `--cpu`, `--gpu`        | `[N]`             | Accepted for consistency with other commands. They have no effect on the result.                                                                  | cpu                                      |

### Voxel size and padding units

`-v 100 %` (the default) keeps the average voxel size; `-v 50 %` halves it, so that the mean space has twice as many voxels along each axis; `-v 1` or `-v 1 mm` gives 1 mm isotropic voxels; `-v 0.8 0.8 1.2` sets anisotropic voxels.

For padding, `-p 10 %` enlarges the field of view by 10% of its size along each axis (5% on each side), and `-p -10 %` shrinks it by the same amount. In the current implementation, a unit other than `%` is not converted: the value is then added to each side of the bounding box in **voxels of the mean space**, not in millimetres. Prefer `%`, or check the printed shape.

## Examples

Compute the mean space of three subjects and write it next to the first one as `meanspace.nii.gz`:

```shell
nitorch meanspace sub-01_T1w.nii.gz sub-02_T1w.nii.gz sub-03_T1w.nii.gz
```

Choose 1 mm isotropic voxels and a name for the output:

```shell
nitorch meanspace sub-*_T1w.nii.gz -v 1 mm -o group_space.nii.gz
```

Compute the mean space with 5% of margin around the union of the fields of view, and reslice every image onto it. Each resliced image is written next to its input, with the suffix `.meanspace`:

```shell
nitorch meanspace sub-01_T1w.nii.gz sub-02_T1w.nii.gz -p 5 % -r
```

Do the same, but choose where the resliced images go, and keep the mean-space image in a work directory:

```shell
nitorch meanspace sub-01_T1w.nii.gz sub-02_T1w.nii.gz \
    -o work/space.nii.gz -r 'work/{base}_group{ext}'
```

Resample an image onto the mean space yourself, to choose the interpolation order (here cubic splines):

```shell
nitorch reslice sub-01_T1w.nii.gz -t work/space.nii.gz -i 3 -o work/sub-01_T1w_cubic.nii.gz
```

## See also

- [`reslice`](reslice.md) resamples images onto a target space such as the one computed here.
- [`resize`](resize.md) changes the voxel size of an image on a grid derived from its own.
- [`info`](../images/info.md) prints the shape and orientation matrix of a file.
