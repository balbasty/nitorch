---
icon: octicons/terminal-24
---

# resize

Change the resolution of a volume by a zoom factor, to a target voxel size, or to a target shape, and update the orientation matrix accordingly. Use it to up- or down-sample an image on a grid that is *derived from its own*, when you do not need to bring it into the space of another image (for that, see [`reslice`](reslice.md)).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch resize FILE... {-f FACTOR... | -v VX... | -s SHAPE...}
                   [-a ANCHOR...] [-i ORDER...] [-b BND...] [-p BOOL] [-g {d,t}]
                   [-o FILE...] [--cpu | --gpu [ID]]
    ```

## Description

`resize` resamples each input volume onto a new regular grid that covers (nearly) the same part of space. The size of the new grid is given in one of three ways: a **zoom factor** (`-f`), a **voxel size** (`-v`) or a **shape** (`-s`). The orientation matrix of the output is computed so that the resized image stays aligned with the input in world space: only the sampling changes, and the image is neither shifted nor rotated.

Unlike [`reslice`](reslice.md), `resize` has no target space and applies no transformation. If you need the output to match the grid of another image, or want to apply a transformation, use `reslice`.

### Choosing the size

At least one of `-f`, `-v` and `-s` is required. Without one, `resize` prints its usage followed by an `[ERROR]` line and writes nothing. The three options are used in this order of precedence:

- `-v` gives the voxel size of the output, in millimetres (the units of the orientation matrix). The zoom factor is the input voxel size divided by this value, axis by axis. It cannot be combined with `-f`: using both is an error.
- `-f` gives the zoom factor directly. A factor above 1 yields more, smaller voxels; a factor below 1 yields fewer, larger voxels. Fractions such as `1/2` are accepted.
- `-s` gives the shape of the output, in voxels. Used alone, the zoom factor is the ratio of output to input shape.

One value applies to all three axes; otherwise give one value per axis. Unless `-s` is given, the output shape follows from the input shape $N$ and the zoom factor $f$, and depends on the anchor (see [below](#anchors)). It is truncated to an integer and at least 1.

When `-s` is given together with `-f` or `-v`, it fixes the output shape explicitly. With the `centers` and `edges` anchors, the factor is then irrelevant, since the voxel size follows from the two shapes. With the `first` and `last` anchors, the factor still sets the voxel size exactly, and the shape sets how many voxels are generated: if it is larger than the zoomed input, the extra voxels at the far end are padding.

### Anchors

The anchor says which points of the input and output grids coincide.

| Anchor    | What is aligned                                      | Output shape (no `-s`)         | Voxel size of the output                          | Samples outside the field of view |
| --------- | ---------------------------------------------------- | ------------------------------ | ------------------------------------------------- | --------------------------------- |
| `centers` | The centres of the first and last voxels.            | $\lfloor N f \rfloor$          | Derived, so that both shapes span the same distance. | Never. |
| `edges`   | The outer edges of the first and last voxels.        | $\lfloor N f \rfloor$          | Input size multiplied by the ratio of input to output shape: the field of view is exactly preserved. | Up to half an input voxel at each end when up-sampling. |
| `first`   | The centre of the first voxel only.                  | $\lfloor (N-1) f \rfloor + 1$  | Input size divided by the factor, exactly.        | Never. |
| `last`    | The centre of the last voxel only.                   | $\lfloor (N-1) f \rfloor + 1$  | Input size divided by the factor, exactly.        | Never. |

For example, 10 voxels with a factor of 2 give 20 voxels with `centers` or `edges`, but 19 with `first` or `last`: the 20th voxel would lie beyond the last input voxel. Likewise, a factor of 1/3 gives 4 voxels with `first`, i.e. every third voxel of the input, and 256 voxels resized from 1 mm to 0.5 mm give 511. An explicit shape (`-s`) always takes precedence over these rules.

The default, `centers`, is the usual choice for resizing images or fields. With `centers` or `edges`, the voxel size you get can differ slightly from the one you asked for with `-v`, because the output shape is an integer: use `-a first` if you need exactly the requested voxel size. With `first`, a factor of 1/2 and an interpolation order of 0 amount to keeping every second voxel. Axes of length one are always treated with `edges`.

You may give one anchor per axis. Only the first letter of an anchor is read, so `c`, `e`, `f` and `l` are accepted.

### Interpolation and boundaries

Values are resampled with B-splines of order `-i`: 0 is nearest neighbour, 1 is trilinear (the default), and 2 and above are higher-order splines. Orders can be given as numbers from 0 to 6, or as names (`nearest`, `linear`, `quadratic`, `cubic`, `fourth`, `fifth`, `sixth`, `seventh`). By default, the spline pre-filter is applied (`-p yes`), so that for orders above 1 the resampled image really interpolates the input rather than blurring it. A different order can be given for each axis.

The boundary condition `-b` describes how the image is extended beyond its edges, which matters for the spline pre-filter and for the voxels close to an edge. The default `dct2` mirrors the image about its outer edge. The accepted values are listed under [boundary conditions](#boundary-conditions). It also applies to samples that fall outside the field of view of the input (see the [table of anchors](#anchors)), which are filled according to `-b` rather than set to zero. This holds for images and, with `-g`, for displacement and transformation fields.

### Output type

Resizing works on a floating-point version of the data. The result is written back with the **on-disk data type of the input**, by a plain cast with no rescaling, so intensities are in the same units as before. For images stored as integers, the interpolated values are truncated by the cast; for label maps, use `-i 0` so that no new values appear, or convert the image to floating point first with [`convert`](../images/convert.md). Images with several channels (a fourth dimension) are resized channel by channel.

## Arguments

| Name   | Description                                                                |
| ------ | -------------------------------------------------------------------------- |
| `FILE` | One or more input volumes. Any format NITorch reads and writes (see [file formats](../../conventions.md#file-formats)). |

## Options

### Size

At least one of the following is required. `-f` and `-v` are mutually exclusive.

| Option                 | Value         | Description                                                                                          | Default |
| ---------------------- | ------------- | ---------------------------------------------------------------------------------------------------- | ------- |
| `-f`, `--factor`       | `FACTOR...`   | Zoom factor, one value or one per axis. Fractions such as `1/2` are accepted. Above 1: more, smaller voxels. Below 1: fewer, larger voxels. | *required*, unless `-v` or `-s` is given |
| `-v`, `--voxel-size`   | `VX...`       | Voxel size of the output, in mm, one value or one per axis.                                           |         |
| `-s`, `--shape`        | `SHAPE...`    | Shape of the output, in voxels, one value or one per axis. May be combined with `-f` or `-v`: see above. |      |

### Resampling

| Option                   | Value        | Description                                                                                                           | Default   |
| ------------------------ | ------------ | --------------------------------------------------------------------------------------------------------------------- | --------- |
| `-a`, `--anchor`         | `ANCHOR...`  | `centers`, `edges`, `first` or `last`; one per axis if needed. [See above](#anchors).                                  | `centers` |
| `-i`, `--interpolation`  | `ORDER...`   | Interpolation order, 0 to 6 as a number, or a name from `nearest` to `seventh`; one per axis if needed.               | `1`       |
| `-b`, `--bound`          | `BND...`     | Boundary condition, for the pre-filter and for samples outside the field of view; one per axis if needed. [See below](#boundary-conditions).                                         | `dct2`    |
| `-p`, `--prefilter`      | `BOOL`       | Apply the spline pre-filter: `yes`, `no` (also `true`, `false`, `1`, `0`).                                             | `yes`     |
| `-g`, `--grid`           | `{d,t}`      | Treat the input as a displacement (`d`) or transformation (`t`) field. [See below](#resizing-displacement-and-transformation-fields). | off       |

### Output and hardware

| Option             | Value        | Description                                                                      | Default                           |
| ------------------ | ------------ | -------------------------------------------------------------------------------- | --------------------------------- |
| `-o`, `--output`   | `FILE...`    | Output filenames, one per input. [Templates](../../conventions.md#output-filenames) with `{dir}`, `{sep}`, `{base}` and `{ext}` are accepted. | `{dir}{sep}{base}.resized{ext}`   |
| `--cpu`            |              | Use the CPU.                                                                     | CPU                               |
| `--gpu`            | `[ID]`       | Use a GPU, optionally the one with this index (see [hardware](../../conventions.md#hardware)). | off                               |

The device flags of this command must be written with two dashes (`--cpu`, `--gpu`); the single-dash forms are not recognised.

### Boundary conditions

| Value        | Aliases                                  | Extension of a row of four voxels `a b c d`  |
| ------------ | ---------------------------------------- | ---------------------------------------------------------- |
| `dct2`       | `reflect`, `reflection`, `neumann`       | Mirror about the outer edge of the border voxel: `b a │ a b c d │ d c`. |
| `dct1`       | `mirror`                                 | Mirror about the centre of the border voxel: `c b │ a b c d │ c b`. |
| `replicate`  | `repeat`, `border`, `nearest`            | Repeat the border voxel: `a a │ a b c d │ d d`.              |
| `zero`       | `zeros`, `constant`                      | Pad with zeros.                                              |
| `dft`        | `wrap`, `circular`                       | Wrap around: `c d │ a b c d │ a b`.                          |
| `dst2`       | `antireflect`, `dirichlet`               | Mirror about the outer edge and change sign: `-b -a │ a b c d │ -d -c`. |
| `dst1`       | `antimirror`                             | Mirror about the border voxel and change sign: `-a 0 │ a b c d │ 0 -d`. |

### Resizing displacement and transformation fields

With `-g`, the input is a field with three components (stored as the last dimension), expressed in voxels of the input grid. Samples outside the field of view follow the boundary condition `-b`, applied to the displacements themselves, so coordinates are not reflected. After resampling, its values are rescaled so that they are expressed in voxels of the output grid. With `-g d` (displacement), each component is divided by the ratio of output to input voxel size: for example, with `-f 2` and `-a first` a displacement of 1 voxel becomes 2 voxels. With `-g t` (transformation, i.e. absolute voxel coordinates), the coordinates are also shifted so that they index the output grid.

## Examples

Halve the resolution of an image, getting a volume with half as many voxels along each axis:

```shell
nitorch resize T1w.nii.gz -f 0.5
```

The same, written with a fraction and a chosen output name:

```shell
nitorch resize T1w.nii.gz -f 1/2 -o T1w_half.nii.gz
```

Resample to 1 mm isotropic voxels, keeping exactly the requested voxel size by anchoring on the first voxel, and using cubic splines. The output has $\lfloor (N-1) f \rfloor + 1$ voxels along an axis of $N$ input voxels, where $f$ is the input voxel size in mm:

```shell
nitorch resize sub-01_T1w.nii.gz -v 1 -a first -i 3 -o sub-01_T1w_1mm.nii.gz
```

Resample an image so that it has exactly $128 \times 128 \times 64$ voxels over the same field of view:

```shell
nitorch resize sub-01_FLAIR.nii.gz -s 128 128 64 -a edges
```

Down-sample a segmentation by a factor of 2 without creating new label values (nearest neighbour, no pre-filter needed):

```shell
nitorch resize aseg.nii.gz -f 0.5 -i 0 -o aseg_half.nii.gz
```

Resize several files at once, each written next to its input with the default suffix:

```shell
nitorch resize sub-01_T1w.nii.gz sub-02_T1w.nii.gz -v 2
```

## See also

- [`reslice`](reslice.md) resamples a volume onto the grid of another image, optionally through a chain of transformations.
- [`crop`](../images/crop.md) and [`pad`](../images/pad.md) change the field of view without resampling.
- [`pool`](../images/pool.md) down-samples by local averaging or other pooling operations.
- [`smooth`](../images/smooth.md) low-pass filters an image, for instance before down-sampling.
