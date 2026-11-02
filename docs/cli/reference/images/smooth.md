---
icon: octicons/terminal-24
---

# smooth

Blur a volume with a Gaussian, box or triangular filter whose width you give
in millimetres or voxels. Use it to suppress noise, to match the resolution of
two images, or to prepare data for an algorithm that expects smooth input.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch smooth FILE... [-f FWHM... [UNIT]] [-m METHOD] [-b BASIS]
                           [-p BOUND] [-o FILE] [--cpu | --gpu [ID]]
    ```

## Description

`smooth` convolves each input volume with a separable filter, one
one-dimensional filter per spatial axis. The output has the same grid as the
input: the same shape, orientation matrix, header and data type. Only the
first three dimensions of a file are smoothed; any further dimension, such as
the channels or time points of a 4D file, is treated as a set of independent
volumes, each smoothed separately.

The width of the filter is its full width at half maximum (FWHM), which you
give with `-f`. For the default Gaussian filter, the FWHM relates to the
standard deviation by $\sigma = \text{FWHM} / \sqrt{8 \ln 2} \approx
\text{FWHM} / 2.355$.

### Giving the FWHM

`-f` takes one to three numbers and an optional unit, in this order:

```text
-f 6            # 6 mm along every axis
-f 4 4 8        # 4 mm, 4 mm and 8 mm along the first, second and third axis
-f 3 vox        # 3 voxels along every axis
-f 2 2 4 vox    # 2, 2 and 4 voxels
```

If you give fewer than three numbers, the last one is repeated. The unit is
the last word of the option and is read as follows:

- `mm` (the default when no unit is given): the numbers are physical widths.
  For each axis, `smooth` divides the width by the voxel size of the image
  along that axis, so that the filter has the same physical width in every
  direction even if the voxels are anisotropic. The voxel size is read from
  the orientation matrix of each file separately.
- anything else, conventionally `vox`: the numbers are widths in voxels, and
  the voxel size is ignored.

The unit applies to all numbers. The comparison with `mm` is exact and case
sensitive, so write the unit in lower case; any other spelling is taken to
mean voxels. Positions in the list refer to the voxel axes in storage order,
not to anatomical directions.

### Filter shapes

The filter is built from two ingredients: a smoothing function, chosen with
`-m`, and a basis function, chosen with `-b`, that models how the image was
sampled. Convolving them gives a discrete kernel that is evaluated at integer
voxel offsets and sums to one, so the mean intensity of the image is preserved.

| Method  | Smoothing function                                                         |
| ------- | -------------------------------------------------------------------------- |
| `gauss` | Gaussian (the default). Truncated at about four standard deviations.       |
| `rect`  | Box (rectangular) function whose width is the FWHM.                         |
| `tri`   | Triangular function whose width at half maximum is the FWHM.               |

The basis (`-b`) can be `tri`, the default, which treats the image as a linear
interpolation of the voxel values, or `rect`, which treats each voxel as a
constant block.

!!! warning "Two options do not currently do what they say"
    `-m tri` is accepted by the parser but stops with an error
    (`KeyError: 'tri'`) when the kernel is built. And `-b rect` is accepted but
    has the same effect as `-b tri`, because any non-empty basis name is
    treated as the linear basis. Use `gauss` or `rect` for the method, and
    leave the basis at its default.

### Boundaries

Near the edges of the volume, the filter needs voxels that lie outside the
image. `-p` says how they are filled in. The default, `replicate`, repeats the
edge voxel; `zero` assumes the image is surrounded by zeros, which darkens the
edges; `dct2` mirrors the image about its edge; `circular` wraps the image
around. The full list of accepted names is given in the table below.

| Name                                                    | Value of the missing voxels           |
| ------------------------------------------------------- | ------------------------------------- |
| `replicate`, `border`, `nearest`, `repeat`              | the nearest voxel of the image        |
| `zero`, `zeros`, `constant`                             | zero                                  |
| `dct2`, `reflect`, `reflection`, `neumann`              | mirror about the edge of the image (the edge voxel is repeated) |
| `dct1`, `mirror`                                        | mirror about the edge voxel (the edge voxel is not repeated) |
| `dft`, `wrap`, `circular`                               | the image, periodically repeated      |
| `dst2`, `antireflect`, `dirichlet`                      | mirror, with opposite sign            |
| `dst1`, `antimirror`                                    | mirror about the edge voxel, with opposite sign |

## Arguments

| Name   | Description                                                                                                  |
| ------ | ------------------------------------------------------------------------------------------------------------ |
| `FILE` | One or more input volumes, in any [format NITorch reads](../../conventions.md#file-formats). They must come before the options. |

## Options

| Option                | Value              | Description                                                                                             | Default                          |
| --------------------- | ------------------ | ------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `-f`, `--fwhm`        | `FWHM... [UNIT]`   | Full width at half maximum, one to three numbers, followed by an optional unit. [See above](#giving-the-fwhm). | `1 mm`                           |
| `-m`, `--method`      | `METHOD`           | Smoothing function: `gauss`, `rect` or `tri` (`tri` currently fails, [see above](#filter-shapes)).       | `gauss`                          |
| `-b`, `--basis`       | `BASIS`            | Basis function: `tri` or `rect` (no practical difference at the moment, [see above](#filter-shapes)).    | `tri`                            |
| `-p`, `--padding`     | `BOUND`            | Boundary condition used outside the image. [See above](#boundaries).                                     | `replicate`                      |
| `-o`, `--output`      | `FILE`             | Output file name or template. Only one value is accepted; it is reused for every input.                 | `{dir}{sep}{base}.smooth{ext}`   |
| `--cpu`               | `[N]`              | Compute on the CPU. The optional number is read but has no effect.                                      | CPU                              |
| `--gpu`               | `[ID]`             | Compute on a GPU. `ID` is the index of the device. Falls back silently to the CPU if CUDA is not available. |                                  |
| `-h`, `--help`        |                    | Print the built-in help and exit.                                                                       |                                  |

### Output names

The output is a template with the placeholders `{dir}` (directory of the
input, `.` if none), `{base}` (name without extension; `.nii.gz` counts as one
extension), `{ext}` (extension with its dot) and `{sep}` (the path separator).
Because `-o` takes a single value, a name without placeholders would be
overwritten by each successive input; with several inputs, always use a
template. See [Output filenames](../../conventions.md#output-filenames) and,
for devices, [Hardware](../../conventions.md#hardware).

The result is written with the data type of the input file. Smoothing
produces fractional values, so an integer-typed image loses them on writing;
convert such an image to a floating-point type first (see
[`convert`](convert.md)).

## Examples

Blur a T1-weighted image with the default 1 mm Gaussian. The result is
`T1w.smooth.nii.gz`:

```shell
nitorch smooth T1w.nii.gz
```

Smooth with a Gaussian of 6 mm FWHM:

```shell
nitorch smooth T1w.nii.gz -f 6
```

Smooth by different amounts along each axis, for example when the slices are
much thicker than the in-plane resolution:

```shell
nitorch smooth sub-01_FLAIR.nii.gz -f 2 2 6 -o sub-01_FLAIR_blur.nii.gz
```

Express the width in voxels instead of millimetres:

```shell
nitorch smooth T1w.nii.gz -f 1.5 vox
```

Smooth several images and name the outputs after their inputs:

```shell
nitorch smooth sub-01_T1w.nii.gz sub-02_T1w.nii.gz -f 4 -o '{dir}{sep}{base}_fwhm4{ext}'
```

Apply a box filter that assumes the image is zero outside the field of view:

```shell
nitorch smooth T1w.nii.gz -f 3 vox -m rect -p zero
```

## See also

- [`pool`](pool.md) — reduce the resolution by window statistics.
- [`resize`](../spatial/resize.md) — change the voxel size of an image.
- [`inpaint`](inpaint.md) — fill in missing voxels.
