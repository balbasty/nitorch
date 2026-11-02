---
icon: octicons/terminal-24
---

# pool

Downsample a volume by reducing each window of voxels to a single value, for
example by averaging or taking the maximum. The orientation matrix of the
output is updated so that it stays aligned with the input in world space.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch pool FILE... [-w WINDOW...] [-s STRIDE...] [-p [BOUND] PAD [BOUND] [PAD [BOUND] ...]]
                         [-m METHOD] [-d DIM] [-o FILE...] [-cpu | -gpu [ID]]
    ```

## Description

`pool` slides a window over the volume and replaces the voxels it covers by one
number, chosen by the pooling [method](#pooling-methods). The window moves by
`STRIDE` voxels at a time, and by default the stride equals the window size.
The windows then tile the image without overlap, and the output has roughly
`1 / WINDOW` of the voxels along each axis. With `-w 2` an image of
`256 x 256 x 160` voxels becomes `128 x 128 x 80`.

This is the same operation as a pooling layer in a convolutional network. It
differs from [`resize`](../spatial/resize.md) and
[`reslice`](../spatial/reslice.md) in that it never interpolates: the output
voxels are statistics of exact groups of input voxels.

Each file is pooled independently, and the result is written next to the
input unless `-o` says otherwise. Only the first `DIM` dimensions of a file
(three by default) are pooled; further dimensions, such as the channels of a 4D
file, are kept and treated as independent channels. The output file uses
the header of the input as a template and has the same data type as the input:
values are converted back to that type without rescaling, so pooling an
integer-typed image with `mean` loses the fractional part of the result. Convert the image to
a floating-point type first (see [`convert`](convert.md)) if you need to keep
fractional values.

### Window and stride

`WINDOW` and `STRIDE` are given in voxels, one value per spatial dimension. If
you give fewer values than there are dimensions, the last value is repeated, so
`-w 2` means `-w 2 2 2`. A window value of `0` (or less) means "the whole
axis": the dimension is reduced to a single voxel.

Without padding, a window is only placed where it fits entirely inside the
image. The output size along an axis of length $L$ is
$\lfloor (L - \text{window}) / \text{stride} \rfloor + 1$, so voxels that
remain after the last complete window are dropped. For example, pooling 100
voxels with a window of 3 gives 33 voxels and drops the last one. You can
avoid this with [padding](#padding).

### Orientation matrix

The output is not simply the input with fewer voxels: it is resampled on a
coarser lattice, and the orientation matrix is updated to describe it. Output
voxel $i$ is centred on the centre of the window that produced it. In voxel
coordinates of the input along one axis, with window size $K$, stride $S$ and
padding $P$ on the low side of the axis (the amount added before the first
voxel), this is the position $S\,i + (K-1)/2 - P$. As a consequence:

- the voxel size is multiplied by the **stride** (not by the window size) along
  each axis;
- the axes keep their direction;
- the origin is shifted by half a window, so that, say, a mean over a
  $2 \times 2 \times 2$ window is placed between the four voxels it averages.

Because of this, the pooled image overlays the original correctly in a viewer,
and can be used as a lower-resolution copy of it. When the window is larger
than the stride, the windows overlap and the voxels are correlated, but the
geometry follows the same rule.

### Padding

`-p` adds voxels around the image before pooling. Each padding size may be
followed by the boundary condition to use along that axis, and an optional
boundary condition placed first applies to all the axes that do not specify
their own:

```text
-p [BOUND] PAD [BOUND] [PAD [BOUND] ...]
```

For example, `-p 1 zero 2 dft` pads the first axis by 1 voxel with zeros and
the remaining axes by 2 voxels with wrap-around, whereas `-p zero 1 2 dft`
pads the first two axes with zeros and the third one with wrap-around. A
boundary condition that does not follow a size is an error, unless it comes
first.

Each `PAD` is a number of voxels added on both sides of an axis, or the word
`same`. With `same`, the smallest padding is chosen that gives an output of
$\lfloor L / \text{stride} \rfloor$ voxels along that axis. When the window
equals the stride, the unpadded output already has that size, so `same` adds
nothing; it matters for overlapping windows, for instance a window of 3 with a
stride of 1, which then keeps the size of the image. If the amount of padding
$P$ needed is odd, it is split unevenly: $\lfloor P / 2 \rfloor$ voxels are
added before the image and the remaining $\lceil P / 2 \rceil$ after it, and
the orientation matrix accounts for the voxels added before. As for the window,
the last value is repeated if fewer than three are given, whether it is a size
or a boundary condition.

`BOUND` says what values the padding voxels take. The default is `dct2`, which
mirrors the image at its edge (the edge voxel is repeated, so `a b c d` is padded to `b a a b c d d c`).
The other values accepted are the boundary conditions listed in
[Boundary conditions](#boundary-conditions) below.

### Pooling methods

| Method   | Value written for each window                                 |
| -------- | ------------------------------------------------------------- |
| `mean`   | The arithmetic mean (the default).                            |
| `sum`    | The sum.                                                      |
| `min`    | The smallest value.                                           |
| `max`    | The largest value.                                            |
| `median` | The median.                                                   |
| `ssq`    | The sum of the squared values.                                |

The method is not validated by the parser: an unknown name is reported as an
error only once the computation starts.

### Boundary conditions

The names below can be used with `-p`. Several names exist for most of them
(the Python/SciPy and PyTorch spellings are accepted too).

| Name        | Pads with                                                  | Before `a b c d` | After `a b c d` |
| ----------- | ---------------------------------------------------------- | ---------------- | --------------- |
| `dct2`      | mirror about the edge (also `reflect`, `neumann`)          | `b a`            | `d c`           |
| `dct1`      | mirror about the edge voxel (also `mirror`)                | `c b`            | `c b`           |
| `replicate` | the edge value (also `border`, `nearest`, `repeat`)        | `a a`            | `d d`           |
| `zero`      | zeros (also `zeros`, `constant`)                           | `0 0`            | `0 0`           |
| `dft`       | wrap-around (also `wrap`, `circular`)                      | `c d`            | `a b`           |
| `dst2`      | antisymmetric mirror (also `antireflect`, `dirichlet`)     | `-b -a`          | `-d -c`         |
| `dst1`      | antisymmetric mirror (also `antimirror`)                   | `-a 0`           | `0 -d`          |

## Arguments

| Name   | Description                                                      |
| ------ | ---------------------------------------------------------------- |
| `FILE` | One or more input volumes, in any [format NITorch reads](../../conventions.md#file-formats). They must come before the options. |

## Options

| Option                  | Value              | Description                                                                              | Default                          |
| ----------------------- | ------------------ | ---------------------------------------------------------------------------------------- | -------------------------------- |
| `-w`, `--window`        | `WINDOW...`        | Window size, in voxels, per dimension. `0` pools a whole axis. [See below](#window-and-stride). | `3`                              |
| `-s`, `--stride`        | `STRIDE...`        | Step between windows, in voxels, per dimension.                                          | same as the window               |
| `-p`, `--padding`       | `[BOUND] PAD [BOUND] ...` | Padding added on both sides, in voxels, or `same`, each optionally followed by a boundary condition; a leading one applies to all axes. [See below](#padding). | `0`                              |
| `-m`, `--method`        | `METHOD`           | Pooling function: `mean`, `sum`, `min`, `max`, `median` or `ssq`. [See above](#pooling-methods). | `mean`                           |
| `-d`, `--dim`, `--dimension` | `DIM`         | Number of spatial dimensions. It has to agree with the orientation matrix of the file, so leave it at `3` for ordinary volumes. `0` takes the value from the file. | `3`                              |
| `-o`, `--output`        | `FILE...`          | Output file names or templates, one per input. A single template is reused for every input. | `{dir}{sep}{base}.pool{ext}`     |
| `-cpu`, `--cpu`         |                    | Compute on the CPU.                                                                      | CPU                              |
| `-gpu`, `--gpu`         | `[ID]`             | Compute on a GPU; `ID` selects the device (`cuda:ID`).                                   |                                  |

### Output names

Templates accept `{dir}` (directory of the input, `.` if none), `{base}` (name
without extension; `.nii.gz` counts as one extension), `{ext}` (extension with
its dot) and `{sep}` (the path separator). See
[Output filenames](../../conventions.md#output-filenames). If you give fewer
names than inputs, the last name is repeated, which makes later outputs
overwrite earlier ones; give a template, or one name per input. Device options
are described in [Hardware](../../conventions.md#hardware).

## Examples

Halve the resolution of a volume by averaging $2 \times 2 \times 2$ blocks. The
result is `T1w.pool.nii.gz`:

```shell
nitorch pool T1w.nii.gz -w 2
```

Average over a window of 4 voxels, but move by 2, so that windows overlap:

```shell
nitorch pool T1w.nii.gz -w 4 -s 2
```

Downsample a binary mask so that a voxel is set if any voxel of its block was
set:

```shell
nitorch pool brain_mask.nii.gz -w 2 -m max -o brain_mask_lowres.nii.gz
```

Pool only in the first two directions, leaving the third at full resolution
(a window of 1 along the last axis):

```shell
nitorch pool bold.nii.gz -w 2 2 1
```

Collapse the whole third axis into one slice, taking the median along it:

```shell
nitorch pool T1w.nii.gz -w 1 1 0 -m median -o '{dir}{sep}{base}.median-z{ext}'
```

Compute a local mean over a 3-voxel window that moves one voxel at a time,
padding the edges so the image keeps its size and grid:

```shell
nitorch pool T1w.nii.gz -w 3 -s 1 -p same -o T1w.localmean.nii.gz
```

Pool several images at once. Each result is written next to its input, as
`sub-01_T1w.pool.nii.gz` and `sub-02_T1w.pool.nii.gz`:

```shell
nitorch pool sub-01_T1w.nii.gz sub-02_T1w.nii.gz -w 2
```

## See also

- [`resize`](../spatial/resize.md) — change the resolution by interpolation.
- [`reslice`](../spatial/reslice.md) — resample an image onto another lattice.
- [`smooth`](smooth.md) — low-pass filter an image without changing its grid.
