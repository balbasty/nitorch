---
icon: octicons/terminal-24
---

# inpaint

Fill in missing voxels of a volume from the values that surround them. Use it
to repair images with holes, for example voxels masked out as `NaN` after a
lesion or artefact was removed, or regions set to zero by a previous processing
step.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch inpaint FILE... [-m VALUE...] [-o FILE...]
                            [-nrls N] [-trls TOL] [-ncg N] [-tcg TOL]
                            [-v [LEVEL]] [-cpu | -gpu [ID]]
    ```

## Description

`inpaint` leaves every observed voxel untouched and estimates only the missing
ones. The estimate is the one that makes the completed image as smooth as
possible in the sense of **total variation**: it penalises the size of the
image gradient, but, unlike a Gaussian blur, it does not penalise a few large
jumps as much as many small ones. Edges that run into a hole are therefore
continued across it rather than smeared, and flat regions stay flat.

When you give several files, they are treated as the channels of a single
image and are inpainted **jointly**. The gradient magnitude that enters the
penalty is computed over all channels together, so all channels share the same
edges. This is the right choice for a set of co-registered contrasts of the
same subject (say a T1-weighted and a T2-weighted scan), and it makes the
filling of each channel benefit from the structure visible in the others. The
files must have the same shape. The voxel size is read from the orientation
matrix of the first file. Each channel keeps its own set of missing voxels: a
voxel that is missing in one file and observed in another is filled in the
first and kept in the second.

Each input produces its own output file, with the header and data type of that
input. The data type matters: the values estimated for the missing voxels are
real numbers, and they are converted back to the data type of the input
without rescaling. If you work with an integer-typed image, convert it to
floating point first (see [`convert`](convert.md)) so that the filled values
are not rounded off.

### What counts as missing

By default, a voxel is missing if its value is not finite: `NaN`, `+Inf` or
`-Inf`. With `-m`, you give the values that mark missing data in your images.
Every voxel equal to one of them is then treated as missing, in addition to
non-finite voxels. The comparison is an exact equality on the (floating-point)
voxel value, applied to every channel. The typical use is `-m 0`, for images in
which holes were filled with zero. Be careful with `-m 0` on an image whose
background is also zero: the background would be treated as missing and filled
in.

### The algorithm

The problem is solved by *iteratively reweighted least squares*. Missing voxels
are first set to the median of the observed voxels of their channel. Each
iteration then does the following.

1. From the current estimate, compute a weight at every voxel. The weight is
   the inverse of the gradient magnitude (computed across all channels and in
   physical units), so that voxels on strong edges count for less.
2. With these weights fixed, the total variation penalty is a quadratic
   (membrane) energy. Solve for the missing voxels, with the observed voxels as
   fixed boundary values, using a few steps of the preconditioned conjugate
   gradient method.
3. Update the missing voxels and recompute the objective.

In the first iteration all weights are equal, which amounts to a plain
Laplacian interpolation; later iterations sharpen the result around edges. The
loop stops when the relative change of the objective drops below `--tol-rls`,
or after `--max-rls` iterations. In each iteration, the conjugate gradient
solver stops after `--max-cg` steps or when it reaches `--tol-cg`. At the
outer edges of the field of view, the image is assumed to be flat across the
boundary. The voxel sizes only enter through their ratios: the gradient along
an axis with larger voxels is scaled down accordingly, so anisotropic images
are interpolated correctly.

The defaults are enough for holes of moderate size. Larger holes, or images in
which most of the volume is missing, may need more iterations.

### Progress output

At verbosity 1 (the default) the command prints one line per reweighting
iteration, giving the iteration number, the two terms of the objective and
their sum, and prints `Converged` when it stops because the tolerance was met.
Verbosity 2 additionally prints the progress of the conjugate gradient solver.
At verbosity 0 nothing is printed.

## Arguments

| Name   | Description                                                                                                       |
| ------ | ----------------------------------------------------------------------------------------------------------------- |
| `FILE` | One or more input volumes, in any [format NITorch reads](../../conventions.md#file-formats). They must come before the options, and must have the same shape. |

## Options

| Option                  | Value         | Description                                                                                              | Default                              |
| ----------------------- | ------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `-m`, `--missing`       | `VALUE...`    | Values that mark missing voxels, in addition to non-finite voxels. [See above](#what-counts-as-missing). | only non-finite voxels               |
| `-o`, `--output`        | `FILE...`     | Output file names or templates, one per input. A single template is reused for every input.              | `{dir}{sep}{base}.inpaint{ext}`      |
| `-nrls`, `--max-rls`    | `N`           | Maximum number of reweighting (RLS) iterations.                                                          | `10`                                 |
| `-trls`, `--tol-rls`    | `TOL`         | Stop when the relative change of the objective falls below this value.                                   | `1e-5`                               |
| `-ncg`, `--max-cg`      | `N`           | Maximum number of conjugate gradient iterations per RLS iteration.                                       | `32`                                 |
| `-tcg`, `--tol-cg`      | `TOL`         | Tolerance of the conjugate gradient solver.                                                              | `1e-5`                               |
| `-v`, `--verbose`       | `[LEVEL]`     | Verbosity: 0 is silent, 1 prints the objective, 2 adds solver details. Without a value, level 1.        | `1`                                  |
| `-cpu`, `--cpu`         |               | Compute on the CPU.                                                                                      | CPU                                  |
| `-gpu`, `--gpu`         | `[ID]`        | Compute on a GPU; `ID` selects the device (`cuda:ID`).                                                   |                                      |
| `-h`, `--help`          |               | Print the built-in help and exit.                                                                        |                                      |

### Output names

Templates accept `{dir}` (directory of the input, `.` if none), `{base}`
(name without extension; `.nii.gz` counts as one extension), `{ext}`
(extension with its dot) and `{sep}` (the path separator), each taken from the
corresponding input file. See
[Output filenames](../../conventions.md#output-filenames). If you pass fewer
names than inputs, the last one is repeated, and later outputs overwrite
earlier ones; pass a template or one name per input. Device options are
described in [Hardware](../../conventions.md#hardware).

## Examples

Fill the `NaN` voxels of a volume. The result is `T1w_masked.inpaint.nii.gz`:

```shell
nitorch inpaint T1w_masked.nii.gz
```

Treat zeros as missing data, and choose the output name:

```shell
nitorch inpaint T1w_holes.nii.gz -m 0 -o T1w_filled.nii.gz
```

Treat both zero and a sentinel value as missing:

```shell
nitorch inpaint sub-01_T2w.nii.gz -m 0 -9999
```

Inpaint two co-registered contrasts jointly, so that they share their edges:

```shell
nitorch inpaint sub-01_T1w.nii.gz sub-01_T2w.nii.gz \
    -o '{dir}{sep}{base}_filled{ext}'
```

Allow more iterations for a large hole, run on the first GPU and keep the
output quiet:

```shell
nitorch inpaint sub-01_FLAIR.nii.gz -nrls 30 -ncg 64 -gpu 0 -v 0
```

## See also

- [`smooth`](smooth.md) — blur an image.
- [`convert`](convert.md) — change the data type of an image.
