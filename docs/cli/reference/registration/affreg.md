---
icon: octicons/terminal-24
---

# affreg

Affine registration of a moving image to a fixed image, with a classical
`-m MOVING -f FIXED` interface. Use it for the most common task, aligning
two scans of the same subject rigidly or with a full affine, when you do not
need the control (or the nested syntax) of [`register`](register.md).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch affreg -m FILE -f FILE [-i FILE] [-o FILE] [-x FILE] [-r FILE]
                   [-l LOSS] [-k SIZE... [UNIT]] [-s [BOOL]]
                   [-a [BOOL]] [-p LEVEL...] [-z {gn,lbfgs,powell}]
                   [-n N] [-t TOL] [--line-search N]
                   [-v [LEVEL]] [--cpu [THREADS] | --gpu [ID]]
    ```

## Description

`affreg` is a thin wrapper: it does not implement any registration of its
own. It reads its options, builds the equivalent
[`register`](register.md) command line, and runs it. The command

```shell
nitorch affreg -m FLAIR.nii.gz -f T1w.nii.gz
```

is equivalent to

```shell
nitorch register --verbose 0 --cpu \
    @loss nmi @@fix T1w.nii.gz -o false @@mov FLAIR.nii.gz \
        -o '{dir}/{base}.moved{ext}' -r False \
    @affine -o '{dir}/{name}.lta' \
        @@optim powell -n 50 -t 1e-05 -s wolfe \
    @pyramid --levels 0 1 2
```

so everything said on the `register` page about losses, the position of the
affine, the pyramid and the output files applies here. What `affreg` fixes
for you:

- The transformation is **rigid** (6 parameters), or a full **affine** (12
  parameters) with `-a`. There is no initialisation by progressive
  freeing of parameters, and no prior on the parameters.
- The affine is in the **symmetric** position: it is parameterised by its
  square root. The `.moved` image still receives the full affine in its
  header, which carries it into the space of the fixed image. The matrix
  written to disk is the full transformation, which maps
  fixed world coordinates to moving world coordinates, in RAS convention. See
  [Position of the affine](register.md#position-of-the-affine).
- The default loss is **normalised mutual information** (Studholme's
  normalisation), which works across contrasts, and the optimiser is
  **Powell's** derivative-free method when this loss is used, **Gauss-Newton**
  otherwise.
- The pyramid has **three levels**: full resolution, and two coarser levels
  that are subsampled by 2 and 4.
- Images are rescaled to their 0th to 95th percentiles, and voxels equal
  to 0 are treated as missing, as in `register`.

For anything else (a different affine model such as translations only,
several image pairs, label maps, a nonlinear part, an initial `com` or `fov`
alignment, another parametrisation of the penalty) use `register`.

### Outputs

Three kinds of files can be written:

- **The transformation**, an affine matrix in LTA format, written to
  `{dir}/{name}.lta`: `rigid.lta` or `affine.lta`, in the directory of the
  fixed image. Use `-o` to choose another path, or `-o false` to skip it.
- **The moved moving image**, `{base}.moved{ext}`, in the directory of the
  moving image. The moving image is carried into the space of the fixed
  image by changing only its orientation matrix, so that it overlaps the
  fixed image without any interpolation. It is
  written by default. `-x` changes its path, and `-x false` suppresses it.
  The fixed image is never rewritten.
- **The resliced moving image**, `-r PATH`, which is the moving image
  resampled on the grid of the fixed image. This is the file most people want.
  It is not written unless you ask for it.

Paths given to `-o`, `-x` and `-r` can use the fields `{dir}`, `{base}`,
`{ext}` and, for `-o`, `{name}`. See
[Output filenames](../../conventions.md#output-filenames).

To apply the transformation to other images, use
[`reslice`](../spatial/reslice.md):

```shell
nitorch reslice FLAIR_mask.nii.gz -l rigid.lta -t T1w.nii.gz -i 0
```

### Chaining registrations

With `-i`, the initial transformation is folded into the orientation matrix
of the moving image, and the registration then estimates what is left. The
file written by `-o` is therefore the *residual* transformation $R$, and the
complete transformation from the fixed to the moving space is the product of
the initial transformation $T_0$ and $R$. [`reslice`](../spatial/reslice.md)
multiplies consecutive `-l` transformations in the order they are given, so
`-l init.lta -l result.lta` applies the complete transformation. The moved
and resliced images written by `affreg` already include the initial
transformation.

## Options

### Input

| Option                | Value  | Description                                                                                                                                                  | Default    |
| --------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `-m`, `--moving`      | `FILE` | Moving image.                                                                                                                                                | *required* |
| `-f`, `--fixed`       | `FILE` | Fixed image.                                                                                                                                                 | *required* |
| `-i`, `--init`        | `FILE` | Initial affine transformation, for instance the output of a previous registration. It is applied to the orientation matrix of the moving image before the registration starts. The matrix written by `affreg` then describes the *remaining* transformation and does not include it; see [below](#chaining-registrations). | identity   |

### Output

| Option                | Value           | Description                                                                                      | Default                          |
| --------------------- | --------------- | ------------------------------------------------------------------------------------------------ | -------------------------------- |
| `-o`, `--output`      | `PATH` &vert; `false` | Where to write the affine matrix. `{dir}` is the directory of the fixed image. `false` disables it. | `{dir}/{name}.lta`               |
| `-x`, `--moved`       | `PATH` &vert; `false` | Where to write the moved moving image (header change only). `false` disables it.          | `{dir}/{base}.moved{ext}`        |
| `-r`, `--resliced`    | `PATH`          | Where to write the moving image resliced on the fixed image grid.                                | not written                      |

### Loss

| Option                  | Value                | Description                                                                                                                                                                   | Default   |
| ----------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `-l`, `--loss`          | `NAME`               | Matching term. One of `nmi`, `mse`, `mad`, `tuk`, `cc`, `lcc`, `gmm`, `lgmm`. See [Choosing a loss](register.md#choosing-a-loss).                                              | `nmi`     |
| `-k`, `--kernel`        | `SIZE... [UNIT]`     | Size of the local window of `lcc` and `lgmm`: one value, or one per axis, then a unit `vox`, `mm` or `pct` (percentage of the image size). Ignored by the other losses.        | `10 pct`  |
| `-s`, `--symmetric`     | `[BOOL]`             | Make the loss symmetric, by also comparing the fixed image warped to the moving image.                                                                                         | off       |

### Optimisation

| Option                  | Value                       | Description                                                                                                                                  | Default                                    |
| ----------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `-a`, `--affine`        | `[BOOL]`                    | Fit a full 12-parameter affine instead of a 6-parameter rigid transformation.                                                                 | rigid                                      |
| `-p`, `--pyramid`       | `LEVEL...`                  | Pyramid levels to process, as indices or ranges `START:STOP[:STEP]` (the stop is required). Level 0 is full resolution. For a single level, `-p 0`. | `0:3`                                      |
| `-z`, `--optimizer`     | `gn` &vert; `lbfgs` &vert; `powell` | Optimiser: Gauss-Newton, L-BFGS, or Powell's derivative-free method.                                                                   | `powell` with `nmi`, `gn` otherwise        |
| `-n`, `--max-iter`      | `N`                         | Maximum number of iterations at each pyramid level.                                                                                          | `50`                                       |
| `-t`, `--tolerance`     | `TOL`                       | Tolerance of the stopping criterion.                                                                                                         | `1e-5`                                     |
| `--line-search`         | `N` &vert; `wolfe`          | Backtracking line search with at most `N` steps, or a line search until the Wolfe conditions hold.                                           | `wolfe`                                    |
| `--no-pyramid`          |                             | Process a single level, at full resolution (same as `-p 0`).                                                                                  |                                            |

### Other

| Option                  | Value         | Description                                                                                          | Default |
| ----------------------- | ------------- | ---------------------------------------------------------------------------------------------------- | ------- |
| `-v`, `--verbose`       | `[LEVEL]`     | Verbosity: `0` is silent, `1` shows progress, `2` more detail, `3` also plots (requires `matplotlib`). With no value, `1`. | `0`     |
| `--cpu`                 | `[THREADS]`   | Run on the CPU, optionally with this number of threads.                                              | CPU     |
| `--gpu`                 | `[ID]`        | Run on a GPU (the one with this index, if given).                                                    | CPU     |
| `-h`, `--help`          |               | Print the built-in help.                                                                             |         |

See [Hardware](../../conventions.md#hardware) for general notes on devices.

## Examples

Rigidly align a FLAIR scan to a T1-weighted scan. This writes `rigid.lta`
next to `T1w.nii.gz`, and `FLAIR.moved.nii.gz` next to `FLAIR.nii.gz`:

```shell
nitorch affreg -m FLAIR.nii.gz -f T1w.nii.gz
```

Fit a full affine, write the FLAIR image resampled on the T1w grid, and do
not write the moved FLAIR image:

```shell
nitorch affreg -m FLAIR.nii.gz -f T1w.nii.gz -a \
    -r FLAIR_in_T1w.nii.gz -x false
```

Register two T1-weighted scans of the same subject using mean squared
error, which is faster than mutual information when the contrast is the
same, and print progress:

```shell
nitorch affreg -m sub-01_ses-2_T1w.nii.gz -f sub-01_ses-1_T1w.nii.gz \
    -l mse -v
```

Refine a previous registration with a full affine, starting from its result
and using only the two finest pyramid levels. The new matrix is the
correction on top of `rigid.lta`, so both are needed to resample an image:

```shell
nitorch affreg -m FLAIR.nii.gz -f T1w.nii.gz -i rigid.lta -a -p 0:2 \
    -o affine_refined.lta
nitorch reslice FLAIR.nii.gz -l rigid.lta -l affine_refined.lta -t T1w.nii.gz
```

Use a local correlation loss with a 6 mm window, on a GPU:

```shell
nitorch affreg -m sub-01_T1w.nii.gz -f template_T1w.nii.gz \
    -l lcc -k 6 mm --gpu -a
```

## See also

- [`register`](register.md) — the command `affreg` wraps, with full control.
- [`simplereg`](simplereg.md) — a similar interface with an added nonlinear
  part. Beware that `-a` means something else there: it names the output
  affine, and the model is chosen with `--affine [TYPE]`.
- [`reslice`](../spatial/reslice.md) — apply the resulting matrix to other
  images.
- [Registering two images](../../tutorials/registration.md) — a tutorial.
