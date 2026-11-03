---
icon: octicons/terminal-24
---

# simplereg

Pairwise **affine + nonlinear** registration of a moving image to a fixed
image, with a classical `-m MOVING -f FIXED` interface. It is meant as the
entry point for people who want a deformable registration without learning
the nested syntax of [`register`](register.md). For an affine-only
registration, use [`affreg`](affreg.md) (or `simplereg --nonlin none`).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch simplereg -m FILE -f FILE [-i FILE]
                      [--fwhm VAL [UNIT]] [--pad VAL [UNIT]] [--bound BND]
                      [--order N] [--rescale [MN] MX] [--discretize [N]]
                      [-a FILE] [-w FILE] [-x FILE] [-r FILE]
                      [-l LOSS] [-k SIZE... [UNIT]] [-s [BOOL]]
                      [--affine [{translation,rigid,similitude,affine,none}]]
                      [--no-affine] [-z {gn,lbfgs,powell}]
                      [-n N] [-t TOL] [--line-search N]
                      [--nonlin {svf,shoot,smalldef,none}] [--no-nonlin]
                      [--penalty F] [--absolute F] [--membrane F]
                      [--bending F] [--lame F [F]]
                      [--nonlin-optimizer {gn,lbfgs}] [--nonlin-iter N]
                      [--outer-iter N] [-p LEVEL... | --no-pyramid]
                      [-v [LEVEL]] [--cpu [THREADS] | --gpu [ID]]
    ```

## Description

`simplereg` is a thin wrapper: it reads its options, builds the equivalent
[`register`](register.md) command line, and runs it. The command

```shell
nitorch simplereg -m FLAIR.nii.gz -f T1w.nii.gz
```

is equivalent to

```shell
nitorch register --verbose 0 --cpu \
    @loss nmi @@fix T1w.nii.gz -o false @@mov FLAIR.nii.gz \
        -o '{dir}/{base}.moved{ext}' -r False \
    @affine rigid -o '{dir}/{name}.lta' \
        @@optim powell -n 50 -t 1e-05 -s wolfe \
    @nonlin svf -o '{dir}/{name}.nii.gz' \
        @@optim unset \
    @optim interleaved \
    @pyramid --levels 0 1 2
```

so everything said on the `register` page about losses, the position of the
affine, the regularisation of the nonlinear part, the pyramid and the output
files applies here. What `simplereg` fixes for you:

- One pair of images (one `-m`, one `-f`) and one loss.
- An **affine** part, **rigid** (6 parameters) by default, placed
  symmetrically (`sym`). `--affine TYPE` chooses another model
  (`translation`, `similitude`, or `affine`); a bare `--affine` (no value)
  fits a full **affine** (12 parameters).
- A **nonlinear** part, a stationary velocity field (`svf`) by default, or
  geodesic shooting (`shoot`) or a small deformation field (`smalldef`) with
  `--nonlin`. Its regularisation defaults to those of `register`.
- Both parts are optimised **jointly** (`@optim interleaved`): at each
  pyramid level, `--outer-iter` passes alternate between the affine and the
  nonlinear parts.
- The default loss is **normalised mutual information**. The affine
  optimiser is **Powell's** method with this loss; otherwise both
  optimisers are chosen by `register` (Gauss-Newton when the loss has a
  Hessian, L-BFGS otherwise).
- The preprocessing options (`--fwhm`, `--pad`, `--bound`, `--order`,
  `--rescale`, `--discretize`) apply to **both** images. When they are not
  given, the defaults of `register` apply.

Use `--nonlin none` (or its alias `--no-nonlin`) for an affine-only
registration (then `simplereg` behaves like [`affreg`](affreg.md)), or
`--affine none` (or `--no-affine`) for a nonlinear-only one. The two cannot
be combined.

### Outputs

- **The affine**, in LTA format, written to `{dir}/{name}.lta`
  (`rigid.lta`, `affine.lta`, ... after the `--affine` type) in the
  directory of the fixed image. Use `-a` (`--output-affine`) to choose
  another path. Not written with `--affine none`.
- **The nonlinear field**, written to `{dir}/{name}.nii.gz` (`svf.nii.gz`,
  `shoot.nii.gz` or `smalldef.nii.gz`) in the directory of the fixed image.
  It is a velocity field for `svf` and `shoot` (with a `shoot.json` file
  next to it), a displacement field for `smalldef`. Use `-w`
  (`--output-warp`) to choose another path. Not written with `--nonlin none`.
- **The moved moving image**, `{base}.moved{ext}`, in the directory of the
  moving image: the image carried into the space of the fixed one (*minimal
  reslicing*; see [Outputs of `register`](register.md#outputs)). It is
  written by default. `-x` changes its path, and `-x false` suppresses it.
  The fixed image is never rewritten.
- **The resliced moving image**, `-r PATH`: the moving image resampled on
  the grid of the fixed image through the full (affine + nonlinear)
  transformation. It is not written unless you ask for it.

To apply the transformations to other images, use
[`reslice`](../spatial/reslice.md) with the chain for the `sym` position
given in [Applying the transformations](register.md#applying-the-transformations):

```shell
nitorch reslice FLAIR_mask.nii.gz -l2 rigid.lta -v svf.nii.gz -l2 rigid.lta \
    -t T1w.nii.gz -i 0
```

## Options

### Input

| Option                | Value  | Description                                                                                       | Default    |
| --------------------- | ------ | ------------------------------------------------------------------------------------------------- | ---------- |
| `-m`, `--moving`      | `FILE` | Moving image.                                                                                     | *required* |
| `-f`, `--fixed`       | `FILE` | Fixed image.                                                                                      | *required* |
| `-i`, `--init`        | `FILE` | Initial affine transformation, applied to the orientation matrix of the moving image beforehand. | identity   |

### Preprocessing

Applied to both images. When omitted, the default of `register` is used.

| Option            | Value          | Description                                                                                               | Default        |
| ----------------- | -------------- | --------------------------------------------------------------------------------------------------------- | -------------- |
| `--fwhm`          | `VAL [UNIT]`   | Smooth the images (FWHM), in `vox` or `mm`.                                                               | `0` (none)     |
| `--pad`           | `VAL [UNIT]`   | Pad the images, in `vox` or `mm`.                                                                         | `0` (none)     |
| `--bound`         | `BND`          | Boundary condition: `dct2`, `dct1`, `dst2`, `dst1`, `dft`, `zero`, `replicate` or `nearest` (also `border`), the last two being the same condition. | `dct2`         |
| `--order`         | `N`            | Interpolation order (0 to 7).                                                                             | `1`            |
| `--rescale`       | `[MN] MX`      | Rescale the images so that their `MN`/`MX` percentiles map to (0, 1). `--rescale 0` disables rescaling.   | `0 95`         |
| `--discretize`    | `[N]`          | Discretize the images into `N` bins (`256` if no value).                                                  | not discretized |

### Output

| Option                | Value                 | Description                                                                                  | Default                    |
| --------------------- | --------------------- | -------------------------------------------------------------------------------------------- | -------------------------- |
| `-a`, `--output-affine` | `PATH` &vert; `false` | Where to write the affine. `{dir}` is the directory of the fixed image, `{name}` the model. `false` disables it. | `{dir}/{name}.lta`         |
| `-w`, `--output-warp` | `PATH` &vert; `false` | Where to write the nonlinear field. `false` disables it.                                                       | `{dir}/{name}.nii.gz`      |
| `-x`, `--moved`       | `PATH` &vert; `false` | Where to write the moved moving image (minimal reslicing). `false` disables it.              | `{dir}/{base}.moved{ext}`  |
| `-r`, `--resliced`    | `PATH`                | Where to write the moving image resliced on the fixed image grid.                            | not written                |

### Loss

| Option              | Value            | Description                                                                                                                                  | Default  |
| ------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `-l`, `--loss`      | `NAME`           | Matching term. One of `nmi`, `mse`, `mad`, `tuk`, `cc`, `lcc`, `gmm`, `lgmm`, `emmi`. See [Choosing a loss](register.md#choosing-a-loss).    | `nmi`    |
| `-k`, `--kernel`    | `SIZE... [UNIT]` | Size of the local window of `lcc` and `lgmm`, with a unit `vox`, `mm` or `pct`. Ignored by the other losses.                                 | `10 pct` |
| `-s`, `--symmetric` | `[BOOL]`         | Make the loss symmetric.                                                                                                                     | off      |

### Affine

| Option                | Value                               | Description                                                                                      | Default                                  |
| --------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| `--affine`            | `[TYPE]`                            | Affine model: `translation`, `rigid`, `similitude`, `affine` or `none` (no affine). With no value, `affine` (12 parameters). | `rigid`                                  |
| `--no-affine`         |                                     | Do not estimate an affine transformation (same as `--affine none`).                              |                                          |
| `-z`, `--optimizer`   | `gn` &vert; `lbfgs` &vert; `powell` | Affine optimiser.                                                                                | `powell` with `nmi`, else `register`'s   |
| `-n`, `--max-iter`    | `N`                                 | Maximum number of affine iterations per outer iteration.                                        | `50`                                     |
| `-t`, `--tolerance`   | `TOL`                               | Tolerance of the affine stopping criterion.                                                      | `1e-5`                                   |
| `--line-search`       | `N` &vert; `wolfe`                  | Affine line search.                                                                              | `wolfe`                                  |

### Nonlinear

| Option                | Value                                  | Description                                                                              | Default              |
| --------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------- |
| `--nonlin`            | `svf` &vert; `shoot` &vert; `smalldef` &vert; `none` | Nonlinear model; `none` for affine only.                                   | `svf`                |
| `--no-nonlin`         |                                        | Do not estimate a nonlinear transformation (same as `--nonlin none`).                   |                      |
| `--penalty`           | `F`                                    | Global regularisation factor, multiplying all the penalties below.                      | `1`                  |
| `--absolute`          | `F`                                    | Penalty on absolute displacements.                                                       | `1e-4`               |
| `--membrane`          | `F`                                    | Penalty on the membrane energy (first derivatives).                                     | `1e-3`               |
| `--bending`           | `F`                                    | Penalty on the bending energy (second derivatives).                                     | `0.2`                |
| `--lame`              | `F [F]`                                | Penalties on the linear-elastic energy.                                                 | `0.05 0.2`           |
| `--nonlin-optimizer`  | `gn` &vert; `lbfgs`                    | Nonlinear optimiser.                                                                     | chosen by `register` |
| `--nonlin-iter`       | `N`                                    | Maximum number of nonlinear iterations per outer iteration.                             | `10`                 |

### Joint optimisation

| Option              | Value      | Description                                                                                                         | Default |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------- | ------- |
| `--outer-iter`      | `N`        | Number of interleaved affine/nonlinear passes at each pyramid level.                                                | `5`     |
| `-p`, `--pyramid`   | `LEVEL...` | Pyramid levels to process, as indices or ranges `START:STOP[:STEP]`. Level 0 is full resolution.                    | `0:3`   |
| `--no-pyramid`      |            | Register at full resolution only (same as `-p 0`).                                                                  |         |

### Other

| Option              | Value       | Description                                                                                         | Default |
| ------------------- | ----------- | --------------------------------------------------------------------------------------------------- | ------- |
| `-v`, `--verbose`   | `[LEVEL]`   | Verbosity: `0` is silent, `1` shows progress, `2` more detail, `3` also plots. With no value, `1`. | `0`     |
| `--cpu`             | `[THREADS]` | Run on the CPU, optionally with this number of threads.                                             | CPU     |
| `--gpu`             | `[ID]`      | Run on a GPU (the one with this index, if given).                                                   | CPU     |
| `-h`, `--help`      |             | Print the built-in help.                                                                            |         |

## Examples

Rigid + SVF registration of a FLAIR scan to a T1-weighted scan, writing the
FLAIR resliced on the T1w grid:

```shell
nitorch simplereg -m FLAIR.nii.gz -f T1w.nii.gz -r FLAIR_in_T1w.nii.gz -v
```

This writes `rigid.lta` and `svf.nii.gz` next to `T1w.nii.gz`, and
`FLAIR.moved.nii.gz` next to `FLAIR.nii.gz`.

Full affine, a smoother deformation, and local correlation:

```shell
nitorch simplereg -m sub01.nii.gz -f template.nii.gz --affine \
    -l lcc -k 5 vox --penalty 5 -r sub01_in_template.nii.gz
```

Affine only (equivalent to `affreg`):

```shell
nitorch simplereg -m FLAIR.nii.gz -f T1w.nii.gz --nonlin none
```

Similitude + SVF, with explicit output paths:

```shell
nitorch simplereg -m sub01.nii.gz -f template.nii.gz --affine similitude \
    -a sub01_to_template.lta -w sub01_to_template.nii.gz
```

## See also

- [`affreg`](affreg.md) — affine-only registration with a similar interface.
- [`register`](register.md) — full control: several image pairs or losses,
  per-image preprocessing, label maps, masks, other optimisation strategies.
- [`reslice`](../spatial/reslice.md) — apply the transformations to other
  images.
