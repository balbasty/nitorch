---
icon: octicons/terminal-24
---

# vexp

Exponentiate a stationary velocity field, by scaling and squaring, into a dense displacement field or transformation field. Use it to turn the velocity output of a diffeomorphic registration into a deformation that other tools can read.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch vexp FILE... [-t {d,t}] [-u {v,mm}] [-n STEPS] [-b BND]
                         [-o FILE...] [-cpu | -gpu [ID]]
    ```

## Description

A *stationary velocity field* $v(x)$ is a vector field that does not change with time. Following the flow of $v$ for one unit of time, that is, solving $\dot\phi_t = v(\phi_t)$ with $\phi_0 = \mathrm{id}$, yields a transformation $\phi_1 = \exp(v)$. Because it is the end point of a smooth flow, this transformation is invertible and has an invertible, smooth inverse (it is a *diffeomorphism*): $\exp(v)^{-1} = \exp(-v)$. This is why registration methods such as the one in [`register`](../registration/register.md) optimise a velocity field rather than a displacement field directly.

`vexp` computes $\exp(v)$ with **scaling and squaring**. The velocity is first divided by $2^N$, which makes it small enough for $\exp(v/2^N) \approx \mathrm{id} + v/2^N$ to be accurate. The resulting transformation is then composed with itself $N$ times, each composition doubling the time that has been integrated, until the full unit of time is reached. $N$ is set with `-n` (default 8, so the flow is integrated in 256 small steps). More steps give a more accurate result and cost one more resampling of the field each.

The velocity file must hold the three components of the vector in its last dimension (a volume of $X \times Y \times Z \times 3$ values), expressed in **voxels** of the lattice of the file itself. Each input file is processed independently; its orientation matrix is copied to the output.

### What is written

Two choices define the output.

**Type (`-t`).** With `d` (the default), the output is a *displacement field* $u$, such that the transformation is $\phi(x) = x + u(x)$. With `t`, the output is the *transformation field* $\phi(x)$ itself, that is, the coordinates to which every voxel is sent. Only the first letter is read, so `disp` and `trans` work too.

**Unit (`-u`).** With `v` (the default), values are in voxels of the lattice of the file: displacements are voxel shifts, and coordinates are voxel indices. With `mm`, the output is multiplied by the orientation matrix of the input. For a transformation field (`-t t`), the whole matrix is applied, which turns voxel coordinates into world coordinates in millimetres. For a displacement field (`-t d`), only the linear part of the matrix is applied, since a displacement is a vector and is not affected by the translation. Any value other than `mm` is treated as voxels.

The exponential of a field is a *forward* transformation in the sense that it maps coordinates of the space in which $v$ lives to new coordinates in the same space. When you use the output with [`reslice`](reslice.md) or [`compose`](compose.md), it plays the role of a displacement field (`-d FILE`), mapping coordinates of the output space to coordinates of the input space; see [the conventions of `reslice`](reslice.md#how-transformations-are-chained). You rarely need to compute the inverse separately: `reslice` and `compose` accept the velocity field directly (`-v` and `-iv`) and exponentiate it themselves.

The output is written with the data type of the input file.

## Arguments

| Name   | Description                                                                                     |
| ------ | ----------------------------------------------------------------------------------------------- |
| `FILE` | One or more velocity fields, as volumes with three components in the last dimension. Any format NITorch reads (see [file formats](../../conventions.md#file-formats)). |

## Options

| Option                  | Value          | Description                                                                                                  | Default                    |
| ----------------------- | -------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `-t`, `--type`          | `{d,t}`        | Output a displacement field (`d`, `disp`...) or a transformation field (anything starting with another letter, e.g. `t`). | `d`                        |
| `-u`, `--unit`          | `{v,mm}`       | Voxels (`v`, or anything other than `mm`) or millimetres (`mm`). See above.                                   | `v`                        |
| `-n`, `--nb-steps`      | `STEPS`        | Number of scaling-and-squaring steps $N$; the field is divided by $2^N$ before squaring.                      | `8`                        |
| `-b`, `--bound`         | `BND`          | Boundary condition used when the field is resampled during squaring: `dft`, `dct1`, `dct2`, `dst1`, `dst2`, `zero` (aliases listed in [`resize`](resize.md#boundary-conditions) also work). | `dft` (circular)           |
| `-inv`, `--inverse`     | `[BOOL]`       | Output the inverse transformation, the exponential of $-v$. The flag alone means `yes`; it also accepts `yes`/`no`, `true`/`false` or `1`/`0`. | off                        |
| `--inversion`           | `{default,push,newton}` | Method used to compute the inverse (with `-inv`): the numerical inverse of $\exp(v)$ by `push` or `newton`, as in [`register --inversion`](../registration/register.md#choosing-how-the-warp-is-inverted-with-inversion), instead of $\exp(-v)$. | the value in the `.json` sidecar of the `svf`, else `default` |
| `-o`, `--output`        | `FILE...`      | Output filenames, one per input. [Templates](../../conventions.md#output-filenames) with `{dir}`, `{sep}`, `{base}` and `{ext}` are accepted. | `{dir}{sep}{base}.vexp{ext}` |
| `-cpu`, `--cpu`         |                | Use the CPU.                                                                                                  | CPU                        |
| `-gpu`, `--gpu`         | `[ID]`         | Use a GPU, optionally the one with this index (see [hardware](../../conventions.md#hardware)).               | off                        |

If fewer output names than inputs are given, the last one is repeated (and would be overwritten), so give a template when you process several files.

## Examples

Exponentiate a velocity field into a displacement field in voxel units. The result is written to `velocity.vexp.nii.gz`:

```shell
nitorch vexp velocity.nii.gz
```

Get the transformation as world coordinates in millimetres, with a name of your choice:

```shell
nitorch vexp velocity.nii.gz -t t -u mm -o coordinates_mm.nii.gz
```

Use fewer squaring steps for a quick, approximate result on a large field:

```shell
nitorch vexp velocity.nii.gz -n 5 -o disp_fast.nii.gz
```

Process two fields on the GPU, keeping their directories:

```shell
nitorch vexp sub-01_velocity.nii.gz sub-02_velocity.nii.gz -gpu -o '{dir}{sep}{base}_disp{ext}'
```

## See also

- [`reslice`](reslice.md) applies velocity fields (`-v`, `-iv`) directly to images.
- [`compose`](compose.md) combines several transformations, including velocity fields, into one.
- [`register`](../registration/register.md) and [`autoreg`](../registration/autoreg.md) produce velocity fields.
