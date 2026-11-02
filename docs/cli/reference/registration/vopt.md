---
icon: octicons/terminal-24
---

# vopt

Turns a set of pairwise nonlinear registrations between subjects, each
encoded as a stationary velocity field, into one consistent
"template-to-subject" velocity field per subject. It is the nonlinear
counterpart of [`affopt`](affopt.md).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch vopt -i FIX MOV FILE [-i FIX MOV FILE]... [-w FIX MOV FILE]...
                 [-o FILE] [-x BOUND] [-a VAL] [-m VAL] [-b VAL] [-l VAL [VAL]]
    ```

## Description

When you register several images of a subject (several contrasts, or several
time points) in pairs with a diffeomorphic method, the pairwise results do not
agree with one another: going from `A` to `B` and then from `B` to `C` does
not land where the direct registration from `A` to `C` does. `vopt` reconciles
the pairwise velocity fields. It looks for one velocity field $v_i$ per
image, such that for every pair you provide

$$
v_{ij} \approx v_i - v_j,
$$

where $v_{ij}$ is the field that registers image $j$ (moving) to image $i$
(fixed). Velocity fields live in a vector space, so this is a linear
least-squares problem. The solution is defined up to a common additive field;
`vopt` removes this ambiguity by centring the $v_i$, so that they sum to
zero. The space they refer to is the "mean space" of the subjects.

Read the other way round, $v_i$ is the field you would estimate by registering
a virtual template, as the moving image, to subject $i$, taken as the fixed
image. This is why the outputs are called template-to-subject fields.

The idea comes from the USLR and JUMP pipelines of Casamitjana et al., which
fit mean-to-subject fields with an L1 objective, and from the longitudinal
model of Ashburner and Ridgway (2012). `vopt` instead uses a weighted L2
objective with fixed weights, which you can supply with `-w`. These weights are
the precisions (inverse variances, or squared weights) of the pairwise fields
and can come from the Hessian of the registration loss, written at the same
time as the field by [`register`](register.md). The estimated fields are then
smoothed by a regulariser built from the absolute, membrane, bending and
linear-elastic energies of the field (`-a`, `-m`, `-b`, `-l`).

### Inputs

Each `-i` option describes one pair: the *label* of the fixed image, the label
of the moving image, and the path to a stationary velocity field that warps the
moving image to the fixed one. Labels are arbitrary strings (`mtw`, `01`,
`sub-02_T1w`); they only serve to tell which pairs involve the same image, and
they are used in the names of the output files. If you give the same pair
twice, the last file wins. If only one direction of a pair is given, the other
is assumed to be its negative (the inverse transformation). Leaving pairs out
can bias the estimate, but the bias averages out when there are many subjects
and the pairs you evaluate are sampled at random.

All velocity fields must be defined on the same voxel grid, as 3D vector
fields with three components (see
[Spatial transformations](../../conventions.md#spatial-transformations)).
The pairs should link all images together: every image should appear in at
least one pair, and the graph formed by the pairs should be connected.

Each `-w` option attaches a weight map to one pair, with the same three
arguments as `-i`. A weight map is an image on the same grid as the fields,
with either one value per voxel (the same weight for the three components),
three values (one weight per component) or six values (the independent
entries of a symmetric $3 \times 3$ precision matrix: three diagonal entries,
followed by the three off-diagonal ones). The weight of a pair is also used for
its reverse. Pairs that have a field but no weight receive the mean of the other
weights. If you give no `-w` option at all, all pairs are weighted equally.

### Regularisation

After the pairwise fields have been combined, each template-to-subject field is
smoothed by solving a regularised problem, using the penalties below. The
voxel size used by the penalties is taken from the first input file.

| Penalty | Option | Effect |
| ------- | ------ | ------ |
| Absolute | `-a` | Penalises large velocities. |
| Membrane | `-m` | Penalises first derivatives; favours spatially constant fields. |
| Bending | `-b` | Penalises second derivatives; favours smooth fields. |
| Linear-elastic | `-l` | Penalises divergence (local volume change) and shears; takes two values. If only one is given, it is used for both. |

`-x` selects the boundary condition used when computing derivatives.
Accepted values are `dct2`, `dct1`, `dst2`, `dst1`, `dft`, `zero` and
`nearest`.

### Output

For every label, `vopt` writes one velocity field. The header (orientation
matrix and grid) is copied from the first input file. The output path is a
template in which these placeholders are expanded:

| Placeholder | Replaced by |
| ----------- | ----------- |
| `{label}` | The label of the image. |
| `{dir}` | The directory of the first input file. |
| `{sep}` | The directory separator of your system. |
| `{ext}` | The extension of the first input file (such as `.nii.gz`). |

!!! warning "Always pass `-o`"
    The built-in default is `{dir}{sep}{label}_optimal.lta`. The `.lta` extension
    selects a linear-transform file format, which cannot store a velocity
    field. Always give an output template with a volume extension, for example
    `-o out/{label}_to_mean{ext}`. Create the output directory beforehand.

While running, `vopt` prints progress messages to the standard output.

## Arguments

`vopt` has no positional arguments.

## Options

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-i`, `--input` | `FIX MOV FILE` | One pair: label of the fixed image, label of the moving image, and path to the velocity field that warps the moving image to the fixed one. Repeat for each pair. | *required* |
| `-w`, `--weight` | `FIX MOV FILE` | Precision (squared weight) map for one pair, with the same labels as `-i`. Repeat for each pair. | all pairs weighted equally |
| `-o`, `--output` | `FILE` | Output path template; see [Output](#output). | `{dir}{sep}{label}_optimal.lta` |
| `-x`, `--bound` | `BOUND` | Boundary condition: `dct2`, `dct1`, `dst2`, `dst1`, `dft`, `zero` or `nearest`. | `dft` |
| `-a`, `--absolute` | `VAL` | Penalty on absolute displacements (0th order). | `1e-4` |
| `-m`, `--membrane` | `VAL` | Penalty on membrane energy (1st order). | `1e-4` |
| `-b`, `--bending` | `VAL` | Penalty on bending energy (2nd order). | `1e-4` |
| `-l`, `--lame` | `VAL [VAL]` | Penalty on linear-elastic energy: one value, or two values (divergence and shear). | `1e-4` |
| `-h`, `--help` | | Print the built-in help and exit. | |

Each of `-o`, `-x`, `-a`, `-m`, `-b` and `-l` may be given at most once.

## Using the outputs

The output for a label is the velocity field you would obtain by registering a
template (moving) to that subject (fixed), in the same convention as your
pairwise fields. To bring a subject into the common space, apply it with
[`reslice`](../spatial/reslice.md) as a velocity field (`-v`, or `-iv` for its
inverse), exactly as you would for a pairwise result, with a target that
defines the grid of the template. You can write that grid with
[`meanspace`](../spatial/meanspace.md), and integrate a field into a
displacement field with [`vexp`](../spatial/vexp.md).

## Examples

Combine the six pairwise fields of three images, writing one field per image
(`A_to_B.nii.gz` registers `A`, as the moving image, to `B`, as the fixed
image):

```shell
nitorch vopt \
  -i mtw pdw mtw_to_pdw.nii.gz \
  -i mtw t1w mtw_to_t1w.nii.gz \
  -i pdw mtw pdw_to_mtw.nii.gz \
  -i pdw t1w pdw_to_t1w.nii.gz \
  -i t1w mtw t1w_to_mtw.nii.gz \
  -i t1w pdw t1w_to_pdw.nii.gz \
  -o out/{label}_to_mean.nii.gz
```

Write the outputs next to the first input file, keeping its extension:

```shell
nitorch vopt \
  -i mtw pdw mtw_to_pdw.nii.gz \
  -i pdw t1w pdw_to_t1w.nii.gz \
  -i t1w mtw t1w_to_mtw.nii.gz \
  -o {dir}{sep}{label}_to_mean{ext}
```

Weight each pair by a precision map, such as the Hessians written by
`register`, and increase the smoothness of the result:

```shell
nitorch vopt \
  -i mtw pdw mtw_to_pdw.nii.gz  -w mtw pdw mtw_to_pdw_hessian.nii.gz \
  -i pdw t1w pdw_to_t1w.nii.gz  -w pdw t1w pdw_to_t1w_hessian.nii.gz \
  -i t1w mtw t1w_to_mtw.nii.gz  -w t1w mtw t1w_to_mtw_hessian.nii.gz \
  -b 0.5 \
  -o out/{label}_to_mean.nii.gz
```

## References

- Casamitjana, Iglesias, Tudela, Niñerola-Baizán, Sala-Llonch. "JUMP: A joint
  multimodal registration pipeline for neuroimaging with minimal
  preprocessing". *ISBI* (2024).
- Casamitjana, Sala-Llonch, Lekadir, Iglesias. "USLR: an open-source tool for
  unbiased and smooth longitudinal registration of brain MR". *arXiv* (2023).
- Ashburner, Ridgway. "Symmetric diffeomorphic modeling of longitudinal
  structural MRI". *Frontiers in Neuroscience* (2012).

## See also

- [`affopt`](affopt.md): the same problem for affine transformations.
- [`register`](register.md): produces the pairwise velocity fields and
  Hessians.
- [`reslice`](../spatial/reslice.md), [`vexp`](../spatial/vexp.md),
  [`meanspace`](../spatial/meanspace.md): apply and manipulate the results.
