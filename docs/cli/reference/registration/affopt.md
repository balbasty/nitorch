---
icon: octicons/terminal-24
---

# affopt

Turns a set of pairwise affine registrations between subjects into one
consistent "template-to-subject" matrix per subject. Use it when you have
registered several images to one another and want each of them expressed
relative to a common, unbiased mean space.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch affopt -i FIX MOV FILE [-i FIX MOV FILE]...
                   [-o FILE] [-l] [-a | -s | -r]
    ```

## Description

Suppose you have $N$ images of the same brain (several MRI contrasts, or
several time points) and you register some or all pairs of them with a linear
transformation. The pairwise results are redundant and, because every
registration is noisy, mutually inconsistent: composing the transformation
from `A` to `B` with the one from `B` to `C` rarely gives the transformation
from `A` to `C`. `affopt` reconciles them. It looks for $N$ matrices $Y_i$,
one per image, such that for every pair you provide

$$
X_{ij} \approx Y_i \, Y_j^{-1},
$$

where $X_{ij}$ is the matrix registering image $j$ (the moving image) to
image $i$ (the fixed image). The matrices $Y_i$ are defined up to a common
transformation; `affopt` removes this ambiguity by centring them, so that the
average of all $Y_i$ (in the matrix Lie algebra) is the identity. The
reference space they all refer to is therefore the "mean space" of the
subjects, and no subject is privileged.

Read the other way round, $Y_i$ is the transformation you would estimate by
registering a virtual template, as the moving image, to subject $i$, taken as
the fixed image. This is why the outputs are called template-to-subject
transformations.

The approach follows Leung et al. (2011) and Ashburner and Ridgway (2012), with
three differences:

- Some pairs may be missing. When only one direction of a pair is given, the
  other one is assumed to be its inverse. Leaving pairs out can bias the
  estimated mean space, but the bias averages out when there are many subjects
  and the evaluated pairs are sampled at random.
- Pairwise transformations are not symmetrised first. The mean space is fitted
  to all forward and backward transformations directly.
- By default, the fit minimises the squared Frobenius norm of the residuals
  $Y_i Y_j^{-1} - X_{ij}$, that is, an L2 norm in the space of matrices. This
  is more accurate than the L2 norm in the Lie algebra (which the method of
  Leung et al. implicitly uses) when the pairwise transformations are large,
  because composing affine matrices is then poorly approximated by adding
  their matrix logarithms. The Lie-algebra solution is still available with
  `-l`, and it is also used to initialise the Frobenius fit.

### Inputs

Each `-i` option describes one pair: the *label* of the fixed image, the label
of the moving image, and the path to a file holding the matrix that registers
the moving image to the fixed one. Labels are arbitrary strings (`mtw`, `01`,
`sub-02_T1w`); they only serve to tell which pairs involve the same image, and
they are used in the names of the output files. If you give the same pair twice,
the last file wins. The matrices are read as world-to-world (RAS) transformations from
a format NITorch can read, typically `.lta` files such as those written by
[`register`](register.md) or [`affreg`](affreg.md). See
[Spatial transformations](../../conventions.md#spatial-transformations).

The pairs should link all images together: every image should appear in at
least one pair, and the graph formed by the pairs should be connected.

### Choosing the transformation group

The fit is constrained to a group of transformations. The input matrices are
projected onto this group before they are combined, and the outputs belong to
it.

| Option | Group | Degrees of freedom (3D) |
| ------ | ----- | ----------------------- |
| `-r` | rigid: translations and rotations | 6 |
| `-s` | similitude: rigid plus one isotropic scaling | 7 |
| `-a` | affine: translations, rotations, zooms and shears | 12 |

The default is affine. Give at most one of the three.

### Output

For every label, `affopt` writes one RAS-to-RAS `.lta` file that contains
$Y_i$. The output path is a template in which `{label}` is replaced by the
label (it is the only placeholder available). The default is
`{label}_optimal.lta` in the current directory. Create the output directory
beforehand if your template points to one.

While running, the Frobenius fit prints one line per iteration to the standard
output (previous loss, current loss, difference and step size); it stops after
at most 1000 iterations, or earlier once the loss no longer changes.

## Arguments

`affopt` has no positional arguments.

A bare `--` completes the option just before it, so that the last value can start with a dash: `nitorch affopt -i fix mov -- -odd.nii`. Only the last option can take dash-prefixed values this way; write the others as `./-name`.

## Options

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-i`, `--input` | `FIX MOV FILE` | One pair: label of the fixed image, label of the moving image, and path to the transformation that registers the moving image to the fixed one. Repeat for each pair. | *required* |
| `-o`, `--output` | `FILE` | Output path template. `{label}` expands to the image label. | `{label}_optimal.lta` |
| `-l`, `--lie`, `--log` | | Minimise the L2 norm in the Lie algebra instead of the Frobenius norm in matrix space. | off |
| `-a`, `--aff`, `--affine` | | Constrain all transformations to be affine. | on, unless `-r` or `-s` |
| `-s`, `--sim`, `--similitude` | | Constrain all transformations to be similitudes. | |
| `-r`, `--rig`, `--rigid` | | Constrain all transformations to be rigid. | |
| `-h`, `--help` | | Print the built-in help and exit. | |

Each of `-o`, `-l`, `-a`, `-s` and `-r` may be given at most once.

## Using the outputs

The output for a label is the matrix you would obtain by registering a
template (moving) to that subject (fixed). To bring subject `i` into the
common space, resample it with [`reslice`](../spatial/reslice.md), exactly as
you would resample the fixed image into the space of the moving image after a
pairwise registration, and choose a target that defines the grid of the
template. [`meanspace`](../spatial/meanspace.md) can generate such a target
from your images:

```shell
nitorch meanspace pdw.nii.gz mtw.nii.gz t1w.nii.gz -o template.nii.gz
nitorch reslice mtw.nii.gz -l mtw_optimal.lta -t template.nii.gz
```

You can also compose the matrices of two subjects to get a consistent
transformation between them, with [`compose`](../spatial/compose.md).

## Examples

Estimate a consistent transformation for three images from all six pairwise
registrations (the files `A_to_B.lta` register `A`, as the moving image, to
`B`, as the fixed image):

```shell
nitorch affopt \
  -i mtw pdw mtw_to_pdw.lta \
  -i mtw t1w mtw_to_t1w.lta \
  -i pdw mtw pdw_to_mtw.lta \
  -i pdw t1w pdw_to_t1w.lta \
  -i t1w mtw t1w_to_mtw.lta \
  -i t1w pdw t1w_to_pdw.lta \
  -o out/{label}_to_mean.lta
```

This writes `out/mtw_to_mean.lta`, `out/pdw_to_mean.lta` and
`out/t1w_to_mean.lta`.

Use only one direction of each pair. The missing directions are taken to be
the inverses:

```shell
nitorch affopt \
  -i mtw pdw mtw_to_pdw.lta \
  -i mtw t1w mtw_to_t1w.lta \
  -i pdw t1w pdw_to_t1w.lta
```

Assume that the images differ by rigid-body motion only, as for repeated scans
of the same session:

```shell
nitorch affopt --rigid \
  -i scan1 scan2 scan1_to_scan2.lta \
  -i scan2 scan3 scan2_to_scan3.lta \
  -i scan3 scan1 scan3_to_scan1.lta
```

Keep the Lie-algebra solution and skip the Frobenius refinement:

```shell
nitorch affopt --log \
  -i mtw pdw mtw_to_pdw.lta \
  -i pdw t1w pdw_to_t1w.lta \
  -i t1w mtw t1w_to_mtw.lta
```

## References

- Leung, Ridgway, Ourselin, Fox. "Consistent multi-time-point brain atrophy
  estimation from the boundary shift integral". *NeuroImage* (2011).
- Ashburner, Ridgway. "Symmetric diffeomorphic modeling of longitudinal
  structural MRI". *Frontiers in Neuroscience* (2012).

## See also

- [`vopt`](vopt.md): the same problem for velocity fields.
- [`register`](register.md), [`affreg`](affreg.md): produce the pairwise
  transformations.
- [`reslice`](../spatial/reslice.md), [`compose`](../spatial/compose.md),
  [`meanspace`](../spatial/meanspace.md): apply and combine transformations.
