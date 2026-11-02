---
icon: octicons/terminal-24
---

# uniseg

Segments brain MR images into tissue classes while correcting their intensity
inhomogeneity and aligning them to a template. It is a reimplementation of the
Unified Segmentation of SPM12; reach for it when you want grey matter, white
matter and CSF probability maps from one or several MR contrasts, without
SPM or MATLAB.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch uniseg FILE... [-t FILE... | --no-tpm] [-m FILE] [-o DIR]
                           [OUTPUT OPTIONS] [-k INT...] [-b] [-w] [--mix]
                           [--mrf [MODE]] [--wish] [-c] [-f]
                           [--lam-prior X] [--lam-bias X] [--lam-warp X]
                           [--lam-mix X] [--lam-mrf X] [--lam-wish X]
                           [-s MM...] [-n ITER] [--tolerance TOL]
                           [-v [LEVEL] | -q] [-p [LEVEL]]
                           [--cpu [THREADS] | --gpu [DEVICE]]
    ```

## Description

Unified Segmentation (Ashburner and Friston, 2005) expresses segmentation,
bias correction and spatial normalisation as a single generative model that is
fitted by expectation-maximisation. The intensities of each tissue class are
modelled by a mixture of Gaussians, so a class such as grey matter can absorb
partial-volume voxels with several clusters. The prior probability that a voxel
belongs to a class comes from a **tissue probability map** (TPM), a template
that holds one probability map per class and is itself deformed to fit the
subject. Dividing out a smooth, multiplicative **bias field** accounts for the
intensity non-uniformity of the scanner. The three parts help each other: a
good alignment gives a better segmentation, which in turn gives a better
alignment.

`uniseg` estimates, for the input image(s):

- the posterior probability of each tissue class at each voxel, and the
  corresponding hard labels;
- the mean and covariance of the Gaussian clusters, and the mixing
  proportions;
- a bias field per input channel;
- an affine transformation and a nonlinear warp that align the template to the
  image.

The classes of the default template are those of the SPM12 TPM: grey matter,
white matter, CSF, bone, soft tissue and background, in this order. The sixth
class, the background, is not stored in the template: it is implicitly what is
left for the probabilities to sum to one. By default, the classes are modelled
with 4, 2, 2, 2, 2 and 3 Gaussians.

### Inputs

Pass one or several MR images of the same subject. Several files are treated as
the channels of a single multi-channel image (for example T1w and T2w), and
must have the same voxel grid. The orientation matrix of the first one is used
for all.

- **Template.** `-t` gives the template, either one 4D file with a class per
  volume, or one 3D probability map per class. Without `-t`, the SPM12 TPM is
  used. It is downloaded once from the SPM repository, which needs network
  access the first time, and kept in NITorch's cache directory. A template that
  stores one class fewer than the number of classes has its last class
  implicit. The affine matrix and the grid of the first template file define the
  template space used for the `mni` and `wrp` outputs.
- **No template.** With `--no-tpm`, no prior is used: the model reduces to a
  Gaussian mixture with a bias field. Then there is no alignment or warping,
  you must give the number of classes with `-k`, and only native-space outputs
  exist. Also pass `--no-wish` and `--no-clean`, since the Wishart
  regularisation of the command assumes the SPM template and the clean-up
  assumes a brain.
- **Mask.** `-m` gives a mask or a voxel-wise weight map on the grid of the
  inputs, which sets how much each voxel contributes to the fit.

### Number of classes and clusters

`-k` lists the number of Gaussians for each class; its length is the number of
classes. The default is `4 2 2 2 2 3` (six classes) when the template has five
maps, as the SPM12 TPM does, and one Gaussian per map otherwise. A single value
`K` means `K` classes with one Gaussian each.

### Components

Each part of the model can be switched off.

| Component | On/off | What it does |
| --------- | ------ | ------------ |
| Bias field | `-b`, `--no-bias` | Estimates a smooth multiplicative bias field per channel. |
| Warp | `-w`, `--no-warp` | Estimates a nonlinear deformation of the template. |
| Mixing proportions | `--mix`, `--no-mix` | Updates the global proportion of each class, starting from those of the template. |
| Markov random field | `--mrf [MODE]`, `--no-mrf` | Adds a Markov random field that encourages neighbouring voxels to take the same class. |
| Wishart prior | `--wish`, `--no-wish` | Regularises the covariances of the Gaussians with statistics derived from the image. |
| Clean-up | `-c`, `--no-clean` | Post-processes the posteriors with morphological operations from SPM12, which remove tissue far from the brain. |

The Markov random field has several modes: `once` (default) applies it only at
the end; `always` applies it at each iteration; `learn` applies it at each
iteration and also estimates its weights; `no` is accepted but does not switch
it off, use `--no-mrf` for that. A bare `--mrf` means `learn`, and `yes` is an
alias for `learn`.

The clean-up is tuned for typical head MRI at about 1 mm resolution and may
behave poorly on other data. It also assumes the class layout of the SPM12
template (grey matter, white matter and CSF as the first three classes), so
switch it off with `--no-clean` when you use a different template.

### Alignment of the template

When there is a template, it is first aligned to the image with an affine
transformation and then deformed by the warp, which is refined together with
the other parameters. The affine alignment is always performed; see
[Known issues](#known-issues). With `--flexi`, `uniseg` also tries to find
the right orientation of the image by comparing candidate orientations during
the alignment, which helps when the orientation matrix of the file is wrong.

### Output spaces and files

Every output belongs to one of three spaces, written `SPACE` below:

| Space | Name | Description |
| ----- | ---- | ----------- |
| `nat` | Native | The voxel grid of the first input image. |
| `mni` | Template, affine | The grid of the template, after the estimated affine alignment only. |
| `wrp` | Template, nonlinear | The grid of the template, after the affine alignment and the nonlinear warp. |

Only the native probabilities and labels are written by default. The following
files can be requested. Each option takes an optional file name (a template, see
below) and switches the output on; without a name, the default one is used.

| Output | Options | Content | Default name |
| ------ | ------- | ------- | ------------ |
| Probabilities | `--prob-SPACE [FILE]`, `--post-SPACE [FILE]`, `--no-prob-SPACE` | Posterior probability of each class, as one volume per class (float32). | `{dir}{sep}{base}.prob.SPACE{ext}` |
| Labels | `--labels-SPACE [FILE]`, `--no-labels-SPACE` | Index of the most probable class (starting from 0), as 16-bit integers. | `{dir}{sep}{base}.labels.SPACE{ext}` |
| Priors | `--prior-SPACE [FILE]` | Prior probability of each class at each voxel, from the aligned and warped template. | `{dir}{sep}{base}.prior.SPACE{ext}` |
| Corrected image | `--nobias-SPACE [FILE]` | The input multiplied by the bias field, that is, the bias-corrected image. | `{dir}{sep}{base}.nobias.SPACE{ext}` |
| Bias field | `--bias-SPACE [FILE]` | The multiplicative field that corrects the intensities. | `{dir}{sep}{base}.bias.SPACE{ext}` |
| Warp | `--warp-SPACE [FILE]` | The displacement field of the nonlinear warp. | `{dir}{sep}{base}.warp.SPACE{ext}` |
| Everything | `--all-SPACE` | All the outputs above, in this space. | |

The default native outputs, `--prob-nat` and `--labels-nat`, can be turned off
with `--no-prob-nat` and `--no-labels-nat`. Bias-related outputs exist only if
bias correction is on, and warps only if warping is on. The affine
transformation that aligns the template is not written to disk.

In a file name, `{dir}` is the output directory (`-o`, or the directory of the
first input if `-o` is not given), `{sep}` is the directory separator, and
`{base}` and `{ext}` are the base name and the extension of the first input file
(`.nii.gz` counts as one extension). See also
[Output filenames](../../conventions.md#output-filenames). `--mni-vx` changes
the voxel size of the `mni` and `wrp` grids from that of the template to an
isotropic one.

## Arguments

| Name | Description |
| ---- | ----------- |
| `FILE` | One or more input images, which are the channels of one multi-channel image. They must have the same voxel grid. |

## Options

### Inputs

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-t`, `--tpm` | `FILE...` | Tissue probability maps, as one 4D file or one file per class. | SPM12 TPM |
| `--no-tpm` | | Do not use spatial priors: fit a simple Gaussian mixture. | |
| `-m`, `--mask` | `FILE` | Mask or voxel-wise weight map. | none |

### Hardware

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `--cpu` | `[THREADS]` | Run on the CPU, optionally with this number of threads. | CPU, all threads |
| `--gpu` | `[DEVICE]` | Run on a GPU, optionally the one with this index. | |

See [Hardware](../../conventions.md#hardware) for the shared conventions.

### Outputs

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-o`, `--output` | `DIR` | Directory in which to write the outputs. | directory of the first input |
| `--labels-SPACE` | `[FILE]` | Write the hard labels. | native only |
| `--prob-SPACE`, `--post-SPACE` | `[FILE]` | Write the posterior probabilities. | native only |
| `--prior-SPACE` | `[FILE]` | Write the prior probabilities. | |
| `--nobias-SPACE` | `[FILE]` | Write the bias-corrected images. | |
| `--bias-SPACE` | `[FILE]` | Write the bias fields. | |
| `--warp-SPACE` | `[FILE]` | Write the warp field. | |
| `--all-SPACE` | | Write all outputs in this space. | |
| `--no-labels-SPACE`, `--no-prob-SPACE` | | Do not write the labels or the probabilities in this space. | |
| `--mni-vx` | `FLOAT` | Voxel size of the `mni` and `wrp` grids, in millimetres (isotropic). | that of the template |

### Components

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-k`, `--clusters` | `INT...` | Number of Gaussians per class. | `4 2 2 2 2 3` with the SPM12 TPM |
| `-b`, `--bias` | `[BOOL]` | Estimate a bias field. `--no-bias` switches it off. | yes |
| `-a`, `--align` | `[MODE]` | Align the template (`always`, `once` or `no`). `--no-align` switches it off. **No effect** in the current version. | |
| `-w`, `--warp` | `[BOOL]` | Estimate a nonlinear warp. `--no-warp` switches it off. | yes |
| `--mix` | `[BOOL]` | Update the mixing proportions. `--no-mix` switches it off. | yes |
| `--mrf` | `[MODE]` | Markov random field: `learn`, `always`, `once` or `no`. `--no-mrf` switches it off. | `once` |
| `--wish` | `[BOOL]` | Wishart regularisation of the covariances. `--no-wish` switches it off. | yes |
| `-c`, `--clean` | `[BOOL]` | Clean up the posteriors. `--no-clean` switches it off. | yes |
| `-f`, `--flexi` | `[BOOL]` | Try to find the correct orientation of the input. | no |

A `BOOL` can be `true`, `yes`, `false`, `no`, `1` or `0`. A bare flag, such as `-b`, means yes.

### Regularisation

Larger values give stiffer or more strongly constrained estimates.

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `--lam-prior` | `FLOAT` | Strength of the spatial prior. 0 switches the prior off. | `1` |
| `--lam-bias` | `FLOAT` | Regularisation of the bias field. | `0.1` |
| `--lam-warp` | `FLOAT` | Regularisation of the warp. | `0.1` |
| `--lam-mix` | `FLOAT` | Regularisation of the mixing proportions. Larger keeps them closer to the template. | `100` |
| `--lam-mrf` | `FLOAT` | Regularisation of the MRF weights. Larger gives a less smooth segmentation. | `10` |
| `--lam-wish` | `FLOAT` | Regularisation of the covariances. Larger keeps the variances closer to the fixed ones. | `1` |

### Optimisation

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-s`, `--spacing` | `MM...` | Distance, in millimetres, between the points at which the parameters are estimated. With several values, the fit goes from the largest to the smallest. Smaller is more accurate and slower. | `3` |
| `-n`, `--max-iter` | `ITER` | Maximum number of EM iterations. | `30` |
| `--tolerance` | `TOL` | Tolerance for early stopping. | `1e-3` |

### Verbosity

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-v`, `--verbose` | `[LEVEL]` | 0 prints nothing, 1 prints a summary, 2 prints the convergence of the outer iterations, 3 also that of the Gaussian mixture. A bare `-v` means 2. Levels above 3 also print the parsed options. | `1` |
| `-q`, `--quiet` | | Same as `-v 0`. | |
| `-p`, `--plot` | `[LEVEL]` | Plotting: 0 none, 1 final lower bound and images, 2 live lower bound, 3 live images, 4 live images including the mixture. A bare `-p` means 1. Needs `matplotlib`. | `0` |
| `-h`, `--help` | | Print the built-in help and exit. | |

## Known issues

The following behaviours of the current version differ from what the option
names suggest.

- `-a`/`--align` and `--no-align` are parsed but ignored: when a template is
  used, the affine alignment is always performed.
- `--all-mni` writes the priors, corrected images, bias fields and warp of the
  `mni` space, but not its probabilities and labels: request those explicitly
  with `--prob-mni` and `--labels-mni`.
- The `wrp` outputs are unreliable. `--prob-wrp` and `--labels-wrp` on their
  own write nothing, and `--prior-wrp` and `--all-wrp` stop with an error after
  writing their first files. Prefer the `nat` and `mni` outputs.
- `--prior-mni` takes its file name from `--prob-mni` when that option is
  present (it would overwrite the probabilities), and `--nobias-mni` takes its
  file name from `--bias-mni`. Do not combine these options with custom names.
- With several input images, the per-channel files of `--bias-SPACE` are all
  written under the name derived from the first input, each overwriting the
  previous one, and `--nobias-SPACE` does not work. Use a single input if you
  need these outputs.

## Examples

Segment a T1-weighted image with the SPM12 template. This writes
`T1w.prob.nat.nii.gz` (six probability volumes) and `T1w.labels.nat.nii.gz` next
to the input:

```shell
nitorch uniseg T1w.nii.gz
```

Segment two contrasts of the same subject jointly and collect the results in
a directory:

```shell
nitorch uniseg T1w.nii.gz T2w.nii.gz -o seg
```

Also write the bias field, the bias-corrected image and the warp in native
space:

```shell
nitorch uniseg T1w.nii.gz --bias-nat --nobias-nat --warp-nat
```

Write the probabilities and labels in the space of the template, resampled to a
1 mm grid:

```shell
nitorch uniseg T1w.nii.gz --prob-mni --labels-mni --mni-vx 1
```

Use a custom template made of one probability map per class, give the
number of Gaussians per class, and skip the clean-up, which assumes the class
layout of the SPM12 template:

```shell
nitorch uniseg T1w.nii.gz -t tpm_gm.nii.gz tpm_wm.nii.gz tpm_csf.nii.gz -k 2 2 2 --no-clean
```

Make the fit more accurate, at the cost of speed, by sampling every 2 mm and
allowing more iterations:

```shell
nitorch uniseg T1w.nii.gz -s 2 -n 50
```

Fit a plain three-class Gaussian mixture without any template:

```shell
nitorch uniseg T2w.nii.gz --no-tpm -k 3 --no-wish --no-clean
```

## References

- Ashburner, J. and Friston, K. J. (2005). "Unified segmentation".
  *NeuroImage* 26(3):839–851.
  [doi:10.1016/j.neuroimage.2005.02.018](https://doi.org/10.1016/j.neuroimage.2005.02.018)

## See also

- [`denoise_mri`](denoise_mri.md): denoise MR images before segmenting them.
- [`register`](../registration/register.md): register images with more control
  over the transformations.
