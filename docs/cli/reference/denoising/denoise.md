---
icon: octicons/terminal-24
---

# denoise

Removes noise from one or several 3D images with a total-variation prior, with
the regularisation strength and noise level under your control. Use it as a
general-purpose denoiser; for MR images whose parameters you would rather not
choose, see [`denoise_mri`](denoise_mri.md).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch denoise FILE... [-o FILE...] [-j [BOOL]] [-l LAM...] [-s SIGMA...]
                            [-n ITER...] [-t TOL...] [-x OPTIM] [-v [LEVEL]]
                            [--cpu [THREADS] | --gpu [DEVICE]]
    ```

## Description

`denoise` finds the image $y$ that is close to the noisy input $x$ while having
a small total variation, that is, a small sum of gradient magnitudes. Total
variation penalises noise but, unlike a Gaussian blur, tolerates sharp edges, so
the result is smooth inside regions and keeps the boundaries between them. The
noise is assumed to be Gaussian with a known standard deviation $\sigma$ for
each channel, which weights the data term; the weight of the prior is set by a
regularisation factor $\lambda$. The problem is solved by iteratively
reweighted least squares: each iteration turns the total variation into a
weighted membrane (first-derivative) penalty and solves the resulting linear
system with conjugate gradients (other solvers can be chosen with `--optim`).

### Channels and joint total variation

Each input file is a 3D volume, or a 4D volume whose last dimension holds
channels. All files are stacked into one multi-channel image, so they must
have the same voxel grid. By default (`--joint`), the total variation is
computed jointly across channels: it is the sum over voxels of the norm of the
gradients of all channels together. Channels then share their edges, which is
the natural choice for several contrasts of the same subject, such as a T1w, a
T2w and a FLAIR image that are in register. With `--joint false`, every channel
is denoised independently, although in a single call.

### Noise level

If you do not provide `--sigma`, the noise standard deviation of each channel is
estimated from the image: a mixture of Rician distributions (or of Gaussians if
the image has negative values) is fitted to the intensity histogram, and the
standard deviation of the background class is taken as the noise level. Pass
`--sigma` when you know the noise level, or when the estimate is not reliable,
for example if there is no background in the field of view.

### Regularisation strength

The regularisation factor `--lam` is not normalised by the intensity of the
image: the same value does not give the same amount of smoothing on images
with different intensity scales. If you denoise images of various origins,
adjust it by inspecting the result, or use [`denoise_mri`](denoise_mri.md), which
computes it from the image. With joint total variation, the factor is divided
by the number of channels.

### Output

One output image is written per input file, with the header (orientation
matrix, data type) of that input. Images with several channels keep them, in
the same order. By default, the output is written next to the input, with
`.denoised` added to the base name. While running, `denoise` prints a progress
line with the value of the objective at each iteration.

!!! note "Voxel size"
    The gradients are computed in voxel units: the voxel size in the header is
    ignored, so anisotropic voxels are treated as if they were isotropic.

## Arguments

| Name | Description |
| ---- | ----------- |
| `FILE` | One or more input volumes. Each can be 3D or 4D (channels last); singleton dimensions beyond the fourth are dropped. All must have the same spatial shape. |

## Options

### Input and output

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-o`, `--output` | `FILE...` | Output path templates, one per input file. If fewer templates than inputs are given, the last one is reused. Placeholders: `{dir}`, `{sep}`, `{base}`, `{ext}`, taken from the corresponding input file (see [Output filenames](../../conventions.md#output-filenames)). | `{dir}{sep}{base}.denoised{ext}` |

### Model

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-j`, `--joint` | `[BOOL]` | Use a total variation that is joint across channels. A bare `-j` means yes. Booleans can be `true`, `false`, `yes`, `no`, `1` or `0`. | yes |
| `-l`, `--lam` | `LAM...` | Regularisation factor of each channel. If fewer values than channels are given, the last one is reused. | `10` |
| `-s`, `--sigma` | `SIGMA...` | Noise standard deviation of each channel. If fewer values than channels are given, the last one is reused. | estimated |

### Optimisation

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-n`, `--max-iter` | `ITER...` | Maximum number of reweighting iterations, then of conjugate-gradient iterations per reweighting. A single value is used for both. | `10 32` |
| `-t`, `--tolerance` | `TOL...` | Stopping tolerance of the reweighting iterations, then of the conjugate-gradient iterations. A single value is used for both. | `1e-4` |
| `-x`, `--optim` | `OPTIM` | Solver for the linear system of each reweighting: conjugate gradients (`cg`), relaxation (`relax`), or either of them started from a full multigrid initialisation (`fmg+cg`, `fmg+relax`; `fmg` alone means `fmg+cg`). The first nine reweighting iterations always use conjugate gradients, so `relax` only takes over from the tenth one: with the default `-n 10`, in the last iteration only. | `cg` |

### Interface and hardware

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-v`, `--verbose` | `[LEVEL]` | A bare `-v` means 1. Levels above 2 plot the intermediate images (needs `matplotlib`). The progress line is printed whatever the level. | `0` |
| `--cpu` | `[THREADS]` | Run on the CPU, optionally with this number of threads. | CPU |
| `--gpu` | `[DEVICE]` | Run on a GPU, optionally the one with this index. Falls back to the CPU, with a warning, if CUDA is not available. | |
| `-h`, `--help` | | Print the built-in help and exit. | |

See [Hardware](../../conventions.md#hardware) for the shared conventions.

## Examples

Denoise a T1-weighted image, estimating the noise level automatically. This
writes `T1w.denoised.nii.gz` next to the input:

```shell
nitorch denoise T1w.nii.gz
```

Smooth more strongly by increasing the regularisation factor:

```shell
nitorch denoise T1w.nii.gz -l 30
```

Provide the noise standard deviation instead of estimating it:

```shell
nitorch denoise T1w.nii.gz -s 12.5
```

Denoise three contrasts of the same subject jointly, so that they share their
edges, and write the results to another directory:

```shell
nitorch denoise T1w.nii.gz T2w.nii.gz FLAIR.nii.gz -o 'out/{base}.denoised{ext}'
```

Denoise the same files independently, one regulariser value per image:

```shell
nitorch denoise T1w.nii.gz T2w.nii.gz --joint false -l 10 20
```

Run on the first GPU, with more reweighting iterations:

```shell
nitorch denoise T1w.nii.gz -n 20 --gpu 0
```

## See also

- [`denoise_mri`](denoise_mri.md): total-variation denoising of MR images with
  automatic parameters.
- [`smooth`](../images/smooth.md): linear smoothing.
- [`inpaint`](../images/inpaint.md): fill missing data.
