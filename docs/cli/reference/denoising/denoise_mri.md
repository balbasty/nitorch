---
icon: octicons/terminal-24
---

# denoise_mri

Denoises one or several magnetic resonance images with joint total variation,
estimating the noise level and the regularisation of each image from its
intensities. Use it when you have co-registered MR sequences of the same subject
and want a denoised version of each without tuning parameters by hand.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch denoise_mri FILE... [-s SCALE] [-lr RATE] [-n ITER] [-t TOL]
                                [-v VALUE] [-o DIR] [-cpu | -gpu [N]]
    ```

## Description

Each input file is one MR volume, and the files together form one multi-channel
image: `T1w.nii T2w.nii` is denoised as a two-channel image. The channels share
their edges through a *joint total variation* prior, so that a boundary visible
in one sequence helps preserve it in the others, as in the multi-channel
super-resolution method of Brudfors et al. (2018). A single file works too, and
is then an ordinary total-variation denoising.

The denoised channels $y_c$ minimise

$$
\sum_c \frac{\tau_c}{2} \lVert x_c - y_c \rVert^2
\;+\; \sum_{\text{voxels}} \sqrt{\sum_c \lVert \lambda_c \nabla y_c \rVert^2},
$$

where $x_c$ are the noisy inputs. The first term is a Gaussian data term. The
second is the joint total variation: at each voxel, the gradients of all
channels are combined before taking the square root. The gradients are
computed with the voxel size given in the header of the first file.

### Automatic parameters

Both $\tau_c$ and $\lambda_c$ are derived from the image. A two-class mixture
model of Rician distributions (or of Gaussians if the image contains negative
values) is fitted to the intensity histogram of every channel, and gives two
numbers: the standard deviation $\sigma_c$ of the background (noise) class, and
the mean $\mu_c$ of the foreground (tissue) class. Then

- the noise precision is $\tau_c = 1 / \sigma_c^2$;
- the regularisation is $\lambda_c = s \,/\, (\sqrt{C}\,\mu_c)$, where $C$
  is the number of channels and $s$ is the scaling factor `-s`.

Normalising by the foreground intensity makes the amount of smoothing
independent of the intensity scale of the image, and normalising by
$\sqrt{C}$ keeps it comparable when you add channels. The parameter you
normally need to adjust is the scaling $s$: increase it for stronger smoothing.

### Optimisation

The loss is minimised with Adam, using PyTorch's automatic differentiation,
starting from a volume of zeros. Every ten iterations, the learning rate is
reduced if the loss has stopped improving (`ReduceLROnPlateau`). After the first
ten iterations, the optimisation stops once the relative gain of the loss stays
below the tolerance for five consecutive iterations, or after the maximum
number of iterations. With `-v`, a line with the iteration number, loss, gain
and learning rate is printed every 25 iterations.

### Input requirements

All files must be 3D volumes with exactly the same dimensions; each file holds
one channel. The orientation matrix of the first file is assumed to apply to all.

### Output

The denoised channel $c$ (counting from 0) is written as
`den_<name>_<c><ext>`, where `<name>` is the part of the file name of the
**last** input file that precedes its first dot, and `<ext>` is the rest, such
as `.nii.gz`. The header (orientation matrix and data type) is taken from the
last input file. By default the files are written in the directory of the last
input file; `-o` selects another directory, which is created if needed. For
example,

```shell
nitorch denoise_mri T1w.nii.gz T2w.nii.gz
```

writes `den_T2w_0.nii.gz` (the denoised T1w image) and `den_T2w_1.nii.gz` (the
denoised T2w image). The channel index refers to the order of the arguments, and
only the file name of the last input is used; this naming is easy to misread, so
check which file is which.

!!! warning "A GPU is assumed by default"
    The default device is `cuda`. On a machine without a GPU, the command
    fails unless you pass `-cpu`.

## Arguments

| Name | Description |
| ---- | ----------- |
| `FILE` | One or more input MR volumes, all with the same dimensions. Give them before any option. |

## Options

Long option names use underscores, as in the code.

### Model

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-s`, `--lam_scl` | `SCALE` | Scaling $s$ of the regularisation of all channels. Larger means smoother. | `10` |

### Optimisation

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-lr`, `--learning_rate` | `RATE` | Learning rate of Adam. | `10` |
| `-n`, `--max_iter` | `ITER` | Maximum number of iterations. | `10000` |
| `-t`, `--tolerance` | `TOL` | Convergence threshold on the relative gain of the loss. | `1e-8` |

### Input, output and hardware

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-o`, `--dir_out` | `DIR` | Directory in which to write the denoised images. | directory of the input |
| `-v`, `--verbose` | `VALUE` | Print the progress every 25 iterations. The option needs a value, and any non-empty value (including `0`) switches printing on. | on |
| `-cpu`, `--cpu` | | Run on the CPU. | |
| `-gpu`, `--gpu` | `[N]` | Run on a GPU, optionally the one with index `N`. | yes (`cuda`) |
| `-h`, `--help` | | Print the built-in help and exit. | |

See [Hardware](../../conventions.md#hardware) for the shared conventions.

## How it differs from `denoise`

Both commands denoise with a total-variation prior and estimate the noise from
a mixture model of the histogram, but they are built for different uses.

| | [`denoise`](denoise.md) | `denoise_mri` |
| --- | --- | --- |
| Intended data | Any 3D image, or 4D with channels | MR volumes, one channel per file |
| Prior | Total variation, joint across channels or not (`-j`) | Always joint total variation |
| Noise level | Estimated, or given with `-s` | Always estimated |
| Regularisation | Given in absolute terms (`-l`), the same for any intensity scale | Computed from the foreground intensity and the number of channels; scaled by `-s` |
| Voxel size | Ignored | Used in the gradients |
| Solver | Reweighted least squares with conjugate gradients | Adam on the loss, with automatic differentiation |
| Output names | Template (`-o`), `.denoised` by default | `den_` prefix, in a directory (`-o`) |
| Default device | CPU | GPU |

Note that `-s` means different things: the noise standard deviation in
`denoise`, the regularisation scaling here.

## Examples

Denoise a T1-weighted image, on the CPU. This writes `den_T1w_0.nii` in the same
directory:

```shell
nitorch denoise_mri T1w.nii -cpu
```

Denoise two sequences of the same subject jointly and write the results in
`output/`:

```shell
nitorch denoise_mri T1w.nii T2w.nii -o ./output
```

Smooth more strongly:

```shell
nitorch denoise_mri T1w.nii -s 30 -cpu
```

Run on the second GPU, with a limit of 500 iterations:

```shell
nitorch denoise_mri T1w.nii T2w.nii FLAIR.nii -gpu 1 -n 500
```

## References

- Brudfors et al. "MRI super-resolution using multi-channel total variation".
  *Annual Conference on Medical Image Understanding and Analysis*, Springer,
  Cham (2018).

## See also

- [`denoise`](denoise.md): generic total-variation denoising with explicit
  parameters.
- [`inpaint`](../images/inpaint.md), [`smooth`](../images/smooth.md): related
  image operations.
