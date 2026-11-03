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
                                [-v [0|1]] [-o DIR] [-cpu | -gpu [N]]
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
number of iterations. By default, a line with the iteration number, loss, gain
and learning rate is printed every 25 iterations; `-v 0` turns this off.

### Input requirements

All files must be 3D volumes with exactly the same dimensions; each file holds
one channel. The orientation matrix of the first file is assumed to apply to all.

### Output

Each denoised image is written as `den_<name><ext>`, where `<name>` is the part
of the name of its own input file that precedes its first dot, and `<ext>` is
the rest, such as `.nii.gz`. The header (orientation matrix and data type) is
that of the same input file. By default, the files are written in the directory
of their input; `-o` selects another directory, which is created if needed. For
example,

```shell
nitorch denoise_mri T1w.nii.gz T2w.nii.gz
```

writes `den_T1w.nii.gz` and `den_T2w.nii.gz`.

When two or more outputs would end up at the same path, which happens when
inputs from different directories share a name and are written to a common
`-o` directory, each of them gets its 1-based position among the inputs before
the extension. Outputs that do not collide keep their plain name. For example,

```shell
nitorch denoise_mri a/t1.nii.gz b/t1.nii.gz -o out
```

writes `out/den_t1.1.nii.gz` and `out/den_t1.2.nii.gz`.

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
| `-o`, `--dir_out` | `DIR` | Directory in which to write the denoised images. Outputs with the same name get their input position as a suffix. | directory of the input |
| `-v`, `--verbose` | `[0\|1]` | Print the progress every 25 iterations. The value is optional; `0`, `false`, `no` or `off` disables printing. | on |
| `-cpu`, `--cpu` | | Run on the CPU. | yes |
| `-gpu`, `--gpu` | `[N]` | Run on a GPU, optionally the one with index `N`. If CUDA is unavailable, a warning is issued and the CPU is used. | |
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
| Output names | Template (`-o`), `.denoised` by default | `den_` prefix, named after each input, in a directory (`-o`) |
| Default device | CPU | CPU |

Note that `-s` means different things: the noise standard deviation in
`denoise`, the regularisation scaling here.

## Examples

Denoise a T1-weighted image. This writes `den_T1w.nii` in the same directory:

```shell
nitorch denoise_mri T1w.nii
```

Denoise two sequences of the same subject jointly and write the results in
`output/`:

```shell
nitorch denoise_mri T1w.nii T2w.nii -o ./output
```

Smooth more strongly:

```shell
nitorch denoise_mri T1w.nii -s 30
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
