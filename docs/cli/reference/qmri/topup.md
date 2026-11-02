---
icon: octicons/terminal-24
---

# topup

Estimates the geometric distortion caused by B0 inhomogeneity from a pair of images acquired with opposite readout polarities, and applies it to correct images. It has two subcommands: `topup fit` estimates a one-dimensional distortion field along the readout direction, and `topup apply` unwarps images with a field estimated earlier. Reach for it when you have reversed-polarity (blip-up/blip-down) acquisitions, or the first two echoes of a bipolar multi-echo acquisition, and no measured field map.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch topup fit POS NEG [-m MASK] [-o TEMPLATE] [-r DIR] [-bw BW]
                      [-l {mse,ncc,lncc,gmm,lgmm}] [-k VAL... [UNIT]] [-b BINS]
                      [-j|--modulation [{yes,no}]] [-d [{yes,no}]]
                      [-p VAL... [TYPE]] [-w VOX...] [-n ITER...] [-t TOL...]
                      [--cpu | --gpu [ID]] [-v [LVL]]

    nitorch topup apply FIELD [-p POS...] [-n NEG...] [-o TEMPLATE] [-r DIR]
                        [-bw BW] [-m [{yes,no}]] [-d [{yes,no}]]
                        [--cpu | --gpu [ID]] [-v [LVL]]
    ```

## Description

`topup` is a re-implementation of the TOPUP method of Andersson, Skare and Ashburner (2003), with some differences. Let $I_+$ and $I_-$ be two images of the same object acquired with opposite readout polarities. B0 inhomogeneity displaces voxels along the readout direction in opposite senses in the two images. The command looks for a displacement field $u(\mathbf{x})$ along the readout direction such that the two images agree once $I_+$ has been displaced by $+u$ and $I_-$ by $-u$:

$$
I_+\big(\mathbf{x} + u(\mathbf{x})\,\mathbf{e}\big)\,\big(1 + \partial_e u\big)
\;\approx\;
I_-\big(\mathbf{x} - u(\mathbf{x})\,\mathbf{e}\big)\,\big(1 - \partial_e u\big),
$$

where $\mathbf{e}$ is the unit vector along the readout direction. The Jacobian factors $1 \pm \partial_e u$ model the pile-up or stretching of the intensity caused by the displacement, and can be switched off with `-j no`. The field is smoothed by a penalty on its bending energy (or membrane energy). It is estimated on a pyramid of increasingly fine grids.

The matching term between the two images is the main difference with FSL TOPUP. With the default `mse`, the squared difference is used, as in TOPUP, with the noise level estimated from the images; this requires images with the same contrast. The other losses allow images with *different* contrasts to be matched, which makes it possible to estimate a distortion field from the first two echoes of a bipolar multi-echo acquisition, for instance: `ncc` (global normalised cross-correlation), `lncc` (local normalised cross-correlation in patches, see `-k`), `gmm` (negative mutual information estimated with a Gaussian mixture model with `-b` classes) and `lgmm` (the same, computed locally in patches).

### The output field

`topup fit` writes a single image with the shape and orientation of `POS`, whose values are the displacement $u$ along the readout direction, in voxels of the input image. With `-bw`, which gives the readout bandwidth in Hz/pixel, the displacement is multiplied by the bandwidth so that the field is a B0 field map in Hz (a displacement of $u$ voxels corresponds to a frequency offset of $u \cdot \mathrm{BW}$ Hz). Without `-bw`, or with `-bw 1`, the field stays in voxels. With `-d yes`, the file contains a *velocity* field, the displacement being its exponential, which guarantees an invertible deformation. Such a file must be applied with `-d yes`.

### Applying the field

`topup apply` unwarps images with such a field. Files given with `-p` have the polarity of `POS`, and are displaced by $+u$. Files given with `-n` have the polarity of `NEG` and are displaced by $-u$. The bandwidth `-bw` must be the one used when fitting, so that the field in Hz is converted back to voxels; if it was omitted when fitting, omit it here too. Intensities are modulated by the Jacobian of the deformation, taken along the readout axis, unless you pass `-m no`. The images to unwarp must be 3D volumes on the same grid as the field.

### Readout direction

The readout direction is set with `-r`, using anatomical axis letters such as `ap`, `lr` or `is`. The command picks the first voxel axis of the (first) input file whose orientation, computed from the orientation matrix, matches one of the letters. Without `-r`, the voxel axis with the most voxels is used. The chosen layout is printed when the command runs.

### Multi-resolution schedule

The field is estimated on a pyramid. The grids are defined by `-w`, the voxel size, in millimetres, of each level, from the coarsest to the finest. Images are reduced by an integer factor to approximate each voxel size. The penalty weight (`-p`), the maximum number of iterations (`-n`) and the tolerance (`-t`) can also be given per level. The number of levels is the length of the longest of these four lists, and shorter lists are completed by repeating their last value. With the defaults, there are three levels, at 8, 4 and 1 mm, each with the same penalty, 50 iterations at most, and a tolerance of $10^{-4}$. The field estimated at one level initialises the next.

## Arguments

### `topup fit`

| Name  | Description |
| ----- | ----------- |
| `POS` | 3D image acquired with the positive readout polarity. |
| `NEG` | 3D image acquired with the negative readout polarity, on the same grid as `POS`. |

### `topup apply`

| Name    | Description |
| ------- | ----------- |
| `FIELD` | Distortion field written by `topup fit`. |

## Options: `topup fit`

### Input and output

| Option                    | Value        | Description | Default |
| ------------------------- | ------------ | ----------- | ------- |
| `-m`, `--mask`            | `FILE`       | Mask in which the matching loss is computed. | whole image |
| `-o`, `--output`          | `TEMPLATE`   | Output path. The template can use `{dir}`, `{sep}`, `{base}` and `{ext}` (directory, separator, base name and extension of `POS`). | `{dir}{sep}{base}_topup_b0{ext}` |
| `-r`, `--readout`         | `DIR`        | Readout direction (for example `ap`). | voxel axis with the most voxels |
| `-bw`, `--bandwidth`      | `BW`         | Readout bandwidth in Hz/pixel. The field is multiplied by it. Use `1` for a field in voxels. | `1` |

### Model

| Option                    | Value                 | Description | Default |
| ------------------------- | --------------------- | ----------- | ------- |
| `-l`, `--loss`            | `{mse,ncc,lncc,gmm,lgmm}` | Matching loss. See [Description](#description). | `mse` |
| `-k`, `--kernel`          | `VAL... [UNIT]`       | Patch size of the local losses (`lncc`, `lgmm`): a single value, or one per voxel axis. The unit is `%` (percentage of the image size along each axis), `mm` (along the R, A and S axes) or `vox` (voxels of the input image, rescaled at each level of the pyramid). Patches are at least 2 voxels wide. | `10 %` |
| `-b`, `--bins`            | `N`                   | Number of classes of the mixture for `gmm` and `lgmm`. With `1`, they reduce to `ncc` and `lncc`. | `3` |
| `-j`, `--modulation`      | `[{yes,no}]`          | Modulate intensities by the Jacobian of the deformation. Alone, the flag means `yes`. | `yes` |
| `-d`, `--diffeo`          | `[{yes,no}]`          | Use a diffeomorphic (velocity field) model instead of a small-displacement model. Alone, the flag means `yes`. | `no` |
| `-p`, `--penalty`         | `VAL... [TYPE]`       | Penalty weight, one per level, followed by the type of penalty: `bend` (bending energy) or `memb` (membrane energy). | `100 bend` |

The short option `-m` means `--mask`, and `-j` means `--modulation`.

Because the value of `-j` and `-d` is optional, a flag written directly before the input files would take the first file as its value. Place such flags after the input files, or give them an explicit value.

### Optimisation

| Option                    | Value          | Description | Default |
| ------------------------- | -------------- | ----------- | ------- |
| `-w`, `--downsample`      | `VOX...`       | Voxel size, in millimetres, of each level of the pyramid, coarsest first. | `8 4 1` |
| `-n`, `--max-iter`        | `N...`         | Maximum number of iterations, per level. | `50` |
| `-t`, `--tolerance`       | `TOL...`       | Tolerance for early stopping, per level. | `1e-4` |

### General

| Option                    | Value     | Description | Default |
| ------------------------- | --------- | ----------- | ------- |
| `--cpu`                   |           | Run on the CPU. | |
| `--gpu`                   | `[ID]`    | Run on a CUDA device. `ID` is the device index (0 if omitted). Falls back to the CPU when CUDA is not available. | `--cpu` |
| `-v`, `--verbose`         | `[LVL]`   | Verbosity. Alone, the flag means `1`. | `1` |
| `-h`, `--help`            |           | Print the built-in help. | |

## Options: `topup apply`

| Option                    | Value        | Description | Default |
| ------------------------- | ------------ | ----------- | ------- |
| `-p`, `--pos`             | `FILE...`    | Files acquired with the positive polarity (displaced by $+u$). | none |
| `-n`, `--neg`             | `FILE...`    | Files acquired with the negative polarity (displaced by $-u$). | none |
| `-o`, `--output`          | `TEMPLATE`   | Output path for every file. The template uses the directory, separator, base name and extension of the file being unwarped: `{dir}`, `{sep}`, `{base}`, `{ext}`. | `{dir}{sep}{base}_topup_unwarped{ext}` |
| `-r`, `--readout`         | `DIR`        | Readout direction. | voxel axis with the most voxels |
| `-bw`, `--bandwidth`      | `BW`         | Readout bandwidth in Hz/pixel, as used by `topup fit`. | `1` |
| `-m`, `--modulation`      | `[{yes,no}]` | Modulate intensities by the Jacobian of the deformation. Alone, the flag means `yes`. | `yes` |
| `-d`, `--diffeo`          | `[{yes,no}]` | The field is a velocity field (it was fitted with `-d yes`). Alone, the flag means `yes`. | `no` |
| `--cpu`                   |              | Run on the CPU. | |
| `--gpu`                   | `[ID]`       | Run on a CUDA device. | `--cpu` |
| `-v`, `--verbose`         | `[LVL]`      | Verbosity. | `1` |
| `-h`, `--help`            |              | Print the built-in help. | |

Because every file is written with its own `{dir}` and `{base}`, the default template never overwrites an input.

## Examples

Estimate the field from a blip-up and a blip-down b=0 image, working in voxel units:

```shell
nitorch topup fit b0_up.nii.gz b0_down.nii.gz -bw 1
```

This writes `b0_up_topup_b0.nii.gz` next to `b0_up.nii.gz`. Estimate the same field in Hz, for a readout bandwidth of 25 Hz/pixel, and name the result explicitly:

```shell
nitorch topup fit b0_up.nii.gz b0_down.nii.gz -bw 25 -r ap -o fieldmap_hz.nii.gz
```

Unwarp the diffusion images acquired with both polarities using that field. The same bandwidth must be given:

```shell
nitorch topup apply fieldmap_hz.nii.gz -bw 25 -r ap \
    -p dwi_up.nii.gz -n dwi_down.nii.gz
```

Estimate a distortion field from the first two echoes of a bipolar multi-echo acquisition, which have different contrasts, with a local correlation loss and patches of 15% of the image size:

```shell
nitorch topup fit echo-1.nii.gz echo-2.nii.gz -bw 1 -l lncc -k 15 %
```

Fit a diffeomorphic field with a membrane penalty, restricted to a brain mask. The resulting velocity field is applied with `-d yes`:

```shell
nitorch topup fit b0_up.nii.gz b0_down.nii.gz -bw 1 -m brainmask.nii.gz \
    -d yes -p 50 memb -o velocity.nii.gz
nitorch topup apply velocity.nii.gz -bw 1 -d yes -p dwi_up.nii.gz -n dwi_down.nii.gz
```

Choose a custom pyramid with four levels, a stronger penalty at the coarse levels and fewer iterations at the finest one:

```shell
nitorch topup fit b0_up.nii.gz b0_down.nii.gz -bw 1 \
    -w 8 4 2 1 -p 400 200 100 50 bend -n 50 50 50 20
```

## References

- Andersson JLR, Skare S, Ashburner J. How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging. *NeuroImage* 20(2):870&ndash;888 (2003).

## See also

- [`epic`](epic.md): denoise and correct bipolar multi-echo data with a field map.
- [`estatics`](estatics.md): estimate R2\* and, with `--meetup`, distortion fields, without reversed-polarity data.
- [Spatial transformations](../../conventions.md#spatial-transformations): how NITorch describes deformations in general.
