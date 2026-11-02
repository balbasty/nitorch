---
icon: octicons/terminal-24
---

# epic

Corrects geometric distortions and denoises multi-echo gradient-echo data acquired with a bipolar readout, using a pre-computed field map, or only denoises them when none is given. It uses the exponential decay of the signal across echoes to make the correction robust to an imperfect field map. Use it when your echoes alternate readout polarity and you have a B0 field map or voxel shift map, for example from [`topup`](topup.md).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch epic ECHOES... [-f FIELDMAP] [-r ECHOES...] [-o FILE...]
                 [-d DIRECTION] [-p {+,-}] [-b BW] [-x] [-e [BOOL]]
                 [-s [CHUNKSIZE]] [--penalty VAL] [-m IRLS [CG]]
                 [-t IRLS [CG]] [-v [LVL] | -q] [--cpu | --gpu [ID]]
    ```

## Description

In a bipolar multi-echo acquisition, consecutive echoes are read out in opposite directions. B0 inhomogeneity therefore shifts voxels along the readout direction in opposite senses in odd and even echoes, by a distance (in voxels) of

$$
\Delta x(\mathbf{x}) = \frac{\Delta B_0(\mathbf{x})}{\mathrm{BW}},
$$

where $\Delta B_0$ is the field map in Hz and $\mathrm{BW}$ is the readout bandwidth per pixel in Hz/pixel. `epic` implements the method of Varadarajan et al. (2020). It recovers one undistorted, denoised image per echo by minimising a data-fidelity term and an edge-preserving smoothness term. Each acquired echo is explained as a shifted version of the unknown image: shifted by $+\Delta x$ for the echoes with the polarity of the first echo, and by $-\Delta x$ for the others. A total-variation penalty is applied to the unknown echoes, and solved by iteratively reweighted least squares.

What makes the method robust to errors in the field map is that it also uses *reversed-polarity* echoes, that is, images in which every echo is distorted in the opposite sense. If you did not acquire them, they are synthesised from the acquired echoes. The reversed-polarity version of an echo has the polarity of its two neighbours, and if the signal decays exponentially and the echo spacing is constant, it can be approximated by the geometric mean $\sqrt{S_{e-1}S_{e+1}}$ of those neighbours. This is where the exponential-decay assumption enters. It requires at least four echoes. The first and last echoes have only one neighbour, so their reversed counterpart has to be extrapolated from the two nearest echoes of opposite polarity, as $S_a^2/S_b$ where $a$ is the nearest. This extrapolation is noisier, so it is only used with `-e`. By default, the first and last echoes are matched to the acquired data only.

### Inputs

`ECHOES` are the acquired echoes, in acquisition order. Each file is one 3D echo, or a 4D file with the echoes stacked along the fourth dimension. Files are concatenated in the order given. Every echo in the series is assumed to alternate in polarity: the first, third, ... echoes have the polarity given by `-p`, and the second, fourth, ... echoes have the opposite one.

The field map `-f` is a 3D volume, either in Hz or already a voxel shift map. The command divides it by `-b`, the bandwidth per pixel, so that

- for a field map in Hz, give the readout bandwidth in Hz/pixel with `-b`;
- for a voxel shift map, leave `-b` at its default of 1.

`-p -` declares that the first echo has the negative polarity, which flips the sign of the field map; use it if your field map was computed for the opposite readout polarity. The field map is optional: without `-f`, no geometric model is applied and the command only denoises the echoes.

Reversed-polarity echoes acquired for real are passed with `-r`, in the same order and with the same number of echoes as the main series. They replace the synthesised echoes, and the first and last echoes are then used as any other.

### Readout direction

The readout direction is set with `-d` using anatomical axis letters, for example `ap`, `lr` or `is` (one letter, such as `a`, also works). Either end of an axis designates the axis, so `l` and `r` both mean left–right. The command picks the first voxel axis of the first echo file whose orientation, as computed from the orientation matrix, matches one of the letters. A letter that matches no axis is an error. Without `-d`, the axis with the most voxels is used. The readout can lie along any of the three voxel axes.

### Output files

Outputs are written as floating-point images that copy the format, orientation matrix and header of the input file they derive from (the first input file when one output holds several files). The `-o` option takes one or several path templates. In a template, `{dir}`, `{sep}`, `{base}` and `{ext}` are replaced by the directory, path separator, base name (without extension) and extension of the input file, and `{n}` by the zero-based index of the echo.

The first matching row below applies.

| Condition | Behaviour |
| --------- | --------- |
| One `-o` value containing `{base}`, and several input files | The template is repeated for each input file, so each input file gives one output file with as many echoes as the input. |
| One `-o` value containing `{n}` | One 3D file per echo, named after the first input file. |
| One `-o` value, otherwise | One 4D file with all echoes along the fourth dimension, named after the first input file. |
| As many `-o` values as input files | Input file *i* is written to the *i*-th template, with as many echoes as the input. |
| As many `-o` values as echoes | One 3D file per echo, named after the first input file. |
| Any other number | An error. |

The default template is `{dir}{sep}{base}_epic{ext}`, so with a single 4D input `echoes.nii.gz`, the result is `echoes_epic.nii.gz`. When the input path has no directory part, `{dir}` is the current directory.

## Arguments

| Name      | Description |
| --------- | ----------- |
| `ECHOES`  | One or more files with the acquired bipolar echoes (3D, or 4D with echoes along the fourth dimension), in acquisition order. At least four echoes are needed to synthesise reversed-polarity echoes. |

## Options

### Inputs

| Option                | Value            | Description | Default |
| --------------------- | ---------------- | ----------- | ------- |
| `-f`, `--fieldmap`    | `FILE`           | B0 field map, in Hz, or voxel shift map along the readout direction. | none (denoising only) |
| `-r`, `--reversed`    | `ECHOES...`      | Echoes acquired with reversed-polarity readout. | synthesised |
| `-d`, `--direction`   | `DIR`            | Readout direction: one of `r`, `l`, `a`, `p`, `i`, `s`, `rl`, `lr`, `ap`, `pa`, `is`, `si`. | voxel axis with the most voxels |
| `-p`, `--polarity`    | `{+,-}`          | Readout polarity of the first echo. `-` flips the sign of the field map. | `+` |
| `-b`, `--bandwidth`   | `BW`             | Readout bandwidth in Hz/pixel. The field map is divided by it. | `1` |

### Model

| Option                  | Value            | Description | Default |
| ----------------------- | ---------------- | ----------- | ------- |
| `-x`, `--no-synth`      |                  | Do not synthesise reversed-polarity echoes. Only the acquired echoes are used, and with no field map and no reversed echoes to constrain them, the edge-preserving penalty does all the regularisation. Ignored if `-r` is given. | synthesise |
| `-e`, `--extrapolate`   | `[BOOL]`         | Also synthesise the first and last reversed echoes by extrapolation, so that they are used in the fit. Alone, the flag means `yes`. Only relevant when echoes are synthesised. | `no` |
| `--penalty`             | `VAL`            | Weight of the total-variation penalty. It is divided by the typical tissue intensity of each echo, so that it does not depend on the scaling of the data. Larger values give smoother echoes. | `100` |

### Optimisation

| Option                  | Value            | Description | Default |
| ----------------------- | ---------------- | ----------- | ------- |
| `-m`, `--maxiter`       | `IRLS [CG]`      | Maximum number of reweighting iterations and of conjugate-gradient iterations per reweighting step. If only one value is given, it is used for both. | `20 32` |
| `-t`, `--tolerance`     | `IRLS [CG]`      | Tolerance for early stopping of the reweighting loop and of the conjugate-gradient solver. If only one value is given, it is used for both. | `1e-5` |
| `-s`, `--slicewise`     | `[CHUNKSIZE]`    | Process the data slice by slice, or in chunks of `CHUNKSIZE` slices, in parallel on the available CPU threads. Alone, the flag means 1 slice at a time. This saves memory, but the penalty then no longer couples neighbouring chunks. The slice direction is the shorter of the two axes perpendicular to the readout direction. | off |

### Output and general

| Option                | Value       | Description | Default |
| --------------------- | ----------- | ----------- | ------- |
| `-o`, `--output`      | `FILE...`   | Output path template(s). See [Output files](#output-files). | `{dir}{sep}{base}_epic{ext}` |
| `--cpu`               |             | Run on the CPU. | `--cpu` |
| `--gpu`               | `[ID]`      | Run on a CUDA device. `ID` is the device index; without it, the default CUDA device. Falls back to the CPU, with a warning, when CUDA is not available. | |
| `-v`, `--verbose`     | `[LVL]`     | Verbosity. Alone, the flag means `2`, which prints one line per iteration. | `1` |
| `-q`, `--quiet`       |             | No progress output. | |
| `-h`, `--help`        |             | Print the built-in help. | |

## Examples

Correct a 4D multi-echo file with a field map in Hz, for a readout bandwidth of 780 Hz/pixel and a readout along the antero-posterior axis:

```shell
nitorch epic ./echoes.nii.gz -f ./fieldmap_hz.nii.gz -b 780 -d ap
```

Use a voxel shift map, which needs no bandwidth (for example one estimated with [`topup`](topup.md) and `-bw 1`), and write each echo to its own file:

```shell
nitorch epic ./echoes.nii.gz -f ./echoes_topup_b0.nii.gz -d ap -o './epic/echo-{n}.nii.gz'
```

Pass echoes that were acquired as separate 3D files, with the polarity of the first echo reversed:

```shell
nitorch epic ./e1.nii.gz ./e2.nii.gz ./e3.nii.gz ./e4.nii.gz \
    -f ./fieldmap_hz.nii.gz -b 780 -p -
```

Use reversed-polarity echoes that were acquired for real, instead of synthesising them:

```shell
nitorch epic ./echoes.nii.gz -r ./echoes_reversed.nii.gz -f ./fieldmap_hz.nii.gz -b 780
```

Process large data slab by slab, four slices at a time, with a stronger penalty:

```shell
nitorch epic ./echoes.nii.gz -f ./fieldmap_hz.nii.gz -b 780 -s 4 --penalty 300
```

## References

- Varadarajan D, et al. A new distortion correction approach for multi-contrast MRI. *Proc. ISMRM* (2020).

## See also

- [`topup`](topup.md): estimate a field map from reverse-polarity images.
- [`estatics`](estatics.md): fit R2\* to multi-echo data, with its own distortion model (`--meetup`).
- [`denoise_mri`](../denoising/denoise_mri.md): denoise MR images without a field map.
