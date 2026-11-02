---
icon: octicons/terminal-24
---

# estatics

Fits a mono-exponential decay to multi-echo gradient-echo data and returns an R2* map together with the echo series extrapolated to TE = 0. When you acquired several contrasts (proton-density, T1 and MT weighted, say), the decay is shared across all of them, which makes the R2* estimate more robust to motion between acquisitions.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch estatics [OPTIONS] -c [NAME] [CONTRAST-OPTIONS] -e FILE...
                     [-c [NAME] [CONTRAST-OPTIONS] -e FILE...]...
    ```

    Global options such as `--likelihood` or `--regularization` must come
    **before** the first `-c`. See [Command-line layout](#command-line-layout).

## Description

`estatics` implements the ESTATICS model of Weiskopf et al. (2014) with the nonlinear, total-variation-regularised fitting scheme of Balbastre et al. (2020). Let $S_{c,e}(\mathbf{x})$ be the magnitude signal of echo $e$ of contrast $c$ at voxel $\mathbf{x}$, acquired at echo time $\mathrm{TE}_{c,e}$. The model is

$$
S_{c,e}(\mathbf{x}) = \exp\!\big(\beta_c(\mathbf{x}) - \mathrm{TE}_{c,e}\, R_2^*(\mathbf{x})\big) + \varepsilon,
$$

where each contrast $c$ has its own log-intercept $\beta_c$ (its signal at TE = 0) but all contrasts share the decay rate $R_2^*$. The original ESTATICS fit is a least-squares fit of the *logarithm* of the data. This command instead fits the model directly in the magnitude domain, with a Gaussian or non-central chi noise model, and regularises the maps with a (joint) total-variation penalty. It estimates the noise level of every contrast itself.

The reconstruction (output) space defaults to the mean space of all contrasts, so images need not share the same field of view or resolution. With `--register`, the contrasts are first aligned rigidly.

With `--meetup`, the command additionally estimates one distortion field per contrast along the readout direction, which corrects the geometric distortions of bipolar multi-echo readouts without a field map or reversed-polarity acquisition. See [MEETUP distortion correction](#meetup-distortion-correction).

### Command-line layout

Each contrast is introduced by `-c` and followed by the options that describe it. The echoes of a contrast are listed after `-e`. A command with two contrasts looks like this:

```shell
nitorch estatics -c pdw -te 2.3 4.6 6.9 ms -e pdw_e1.nii.gz pdw_e2.nii.gz pdw_e3.nii.gz \
                 -c t1w -te 2.3 4.6 6.9 ms -e t1w_e1.nii.gz t1w_e2.nii.gz t1w_e3.nii.gz
```

The optional `NAME` right after `-c` labels the contrast. It is only used to select the reconstruction space with `--recon-space`. A contrast can be as small as a single echo series, and a single contrast is allowed.

!!! warning "Put global options first"
    Options that do not belong to a contrast (everything under [Reconstruction](#reconstruction), [Optimisation](#optimisation) and [General](#general)) must appear **before** the first `-c`. Placed after it, they are read as extra echo file names. The contrast options (`-te`, `-e`, ...) can come in any order after their `-c`.

### Echoes and echo times

Each file after `-e` is one echo (a 3D volume) or a series of echoes stacked along the fourth dimension. All files of a contrast are concatenated in the order you give them, and that order must match the order of the echo times.

The echo times are obtained as follows.

1. With `-te`, you give them explicitly. The values may be followed by a unit, `ms` or `s` (or `sec`). **Without a unit, they are interpreted as seconds.** The list must have as many values as the contrast has echoes.
2. With `-te` and `-sp`, you give the first echo time and the echo spacing. The $n$-th echo then has time $\mathrm{TE}_0 + n\,\Delta$, where the number of echoes is the total number of volumes in the `-e` files. Here the unit of the spacing is mandatory (`ms` or `s`). Remember to give a unit for the first echo time as well.
3. Without `-te`, the echo time is read from the header of each file. NITorch does not read BIDS JSON sidecars. It reads the `TE=` field from the description string of NIfTI headers (for example `TR=24ms/TE=2.3ms/FA=6deg`), and the echo time stored in the header of FreeSurfer `.mgh`/`.mgz` files. A value in that string without a unit is read as seconds. A single header holds a single echo time, so for 4D files you must use `-te` or `-sp`.

All units are converted to seconds internally, and R2* is therefore returned in s<sup>-1</sup>.

## Arguments

`estatics` takes no positional arguments. Input files are given with `-e` inside a contrast.

## Options

### Contrast

These options describe one contrast and apply to the `-c` they follow.

| Option                  | Value                       | Description | Default |
| ----------------------- | --------------------------- | ----------- | ------- |
| `-c`, `--contrast`      | `[NAME]`                    | Starts a new contrast. May be repeated; at least one is required. `NAME` is an optional label. | *required* |
| `-e`, `--echo`          | `FILE...`                   | The echoes of this contrast, in acquisition order. 3D or 4D files. | *required* |
| `-te`, `--echo-time`    | `TE... [UNIT]`              | Echo times, with unit `ms` or `s`. A bare number means seconds. With `-sp`, only the first echo time is used. | read from file headers |
| `-sp`, `--echo-spacing` | `DELTA UNIT`                | Spacing between consecutive echoes. Requires `-te` and a unit. | unused |
| `-rd`, `--readout`      | `DIR`                       | Readout direction, for example `ap`, `lr` or `is`. The first voxel axis whose orientation letter appears in `DIR` is chosen. Only used with `--meetup`. | the voxel axis with the most voxels |
| `-bw`, `--bandwidth`    | `BW [UNIT]`                 | Readout bandwidth. Only read together with `-b0`. | unused |
| `-b0`, `--distortion`   | `FIELD [UNIT]`              | Pre-computed B0 field map, in voxels (`vox`) or Hz (`hz`). **Not functional in this version:** the code that loads the map fails. Use `--meetup` instead. | |

### Reconstruction

| Option               | Value                          | Description | Default |
| -------------------- | ------------------------------ | ----------- | ------- |
| `--likelihood`       | `{gauss,chi}`                  | Noise model of the magnitude images. `gauss` is a Gaussian model; `chi` is a non-central chi model, better suited to low signal-to-noise ratios, whose degrees of freedom are estimated from the data. | `gauss` |
| `--register`         | `[yes&vert;no]`                     | Rigidly align the contrasts before fitting. Alone, the flag means `yes`. Writes the registered echoes, see [Output files](#output-files). | `no` |
| `--recon-space`      | `{mean,NAME,INDEX}`            | Space in which the maps are computed. `mean` is the mean space of all contrasts, with a field of view that covers all of them. Otherwise, give the name of a contrast (as set after `-c`) or its zero-based index to use its orientation and shape. | `mean` |
| `--regularization`   | `{no,tkh,tv,jtv}`              | Penalty on the maps: none, Tikhonov (membrane energy), total variation, or joint total variation across all maps. See [Regularisation](#regularisation). | `jtv` |
| `--lam-intercept`    | `VAL...`                       | Penalty weight of the log-intercepts. Give one value per contrast, or fewer values to repeat the last one. | `50` |
| `--lam-decay`        | `VAL`                          | Penalty weight of the R2* map. | `0.05` |
| `--meetup`           | `[yes&vert;no]`                     | Estimate a distortion field per contrast and correct for it. Alone, the flag means `yes`. | `no` |
| `--lam-meetup`       | `VAL`                          | Weight of the bending-energy penalty on the distortion fields. | `1e5` |

### Optimisation

| Option         | Value   | Description | Default |
| -------------- | ------- | ----------- | ------- |
| `--nb-levels`  | `N`     | Number of resolution levels in the multi-resolution scheme. | `1` |
| `--max-iter`   | `N`     | Maximum number of outer (reweighting) iterations. Without a total-variation penalty, the number of inner Gauss–Newton updates is multiplied by this number instead. | `10` |
| `--tolerance`  | `TOL`   | Stop when the relative gain in the objective function falls below this value. | `1e-5` |

### General

| Option                 | Value     | Description | Default |
| ---------------------- | --------- | ----------- | ------- |
| `-o`, `--output-dir`   | `DIR`     | Directory in which to write the results. | directory of the first echo of each output |
| `--cpu`                |           | Run on the CPU. | |
| `--gpu`                | `[ID]`    | Run on a CUDA device. `ID` is the device index. Falls back to the CPU when CUDA is not available. | `--cpu` |
| `-v`, `--verbose`      | `[LVL]`   | Verbosity. Alone, the flag means `1`. At `2` or more, progress is also plotted. | `1` |
| `--framerate`          | `HZ`      | Accepted but currently has no effect. | `1` |
| `-h`, `--help`         |           | Print the built-in help. | |

## Regularisation

The log-intercepts and the R2* map are smoothed jointly during the fit, so that noise in one echo series does not leak into the others, yet edges are preserved. `--regularization jtv`, the default, penalises the *joint* total variation of all maps: the penalty shares one edge-weight map across them, so an edge that is visible in one contrast protects the same edge in the others. `tv` weights each map separately, `tkh` applies a plain membrane (Tikhonov) penalty that blurs edges, and `no` fits each voxel independently.

The weights `--lam-intercept` and `--lam-decay` act on the log-intercepts and on R2* respectively. If the maps look too noisy, increase them. If fine structure is lost, decrease them. If both are zero, no penalty is applied whatever `--regularization` says.

## MEETUP distortion correction

Bipolar multi-echo readouts alternate the readout direction from one echo to the next, so that B0 inhomogeneity displaces voxels along the readout direction in opposite senses in odd and even echoes. With `--meetup`, `estatics` fits, jointly with the decay model, one displacement field per contrast along the readout direction, under the assumption that consecutive echoes alternate in polarity. The fields are regularised by their bending energy (`--lam-meetup`). The readout direction is that given by `-rd`, or by default the voxel axis with the most voxels.

When the distortion model is on, the command also writes the estimated fields and a corrected version of every echo, see [Output files](#output-files). Please cite Balbastre et al. (ISMRM 2022) if you use this option.

## Output files

All outputs are written as 32-bit floating-point images, in the format of the corresponding input file. Without `-o`, they are written next to the input files. Because the names below do not depend on the command line, a second run in the same directory overwrites the results of the first.

| File | Content |
| ---- | ------- |
| `R2star{ext}` | The R2* map, in s<sup>-1</sup>. Written once, in the directory of the first echo of the first contrast. |
| `{base}_TE0{ext}` | For each contrast, the echo series extrapolated to TE = 0 (the exponential of the log-intercept), in the arbitrary units of the input. `{base}` and `{ext}` come from the first echo file of the contrast. |
| `{base}_B0{ext}` | With `--meetup`, for each contrast, the estimated distortion field, as a voxel displacement along the readout direction. |
| `{base}_unwrapped{ext}` | With `--meetup`, each echo with the distortion corrected (and intensities modulated by the Jacobian of the correction). `{base}` is the name of the file the echo comes from. |
| `{base}_registered{ext}` | With `--register`, each echo file saved with its updated orientation matrix. The data are not resampled. |

The maps are in the reconstruction space (`--recon-space`), whereas `_unwrapped` echoes keep the space of their input.

## Examples

Fit R2* to a single series of five echoes whose echo times are stored in the file headers:

```shell
nitorch estatics -c -e echo-1.nii.gz echo-2.nii.gz echo-3.nii.gz echo-4.nii.gz echo-5.nii.gz
```

Give the echo times explicitly, as a first echo time and a constant spacing, for a 4D file that holds all echoes:

```shell
nitorch estatics -c -te 2.3 ms -sp 2.3 ms -e multiecho.nii.gz
```

Fit the multi-contrast model to proton-density, T1 and MT weighted series. The decay is shared, and each series gets a TE = 0 image:

```shell
nitorch estatics \
    -c pdw -e pdw_e1.nii.gz pdw_e2.nii.gz pdw_e3.nii.gz pdw_e4.nii.gz \
    -c t1w -e t1w_e1.nii.gz t1w_e2.nii.gz t1w_e3.nii.gz t1w_e4.nii.gz \
    -c mtw -e mtw_e1.nii.gz mtw_e2.nii.gz mtw_e3.nii.gz mtw_e4.nii.gz
```

Align the contrasts first, use the space of the T1-weighted series, and write to a separate directory. The global options come before the contrasts:

```shell
nitorch estatics --register --recon-space t1w --likelihood chi -o derivatives/ \
    -c pdw -e pdw_e1.nii.gz pdw_e2.nii.gz pdw_e3.nii.gz \
    -c t1w -e t1w_e1.nii.gz t1w_e2.nii.gz t1w_e3.nii.gz
```

Correct readout distortions of a bipolar acquisition, with the readout along the antero-posterior axis, and weaken the penalty on the R2* map:

```shell
nitorch estatics --meetup --lam-decay 0.01 \
    -c pdw -rd ap -e pdw_e1.nii.gz pdw_e2.nii.gz pdw_e3.nii.gz pdw_e4.nii.gz
```

## References

- Weiskopf N, Callaghan MF, Josephs O, Lutti A, Mohammadi S. Estimating the apparent transverse relaxation time (R2\*) from images with different contrasts (ESTATICS) reduces motion artifacts. *Frontiers in Neuroscience* (2014). [doi:10.3389/fnins.2014.00278](https://doi.org/10.3389/fnins.2014.00278)
- Balbastre Y, Brudfors M, Azzarito M, Lambert C, Callaghan MF, Ashburner J. Joint total variation ESTATICS for robust multi-parameter mapping. *MICCAI* (2020). [arXiv:2005.14247](https://arxiv.org/abs/2005.14247)
- Balbastre Y, et al. Model-based multi-parameter mapping. *Medical Image Analysis* (2021). [arXiv:2102.01604](https://arxiv.org/abs/2102.01604) (optimisation scheme used by this implementation)
- Balbastre Y, et al. Distortion correction in multi-echo MRI without field mapping or reverse encoding. *Proc. ISMRM* (2022). (`--meetup`)

## See also

- [`greeq`](greeq.md): fit PD, R1, R2* and MTsat directly from multi-echo, multi-flip-angle data.
- [`epic`](epic.md): denoise and correct bipolar multi-echo data with a pre-computed field map.
- [`topup`](topup.md): estimate a field map from reverse-polarity images.
- [File formats](../../conventions.md#file-formats): the image formats NITorch reads and writes.
