---
icon: octicons/terminal-24
---

# greeq

Estimates quantitative proton density (PD), longitudinal relaxation rate (R1), apparent transverse relaxation rate (R2\*) and, optionally, magnetisation transfer saturation (MTsat) from multi-echo spoiled gradient-echo data acquired with several flip angles and an optional MT pulse. It fits the signal equation directly to the raw echoes, instead of first computing intermediate maps, and can correct for B1 transmit and receive fields.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch greeq [OPTIONS] [-tf FIELD [MAG] [UNIT]] [-rf FIELD [MAG] [UNIT]]
                  -c [NAME] [-fa FA [UNIT]] [-tr TR [UNIT]] [-te TE... [UNIT]]
                            [-es DELTA UNIT] [-mt [yes|no]]
                            [-tf FIELD [MAG] [UNIT]] [-rf FIELD [MAG] [UNIT]]
                            -e FILE...
                  [-c [NAME] ... -e FILE...]...
    ```

    Global options must come **before** the first `-c`. See
    [Command-line layout](#command-line-layout).

## Description

`greeq` implements the multi-parameter mapping model of Balbastre et al. (2021). Every acquired image series (a *contrast*) is a set of echoes with one flip angle $\alpha$, one repetition time $\mathrm{TR}$ and, possibly, an off-resonance MT pulse. At echo time $\mathrm{TE}$, the signal predicted by the model is

$$
S(\mathrm{TE}) \;=\; \mathrm{PD}\cdot B_1^{-}\;
\frac{\sin(\alpha B_1^{+})\,(1-\delta)\,(1-E_1)}
     {1-\cos(\alpha B_1^{+})\,(1-\delta)\,E_1}\;
e^{-\mathrm{TE}\cdot R_2^*},
\qquad E_1 = e^{-\mathrm{TR}\cdot R_1},
$$

where $B_1^{+}$ and $B_1^{-}$ are the relative transmit and receive fields (1 where absent), and $\delta$ is the MTsat. In contrasts without an MT pulse, $\delta = 0$ and the equation reduces to the usual spoiled gradient-echo (Ernst) signal. $\delta$ only exists in the model when at least one contrast is flagged as MT-weighted.

The fit is a regularised maximum a posteriori estimation. PD, R1 and R2\* are estimated in the log domain (so that they remain positive), and MTsat in the logit domain (so that it stays between 0 and 1). A joint total-variation prior smooths all maps while preserving edges. The noise level of every contrast is estimated from the data, and the noise model can be Gaussian or non-central chi.

For the fit to be determined, you need at least two contrasts without an MT pulse (typically proton-density and T1 weighted, with a low and a high flip angle). These two are also used to initialise the maps. One MT-weighted contrast is needed to estimate MTsat, as in the standard multi-parameter mapping protocol.

### Command-line layout

Each contrast starts with `-c`, followed by the options that describe it. The echoes of a contrast are given after `-e`. A typical multi-parameter mapping command has three contrasts:

```shell
nitorch greeq \
    -c pdw -fa 6  -tr 25 ms -te 2.34 ms -es 2.34 ms -e pdw.nii.gz \
    -c t1w -fa 21 -tr 25 ms -te 2.34 ms -es 2.34 ms -e t1w.nii.gz \
    -c mtw -fa 6  -tr 25 ms -te 2.34 ms -es 2.34 ms -mt -e mtw.nii.gz
```

The label after `-c` is only used to select the reconstruction space with `--recon-space`. The contrast options can come in any order after their `-c`.

!!! warning "Put global options first"
    Options that do not belong to a contrast (reconstruction, optimisation, output and general options, and the shared field maps `-tf`/`-rf`) must be given **before** the first `-c`. Placed after it, they would be read as extra echo file names.

### Sequence parameters

NITorch does not read BIDS JSON sidecars. The sequence parameters of every contrast are obtained as follows.

- **Explicit options.** `-fa`, `-tr` and `-te` (optionally with `-es`) set them. This is the safest route. A flip angle without unit is in degrees (`rad` converts from radians). TR and TE without unit are in **seconds**, so you will usually write `ms` explicitly. Only the first value of `-fa` is used.
- **File headers.** When an option is absent, the value is read from the image header: from the `TR=`, `TE=` and `FA=` fields of the description string of NIfTI files (for example `TR=25ms/TE=2.3ms/FA=6deg`; the repetition time can also be the fourth voxel size), or from the corresponding fields of FreeSurfer `.mgh`/`.mgz` files. A number in the description string without a unit is read as seconds (TR, TE) or degrees (FA). If the header does not contain a required value, the command fails.
- **MT pulse.** The presence of an MT pulse is never read from files. Flag MT-weighted contrasts with `-mt`; any contrast without `-mt` is treated as having no MT pulse.

For echo times, `-te` followed by several values gives the echo times of the contrast in the order of the echoes. With `-es`, `-te` gives only the first echo time and `-es` the spacing, and the echo times are generated for as many echoes as there are volumes in the `-e` files. The unit of the spacing is mandatory. Without `-te`, each file's header gives its own echo time, which means that 4D files (several echoes in one file) need `-te`.

### Field maps

B1 transmit (`-tf`) and receive (`-rf`) maps are optional and can be at any resolution and in any orientation: they are resampled to each contrast through their orientation matrices. Each is given as the path to the map, optionally followed by the path to a magnitude image, optionally followed by a unit:

```text
-tf B1plus.nii.gz [magnitude.nii.gz] [UNIT]
```

The unit is `%` (also `pct` or `p.u.`) when the map is expressed in percent of the nominal value, in which case it is divided by 100, or `a.u.` when the map is already a multiplicative factor around 1. **When no unit is given, the map is taken to be in percent.** The magnitude image is only used to register the field map with the other contrasts when `--register` is on.

A transmit map rescales the nominal flip angle at each voxel. A receive map multiplies the estimated PD by the receive sensitivity, so with a receive map the PD output is free of receive bias. Without one, PD retains the bias field of the receive coil.

A map given inside a contrast applies to that contrast only. A transmit map given before the first `-c` applies to every contrast that has none of its own.

## Arguments

`greeq` takes no positional arguments. Input files are given with `-e` inside a contrast.

## Options

### Contrast

These options describe one contrast and apply to the `-c` they follow.

| Option                           | Value                  | Description | Default |
| -------------------------------- | ---------------------- | ----------- | ------- |
| `-c`, `--contrast`               | `[NAME]`               | Starts a new contrast. May be repeated; at least one is required. `NAME` is an optional label. | *required* |
| `-e`, `--echo`                   | `FILE...`              | The echoes of this contrast, in acquisition order. 3D or 4D files. | *required* |
| `-fa`, `--flip-angle`            | `FA [UNIT]`            | Flip angle, in degrees, or radians with unit `rad`. | read from file header |
| `-tr`, `--repetition-time`       | `TR [UNIT]`            | Repetition time, with unit `ms` or `s`. A bare number means seconds. | read from file header |
| `-te`, `--echo-time`             | `TE... [UNIT]`         | Echo times, with unit `ms` or `s`. A bare number means seconds. | read from file headers |
| `-es`, `--echo-spacing`          | `DELTA UNIT`           | Echo spacing. Requires `-te` and a unit. | unused |
| `-mt`, `--mt-pulse`              | `[yes|no]`        | Flags the contrast as MT-weighted. Alone, the flag means `yes`. | no MT pulse |
| `-tf`, `--transmit`              | `FIELD [MAG] [UNIT]`   | B1 transmit map for this contrast. | none |
| `-rf`, `--receive`               | `FIELD [MAG] [UNIT]`   | B1 receive map for this contrast. | none |
| `-b0`, `--b0-field`              | `FIELD [MAG] [UNIT]`   | B0 map. Accepted but **not used**: B0 correction is not implemented. | none |
| `-bw`, `--bandwidth`             | `BW [UNIT]`            | Accepted but currently has no effect. | unused |

### Shared field maps

These are given before the first `-c`.

| Option                | Value                  | Description | Default |
| --------------------- | ---------------------- | ----------- | ------- |
| `-tf`, `--transmit`   | `FIELD [MAG] [UNIT]`   | B1 transmit map used by all contrasts that do not have their own. | none |
| `-rf`, `--receive`    | `FIELD [MAG] [UNIT]`   | B1 receive map used by all contrasts that do not have their own. | none |
| `-b0`, `--b0-field`   | `FIELD [MAG] [UNIT]`   | B0 map used by all contrasts that do not have their own. Accepted but **not used**: B0 correction is not implemented. | none |

### Reconstruction

| Option               | Value                         | Description | Default |
| -------------------- | ----------------------------- | ----------- | ------- |
| `--likelihood`       | `{gauss,chi}`                 | Noise model of the magnitude images: Gaussian, or non-central chi (suited to low signal-to-noise ratios; its degrees of freedom are estimated). | `gauss` |
| `--register`         | `[yes|no]`               | Rigidly align the contrasts (and the magnitude images of field maps) before fitting. Alone, the flag means `yes`. | `no` |
| `--recon-space`      | `{mean,NAME,INDEX}`           | Space in which the maps are computed. `mean` is the mean space of all contrasts, with a field of view that covers all of them. Otherwise, give the name of a contrast or its zero-based index to use its orientation and shape. | `mean` |
| `--crop-space`       | `PCT`                         | Accepted but currently has no effect. | `0` |
| `--regularization`   | `{no,tkh,tv,jtv}`             | Penalty on the (log) maps: none, Tikhonov (membrane energy), total variation, or joint total variation. | `jtv` |
| `--lam`              | `VAL`                         | Penalty weight shared by all maps. | `10` |
| `--lam-pd`           | `VAL`                         | Penalty weight of PD, overriding `--lam`. | `--lam` |
| `--lam-t1`           | `VAL`                         | Penalty weight of R1, overriding `--lam`. | `--lam` |
| `--lam-t2`           | `VAL`                         | Penalty weight of R2\*, overriding `--lam`. | `--lam` |
| `--lam-mt`           | `VAL`                         | Penalty weight of MTsat, overriding `--lam`. | `--lam` |
| `--meetup`           | `[yes|no]`               | Parsed, but distortion correction is not implemented in `greeq`, so this has no effect. Use [`estatics`](estatics.md) for MEETUP. | `no` |
| `--lam-meetup`       | `VAL`                         | Parsed but has no effect, for the same reason. | `1e5` |

A per-map weight of `0` switches the penalty off for that map only. To switch regularisation off for all maps, use `--regularization no`.

### Optimisation

| Option         | Value                  | Description | Default |
| -------------- | ---------------------- | ----------- | ------- |
| `--nb-levels`  | `N`                    | Number of resolution levels in the multi-resolution scheme. | `1` |
| `--max-iter`   | `N`                    | Maximum number of outer (reweighting) iterations. Without a total-variation penalty, the number of Gauss–Newton updates is multiplied by this number instead. | `10` |
| `--tolerance`  | `TOL`                  | Stop when the relative gain in the objective function falls below this value (also used by the linear solver). | `1e-4` |
| `--solver`     | `{cg,fmg} [N]`         | Linear solver for the regularised Newton step: conjugate gradients, or full multigrid. With `cg`, `N` is the maximum number of iterations. With `fmg`, the number of multigrid cycles is fixed and `N` has no effect. | `cg 32` |

### Output and general

| Option                 | Value     | Description | Default |
| ---------------------- | --------- | ----------- | ------- |
| `-o`, `--output-dir`   | `DIR`     | Directory in which to write the maps. | directory of the first echo of the first contrast |
| `-u`, `--uncertainty`  | `[yes|no]` | Also write the uncertainty of every map. Alone, the flag means `yes`. | `no` |
| `--cpu`                |           | Run on the CPU. | |
| `--gpu`                | `[ID]`    | Run on a CUDA device. `ID` is the device index. Falls back to the CPU when CUDA is not available. | `--cpu` |
| `-v`, `--verbose`      | `[LVL]`   | Verbosity. Alone, the flag means `1`. Use `-v 0` for a silent run. | `1` |
| `--framerate`          | `HZ`      | Accepted but currently has no effect. | `1` |
| `-h`, `--help`         |           | Print the built-in help. | |

## Output files

The maps are saved in 32-bit floating point, in the reconstruction space and in the format of the first input file. Their names are fixed, so a second run in the same directory overwrites the first. Use `-o` to keep runs apart.

| File | Content | Unit |
| ---- | ------- | ---- |
| `PD{ext}` | Proton density. | arbitrary units |
| `R1{ext}` | Longitudinal relaxation rate, 1/T1. | s<sup>-1</sup> |
| `R2star{ext}` | Apparent transverse relaxation rate. | s<sup>-1</sup> |
| `MTsat{ext}` | Magnetisation transfer saturation, written only if at least one contrast has `-mt`. | percent units (0 to 100) |
| `log_PD_uncertainty{ext}`, `log_R1_uncertainty{ext}`, `log_R2star_uncertainty{ext}` | With `-u`: the approximate posterior variance (Laplace approximation) of the *logarithm* of each map. | |
| `logit_MTsat_uncertainty{ext}` | With `-u` and MTsat: the same, for the logit of MTsat. | |

Remember that the R1 map is a rate: to obtain T1 in seconds, take its reciprocal.

## Examples

Estimate PD, R1 and R2\* from two series whose flip angle, repetition time and echo times are all stored in the file headers:

```shell
nitorch greeq -c pdw -e pdw.nii.gz -c t1w -e t1w.nii.gz
```

Give the sequence parameters explicitly, for 4D files that hold all echoes, and estimate MTsat from a third, MT-weighted series:

```shell
nitorch greeq \
    -c pdw -fa 6  -tr 25 ms -te 2.34 ms -es 2.34 ms -e pdw.nii.gz \
    -c t1w -fa 21 -tr 25 ms -te 2.34 ms -es 2.34 ms -e t1w.nii.gz \
    -c mtw -fa 6  -tr 25 ms -te 2.34 ms -es 2.34 ms -mt -e mtw.nii.gz
```

Correct for B1 transmit inhomogeneity with a map expressed in percent of the nominal flip angle, shared by all contrasts, and align the series first. Global options go before the contrasts:

```shell
nitorch greeq --register -tf B1map.nii.gz B1magnitude.nii.gz % \
    -c pdw -fa 6  -tr 25 ms -te 2.34 ms -es 2.34 ms -e pdw.nii.gz \
    -c t1w -fa 21 -tr 25 ms -te 2.34 ms -es 2.34 ms -e t1w.nii.gz
```

Apply a receive-sensitivity map to the proton-density series, so that the PD map is bias-free, and write the maps and their uncertainty to a new directory:

```shell
nitorch greeq -u -o mpm/ \
    -c pdw -fa 6  -tr 25 ms -te 2.34 ms -es 2.34 ms -rf RFsens.nii.gz a.u. -e pdw.nii.gz \
    -c t1w -fa 21 -tr 25 ms -te 2.34 ms -es 2.34 ms -rf RFsens.nii.gz a.u. -e t1w.nii.gz
```

Use a stronger penalty on R2\* than on the other maps, compute on a GPU, and reconstruct in the space of the T1-weighted series:

```shell
nitorch greeq --gpu --lam 10 --lam-t2 30 --recon-space t1w \
    -c pdw -fa 6  -tr 25 ms -te 2.34 ms -es 2.34 ms -e pdw.nii.gz \
    -c t1w -fa 21 -tr 25 ms -te 2.34 ms -es 2.34 ms -e t1w.nii.gz
```

## References

- Balbastre Y, et al. Model-based multi-parameter mapping. *Medical Image Analysis* (2021). [arXiv:2102.01604](https://arxiv.org/abs/2102.01604)
- Weiskopf N, Callaghan MF, Josephs O, Lutti A, Mohammadi S. Estimating the apparent transverse relaxation time (R2\*) from images with different contrasts (ESTATICS) reduces motion artifacts. *Frontiers in Neuroscience* (2014). [doi:10.3389/fnins.2014.00278](https://doi.org/10.3389/fnins.2014.00278) (the shared-decay principle that `greeq` also uses)

## See also

- [`estatics`](estatics.md): fit only R2\* and the TE = 0 images, optionally with distortion correction.
- [`epic`](epic.md) and [`topup`](topup.md): correct geometric distortions before fitting.
- [File formats](../../conventions.md#file-formats): the image formats NITorch reads and writes.
