---
icon: octicons/beaker-24
---

# Quantitative MRI

The commands in this group turn raw multi-echo and multi-contrast MR images into quantitative maps, and correct the geometric distortions that get in the way. They are aimed at the gradient-echo protocols used for multi-parameter mapping (proton density, R1, R2\* and magnetisation transfer saturation), and at the distortion that bipolar or echo-planar readouts add to them.

## What they are for

Everything here starts from images acquired at several echo times. The signal of a spoiled gradient-echo sequence decays with the echo time $\mathrm{TE}$ as $S_0\,e^{-\mathrm{TE}\,R_2^*}$, and its initial value $S_0$ depends on proton density, R1, the flip angle, the repetition time and, if an MT pulse is played, on the MT saturation. The two mapping commands fit this physics directly to the images:

- [`estatics`](estatics.md) fits the decay alone. It estimates one R2\* map from any number of contrasts that share it, and returns each contrast extrapolated to TE = 0. It is the right tool when you want R2\* (or R2\*-corrected images to feed another pipeline) and do not need the full signal model.
- [`greeq`](greeq.md) inverts the complete signal equation. From two or more flip angles and an optional MT-weighted series, it estimates PD, R1, R2\* and MTsat in one go, with optional B1 transmit and receive corrections.

Both read the sequence parameters (TE, and for `greeq` also TR, flip angle and the MT flag) from explicit options, or from the image headers when you leave the options out. They do not read BIDS JSON sidecars. Their pages explain what is read and in which units.

The other two commands deal with geometric distortion along the readout direction, caused by B0 inhomogeneity:

- [`topup`](topup.md) estimates a distortion field from two images acquired with opposite readout polarities, in the manner of FSL TOPUP. A choice of matching losses lets it work with images of different contrast, such as the first two echoes of a bipolar multi-echo acquisition.
- [`epic`](epic.md) uses a pre-computed field map to undistort and denoise bipolar multi-echo data, exploiting the exponential decay across echoes.

## How they fit together

A typical multi-parameter mapping pipeline corrects distortions first and maps afterwards. Which route you take depends on what you acquired:

```text
opposite-polarity pair ──► topup fit ──► field ──┐
                                                  ├──► epic ──► estatics / greeq
measured B0 field map ────────────────────────────┘

bipolar multi-echo data, nothing else ──► estatics --meetup
undistorted data ──────────────────────► estatics / greeq
```

`topup` produces a field that `epic` can consume, and `epic` produces corrected echoes that either mapping command can read. `estatics` can also estimate and correct distortions on its own with `--meetup`, without a field map or reversed-polarity data. `estatics` and `greeq` answer different questions. Use `estatics` for a robust R2\* map from several contrasts, and `greeq` when you want PD, R1, R2\* and MTsat from a single model.

## Commands

| Command | Summary |
| ------- | ------- |
| [`estatics`](estatics.md) | Fit a shared mono-exponential R2\* decay to multi-echo, multi-contrast data, with optional distortion correction. |
| [`greeq`](greeq.md) | Estimate PD, R1, R2\* and MTsat from multi-echo, multi-flip-angle gradient-echo data. |
| [`epic`](epic.md) | Denoise and undistort bipolar multi-echo data with a pre-computed field map. |
| [`topup`](topup.md) | Estimate and apply a distortion field from reversed-polarity images. |

## Conventions shared by these commands

- **Contrasts and echoes.** `estatics` and `greeq` describe their input with repeated `-c` groups, one per contrast, each listing its echoes after `-e`. Global options must come before the first `-c`.
- **Units.** Times given without a unit are read as seconds, flip angles as degrees. Add `ms` where you mean milliseconds.
- **Output files.** The mapping commands write fixed names (`R2star`, `PD`, ...) next to the first input unless you pass `-o`. See each page for the list and units.
- **Devices.** The commands run on the CPU by default; see [Hardware](../../conventions.md#hardware) for the GPU options.
- **General conventions.** Image formats, option parsing and filename templates are described in [Conventions](../../conventions.md).
