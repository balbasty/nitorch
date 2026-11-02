---
icon: octicons/terminal-24
---

# metrics

Compare a predicted image with a reference image and write the scores to a JSON
file. Use it to evaluate a segmentation (Dice, Jaccard, Hausdorff distance and
related overlap scores) or a reconstruction or denoising result (squared error,
peak signal-to-noise ratio, and so on).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch metrics -p FILE... -r FILE... -m METRIC... [-s AVERAGE...]
                    [-x LABEL...] [-o FILE] [-a [BOOL]]
                    [--ssim-fwhm FWHM... [UNIT] | full] [--ssim-gauss]
                    [--range [[MIN=0] MAX]]
                    [--cpu [N] | --gpu [ID]]
    ```

## Description

`metrics` has two families of scores:

- **label metrics** treat the images as label maps, in which every voxel holds
  an integer class. They are computed for each label separately, by asking how
  well the voxels predicted as that label coincide with the voxels that carry
  that label in the reference;
- **image metrics** treat the images as continuous intensities and measure the
  voxel-wise difference between them.

You choose the metrics with `-m`, and a single call can mix both families. The
groups `labels` and `images` stand for every label metric and every image metric,
respectively, and can be combined with each other and with individual metrics, as
in `-m labels mse`. The data are loaded once per family.

### Inputs

`--pred` (`-p`) is the image to evaluate and `--ref` (`-r`) is the ground truth.
Each can be given as several files. Every file is read as a volume with up to
three spatial dimensions and, optionally, a channel dimension: a 3D file has one
channel, and a 4D file has one channel per volume along its fourth dimension. The
channels of all the files you list are stacked, in order, to form one multi-channel
image. The stacked prediction and reference must have exactly the same shape.
The command compares shapes only. It does not check that the orientation
matrices agree, so make sure that both images are on the same voxel grid; the
orientation matrix of the reference is used only to convert millimetres to
voxels for the [SSIM window](#ssim).

For label metrics, the data are read as stored (not rescaled) and converted to
integers. For image metrics they are read as floating-point values, with the
scaling stored in the file applied. A call that asks for both families therefore
reads each image twice, once in each way.

### Output

The scores are written to a JSON file. Its name is built from the output
template, which by default is `{dir}{sep}{base}.metrics.json`; the placeholders
refer to the first file given to `--pred` (see [Options](#inputs-and-outputs)). The file contains an object with one
entry per requested metric. Each entry is itself an object that maps either a
label (label metrics) or a channel number starting at 0 (image metrics) to the
score, and that carries one more entry for every averaging mode you asked for
with `-s`. The keys are strings, since JSON has no numeric keys. For example,

```shell
nitorch metrics -p seg.nii.gz -r manual.nii.gz -m dice jaccard -s macro
```

writes `seg.metrics.json` with a content of the form

```json
{
    "dice": {
        "1": 0.94,
        "2": 0.87,
        "macro": 0.905
    },
    "jaccard": {
        "1": 0.89,
        "2": 0.77,
        "macro": 0.83
    }
}
```

If the output file already exists, it is overwritten, unless you pass `--append`:
the new scores are then merged into the existing file, metric by metric. Entries
of a metric that are not recomputed (other labels, or averages you did not ask
for this time) are kept, and entries that are recomputed are replaced. Metrics that
the file does not contain yet are added, and the file is created if it does not
exist.

The values above only illustrate the layout. Ratios that are undefined, such as
a precision for a label that the prediction never uses, are written as `NaN`.

If anything goes wrong, for example a shape mismatch between the prediction and
the reference, `metrics` prints a single `[ERROR] ...` line to the standard error
stream and exits with status 1.

## Label metrics

For each label $\ell$ that occurs in the reference image, the prediction and
the reference are reduced to two binary masks, "is $\ell$" or "is not $\ell$",
and the voxels of the whole image (all channels together) are counted:

- **TP** (true positives): labelled $\ell$ in both images;
- **FP** (false positives): labelled $\ell$ in the prediction only;
- **FN** (false negatives): labelled $\ell$ in the reference only;
- **TN** (true negatives): not labelled $\ell$ in either.

Let $P = TP + FN$ be the number of voxels of the reference with label $\ell$ and
$N = TN + FP$ the number of the other voxels. Only labels present in the
reference are scored; a label that appears only in the prediction is ignored.
The labels listed with `-x` are never scored and never count as foreground
(by default, label 0, which usually stands for the background).

| Metric        | Name                                | Definition                           |
| ------------- | ----------------------------------- | ------------------------------------ |
| `dice`        | Dice coefficient (F1 score)         | $2\,TP / (2\,TP + FP + FN)$          |
| `jaccard`     | Jaccard index (intersection over union) | $TP / (TP + FP + FN)$            |
| `accuracy`    | Accuracy                            | $(TP + TN) / (P + N)$                |
| `ppv`         | Positive predictive value (precision) | $TP / (TP + FP)$                   |
| `npv`         | Negative predictive value           | $TN / (TN + FN)$                     |
| `fdr`         | False discovery rate                | $FP / (TP + FP)$                     |
| `tpr`         | True positive rate (sensitivity, recall) | $TP / P$                        |
| `fpr`         | False positive rate                 | $FP / N$                             |
| `fnr`         | False negative rate (miss rate)     | $FN / P$                             |
| `tnr`         | True negative rate (specificity)    | $TN / N$                             |
| `plr`         | Positive likelihood ratio           | $\text{TPR} / \text{FPR}$            |
| `nlr`         | Negative likelihood ratio           | $\text{FNR} / \text{TNR}$            |
| `hausdorff`   | Hausdorff distance                  | [See below](#hausdorff-distance)     |
| `hausdorff95` | Hausdorff distance, 95th percentile | [See below](#hausdorff-distance)     |

### Hausdorff distance

For a label $\ell$, `hausdorff` is the largest distance, over all voxels that the
prediction assigns to $\ell$, to the closest voxel that the reference assigns to
$\ell$. It is therefore a *directed* distance, from prediction to reference: a
reference structure that the prediction misses entirely does not increase it.
`hausdorff95` replaces the maximum with the 95th percentile of these distances,
which is much less sensitive to a few stray voxels. Distances are Euclidean and
expressed in **voxels**; the voxel size is not applied. If the prediction does not
contain the label at all, the distance is reported as 0.

The distances are computed on the whole stacked image, with the channel axis
treated as a fourth dimension with unit spacing. For meaningful results, evaluate
single-channel label maps.

## Image metrics

Image metrics are computed for each channel separately, from the difference
between prediction and reference, then optionally averaged.

| Metric | Value computed for each channel                                              |
| ------ | ---------------------------------------------------------------------------- |
| `mse`  | Mean squared error, the mean of $(\text{pred} - \text{ref})^2$.              |
| `rmse` | Root mean squared error, the square root of `mse`.                           |
| `mrse` | Mean of the absolute difference, $\text{mean}\,\lvert \text{pred} - \text{ref} \rvert$. The built-in help calls it "mean root squared error". |
| `mad`  | Median of the absolute difference, $\text{median}\,\lvert \text{pred} - \text{ref} \rvert$. The built-in help calls it "median absolute deviation", but it is the median *error* between the two images, not the deviation from their own median. |
| `psnr` | Peak signal-to-noise ratio in decibels, $10 \log_{10}(L^2 / \text{mse})$, where $L$ is the width of the [dynamic range](#dynamic-range). |
| `ssim` | Structural similarity index. [See below](#ssim).                             |

### Dynamic range

`psnr` and `ssim` need the dynamic range of the data, whose width is
$L = \text{max} - \text{min}$. It is set with `--range [[MIN=0] MAX]`: a single
number is taken as `MAX`, with `MIN` equal to 0. Without `--range`, the range
is that of the reference, from its minimum to its maximum. It is measured
in each channel separately, or over all channels together for the `micro` score.

### SSIM

The structural similarity index compares, in a local window, the mean ($\mu$),
variance ($\sigma^2$) and covariance ($\sigma_{xy}$) of the prediction $x$ and the
reference $y$ (Wang et al., 2004):

$$
\text{SSIM} =
\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
     {(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)},
\qquad C_1 = (0.01\,L)^2,\quad C_2 = (0.03\,L)^2 .
$$

The score is the mean of the local indices over the image. `--ssim-fwhm` sets
the size of the window, as one value, or one per axis, optionally followed by
a unit, `vox` (voxels, the default) or `mm` (millimetres, converted to voxels
with the voxel size of the reference). Three variants exist:

- by default, a **sliding square window** whose side is the window size, rounded up
  to an integer number of voxels (3 by default). The statistics are computed
  where the window fits entirely inside the image;
- with `--ssim-gauss`, a **Gaussian window** whose full width at half maximum is
  the window size;
- with `--ssim-fwhm full`, **global statistics**: the whole image is a single
  window.

## Averaging

Without `-s`, the output contains one score per label or channel. `-s` adds a
summary score to every metric. It can be given several modes at once.

| Mode       | Meaning                                                                                       |
| ---------- | --------------------------------------------------------------------------------------------- |
| `macro`    | The unweighted mean of the per-label (or per-channel) scores. Every label counts equally.     |
| `micro`    | The score recomputed after pooling the counts (or voxels) of all labels or channels. Large structures dominate. |
| `weighted` | The mean of the per-label scores, weighted by the number of reference voxels of each label ($P$). Available for label metrics other than the Hausdorff distances; ignored by the other metrics. |

The `micro` score of a label metric uses the sums of TP, TN, FP and FN over the
labels. For `hausdorff`, it is the maximum (or percentile) of the distances
pooled over all labels. For image metrics, `micro` of `mse` is the mean over all
voxels of all channels, and of `mad` the median over all of them; for `mrse` it
is the mean, over voxels, of the Euclidean norm of the difference vector across
channels, which is a different quantity from the per-channel mean of absolute
errors.

## Arguments

`metrics` takes no positional arguments. Everything is given through options.

## Options

### Inputs and outputs

| Option             | Value                  | Description                                                                                       | Default                         |
| ------------------ | ---------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------- |
| `-p`, `--pred`     | `FILE...`              | Predicted image(s). Files are stacked as channels. [See above](#inputs).                          | *required*                      |
| `-r`, `--ref`      | `FILE...`              | Reference image(s), with the same total shape as the prediction.                                  | *required*                      |
| `-m`, `--metrics`  | `METRIC...`            | Metrics to compute: `dice`, `jaccard`, `accuracy`, `ppv`, `npv`, `fdr`, `tpr`, `fpr`, `fnr`, `tnr`, `plr`, `nlr`, `hausdorff`, `hausdorff95` (label metrics, or `labels` for all of them); `mse`, `rmse`, `mrse`, `mad`, `psnr`, `ssim` (image metrics, or `images` for all of them). The two families can be mixed. [See above](#description). | *required*                      |
| `-s`, `--average`  | `AVERAGE...`           | Summary scores to add: `micro`, `macro`, `weighted`. [See above](#averaging).                     | none                            |
| `-x`, `--exclude`  | `LABEL...`             | Labels that are never scored as foreground. Use `none` to exclude nothing. Label metrics only.    | `0`                             |
| `-o`, `--output`   | `FILE`                 | Name or template of the JSON file. Only one value is accepted.                                    | `{dir}{sep}{base}.metrics.json` |
| `-a`, `--append`   | `[BOOL]`               | Merge the scores, metric by metric, into the output file if it exists, instead of overwriting it. [See above](#output). | off                             |

The output template accepts `{dir}` (directory of the first prediction, `.` if none),
`{base}` (its name without extension; `.nii.gz` counts as one extension), `{ext}`
(its extension) and `{sep}` (the path separator). See
[Output filenames](../../conventions.md#output-filenames).

### Image metric parameters

| Option           | Value               | Description                                                                                                | Default                                  |
| ---------------- | ------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `--range`        | `[[MIN=0] MAX]`     | Dynamic range of the data, used by `psnr` and `ssim`. A single value is taken as `MAX`, with `MIN` equal to 0. [See above](#dynamic-range). | min and max of the reference |
| `--ssim-fwhm`    | `FWHM... [UNIT]` or `full` | Window size for `ssim`, one or more numbers and an optional unit, `vox` (default) or `mm`; or `full` for global statistics. [See above](#ssim). | `3`                                      |
| `--ssim-gauss`   | `[BOOL]`            | Use a Gaussian window with the given FWHM for `ssim`, instead of a sliding square window.                  | off                                      |

### Hardware

| Option    | Value  | Description                                                                                      | Default |
| --------- | ------ | ------------------------------------------------------------------------------------------------ | ------- |
| `--cpu`   | `[N]`  | Compute on the CPU. `N` is the number of threads; 0 (the default) keeps the PyTorch default.| CPU     |
| `--gpu`   | `[ID]` | Compute on a GPU; `ID` is the device index. Falls back to the CPU, with a warning, if CUDA is not available. |         |

See [Hardware](../../conventions.md#hardware) for the general behaviour of device options.

## Known issues

- `-s weighted` is ignored by the image metrics and by the Hausdorff distances.
- `mad` is the median absolute *error*, not the median absolute deviation that
  its name and the built-in help suggest.
- Only the first prediction file names the output, so a call with several
  prediction files writes a single JSON file, and `--append` has no per-file key
  to tell their results apart.

## Examples

Compute the Dice coefficient of each label of a segmentation against a manual
reference. The scores go to `seg.metrics.json`, and label 0 is skipped:

```shell
nitorch metrics -p seg.nii.gz -r manual_seg.nii.gz -m dice
```

Add the Jaccard index and the 95th-percentile Hausdorff distance, with the macro
and micro averages:

```shell
nitorch metrics -p seg.nii.gz -r manual_seg.nii.gz \
    -m dice jaccard hausdorff95 -s macro micro
```

Treat both 0 and 4 as labels that should not be scored:

```shell
nitorch metrics -p seg.nii.gz -r manual_seg.nii.gz -m dice -x 0 4
```

Score every label, including the background, and choose the output file:

```shell
nitorch metrics -p seg.nii.gz -r manual_seg.nii.gz -m dice tpr ppv -x none \
    -o results/seg_scores.json
```

Evaluate a denoised image against a clean one, with intensities in a range of
0 to 1000:

```shell
nitorch metrics -p denoised.nii.gz -r clean.nii.gz -m mse rmse psnr --range 1000
```

Evaluate a two-contrast result, giving each contrast its own file, and add the
mean over the contrasts:

```shell
nitorch metrics -p pred_T1w.nii.gz pred_T2w.nii.gz \
                -r ref_T1w.nii.gz ref_T2w.nii.gz \
                -m mse mrse -s macro
```

Compute SSIM with a Gaussian window of 5 mm FWHM, and with global statistics,
using the range of the reference:

```shell
nitorch metrics -p denoised.nii.gz -r clean.nii.gz -m ssim --ssim-fwhm 5 mm --ssim-gauss
nitorch metrics -p denoised.nii.gz -r clean.nii.gz -m ssim --ssim-fwhm full
```

Compute all the label metrics, then add an image metric to the same file:

```shell
nitorch metrics -p seg.nii.gz -r manual_seg.nii.gz -m labels -s macro -o scores.json
nitorch metrics -p seg.nii.gz -r manual_seg.nii.gz -m mse -a -o scores.json
```

## See also

- [`info`](info.md) — check the shape and orientation of the images you compare.
- [`smooth`](smooth.md) — blur an image.
