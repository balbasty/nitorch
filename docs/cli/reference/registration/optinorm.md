---
icon: octicons/terminal-24
---

# optinorm

Unbiased, groupwise alignment ("normalisation") of several scans of the same
subject, for example sessions acquired over time or different contrasts
acquired in one session. Use it when you have three or more images of one
brain and want them all in a common space without privileging any one of
them.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch optinorm INPUT INPUT... [-o DIR] [-s STEP...] [--both] [--cleanup]
                     [--loss NAME] [--affine {rigid,similitude,affine}]
                     [--affine-levels LEVELS] [--nonlin-levels LEVELS]
                     [-a VAL] [-m VAL] [-b VAL] [-l VAL...]
                     [-i ORDER] [--no-clip]
                     [-v [LEVEL]] [--cpu [THREADS] | --gpu [ID]]
    ```

## Description

The usual way to align several scans is to choose one as a reference and
register all the others to it. This is simple, but biased: the reference is
never resampled and its anatomy is imposed on the group, which matters when
you compare scans quantitatively, for example to measure longitudinal change.

`optinorm` avoids the choice. It registers **every ordered pair** of
images, and then finds, for each image, the transformation to a mean space
that is, in the least-squares sense, most consistent with all the pairwise
transformations. No image plays a special role. The same procedure is used
twice, first with affine transformations and then with nonlinear (stationary
velocity field) transformations, and the images are finally resampled into
the mean space.

Inputs are given first, then options. At least two images are required,
and their filenames can have any extension NITorch reads. With *N* images, the
command runs *N*(*N* − 1) pairwise registrations in each of the two
registration stages, so the cost grows quickly with *N*; it is intended for
a handful of scans.

The command is a driver that chains other NITorch commands. It performs
six steps, which you can run in full or selectively:

| Step        | What it does                                                                                                         | Command it corresponds to                                    |
| ----------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `aff`       | Pairwise affine registrations of all ordered pairs of images, rigid by default, with progressive initialisation.      | [`register`](register.md)                                    |
| `affopt`    | Optimal "mean-to-image" affine of each image, computed from the pairwise matrices. Applies it to each header.         | [`affopt`](affopt.md), [`reslice`](../spatial/reslice.md)    |
| `meanspace` | Mean space (voxel grid and orientation) of the aligned images.                                                        | [`meanspace`](../spatial/meanspace.md)                       |
| `svf`       | Pairwise nonlinear registrations (stationary velocity fields) of all ordered pairs, after aligning each image with its optimal affine. | [`register`](register.md)                                    |
| `svfopt`    | Optimal "mean-to-image" velocity field of each image, computed from the pairwise fields.                              | [`vopt`](vopt.md)                                            |
| `norm`      | Resample every image into the mean space with its optimal affine and velocity field.                                  | [`reslice`](../spatial/reslice.md)                           |

Each step reads the outputs of the previous steps from the output directory.
This makes a run **resumable**: if it stops or you want to change the
settings of the nonlinear stage, rerun it with `-s` listing only the
steps that remain. Before a step starts, the files it needs are checked, and
the command stops with a message if any is missing.

By default, the nonlinear stage weights the pairwise velocity fields with
the Hessian of the matching term (written by `register`), which gives more
influence to regions where a registration was well constrained. With `--both`
the unweighted combination is computed as well.

All of the intermediate results are kept, unless `--cleanup` is given. The
affine and nonlinear pairwise registrations make up most of the disk usage.

### Output layout

Everything is written under the output directory `DIR` (`norm` by default).
Indices *i* and *j* are positions in the list of inputs, starting at 1.
`{base}` and `{ext}` are the name and extension of input file *i*. If two
inputs have the same name, `{base}` is prefixed by the position of the image.

| File                                         | Content                                                                         |
| -------------------------------------------- | ------------------------------------------------------------------------------- |
| `aff/{i}_{j}/{affine}.lta`                   | Affine from image *j* (as fixed) to image *i* (as moving), with `{affine}` the model name. |
| `svf/{i}_{j}/svf.nii.gz`                     | Velocity field from image *j* (fixed) to image *i* (moving).                   |
| `svf/{i}_{j}/hessian.nii.gz`                 | Hessian of the matching term of this nonlinear registration.                    |
| `opt/aff/{affine}_optim_{i}.lta`             | Optimal mean-to-image affine of image *i*.                                      |
| `opt/aff/{affine}_optim_{i}_inv.lta`         | Its inverse.                                                                    |
| `opt/{base}.aligned{ext}`                    | Image *i* aligned to the mean (orientation matrix changed, no resampling).      |
| `opt/meanspace.nii.gz`                       | The mean space.                                                                 |
| `opt/{base}.aligned.resliced{ext}`           | Image *i* resampled into the mean space after the affine alignment only.        |
| `opt/svf/weighted/svf_optim_weighted_{i}.nii.gz`     | Optimal mean-to-image velocity field of image *i*, Hessian-weighted.    |
| `opt/svf/unweighted/svf_optim_unweighted_{i}.nii.gz` | The same without weights (with `--both`).                                |
| `opt/{base}.norm{ext}`                       | Image *i* resampled into the mean space with the affine and the velocity field. |

The optimal transformations of image *i* can be used to bring any other
image in its native space, such as a segmentation, into the mean space with
[`reslice`](../spatial/reslice.md), exactly as the `norm` step does. See the
last example.

## Arguments

| Name    | Description                                                                          |
| ------- | ------------------------------------------------------------------------------------ |
| `INPUT` | Two or more images of the same subject. They must come before any option.            |

## Options

### Input and output

| Option                | Value        | Description                                                                                                  | Default |
| --------------------- | ------------ | ------------------------------------------------------------------------------------------------------------ | ------- |
| `-o`, `--output`      | `DIR`        | Output directory. Created if needed.                                                                         | `norm`  |
| `-s`, `--steps`       | `STEP...`    | Steps to run, among `aff`, `affopt`, `meanspace`, `svf`, `svfopt` and `norm`. Each step reads the outputs of the earlier ones from `DIR`. | all     |
| `--both`              |              | Also compute the unweighted optimal velocity fields.                                                          | Hessian-weighted only |
| `--cleanup`           |              | Delete the pairwise registrations (the `aff` and `svf` directories) once all the steps have succeeded.        | keep everything |

### Registration

| Option                 | Value                                   | Description                                                                                                                                | Default    |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `--loss`               | `NAME`                                  | Matching term of the pairwise registrations, used in both stages. Any loss name of `register`; see [Choosing a loss](register.md#choosing-a-loss). Loss options cannot be passed. | `lcc`      |
| `--affine`             | `rigid` &vert; `similitude` &vert; `affine` | Affine model of the first stage.                                                                                                       | `rigid`    |
| `--affine-levels`      | `LEVELS`                                | Pyramid levels of the affine registrations, as one range `START:STOP[:STEP]` or an index. `:4` means levels 0 to 3.                         | `:4`       |
| `--nonlin-levels`      | `LEVELS`                                | Pyramid levels of the nonlinear registrations, in the same format. `:6` means levels 0 to 5.                                               | `:6`       |
| `-a`, `--absolute`     | `VAL`                                   | Weight of the penalty on absolute displacements.                                                                                           | `0`        |
| `-m`, `--membrane`     | `VAL`                                   | Weight of the membrane-energy penalty (first derivatives).                                                                                 | `20`       |
| `-b`, `--bending`      | `VAL`                                   | Weight of the bending-energy penalty (second derivatives).                                                                                 | `0`        |
| `-l`, `--lame`         | `VAL...`                                | Weights of the linear-elastic penalty (shears, then volume change).                                                                         | `0 0`      |

The four penalty weights have the same meaning as in the `@nonlin` group of
[`register`](register.md#nonlinear-nonlin). They apply to the pairwise
registrations and to the fit of the optimal velocity fields. The default
is a pure membrane energy, which allows large but smooth deformations.

### Resampling

| Option                     | Value    | Description                                                                         | Default |
| -------------------------- | -------- | ----------------------------------------------------------------------------------- | ------- |
| `-i`, `--interpolation`    | `ORDER`  | Interpolation order of the resampled output images (`aligned.resliced` and `norm`). | `3`     |
| `--no-clip`                |          | Do not clip the resampled values to the range of the input image. High-order interpolation can overshoot, so clipping is on by default. | clip    |

### Other

| Option                | Value        | Description                                                                              | Default |
| --------------------- | ------------ | ---------------------------------------------------------------------------------------- | ------- |
| `-v`, `--verbose`     | `[LEVEL]`    | Verbosity, passed to the pairwise registrations. `0` is quiet. With no value, `1`.        | `1`     |
| `--cpu`               | `[THREADS]`  | Run on the CPU, optionally with this number of threads.                                  | CPU     |
| `--gpu`               | `[ID]`       | Run on a GPU (the one with this index, if given).                                        | CPU     |
| `-h`, `--help`        |              | Print the built-in help.                                                                 |         |

See [Hardware](../../conventions.md#hardware) for general notes on devices.

## Examples

Normalise three sessions of the same subject. All the steps are run, and
the results are written to `norm/`; the normalised images are
`norm/opt/sub-01_ses-1_T1w.norm.nii.gz` and so on:

```shell
nitorch optinorm sub-01_ses-1_T1w.nii.gz sub-01_ses-2_T1w.nii.gz \
    sub-01_ses-3_T1w.nii.gz -o norm
```

Align images with different contrasts, using mutual information instead of
the default local correlation, on a GPU:

```shell
nitorch optinorm sub-01_T1w.nii.gz sub-01_T2w.nii.gz sub-01_FLAIR.nii.gz \
    --loss mi --gpu -o norm
```

Only align the images affinely (no nonlinear deformation), and write them
resliced in the mean space:

```shell
nitorch optinorm sub-01_T1w.nii.gz sub-01_T2w.nii.gz sub-01_FLAIR.nii.gz \
    -s aff affopt meanspace -o norm_affine
```

The result is in `norm_affine/opt/*.aligned.resliced.nii.gz`.

Continue the run above with the nonlinear stages. The outputs of the affine
stage are read from `norm_affine/`, so the affine settings must not change
(the affine model is part of the filenames), but the nonlinear settings can
be chosen now: here a stronger membrane penalty, both the weighted and the
unweighted optimal fields, and removal of the pairwise registrations at the
end:

```shell
nitorch optinorm sub-01_T1w.nii.gz sub-01_T2w.nii.gz sub-01_FLAIR.nii.gz \
    -s svf svfopt norm -m 40 --both --cleanup -o norm_affine
```

Bring a binary brain mask of the first image (`sub-01_T1w.nii.gz`) into the
mean space, using the optimal transformations of image 1. This is what the
`norm` step does for the images themselves:

```shell
nitorch reslice sub-01_T1w_mask.nii.gz \
    -l norm/opt/aff/rigid_optim_1.lta \
    -v norm/opt/svf/weighted/svf_optim_weighted_1.nii.gz \
    -t norm/opt/meanspace.nii.gz -i 0 -o sub-01_T1w_mask_mean.nii.gz
```

## See also

- [`register`](register.md) — the pairwise registration used internally.
- [`affopt`](affopt.md) and [`vopt`](vopt.md) — the steps that turn
  pairwise registrations into mean-to-image transformations, usable on their
  own with registrations obtained elsewhere.
- [`meanspace`](../spatial/meanspace.md) — computes the mean space.
- [`reslice`](../spatial/reslice.md) — resamples images with the resulting
  transformations.
