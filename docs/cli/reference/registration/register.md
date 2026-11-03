---
icon: octicons/terminal-24
---

# register

Pairwise registration of images, with an affine transformation, a nonlinear
transformation, or both. It is NITorch's general-purpose registration tool:
you choose how the images are compared (the *loss*), how they are allowed to
deform (the *transformation model*), and how the optimisation is run. Use it
when [`affreg`](affreg.md) does not give you enough control, for instance to
register several image pairs at once, to mix intensity images and label maps,
or to fit a diffeomorphic deformation.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch register [-v [LEVEL]] [-o DIR] [--cpu [THREADS] | --gpu [ID]]
                     @loss [NAME] [FACTOR...] [LOSS OPTIONS]
                         @@fix FILE... [IMAGE OPTIONS]
                         @@mov FILE... [IMAGE OPTIONS]
                     [@loss ...]
                     [@affine [NAME] [FACTOR] [AFFINE OPTIONS]
                         [@@optim [NAME] [OPTIM OPTIONS]]]
                     [@nonlin [NAME] [FACTOR] [NONLIN OPTIONS]
                         [@@optim [NAME] [OPTIM OPTIONS]]]
                     [@optim [NAME] [OPTIONS]]
                     [@pyramid [NAME] [OPTIONS]]

    nitorch register -h [LEVEL]
    ```

## Description

A registration finds the transformation that makes a **moving** image look
like a **fixed** image. In NITorch the transformation is a *model* with a
handful of parameters (an affine matrix, a dense velocity field), and
registering means choosing the parameters that minimise an objective
function made of two parts: a *matching term* (the loss) that measures how
different the images are once the moving image has been warped, and, for
nonlinear models, a *regularisation term* that penalises implausible
deformations. The minimisation is carried out coarse to fine, on a pyramid
of progressively less smoothed and less subsampled images.

`register` exposes each of these ingredients as a group of options:

| Group       | Controls                                                          |
| ----------- | ----------------------------------------------------------------- |
| `@loss`     | What is compared, and how. One or more image pairs, each with its own loss. |
| `@affine`   | The linear part of the transformation, and its optimiser.         |
| `@nonlin`   | The nonlinear part of the transformation, its regularisation and optimiser. |
| `@optim`    | How the affine and nonlinear parts are alternated.                |
| `@pyramid`  | The multi-resolution scheme.                                      |

At least one `@loss` group, with both a fixed and a moving image, is
required, and at least one of `@affine` and `@nonlin`. Everything else has a
default. A `@loss` group can be repeated, and its sub-options can be
combined freely, which is how you build a composite objective, for example
mutual information between two T1-weighted and FLAIR scans plus a Dice term
between two label maps.

When the command succeeds it writes the transformations (an affine matrix in
LTA format and/or a velocity field in NIfTI format) and, by default, a copy
of each input image carried to the registered space. See
[Outputs](#outputs) for the files, and
[Applying the transformations](#applying-the-transformations) for how to use
them with [`reslice`](../spatial/reslice.md).

!!! note "Orientation conventions"
    Registration is performed in world (RAS) space, so images with different
    voxel sizes, fields of view and orientations can be registered without
    resampling them first. The affine matrix that `register` writes maps
    *fixed* world coordinates to *moving* world coordinates. See
    [Spatial transformations](../../conventions.md#spatial-transformations).

## The command-line grammar

Most NITorch commands take flat lists of options. `register` takes a
*nested* structure, because a registration is made of several components
that each have their own options, and some of those components can occur
several times. Learning the grammar once makes every example below easy to
read.

Two kinds of tags open a group:

- A tag starting with a single `@` (`@loss`, `@affine`, `@nonlin`, `@optim`,
  `@pyramid`) opens a **top-level group**.
- A tag starting with `@@` (`@@fix`, `@@mov`, `@@optim`) opens a **sub-group**
  that belongs to the top-level group written just before it.

Inside a group, things come in a fixed order: first the group's
**positional values** (typically a model name and a numeric factor), then its
**options** (tags starting with `-`, each followed by any number of values),
then its **sub-groups**. A group ends as soon as the next word is not
something it recognises, and control returns to the enclosing group.

Here is an annotated example. It registers a FLAIR image to a T1-weighted
image with mutual information, using a rigid plus diffeomorphic model, and
adds a Dice term on two label maps to guide the deformation:

```shell
nitorch register -v -o results --gpu \
    @loss mi 1 -b 64 \
        @@fix T1w.nii.gz -o false \
        @@mov FLAIR.nii.gz -r \
    @loss dice 0.5 \
        @@fix T1w_aseg.nii.gz -l 2 41 3 42 \
        @@mov FLAIR_aseg.nii.gz -l 2 41 3 42 \
    @affine rigid \
        @@optim gn -n 100 \
    @nonlin svf 1 -b 0.2 -m 0.001 \
        @@optim gn -n 20 \
    @optim interleaved -n 5 \
    @pyramid gaussian -l 0:4
```

Reading it from top to bottom:

1. `-v -o results --gpu` are **global options**. They come first, before any
   group.
2. `@loss mi 1 -b 64` opens a **loss group**. `mi` is the loss name, `1` its
   weight in the global objective, and `-b 64` an option specific to mutual
   information (the number of histogram bins). The options of a loss must
   precede its `@@fix` and `@@mov` sub-groups.
3. `@@fix T1w.nii.gz -o false` is the **fixed image** of this loss. `-o false`
   is an option of the `@@fix` sub-group: it disables the output image that
   would otherwise be written for this file.
4. `@@mov FLAIR.nii.gz -r` is the **moving image**. `-r` asks for a copy
   resliced onto the grid of the fixed image.
5. A **second loss group**: a Dice loss with weight 0.5, applied to
   segmentations. `-l 2 41 3 42` declares that the files are label maps and
   that labels 2, 41, 3 and 42 should be used.
6. `@affine rigid` is the **affine part** of the model: a rigid
   transformation. Its `@@optim gn -n 100` sub-group selects the **optimiser**
   of the affine parameters (Gauss-Newton, at most 100 iterations). A
   `@@optim` always belongs to the group written just before it.
7. `@nonlin svf 1 -b 0.2 -m 0.001` is the **nonlinear part**: a stationary
   velocity field, with a global regularisation factor of 1, a
   bending-energy weight of 0.2 and a membrane-energy weight of 0.001. It has
   its own `@@optim`.
8. `@optim interleaved -n 5` is the **joint strategy**: how the affine and
   nonlinear updates alternate.
9. `@pyramid gaussian -l 0:4` is the **pyramid**: Gaussian smoothing, with
   levels 0 to 3.

A few rules follow from the grammar:

- Global options must be given **before the first `@` group**. Once a group
  has been opened they are no longer recognised.
- The same short tag can mean different things in different groups: `-o` is
  the output directory globally, the output image in `@@fix`/`@@mov`, the
  transformation file in `@affine`/`@nonlin`, and a momentum file in
  `@@optim`. The meaning is always given by the innermost open group.
- Each `@loss` must contain exactly one `@@fix` and one `@@mov`, in either
  order.
- `@affine`, `@nonlin`, `@optim` and `@pyramid` can each appear at most once.
  Every group except `@loss` can be written with no content at all
  (`@nonlin` alone means "a stationary velocity field with default
  settings").
- Several filenames after `@@fix` or `@@mov` are stacked into the channels
  of a single multi-channel image.
- Negative numbers are values, not tags. See
  [How options are parsed](../../conventions.md#how-options-are-parsed).

`nitorch register -h` prints a short summary of the grammar;
`-h 2` and `-h 3` add the list of accepted names and of all sub-options.
The built-in help is occasionally out of date; this page follows the code.

## Matching terms: `@loss`

```text
@loss [NAME] [FACTOR...] [-s] [-z [AXIS]] [LOSS-SPECIFIC OPTIONS]
    @@fix FILE... [IMAGE OPTIONS]
    @@mov FILE... [IMAGE OPTIONS]
```

Each `@loss` group registers one moving image to one fixed image, using
one loss. The total objective is the sum of all `@loss` groups
(weighted by their factors) plus the regularisation.

| Value         | Description                                                                                         | Default |
| ------------- | --------------------------------------------------------------------------------------------------- | ------- |
| `NAME`        | Name of the loss. See [the table below](#choosing-a-loss).                                          | `mi`    |
| `FACTOR...`   | Weight of this loss in the global objective. One value, or one value per pyramid level, starting with the finest level; the last value is repeated for the remaining (coarser) levels. | `1`     |

| Option               | Value    | Description                                                                                                                                  | Default |
| -------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `-s`, `--symmetric`  |          | Make the loss symmetric: add the same loss computed in the other direction (the fixed image warped to the moving image with the inverse transformation). The weight is split in two. |         |
| `-z`, `--slicewise`  | `[AXIS]` | Compute the loss separately in each slice along `AXIS`, and sum. Useful for stacks of slices that have different intensity scales. Without a value, the last axis is used. | off     |

### Choosing a loss

The choice of loss is the most important decision in a registration. It
should reflect how the two images are *related*: are they the same contrast,
or different? Is the relationship linear, or arbitrary? Is the intensity
bias uniform, or does it vary smoothly across the field of view? Are they
images, or segmentations?

| Name(s)               | Measures                                                                                     | Typical use                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `mi`, `nmi`           | Mutual information, estimated from a joint histogram. Normalised by default.                  | Different contrasts or modalities (T1w to FLAIR, MRI to CT, MRI to PET). The default.         |
| `mse`, `l2`           | Mean squared error, with an automatically estimated noise precision.                          | Same contrast and similar intensity profile (longitudinal scans, same-sequence repeats).       |
| `mad`, `l1`           | Median absolute deviation (L1 loss), fitted by reweighting.                                   | Same as `mse`, but robust to a limited amount of outliers such as lesions or artefacts.        |
| `tuk`, `tukey`        | Tukey's biweight, a truncated L2 loss: gross outliers get zero weight.                        | Same as `mad`, when outliers (resection cavities, large lesions) are too large for `mad`.      |
| `cc`, `ncc`           | Squared correlation coefficient (normalised cross-correlation) over the whole image.          | Same contrast, but with an unknown global gain and offset; or contrasts related linearly.      |
| `lcc`, `lncc`         | Squared correlation coefficient computed in a sliding window, then averaged.                  | Same contrast with a smooth intensity bias field; nonlinear registration where the global relationship varies in space. |
| `gmmi`, `gmm`         | Mutual-information-type measure in which the joint intensity distribution is modelled by a Gaussian mixture fitted by EM. | Different contrasts, as an alternative to `mi` that needs no histogram binning.                |
| `lgmmi`, `lgmm`       | The same, with the mixture fitted within local patches.                                       | Different contrasts with spatially varying relationship (bias field).                           |
| `emmi`                | Mutual information computed by an expectation-maximisation estimate of the joint histogram, between a discretised fixed image and a soft-quantised moving image. | Different contrasts. See [Image options](#image-options-fix-mov) for what it does to the inputs. |
| `dot`, `prod`         | Dot product between the two images.                                                           | Mostly useful as a building block; it is not invariant to the scale of the data.               |
| `ndot`                | Normalised dot product (cosine similarity between the two images).                            | Same contrast, when only the direction of the intensity vectors matters.                       |
| `sqz`, `squeezed`     | Squeezed product, weighted by a precision `-w`.                                                | Same contrast, as a weighted alternative to `mse`.                                              |
| `cat`, `cce`          | Categorical cross-entropy between two sets of probability maps (one channel per class).       | Soft segmentations, such as tissue probability maps, or label maps converted to one-hot maps with `-l`. |
| `dice`, `f1`          | One minus the Dice coefficient of each class, summed over classes.                            | Label maps and probability maps, when you care about the overlap of structures rather than voxel intensities. |

Some practical guidance:

- **Same image type, same scanner**: `mse` is simple and fast, and its
  estimated noise precision means the overall scale of the data does not
  matter. `cc` and `lcc` additionally tolerate differences in gain.
- **Different contrasts**: use `mi`. Mutual information makes no assumption
  on the relationship between intensities, only that one image tells you
  something about the other. It is the right choice for registering a T1w
  scan to a FLAIR scan, or a CT to an MRI. A global histogram
  carries little information about local structure, so for a fine nonlinear
  deformation consider `lgmmi`, or add a Dice term on segmentations.
- **Bias field**: the global losses (`mse`, `cc`, `mi`) assume the
  relationship between images is the same everywhere. When it is not, the
  local variants `lcc` and `lgmmi` are more forgiving.
- **Segmentations**: `dice` is the natural choice for label maps. With
  `-l` each label becomes a binary channel, and the moving channels are
  interpolated, so the loss is smooth. `cat` is equivalent in spirit, but
  works with soft probability maps (for example the output of a tissue
  segmentation) and penalises confident disagreement more strongly.
  Combine them with an intensity loss, in separate `@loss` groups, to get a
  semi-supervised registration.
- **Robust losses**: `mad` and `tuk` estimate their own weight maps by
  iterative reweighting, so they down-weight voxels that cannot be matched.
- **Multi-channel images**: when several files are given after `@@fix` and
  `@@mov`, the loss is computed channel by channel (channel *k* of the
  moving image against channel *k* of the fixed image) and summed. For `mi`
  this means one joint histogram per pair of channels, not one over all
  pairs.

One further name, `extra`, is accepted but is not a loss at all: an
`@loss extra` group contributes nothing to the objective, but its moving
image is warped with the final transformation and written out like any
other. It still requires a `@@fix` and `@@mov`, the former being used to
define the target grid.

### Options of specific losses

Each of these options is only accepted after the corresponding loss name,
and before `@@fix`/`@@mov`.

| Loss           | Option                  | Value              | Description                                                                                         | Default            |
| -------------- | ----------------------- | ------------------ | --------------------------------------------------------------------------------------------------- | ------------------ |
| `mi`           | `-m`, `--norm`          | `NAME`             | Normalisation of the mutual information: `studholme` (MI divided by the joint entropy), `arithmetic`, `geometric`, or `no` for plain, unnormalised MI. | `studholme`        |
| `mi`           | `-b`, `--bins`          | `N [N]`            | Number of bins of the joint histogram. A second value gives a different number of bins for the fixed image (the first is for the moving image). | `32`               |
| `mi`           | `-f`, `--fwhm`          | `VAL...`           | Full width at half maximum of the Gaussian smoothing applied to the joint histogram, in bins.        | `0`                |
| `mi`           | `-n`, `--order`         | `N [N]`            | Order of the B-spline (Parzen window) that spreads each voxel over neighbouring bins.                | `3`                |
| `mse`          | `-w`, `--weight`        | `VAL` &vert; `auto` | Precision (inverse variance) of the noise. With `auto`, it is estimated from the data.              | `auto`             |
| `mad`, `tuk`   | `-w`, `--weight`        | `VAL` &vert; `auto` | Precision of the residuals. With `auto`, it is estimated from the data.                             | `auto`             |
| `sqz`          | `-w`, `--weight`        | `VAL`              | Precision used to weight the product. There is no automatic weighting: `auto` is an error.          | `1`                |
| `mad`, `tuk`   | `-j`, `--joint`         |                    | Use a single weight map shared by all channels, rather than one per channel.                         |                    |
| `lcc`          | `-p`, `--patch`         | `VAL... [UNIT]`    | Size of the local window, one value or one per axis. The unit is `vox`, `mm` or `pct` (percentage of the image size). With the default Gaussian kernel the size is a FWHM; with a square kernel it is the full width. | `10 pct`           |
| `lcc`          | `--stride`              | `VAL...`           | Distance between consecutive windows, in voxels. Larger values are faster.                           | `1`                |
| `lcc`          | `-k`, `--kernel`        | `NAME`             | Window shape: `gauss` or `square`.                                                                   | `gauss`            |
| `gmmi`         | `-b`, `--bins`          | `N`                | Number of Gaussian components in the mixture.                                                        | `6`                |
| `gmmi`         | `-n`, `--max-iter`      | `N`                | Maximum number of EM iterations per evaluation.                                                      | `128`              |
| `lgmmi`        | `-b`, `--bins`          | `N`                | Number of Gaussian components in each local mixture.                                                 | `3`                |
| `lgmmi`        | `-n`, `--max-iter`      | `N`                | Maximum number of EM iterations.                                                                     | `128`              |
| `lgmmi`        | `-p`, `--patch`         | `VAL... [UNIT]`    | Patch size, as for `lcc`.                                                                            | `10 pct`           |
| `lgmmi`        | `--stride`              | `VAL...`           | Distance between patches, as for `lcc`.                                                              | `1`                |
| `lgmmi`        | `-k`, `--kernel`        | `NAME`             | Patch shape, as for `lcc`.                                                                           | `gauss`            |
| `dice`         | `-w`, `--weight`        | `[VAL...]`         | Weight each class. With no value, classes are weighted by their frequency in the fixed image; with values, one weight per class. | unweighted         |

!!! note "A patch size in `pct` changes with the pyramid level"
    The local window of `lcc` and `lgmmi` is recomputed at each pyramid level
    so that it covers the same fraction of the image (`pct`) or the same
    physical size (`mm`). A size in `vox` is halved each time the image is
    subsampled by two.

`--stride` has no short tag: in every loss, `-s` is the symmetric flag.
Because that flag takes no value, `lcc -s 2` is an error: the message names
the loss and the option before the stray value, and suggests `--stride`.

### Image options (`@@fix`, `@@mov`)

Each `@@fix` and `@@mov` sub-group takes one or more filenames, followed by
options that describe how to load, preprocess and write that image. The
options are the same for both. The filenames are stacked into channels, so
`@@fix T1w.nii.gz T2w.nii.gz` is a two-channel image; the files must share
the same voxel grid.

| Option                  | Value                       | Description                                                                                                                                | Default |
| ----------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `-c`, `--channels`      | `C...`                      | Channels to use, as indices or ranges `START:STOP[:STEP]` (`:` selects all).                                                                | all     |
| `-l`, `--label`         | `[LABEL...]`                | The file is a label map. It is converted to one binary channel per label. With no value, all non-zero labels are used; otherwise only those given. Required for `dice` and `cat` on label maps. | off     |
| `-m`, `--mask`          | `FILE`                      | Mask of the voxels to include in the loss. Must have the same shape as the image.                                                           | all     |
| `-x`, `--missing`       | `VAL...`                    | Voxel values to treat as missing and exclude from the loss. Non-finite values are always excluded.                                          | `0`     |
| `--no-missing`          |                             | Treat no value as missing.                                                                                                                  |         |
| `-s`, `--rescale`       | `[MN] MX`                   | Rescale intensities linearly so that the `MN`-th and `MX`-th percentiles map to 0 and 1. A single value is the upper percentile. Not applied to label maps. `-s 0` disables rescaling. | `0 95`  |
| `-f`, `--fwhm`          | `[VAL... [UNIT]]`           | Smooth the image with a Gaussian of this FWHM, once, before building the pyramid. The unit is `vox` or `mm`. Without a value, 1 voxel.       | `0`     |
| `-z`, `--pad`           | `VAL... [UNIT]`             | Pad the image on both sides before using it. The unit is `vox`, `mm` or `%`.                                                                | `0`     |
| `-d`, `--discretize`    | `[N` &vert; `auto[:N]]`     | Discretise intensities into `N` integer bins (256 if no value is given). With `auto`, the number of bins is chosen at each pyramid level from the number of voxels, with `N` (default 256) as the upper limit. | off     |
| `--mind`                | `[FWHM [RADIUS]]`           | Replace the image by MIND features (modality-independent neighbourhood descriptors), which turns a multi-modal problem into one that a same-modality loss such as `mse` can handle. Without values, `FWHM` is 1 and `RADIUS` is 2. | off     |
| `-w`, `--world`         | `FILE`                      | Orientation matrix (voxel to world) that replaces the one stored in the image.                                                              |         |
| `-a`, `--affine`        | `FILE...`                   | One or more world-to-world affine transformations to apply to the image's orientation matrix before registration. See below.                |         |
| `-b`, `--bound`         | `NAME`                      | Boundary condition when sampling outside the image, and when padding or smoothing: `dct2`, `dct1`, `dft`, `dst2`, `dst1`, `zero` or `replicate`. | `dct2`  |
| `--extrapolate`         |                             | Extrapolate out-of-bounds data instead of leaving it undefined.                                                                             |         |
| `-n`, `--order`         | `N`                         | Interpolation order used to write the output images. Registration itself always uses linear interpolation. Use `0` for label maps.          | `1`     |
| `--name`                | `NAME`                      | A name for this image, to refer to it in `@nonlin --fov`.                                                                                   | first filename |
| `-o`, `--output`        | `[PATH` &vert; `true` &vert; `false]` | Template of the output image with *minimal reslicing*. With no value or `true`, the default template is used; `false` disables the output. See [Outputs](#outputs). | `{dir}{sep}{base}.moved{ext}` |
| `-r`, `--resliced`      | `[PATH` &vert; `true` &vert; `false]` | Template of the output image resliced onto the grid of the *other* image. With no value or `true` (or `yes`), `{dir}{sep}{base}.resliced{ext}`. | off     |

A few of these deserve a comment.

**Missing values.** By default every voxel exactly equal to 0 is excluded
from the loss, in addition to NaNs and infinities. This is convenient for
skull-stripped or padded images, where the background carries no
information. If your images have meaningful zeros (for example CT in
Hounsfield units, or a quantitative map), disable it with `--no-missing`.

**Intensity scaling.** Rescaling makes losses such as `mse` or `lcc`
independent of the unit of the image. The upper percentile of 95 keeps a
few bright voxels from compressing the rest of the range. Rescaling is
turned off automatically for label maps, for `cat` and `dice`, and for
the moving image of `emmi`.

**Initial transformations (`-a`).** Each transformation $T$ replaces the
orientation matrix $A$ of the image with $T^{-1}A$. This is exactly what
`nitorch reslice FILE -l T` does when no target is given: the voxel data
are untouched and only the header changes. It is a way to feed the result
of a previous registration as a starting point, or to put several images in
a common space before a nonlinear registration. The output images are
computed from the modified orientation matrix too.

**Labels.** With `-l`, the image is first expanded to one binary channel per
label, then resampled with linear interpolation, so overlaps vary smoothly
and can be differentiated. Specifying the labels explicitly (`-l 2 41 3
42`) keeps the number of channels, and therefore memory use, under
control; with many labels, the default of using every non-zero label can be
costly.

**`emmi`.** This loss changes the preprocessing of its inputs. The fixed image
is discretised (into 256 bins unless `-d` says otherwise), and the moving
image is soft-quantised into 16 bins (also adjustable with `-d`, set on
`@@mov`). Intensity rescaling is disabled on the moving image, and no value
is considered missing. Only single-channel moving images are supported.

## Transformation models

A registration needs at least one of `@affine` and `@nonlin`. When both are
given, the total transformation is a nonlinear deformation in the space
between the two images, composed with the affine.

### Affine: `@affine`

```text
@affine [NAME] [FACTOR] [AFFINE OPTIONS]
    [@@optim [NAME] [OPTIM OPTIONS]]
```

The affine transformation is parametrised by its logarithm in a Lie algebra,
so that composition and inversion are exact and the parameters of different
types (translation, rotation, zoom, shear) are on a comparable footing.
`NAME` chooses the number of degrees of freedom.

| Name                | Alias | Parameters (3D) | Description                                              |
| ------------------- | ----- | --------------- | -------------------------------------------------------- |
| `translation`       | `t`   | 3               | Translations only.                                       |
| `rotation`          | `o`   | 3               | Rotations only.                                          |
| `rigid`             | `r`   | 6               | Translations and rotations. The default.                 |
| `similitude`        | `s`   | 7               | Rigid plus one isotropic scaling factor.                 |
| `affine`            | `a`   | 12              | Translations, rotations, anisotropic zooms and shears.   |

`FACTOR` is the precision of a Gaussian prior centred on the identity: it
adds the penalty $\tfrac12\lambda\lVert q\rVert^2$ to the objective, where
$q$ are the Lie parameters of the transformation and $\lambda$ is `FACTOR`.
It is `0` by default, meaning no prior. A small positive value keeps a
poorly constrained affine registration from drifting.

| Option                       | Value         | Description                                                                                                                                  | Default                  |
| ---------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `-p`, `--position`           | `sym` &vert; `mov` &vert; `fix` | Where the affine sits in the forward (fixed-to-moving) model. See [below](#position-of-the-affine). | `sym`                    |
| `-g`, `--progressive`        | `[BOOL]`      | Initialise by progressively freeing degrees of freedom: rigid, then similitude, then affine (up to the chosen model), each on the whole pyramid. | off                      |
| `-i`, `--input`, `--init`    | `FILE` &vert; `fov` &vert; `com` | Initial transformation: a file, or `fov` to align the centres of the fields of view, or `com` to align the centres of mass. | identity                 |
| `-o`, `--output`             | `[PATH` &vert; `true` &vert; `false]` | Where to write the affine matrix (an LTA file). `{dir}` and `{name}` are available. With no value, `true`, `yes` or `1`, the default path is used; `false`, `no` or `0` writes nothing. | `{dir}/{name}.lta`       |
| `-t`, `--translation-penalty`| `F`           | Relative weight of the prior on translations (multiplies `FACTOR`).                                                                          | `1`                      |
| `-r`, `--rotation-penalty`   | `F`           | Relative weight of the prior on rotations.                                                                                                   | `1`                      |
| `-z`, `--zoom-penalty`       | `F`           | Relative weight of the prior on zooms.                                                                                                       | `1`                      |
| `-s`, `--shear-penalty`      | `F`           | Relative weight of the prior on shears.                                                                                                      | `1`                      |
| `-2d`                        | `[AXIS]`      | Restrict the transformation to a 2D transformation in the plane normal to `AXIS` (an index 0 to 2, or `LR`, `AP`, `IS`). Without a value, axis 2. | 3D                       |

#### Position of the affine

The forward model of `register` always maps coordinates of the fixed image to
coordinates of the moving image. When the registration has both an affine and
a nonlinear component, the position says *where the affine sits* in that
composition. With $A$ the affine (the matrix written to `{name}.lta`, which is
always the full fixed-to-moving transformation) and $\varphi$ the nonlinear
transformation, the model is:

| Position        | Forward model                                                   | The nonlinear field acts...        |
| --------------- | --------------------------------------------------------------- | ---------------------------------- |
| `mov`           | $x \mapsto A\bigl(\varphi(x)\bigr)$                            | on fixed-image (world) coordinates |
| `fix`           | $x \mapsto \varphi\bigl(A(x)\bigr)$                            | in the frame of the moving image   |
| `sym` (default) | $x \mapsto A^{1/2}\bigl(\varphi(A^{1/2}(x))\bigr)$              | in a frame halfway between the two |

The position does **not** change where the loss is computed: the moving image
is always pulled through the forward model onto the voxel grid of the fixed
image, and the loss is evaluated there. The only exception is a symmetric loss
(`-s` on a loss, see above), which also warps the fixed image into the moving
space through the inverse model and adds the same loss there. This is
independent of the position.

- **Affine-only registration.** The three positions describe the same
  transformation ($A^{1/2}A^{1/2}=A$), so they have the same optimum. They
  only differ in the parameterisation and in the headers of the `.moved`
  outputs (see below).
- **With a nonlinear component.** The position changes the model, because it
  changes the frame in which the nonlinear field lives, and hence how it is
  regularised. The results can therefore differ. It also determines the order
  in which the transformations must be chained when applying them with
  `reslice` (see [Applying the transformations](#applying-the-transformations)).
- `sym` makes the *model* symmetric: its inverse has the same structure, with
  the roles of the fixed and moving images swapped. Combined with a symmetric
  loss (`-s`), it gives a fully symmetric registration. It is the best choice
  when the two images play comparable roles.

The position also determines the orientation matrices written to the
`{base}.moved{ext}` files (this concerns the output headers only, not the
loss). With `mov`, the orientation matrix of the moving image is adjusted by
the affine and the fixed image keeps its own. With `fix`, it is the other way
around. With `sym`, both images receive half of the affine, and end up in the
halfway space. Full reslicing (`-r`) is not affected by the position: the
output is always on the grid of the other image.

### Nonlinear: `@nonlin`

```text
@nonlin [NAME] [FACTOR] [NONLIN OPTIONS]
    [@@optim [NAME] [OPTIM OPTIONS]]
```

A nonlinear transformation is a dense vector field, defined on a regular
lattice (the *mean space* of the registered images), whose regularity is
controlled by a penalty. `NAME` selects what the field represents.

| Name        | Alias | Model                                                                                                                                                      |
| ----------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `svf`       | `v`   | A **stationary velocity field**. The deformation is its exponential, computed by scaling and squaring, so it is smooth and invertible (diffeomorphic). The default. |
| `shoot`     | `g`   | An **initial velocity** that is integrated by **geodesic shooting** under the regularisation metric. Diffeomorphic by construction, and the field is a compact description of the whole deformation path. More expensive than `svf`. |
| `smalldef`  | `d`   | A **dense displacement field**, used as is. No diffeomorphism guarantee. Cheaper, and adequate for small deformations or when you only need a forward transformation. |

`FACTOR` (default `1`) is a global multiplier of the regularisation, to
which the four penalties below contribute. Larger values give smoother
deformations. The penalty is normalised by the number of voxels of the field,
so factors are comparable across image sizes.

The regularisation is a weighted sum of four energies of the field,
each with its own weight (see the options below):

$$
\lambda\,\Big(
w_a\,\lVert v\rVert^2
+ w_m\,\lVert\nabla v\rVert^2
+ w_b\,\lVert\nabla^2 v\rVert^2
+ \text{linear-elastic}(w_{\mu},w_{\lambda})
\Big)
$$

*Absolute* shrinks the field towards zero, *membrane* penalises its first
derivatives (it discourages stretching), *bending* its second derivatives
(it favours smooth, locally affine deformations), and *linear-elastic*
penalises shears and local volume changes separately, with the two
weights of the `--lame` option.

| Option                      | Value               | Description                                                                                                                         | Default            |
| --------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `-a`, `--absolute`          | `VAL`               | Weight of the absolute (zeroth-order) penalty.                                                                                      | `1e-4`             |
| `-m`, `--membrane`          | `VAL`               | Weight of the membrane (first-order) penalty.                                                                                       | `1e-3`             |
| `-b`, `--bending`           | `VAL`               | Weight of the bending (second-order) penalty.                                                                                       | `0.2`              |
| `-l`, `--lame`              | `SHEAR [DIV]`       | Weights of the two terms of the linear-elastic penalty: shears, and local volume change (divergence). A single value is used for both. | `0.05 0.2`         |
| `-s`, `--steps`             | `N`                 | Number of integration steps for `svf` and `shoot`.                                                                                  | `8`                |
| `-v`, `--voxel-size`        | `VAL... [UNIT]`     | Voxel size of the lattice of the field: one value or one per axis. The unit is `mm` or `%` (percentage of the mean voxel size of the inputs). | `100 %`            |
| `-p`, `--pad`               | `VAL... [UNIT]`     | Pad the field of view of the lattice, by a percentage (`%`, the default unit) or a distance (`mm`).                                  | `0 %`              |
| `-f`, `--fov`               | `NAME...`           | Names of the images (see `--name` of `@@fix`/`@@mov`) whose fields of view define the lattice. Images without a `--name` are named after their first filename. | all images         |
| `-i`, `--input`, `--init`   | `FILE`              | Initial field, or a scalar image whose grid defines the lattice of a zero-initialised field.                                         | zeros on the mean space |
| `-o`, `--output`            | `[PATH` &vert; `true` &vert; `false]` | Where to write the field. `{dir}` and `{name}` are available. With no value, `true`, `yes` or `1`, the default path is used; `false`, `no` or `0` writes nothing. | `{dir}/{name}.nii.gz` |
| `-h`, `--hessian`           | `[PATH]`            | Also write the Hessian of the matching term at the solution (a map of local confidence). With no path, `{dir}/{name}_hessian.nii.gz`. Used to weight [`vopt`](vopt.md). | not written        |
| `-2d`                       | `[AXIS]`            | Restrict the deformation to the plane normal to `AXIS`. Without a value, axis 2.                                                     | 3D                 |

The default lattice has the mean voxel size of the images and covers all
their fields of view, so a registration between a 1 mm and a 2 mm image
produces a field at about 1.5 mm.

## Optimisation

Each of `@affine` and `@nonlin` can contain an `@@optim` sub-group that
selects the algorithm used to update its parameters, and the joint `@optim`
group controls how the two alternate.

### The solver: `@@optim`

```text
@@optim [NAME] [OPTIONS]
```

By default, the solver is chosen from the loss. All the losses of `register`
provide a Gauss-Newton Hessian, so the default is `gn` for both the affine
and the nonlinear part. Gauss-Newton is a second-order method: it converges
in few iterations and needs no learning rate. The other solvers are
first-order methods, which are more general but need more iterations and a
learning rate.

| Name                         | Description                                                                             |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| `gn`, `gauss-newton`         | Gauss-Newton with Levenberg-Marquardt damping. The default.                              |
| `gd`, `gradient-descent`     | Plain gradient descent.                                                                  |
| `cg`, `conjugate-gradient`   | Nonlinear conjugate gradient.                                                            |
| `mom`, `momentum`            | Gradient descent with momentum.                                                          |
| `nes`, `nesterov`            | Gradient descent with Nesterov momentum.                                                 |
| `ogm`, `optimized-gradient`  | Optimised gradient method of Kim and Fessler.                                            |
| `lbfgs`                      | Limited-memory BFGS.                                                                     |
| `pow`, `powell`              | Powell's derivative-free method. For `@affine` only; it is what [`affreg`](affreg.md) uses with mutual information. |
| `none`                       | Do not optimise this transformation: keep it at its initial value (see `-i`).            |

Common options:

| Option                      | Value            | Description                                                                                                    | Default                                     |
| --------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `-l`, `--lr`                | `VAL`            | Learning rate (step size).                                                                                     | `1` (second order), `1e-3` (first order)    |
| `-n`, `--max-iter`          | `N`              | Maximum number of iterations, at each pyramid level.                                                           | affine: 50, nonlinear: 10 (interleaved); affine: 100, nonlinear: 50 (sequential) |
| `-t`, `--tolerance`         | `VAL`            | Tolerance of the stopping criterion.                                                                           | `1e-4`                                      |
| `-c`, `--crit`              | `diff` &vert; `gain` | Stopping criterion: absolute change of the objective (`diff`) or change relative to the total gain so far (`gain`). | `diff`                                      |
| `-s`, `--line-search`       | `N` &vert; `wolfe` | A backtracking line search with at most `N` steps, or a line search until the Wolfe conditions hold.          | `wolfe`                                     |

The default number of iterations depends on the joint strategy (see below),
and applies even if only one of `@affine` and `@nonlin` is used. A nonlinear
registration on its own therefore runs 10 iterations per pyramid level
unless you give `-n` or choose `@optim sequential`.

Options of specific solvers:

| Solver   | Option                 | Value                | Description                                                                                                       | Default            |
| -------- | ---------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------ |
| `gn`     | `-m`, `--marquardt`    | `VAL` &vert; `auto` &vert; `false` | Levenberg-Marquardt damping: a value, `auto`, or `false` for none.                                  | `auto`             |
| `gn`     | `-q`, `--solver`       | `cg` &vert; `relax`  | Linear solver for the nonlinear update: conjugate gradient or relaxation. `@nonlin` only.                         | `cg`               |
| `gn`     | `-j`, `--sub-iter`     | `N`                  | Iterations of the linear solver. `@nonlin` only.                                                                  | 2 per FMG cycle if FMG is used, else 16 |
| `gn`     | `-g`, `--fmg`          | `N`                  | Number of full multigrid cycles used to precondition the solver (0 disables FMG). `@nonlin` only.                  | `2`                |
| `cg`     | `-b`, `--beta`         | `pr` &vert; `fr` &vert; `hs` &vert; `dy` | Update rule: Polak-Ribiere, Fletcher-Reeves, Hestenes-Stiefel, or Dai-Yuan (the long names also work).        | `pr`               |
| `mom`    | `-m`, `--momentum`     | `VAL`                | Momentum factor.                                                                                                  | `0.9`              |
| `nes`    | `-m`, `--momentum`     | `VAL`                | Momentum factor; `0` chooses it automatically.                                                                    | `0` (auto)         |
| `nes`    | `-r`, `--restart`      | `BOOL`               | Restart the momentum automatically when the objective increases.                                                  | `true`             |
| `nes`    | `-o`, `--output`       | `[PATH` &vert; `true` &vert; `false]` | Template of the final momentum: a map for `@nonlin`, a JSON file for `@affine`. With no value, `true` or `yes`, the default path is used (`{dir}{sep}momentum.nii.gz`, or `{dir}{sep}affine_momentum.json` for `@affine`); `false` or `no` writes nothing. | not written        |
| `ogm`    | `-m`, `--momentum`     | `VAL`                | Momentum factor; `0` chooses it automatically.                                                                    | `0` (auto)         |
| `ogm`    | `-x`, `--relax`        | `VAL`                | Relaxation factor; `0` chooses it automatically.                                                                  | `0` (auto)         |
| `ogm`    | `-r`, `--restart`      | `BOOL`               | Restart automatically.                                                                                            | `true`             |
| `ogm`    | `-o`, `--output`       | `[PATH` &vert; `true` &vert; `false]` | Template of the final momentum, as for `nes`.                                                      | not written        |
| `lbfgs`  | `-h`, `--history`      | `N`                  | Number of past steps kept to approximate the Hessian.                                                              | `100`              |

Gauss-Newton is the right choice in most cases. Powell's method is the
default of [`affreg`](affreg.md) when it is used with mutual information.

### The joint strategy: `@optim`

```text
@optim [NAME] [OPTIONS]
```

When an `@affine` and a `@nonlin` are both present, the affine part is first
fitted alone, from coarse to fine through the pyramid, and then the
affine and nonlinear parts are optimised together, again from coarse to fine.
`@optim` selects how the two are optimised together.

| Name                     | Description                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `interleaved`, `i`       | Alternate: run the affine optimiser, then the nonlinear optimiser, and repeat. The default.                                     |
| `sequential`, `s`        | Fit the affine part to convergence, then the nonlinear part to convergence, once (or the number of times `-n` says).            |

| Option                | Value | Description                                                                                  | Default                                  |
| --------------------- | ----- | -------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `-n`, `--max-iter`    | `N`   | Maximum number of alternations (interleaved) or of full affine-then-nonlinear passes (sequential). | 5 (interleaved), 1 (sequential)          |
| `-t`, `--tolerance`   | `VAL` | Tolerance for stopping the alternation early.                                                | `1e-5` (interleaved), `0` (sequential)   |

## Multi-resolution: `@pyramid`

```text
@pyramid [NAME] [OPTIONS]
```

Registration is run on a pyramid of images, in which each level has half
the resolution of the one before. Coarse levels are cheap and have a wide
basin of attraction, so they bring the images close enough that the fine
levels only need to refine the result. The pyramid is used even if you do
not write `@pyramid`: it then has three Gaussian levels.

Level 0 is the full-resolution image; level *n* is subsampled by $2^n$.
When the images in the registration have different resolutions, the levels
are matched across images by voxel size, so a 1 mm image and a 2 mm image
are compared at their closest common resolution at each level.

| Name                | Alias | Description                                                    |
| ------------------- | ----- | -------------------------------------------------------------- |
| `gaussian`          | `g`   | Smooth with a Gaussian, then subsample. The default.           |
| `average`           | `a`   | Average over windows.                                          |
| `median`            | `m`   | Median over windows.                                           |
| `stride`            | `s`   | Keep every second voxel, without smoothing.                    |
| `none`              | `n`   | No smoothing or subsampling; only valid if a single level, `-l 0`, is requested. |

| Option                    | Value                     | Description                                                                                                                              | Default |
| ------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `-l`, `--levels`          | `LEVEL...`                | Levels to use, as indices or ranges `START:STOP[:STEP]` (`3:` means from level 3 to the coarsest valid level, `:3` levels 0, 1 and 2). Levels are processed from the coarsest to the finest. | `0:3`   |
| `--min-size`              | `N`                       | Do not use levels with fewer than `N` voxels along some axis.                                                                            |         |
| `--max-size`              | `N`                       | Do not use levels with more than `N` voxels along some axis. This skips the finest levels.                                               |         |
| `--min-vx`                | `VAL`                     | Do not use levels with a voxel size under `VAL` mm along some axis.                                                                      |         |
| `--max-vx`                | `VAL`                     | Do not use levels with a voxel size over `VAL` mm along some axis.                                                                       |         |
| `-c`, `--concurrent`      | `[BOOL]`                  | Optimise all levels at once, as one objective that sums the levels, rather than one after the other.                                    | `false` |
| `-m`, `--reg-modulation`  | `MOD`                     | Multiply the nonlinear regularisation by `MOD` each time the voxel size of the field doubles across levels. `1` uses the same value at all levels. The finest level always uses the value you set. | `1`     |

The level numbers in `--levels` count *valid* levels: if you give `--max-size`
or `--min-vx`, level 0 is the finest level that satisfies the constraints.
To spend more effort at fine scales, use more levels (`-l 0:5` for a large
image); to save time on a rough affine alignment, drop the finest ones
(`-l 2:5`).

## Outputs

Unless told otherwise, `register` writes the following files. Templates use
the fields `{dir}` (a directory), `{sep}` (the path separator), `{base}`
(the input filename without extension), `{ext}` (its extension, including
`.nii.gz`) and `{name}` (the name of the model); see
[Output filenames](../../conventions.md#output-filenames).

| File                                | Written when                          | Content                                                                                                           |
| ----------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `{dir}/{name}.lta`                  | `@affine` is used                     | The affine matrix, in RAS-to-RAS convention. For example `rigid.lta`, `affine.lta`. It maps fixed to moving world coordinates. Always the *full* transformation, whatever the position. |
| `{dir}/{name}.nii.gz`               | `@nonlin` is used                     | The field, in voxels of its own lattice: a velocity for `svf` (`svf.nii.gz`) and `shoot`, a displacement for `smalldef`. |
| `{dir}/{name}.json`                 | `@nonlin shoot` only                  | Regularisation (penalty and factor) and the number of integration `steps` and `fast` flag used during registration, needed to integrate the shooting equations the same way (`shoot.json`). Read by [`reslice`](../spatial/reslice.md#velocity-fields). |
| `{dir}/{name}_hessian.nii.gz`       | `@nonlin -h`                          | Hessian of the matching term.                                                                                      |
| `{dir}/momentum.nii.gz`, `affine_momentum.json` | `@@optim nes`/`ogm` with `-o` | Final momentum of the optimiser.                                                                              |
| `{dir}/{base}.moved{ext}`           | for each image, unless `-o false`     | The image carried to the registered space, with *minimal reslicing*.                                              |
| `{dir}/{base}.resliced{ext}`        | for each image with `-r`              | The image resampled onto the voxel grid of the other image.                                                       |

For the transformations, `{dir}` is the output directory given by `-o` or,
if there is none, the directory of the first fixed image. For images it is
the output directory, or else the directory of the image itself. Note that
**both** the fixed and moving images are written as `.moved` by default; give
`-o false` to `@@fix` or `@@mov` to avoid it. Output templates are checked
before the fit starts, so a malformed template is reported immediately
rather than after the optimisation has run.

**Minimal reslicing** means that the image keeps its own voxel grid
whenever possible. With a purely affine registration, the voxel data are
copied unchanged and only the orientation matrix in the header is modified;
the result can be viewed on top of the other image, without any
interpolation. If a nonlinear transformation is present, the image is
resampled on its own grid. The position of the affine only decides which
orientation matrices are modified: the moving image's with `mov`, the fixed
image's with `fix`, and half of the affine on each with `sym`. **Full
reslicing** (`-r`), in contrast, always produces an image on the grid of the
other image: the
moving image on the fixed grid, and the fixed image on the moving grid.

## Applying the transformations

To resample other images (a segmentation, a different contrast, a mask)
with the result of a registration, use [`reslice`](../spatial/reslice.md).
For an affine-only registration, whatever the position:

```shell
# moving space -> fixed space
nitorch reslice aseg.nii.gz -l rigid.lta -t T1w.nii.gz -i 0 -o aseg_in_T1w.nii.gz
# fixed space -> moving space
nitorch reslice T1w.nii.gz -il rigid.lta -t FLAIR.nii.gz
```

(`-i 0` selects nearest-neighbour interpolation, for label maps.)

When a nonlinear transformation is present, the affine is no longer
sufficient and the order in which transformations are chained depends on the
position of the affine. In the following table, `aff` is the `.lta` file,
and `vel` the field. For `smalldef`, replace `-v vel` with `-d disp`; for
`shoot`, the JSON file next to the field is picked up automatically.

| Position | Moving to fixed                  | Fixed to moving                     |
| -------- | -------------------------------- | ----------------------------------- |
| `sym`    | `-l2 aff -v vel -l2 aff`         | `-il2 aff -iv vel -il2 aff`         |
| `mov`    | `-l aff -v vel`                  | `-iv vel -il aff`                   |
| `fix`    | `-v vel -l aff`                  | `-il aff -iv vel`                   |

Here `-l2` applies the square root of the affine, which is what the
symmetric model uses on each side of the deformation. For example, for a
default registration:

```shell
nitorch reslice FLAIR.nii.gz -l2 affine.lta -v svf.nii.gz -l2 affine.lta \
    -t T1w.nii.gz -o FLAIR_in_T1w.nii.gz
```

!!! tip "Let `register` do it"
    If you only need the moving image on the grid of the fixed image, ask for
    it with `-r` at registration time and skip the extra step. The recipes
    above are for other images that were not part of the registration.

A bare `--` is not supported, since `register` has no positional inputs: it is an error. Write a file name that starts with a dash as `./-name`.

## Global options

These must precede the first group.

| Option                    | Value          | Description                                                                                                                | Default                                  |
| ------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `-o`, `--output-dir`      | `DIR`          | Directory where all outputs are written. It is created if needed.                                                          | see [Outputs](#outputs)                  |
| `-v`, `--verbose`         | `[LEVEL]`      | Verbosity. `0` is silent, `1` prints a progress line per iteration, `2` also prints the objects being optimised and each line-search evaluation, `3` and above also plot the registration live (requires `matplotlib`). With no value, `1`. | `0`                                      |
| `-r`, `--framerate`       | `HZ`           | Refresh rate of the live plot.                                                                                             | `1`                                      |
| `--cpu`                   | `[THREADS]`    | Run on the CPU, with an optional number of threads.                                                                        | CPU                                      |
| `--gpu`                   | `[ID]`         | Run on a GPU (the one with this index, if given). Falls back to the CPU with a warning when CUDA is not available.          | CPU                                      |
| `-d`, `--dim`             | `DIM`          | Dimensionality of the registration. Only `3` is accepted; any other value is an error, raised before any file is read. A 2D image is registered as a single-slice volume, using `-2d` in `@affine`/`@nonlin` for in-plane transformations. | `3`                                      |
| `-h`, `--help`            | `[LEVEL]`      | Print the help. Level 1 is a summary, 2 lists the accepted names, 3 lists all options.                                     | `1`                                      |

See [Hardware](../../conventions.md#hardware) for general notes on devices.

## Examples

Rigidly register two T1-weighted scans of the same subject, with mean
squared error. The transformation is written to `rigid.lta` next to the
fixed image, and `{base}.moved.nii.gz` copies of both images are written:

```shell
nitorch register -v \
    @loss mse @@fix sub-01_ses-1_T1w.nii.gz @@mov sub-01_ses-2_T1w.nii.gz \
    @affine rigid
```

Register a FLAIR scan to a T1-weighted scan with mutual information and a
full affine, initialised by aligning the centres of mass and with
progressively freed degrees of freedom. The FLAIR is written resliced onto
the T1w grid in the `reg` directory, and no `.moved` files are produced:

```shell
nitorch register -v -o reg \
    @loss mi @@fix T1w.nii.gz -o false @@mov FLAIR.nii.gz -o false -r \
    @affine affine -i com -g -o reg/FLAIR_to_T1w.lta
```

Register two brain images of a similar contrast diffeomorphically, after a
rigid initialisation. `lcc` compares local patches of 8 % of the image size,
the nonlinear part uses a bending energy of 0.5, and five pyramid levels
are used. The velocity field is written to `warp/svf.nii.gz`:

```shell
nitorch register -v -o warp --gpu \
    @loss lcc -p 8 pct @@fix template_T1w.nii.gz @@mov sub-01_T1w.nii.gz \
    @affine rigid \
    @nonlin svf 1 -b 0.5 \
    @pyramid -l 0:5
```

Register with the affine and nonlinear parts fitted one after the other, and
more iterations for the nonlinear solver. This is often more stable than
the default interleaving when the images are initially far apart:

```shell
nitorch register -o warp \
    @loss mse @@fix fixed.nii.gz @@mov moving.nii.gz \
    @affine affine @@optim gn -n 100 \
    @nonlin svf @@optim gn -n 50 \
    @optim sequential
```

Register a template to a subject using two contrasts jointly, then carry the
template's label map to the subject. The two files after each of `@@fix` and
`@@mov` are channels of a two-channel image, so both contrasts contribute to
the match. The labels are transferred with nearest-neighbour interpolation by
`reslice`:

```shell
nitorch register -v -o reg \
    @loss mi -b 48 \
        @@fix sub-01_T1w.nii.gz sub-01_T2w.nii.gz -o false \
        @@mov template_T1w.nii.gz template_T2w.nii.gz -o false \
    @affine affine -g \
    @nonlin svf

nitorch reslice template_aseg.nii.gz \
    -l2 reg/affine.lta -v reg/svf.nii.gz -l2 reg/affine.lta \
    -t sub-01_T1w.nii.gz -i 0 -o sub-01_aseg.nii.gz
```

Finally, a complete affine plus nonlinear registration that combines mutual
information on T1w and FLAIR scans with a Dice term on a handful of
segmented structures (left and right white matter, and cortex), so that the
deformation is driven by both intensities and anatomy. The Dice term has a
small weight and the labels are read as 2 and 41 (white matter), 3 and 42
(cortex):

```shell
nitorch register -v -o results --gpu \
    @loss mi -b 64 \
        @@fix sub-01_T1w.nii.gz -o false \
        @@mov sub-01_FLAIR.nii.gz -o false -r \
    @loss dice 0.5 \
        @@fix sub-01_T1w_aseg.nii.gz -l 2 41 3 42 -o false \
        @@mov sub-01_FLAIR_aseg.nii.gz -l 2 41 3 42 -o false -r -n 0 \
    @affine affine -g \
    @nonlin svf 1 -b 0.2 -m 0.001 -h \
    @pyramid -l 0:4
```

This writes `results/affine.lta`, `results/svf.nii.gz`, `results/svf_hessian.nii.gz`,
and the resliced FLAIR image and its label map on the T1w grid
(`sub-01_FLAIR.resliced.nii.gz`, `sub-01_FLAIR_aseg.resliced.nii.gz`, in the
directory of the input files). The `-n 0` option writes labels with
nearest-neighbour interpolation.

## See also

- [`simplereg`](simplereg.md), [`affreg`](affreg.md) — simpler wrappers
  around `register` for affine alignment.
- [`optinorm`](optinorm.md) — groupwise alignment of several scans, built on
  `register`.
- [`affopt`](affopt.md) and [`vopt`](vopt.md) — combine pairwise
  registrations into transformations to a common space; `vopt` uses the
  Hessian written by `@nonlin -h`.
- [`autoreg`](autoreg.md) — the older, autograd-based registration tool.
- [`reslice`](../spatial/reslice.md) — apply the transformations to images.
- [Registering two images](../../tutorials/registration.md) — a guided
  tutorial.
