---
icon: octicons/terminal-24
---

# autoreg

Registers one or several pairs of images with a linear transformation, a
nonlinear one, or both, by minimising a composite loss with PyTorch's
automatic differentiation.

!!! note "Older tool"
    `autoreg` is the older, autograd-based registration command of NITorch.
    [`register`](register.md) is the newer one, with a more structured command
    line and a wider choice of losses and optimisers, and is generally preferred
    for new work. This page documents `autoreg` for existing scripts.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch autoreg LOSS [FACTOR] -fix FILE... [OPTIONS] -mov FILE... [OPTIONS]
                         [LOSS-OPTIONS]
                    ...
                    TRANSFORM [FACTOR] [OPTIONS] [REGULARISER [VALUE...]]...
                    ...
                    [OPTIMISER [OPTIONS]]...
                    [-all DEFAULTS]
                    [-prg [BOOL]] [-pad PAD [UNIT]] [-v [LEVEL]] [-cpu | -gpu [N]]
    ```

## Description

An `autoreg` command describes a registration problem in three parts, which you
can write in any order and repeat as needed:

- **What to match.** One or more *loss groups* each name a similarity measure
  (mutual information, mean squared error, ...) and the *fixed* and *moving*
  images it compares. Several loss groups add up to a composite loss, which lets
  you register on several contrasts at once, or mix intensity images and
  segmentations.
- **What to optimise.** One *transformation group* selects the kind of
  transformation: a linear one (translation, rigid, similitude or affine), a
  nonlinear one (a free-form deformation or a diffeomorphism), or one of each.
  The transformation groups also carry the regularisation of the nonlinear
  field.
- **How to optimise.** Optionally, one or more *optimiser groups* choose the
  algorithm (Adam, gradient descent, OGM or L-BFGS) and its settings. Adam is
  the default.

The moving images are warped to the fixed images, and the loss is minimised
with respect to the parameters of the transformation by back-propagation.
Because the loss is evaluated by resampling the moving images, interpolation
and boundary options are available for each image.

By default, the linear parameters are freed in stages (translations, then
rotations, then scaling, then shears) and the nonlinear field is then refined
from a coarse grid to a fine one. See [Optimisation](#optimisation).

`autoreg` always writes the estimated transformations, and by default also
writes each moving image with the transformation applied. See
[Outputs](#outputs).

!!! warning "At least one transformation is needed"
    If the command contains no transformation group (`-tr`, `-rig`, `-sim`,
    `-aff`, `-ffd` or `-dif`), nothing is optimised: the images are only copied
    to their output files. Contrary to what the built-in help suggests, there
    is no default transformation.

## Command-line grammar

The command line is a sequence of **groups**. A group is opened by a *group
tag* and extends over the options that follow it, up to the next tag that the
group does not recognise. The same short tag can therefore mean different
things in different groups (`-o` writes an image inside `-fix` and `-mov`, and
a transformation inside a transformation group; `-lr` sets the learning rate
of an optimiser or a relative learning rate of a transformation; `-h` is the
help except inside an `-lbfgs` group, where it is the history size).

Any argument that begins with a dash is a tag, including negative numbers, and
every argument that is not a tag must belong to a group. File names therefore
cannot start with a dash.

The following tags open a group:

| Group | Tags | Purpose |
| ----- | ---- | ------- |
| Loss | `-mi`, `-mse`, `-jtv`, `-dice`, `-cat`, `-oth` | A similarity measure and the images it compares. |
| Linear transformation | `-tr`, `-rig`, `-sim`, `-aff` | A linear transformation. At most one. |
| Nonlinear transformation | `-ffd`, `-dif` | A nonlinear transformation. At most one. |
| Optimiser | `-adam`, `-gd`, `-ogm`, `-lbfgs` | An optimisation algorithm. |
| Defaults | `-all` | Values that apply to all groups that do not set them. |

Other top-level options (`-prg`, `-pad`, `-v`, `-cpu`, `-gpu`, `-h`) take
effect wherever they appear but also end the group that precedes them.

### Loss groups

A loss group starts with the loss tag, optionally followed by a numeric
`FACTOR` that multiplies this loss in the sum (default 1). It then contains a
`-fix` and a `-mov` *image block*, in any order, and finally options of the loss.

```shell
-mi 0.5 -fix target.nii.gz -mov source.nii.gz -b 64
```

An image block starts with `-fix` (aliases `--fixed`, `-trg`, `--target`) or
`-mov` (aliases `--moving`, `-src`, `--source`), followed by one or more file
names, followed by options that apply to this image only (see the
[image block options](#image-block-options)). The block ends at the first tag
that it does not recognise, so options of the loss itself, such as `-b 64`
above, must come after the image blocks, and the options `-inter`, `-bnd`,
`-ex` and `-pyr` that follow a `-fix` or `-mov` belong to that image.

The `-oth` loss is not a loss. It declares extra moving images that are carried
along: they are warped with the final transformation, but play no role in the
optimisation. It takes a `-mov` block only.

An image can have several channels, and several files given to the same block
are concatenated into one multi-channel image (they must have the same
voxel grid). To select channels inside a 4D file, append their 1-based indices
to the file name, separated by commas: `dwi.nii.gz,1,3` loads the first and the
third volumes. Fixed and moving images of the same group should have the same
number of channels, which are matched one to one.

### Transformation groups

A transformation group starts with its tag, optionally followed by a numeric
`FACTOR` that scales all its regularisers (default 1), then any of the options
of the [transformation tables](#transformations). Regularisers have
the form `-abs`, `-mem`, `-ben`, `-le`, each followed by its strength.

```shell
-dif 2 -mem 0.002 -ben 0.4 -pyr 1:3
```

### Optimiser groups

An optimiser group is its tag followed by its options. Several optimiser groups
are run one after the other, in the order given, at each [stage](#optimisation)
of the registration.

```shell
-adam -nit 100 -lr 0.05  -lbfgs -nit 20
```

### Default values

The `-all` group holds values that apply everywhere (see the
[table](#defaults-all)). A value set in a group wins over the one in `-all`, which
wins over the built-in default. An `-all` group ends at the first tag it does
not recognise, so it can be placed anywhere, for example first:

```shell
nitorch autoreg -all -nit 200 -mi -fix ... -mov ... -aff
```

## Arguments

`autoreg` has no positional arguments. Files are given in `-fix`, `-mov` and
`-init` options.

## Options

### Losses

| Tag | Aliases | Loss |
| --- | ------- | ---- |
| `-mi` | `--mutual-info`, `--mutual-information` | Mutual information between the intensity histograms, for images with different contrasts. |
| `-mse` | `--mean-squared-error` | Mean squared error, for images with the same contrast. |
| `-jtv` | `--total-variation` | Normalised joint total variation: an edge-based measure, meant for images with strong intensity non-uniformity. |
| `-dice` | `--dice` | Dice coefficient, for soft or hard segments. |
| `-cat` | `--categorical` | Categorical cross-entropy, for soft segments (probability maps). The last class may be left implicit. |
| `-oth` | `--other` | Not a loss: additional moving images to warp at the end. |

All images are rescaled before the registration, channel by channel, so that
the 1st percentile of intensities maps to 0 and the 95th percentile to 1.

Two limitations of the current version are worth knowing. For `-dice`, use
floating-point probability maps (one channel per class): integer label volumes
are meant to be converted to one-hot maps on the fly, but that conversion does
not work in the current code. When you use `-oth`, add `-v 0` to your command;
with the default verbosity, the summary printed at the start does not handle
these groups.

#### Options common to all loss groups

These options apply to the group as a whole. They are written after the image blocks.

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-exc`, `--exclude` | `[BOOL]` | Exclude this pair of images from the computation of the mean space. A bare `-exc` means yes. Accepts `true`, `false`, `1`, `0` (only the first letter `t`/`f` is checked). | no |

Options `-inter`, `-bnd`, `-ex` and `-pyr` can also be given at the group level.
To affect both images, put them straight after the loss tag (and its factor),
before `-fix` and `-mov`. They are then the default for both image blocks.

#### Mutual information (`-mi`)

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-p`, `--patch` | `SIZE...` | Patch size for local mutual information. 0 means global mutual information. | global |
| `-b`, `--bins` | `BINS` | Number of bins of the joint histogram. | `32` |
| `-f`, `--fwhm` | `WIDTH` | Full width at half maximum of the histogram bins, in bin units. | `1` |
| `-t`, `--threshold` | `VAL` | Voxels where the fixed image is below this value are excluded. | no threshold |
| `-n`, `--order` | `ORDER` | Order of the splines used to build the histogram. Accepts an integer or `inf`. | `3` |

With several channels, the loss is the average of the mutual information of
each pair of channels.

#### Dice (`-dice`)

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-l`, `--labels` | `LABEL...` | Labels to use. | all labels except 0 |
| `-w`, `--weights` | `WEIGHT...` | Weight of each label. Only the first value is retained in the current version. | unweighted |

### Image block options

These options follow a file list in a `-fix` or `-mov` block.

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-o`, `--output` | `[FILE...]` | Write an "updated" version of the image: same grid and data, with the transformation applied. With no value, use the default name; otherwise give one name per file. See [Outputs](#outputs). | moving: yes; fixed: no |
| `-r`, `--resliced` | `[FILE...]` | Write a "resliced" version of the image, resampled to the grid of the other image of the pair. With no value, use the default name. | no |
| `-inter`, `--interpolation` | `ORDER` | Interpolation order used to resample this image. | `1` |
| `-bnd`, `--bound` | `BOUND` | Boundary condition when resampling this image. | `dct2` |
| `-ex`, `--extrapolate` | | Extrapolate out-of-bounds data using the boundary condition instead of filling with zeros. | off |
| `-pyr`, `--pyramid` | `LEVEL...` | Pyramid levels at which this image is compared; see [Pyramids](#pyramids). | `1` |

### Transformations

#### Linear transformations

| Tag | Aliases | Parameters |
| --- | ------- | ---------- |
| `-tr` | `--translation` | Translations. |
| `-rig` | `--rigid` | Translations and rotations. |
| `-sim` | `--similitude` | Translations, rotations and one isotropic scaling. |
| `-aff` | `--affine` | Translations, rotations, zooms and shears. |

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-z`, `--zero` | `[BOOL]` | Rotate and scale about the centre of the field of view of the fixed images, rather than about the origin of world space. A bare `-z` means yes. | yes |

The linear transformation maps world coordinates of the moving images to world
coordinates of the fixed images.

#### Nonlinear transformations

| Tag | Aliases | Transformation |
| --- | ------- | -------------- |
| `-ffd` | `--free-form` | Free-form deformation: a displacement field encoded with cubic splines. |
| `-dif` | `--diffeo` | Diffeomorphism: a stationary velocity field, integrated to give the deformation. |

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-sd`, `--smalldef` | `[BOOL]` | (`-dif` only) Penalise the exponentiated transformation (the displacement) instead of the velocity field. A bare `-sd` means yes. | no |
| `-g`, `--grid` | `SIZE...` | (`-ffd` only) Number of nodes of the grid. Accepted, but it has no effect in the current version: the grid has the resolution of the current pyramid level. | |

The nonlinear field lives on the mean space of all the fixed and moving images
that are not excluded with `-exc`, at the resolution of its pyramid level (see
[Pyramids](#pyramids)). Its displacements are expressed in voxels of that grid. The
extent of the grid can be enlarged with `-pad`.

#### Options common to all transformations

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-lr`, `--learning-rate` | `LR` | Learning rate of this transformation, relative to the learning rate of the optimiser (the two are multiplied). | `1` |
| `-wd`, `--weight-decay` | `WD` | Weight decay of this transformation, relative to the optimiser's (the two are multiplied). | `0` |
| `-init`, `--init` | `FILE` | Initial value, read from a file. Accepted by the parser, but its handling is incomplete in the current version; do not rely on it. | zero (identity) |
| `-o`, `--output` | `[FILE]` | Output file for the estimated transformation; see [Outputs](#outputs). With no value, use the default. | `{name}{ext}` |
| `-pyr`, `--pyramid` | `LEVEL...` | Resolution levels of a nonlinear field; see [Pyramids](#pyramids). Ignored for linear transformations. | `1` |

The `-stop` option is also accepted in a transformation group but has no effect there.

#### Regularisers

Regularisers apply to the nonlinear field (to the velocity field for `-dif`,
unless `-sd` is given). Each one is followed by an optional strength; a bare
tag means a strength of 1. The strengths are multiplied by the `FACTOR` of the
transformation group.

| Tag | Aliases | Value | Penalises |
| --- | ------- | ----- | --------- |
| `-abs` | `--absolute` | `[VAL]` | Absolute values of the field (0th order). |
| `-mem` | `--membrane` | `[VAL]` | Membrane energy (first derivatives). |
| `-ben` | `--bending` | `[VAL]` | Bending energy (second derivatives). |
| `-le` | `--linear-elastic` | `[VAL [VAL]]` | Linear-elastic energy: zooms (divergence) and shears. With one value, it is used for both. |

If you give none of them, the following set is used: `-abs 0.0001 -mem 0.001
-ben 0.2 -le 0.05 0.2`. As soon as you give at least one regulariser, the
others are switched off: `-dif -ben 0.5` penalises the bending energy only. The
built-in help says that only the bending energy is penalised by default, which
is not what the code does. No regulariser is placed on linear transformations.

### Optimisers

| Tag | Aliases | Algorithm |
| --- | ------- | --------- |
| `-adam` | `--adam` | Adam. This is the default if you give no optimiser. |
| `-gd` | `--gradient-descent` | Gradient descent. |
| `-ogm` | `--optimized-gradient-descent` | Optimised gradient method (Kim and Fessler). |
| `-lbfgs` | `--lbfgs` | L-BFGS. |

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-nit`, `--max-iter` | `N` | Maximum number of iterations per [stage](#optimisation). | `1000` |
| `-lr`, `--learning-rate` | `LR` | Initial learning rate. | `0.1` |
| `-wd`, `--weight-decay` | `WD` | Weight decay. | `0` |
| `-stop`, `--stop` | `VAL` | Stop the stage when the learning rate has decayed to this fraction of its initial value. | `1e-4` |
| `-ls`, `--line-search` | `[N]` | Maximum number of backtracking line-search steps. A bare `-ls` means 6; 0 disables the line search. | `0` |
| `-h`, `--history` | `N` | (`-lbfgs` only) History size. | `100` |

### Defaults (`-all`)

| Option | Value | Applies to | Default |
| ------ | ----- | ---------- | ------- |
| `-nit`, `--max-iter` | `N` | Optimisers | `1000` |
| `-lr`, `--learning-rate` | `LR` | Optimisers | `0.1` |
| `-stop`, `--stop` | `VAL` | Optimisers | `1e-4` |
| `-ls`, `--line-search` | `[N]` | Optimisers | `0` |
| `-inter`, `--interpolation` | `ORDER` | Images | `1` |
| `-bnd`, `--bound` | `BOUND` | Images | `dct2` |
| `-ex`, `--extrapolate` | | Images | off |
| `-pyr`, `--pyramid` | `LEVEL...` | Images and nonlinear fields | `1` |
| `-o`, `--output` | `FILE` | Output name of the transformations | `{name}{ext}` |

!!! warning "`-stop` inside `-all`"
    In the current version, `-stop` in an `-all` group also discards the two
    arguments that follow it. Put `-stop` in an optimiser group instead, or make
    it the last thing on the command line.

The built-in help also lists `-r`, `-init` and `-exc` as default options; the
code does not accept them in an `-all` group.

### Other options

| Option | Value | Description | Default |
| ------ | ----- | ----------- | ------- |
| `-prg`, `--progressive` | `[BOOL]` | Free the parameters progressively. A bare `-prg` means yes. Accepts `yes`/`no`, `true`/`false`, `1`/`0` (first letter checked). | yes |
| `-pad`, `--padding` | `PAD [UNIT]` | Pad the field of view of the mean space on which the nonlinear field is defined. `UNIT` is `%` (of the field of view) or `mm`. | `0 %` |
| `-v`, `--verbose` | `[LEVEL]` | Verbosity. A bare `-v` means 2. 0 is silent; 1 prints the losses, transformations and the loss at each iteration; 2 also prints all parsed options. | `1` |
| `-cpu`, `--cpu` | | Run on the CPU. This is the default device. | |
| `-gpu`, `--gpu` | `[N]` | Run on a GPU, optionally the one numbered `N`. | |
| `-h`, `--help` | | Print the built-in help and exit. | |

Do not turn off progressive freeing (`-prg false`) when you give several pyramid levels
to a nonlinear transformation: the registration would not terminate.

## Optimisation

The parameters are not all optimised from the start. With progressive freeing,
the registration runs through **stages**. At each stage, every optimiser group
is run in turn, each up to its maximum number of iterations, over the
parameters that are free so far, and then more parameters are freed. The order is:

1. The linear transformation, in steps. Translations first; then rotations
   (for `-rig`, the final step); then isotropic scaling (for `-sim`, the final
   step); then zooms and shears (for `-aff`). A step is skipped if the chosen
   group does not contain it.
2. The nonlinear field, from the coarsest to the finest resolution level (see
   [Pyramids](#pyramids)). Once free, the linear parameters keep being
   optimised.

With `-prg false`, all parameters are free from the first stage and there is a
single stage.

Within a stage, the learning rate is reduced automatically when the loss stops
improving (PyTorch's `ReduceLROnPlateau`). The stage stops after `-nit`
iterations, or as soon as the learning rate of every parameter group has
decayed to `-stop` times its initial value. With `-ls`, each step is shortened
by backtracking until it lowers the loss, and discarded if no shortened step
does. For `-dif`, the gradient is preconditioned with the Green's function of
the regulariser.

### Pyramids

A **pyramid level** $l$ is an image downsampled by a factor $2^{l-1}$ with
average pooling, so that level 1 is the full resolution. `-pyr` takes a list of
levels, or ranges written `[START]:STOP[:STEP]` with both ends included:
`-pyr 1:3` is `-pyr 1 2 3`. Ranges and single levels can be mixed.

The `-pyr` option has two roles.

- On an *image* (in a `-fix` or `-mov` block, in a loss group, or in `-all`),
  it lists the resolutions at which the images are compared. The loss is the
  average of the losses over all listed levels, which are evaluated at every
  iteration. The levels are not a coarse-to-fine schedule.
- On a *nonlinear transformation*, it lists the resolution levels of the
  deformation field. The field is created at the coarsest listed level, and
  progressive freeing then moves to each finer level in turn, up-sampling the
  current estimate.

To get a classic coarse-to-fine registration, pass the same levels to both,
which `-all -pyr 1 2 3` does.

## Outputs

**Transformations.** The estimated transformations are always written. The
default names are written in the current directory.

| Transformation | Default file | Content |
| -------------- | ------------ | ------- |
| `-tr` | `translation.lta` | The matrix, in an `.lta` file. |
| `-rig` | `rigid.lta` | The matrix, in an `.lta` file. |
| `-sim` | `similitude.lta` | The matrix, in an `.lta` file. |
| `-aff` | `affine.lta` | The matrix, in an `.lta` file. |
| `-ffd` | `ffd.nii.gz` | The grid of FFD parameters (the spline coefficients of the displacement field). |
| `-dif` | `diffeo.nii.gz` | The velocity field, with the orientation matrix of the mean space. |

You can change the name with `-o` in the transformation group or in `-all`. The
name is a template in which `{name}` is replaced by the name of the
transformation (`rigid`, `affine`, `ffd`, `diffeo`, ...) and `{ext}` by its
default extension; a directory must be part of the file name, so
`-o results/{name}{ext}` works if `results` exists. The linear matrix, which is
left-multiplied onto the orientation matrix of the moving image to align it with the fixed
image, can be applied to images with [`reslice`](../spatial/reslice.md); see
that page for the options that apply linear and velocity transformations, and
their inverses.

**Images.** Each image block can write two kinds of output.

- The *updated* image keeps its voxel grid and its data (except for the
  nonlinear part, which is applied by resampling in the original space). The
  linear transformation is folded into the orientation matrix in the header.
  By default, the updated version of every moving image is written, to
  `{dir}/{base}.registered{ext}`. Fixed images are not written unless you give
  `-o`. For a fixed image, the inverse transformation is applied, so that it
  is aligned with the moving image.
- The *resliced* image is resampled onto the grid of the other image of the
  pair: a moving image is resliced to the fixed grid and a fixed image to the
  moving grid. It is not written unless you give `-r`. Its default name is
  `{dir}/{base}.resliced{ext}`.

In names, `{dir}`, `{base}` and `{ext}` are replaced by the directory, the base
name and the extension (including `.nii.gz`) of the corresponding input file.
When you give names explicitly, give one per file in the block.

## Examples

Register a FLAIR image to a T1-weighted image with a rigid transformation. This
writes `rigid.lta` and `FLAIR.registered.nii.gz`:

```shell
nitorch autoreg -mi -fix T1w.nii.gz -mov FLAIR.nii.gz -rig
```

Add a nonlinear component on top of a similitude. A diffeomorphic field is
estimated after the linear parameters, with the default regularisation:

```shell
nitorch autoreg -mi -fix T1w.nii.gz -mov FLAIR.nii.gz -sim -dif
```

Also write the moving image on the grid of the fixed image, with a rigid
transformation and the output name of the transformation chosen explicitly:

```shell
nitorch autoreg -mi -fix T1w.nii.gz -mov FLAIR.nii.gz -r -rig -o FLAIR_to_T1w.lta
```

Register a template to tissue maps of a subject, and write the subject's maps in
the space of the template. The template is the moving image, so `-o` in the
`-fix` block warps the fixed images to it:

```shell
nitorch autoreg -cat -fix c1T1w.nii.gz c2T1w.nii.gz -o -mov TPM.nii -dif -rig
```

Run a multi-resolution registration with at most 200 iterations per stage. The
same three levels apply to the images and to the deformation field:

```shell
nitorch autoreg -all -nit 200 -pyr 1:3 \
        -mi -fix T1w.nii.gz -mov FLAIR.nii.gz \
        -aff -dif -ben 0.4
```

Optimise with gradient descent and a backtracking line search:

```shell
nitorch autoreg -mi -fix T1w.nii.gz -mov FLAIR.nii.gz -rig -gd -ls
```

Use Adam first and refine with L-BFGS:

```shell
nitorch autoreg -mi -fix T1w.nii.gz -mov FLAIR.nii.gz -aff \
        -adam -nit 100 -lbfgs -nit 20 -h 50
```

Register intensity images and carry a segmentation along, with nearest-neighbour
interpolation. The segmentation is not part of the loss:

```shell
nitorch autoreg -v 0 -mi -fix T1w.nii.gz -mov FLAIR.nii.gz \
        -oth -mov FLAIR_seg.nii.gz -inter 0 \
        -rig
```

## See also

- [`register`](register.md): the newer registration command.
- [`reslice`](../spatial/reslice.md): apply the estimated transformations to
  images.
- [`compose`](../spatial/compose.md), [`vexp`](../spatial/vexp.md): combine and
  integrate transformations.
- [Spatial transformations](../../conventions.md#spatial-transformations).
