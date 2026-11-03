---
icon: octicons/book-24
---

# Registering two images

Registration finds the spatial transformation that brings one image into
alignment with another. You need it whenever two images of the same anatomy
are not on the same grid: scans of the same person taken in different
sessions, scans of different people that you want to compare voxel by voxel,
or an atlas whose labels you want to carry onto a subject. In this tutorial
you register one T1-weighted scan to another, first with an affine
transformation and then with an added nonlinear deformation, and you use the
result to resample both an intensity image and a label map.

## What you will learn

- how to read a `register` command, which is built from groups introduced by
  `@` tags rather than from a flat list of options;
- how to run an affine registration and what the command writes;
- how to add a nonlinear component and what changes in the output;
- how to apply the result to another image, and in particular to a label map,
  with [`reslice`](../reference/spatial/reslice.md);
- which shortcuts exist, and where to go for multi-contrast and
  segmentation-driven registration.

## Before you start

You need two T1-weighted scans, which this tutorial calls `sub-01_T1w.nii.gz`
and `sub-02_T1w.nii.gz`, and, for the last step, a label map (a segmentation)
of the second scan, called `sub-02_aseg.nii.gz`, on the same voxel grid as
`sub-02_T1w.nii.gz`. The label map is optional. The
[tutorial index](index.md#getting-sample-images) suggests where to find images.
If you have not looked at the headers of your images yet, do so with
[`info`](../reference/images/info.md), as in
[Inspecting and preparing images](preparing.md). Then open both images
together:

```shell
nitorch view sub-01_T1w.nii.gz sub-02_T1w.nii.gz
```

The viewer places the images in world space using their orientation matrices.
If the two brains are roughly on top of each other, you are ready to start.
If one is flipped, rotated by 90 degrees, or hundreds of millimetres away,
the orientation matrix of one of them is wrong; fix that first (see
[when `orient` is the right tool](preparing.md#when-orient-is-the-right-tool-and-the-danger)
in the preparation tutorial), because a registration cannot be expected to
recover from it.

**Vocabulary.** The *fixed* image is the one that stays where it is and defines
the space you want to end up in. The *moving* image is the one that is
transformed to match it. Here, `sub-01_T1w.nii.gz` is fixed and
`sub-02_T1w.nii.gz` is moving.

## Step 1: an affine registration

An affine transformation combines translations, rotations, zooms and shears,
twelve parameters in total. It is the right first step between two different
heads, because it removes the differences in position, orientation and overall
size. (Between two scans of the same person, a rigid transformation, with six
parameters, is often enough: use `@affine rigid`, which is also the default
model.)

```shell
nitorch register -v -o reg-affine \
    @loss cc \
        @@fix sub-01_T1w.nii.gz \
        @@mov sub-02_T1w.nii.gz -r \
    @affine affine
```

This is a single command, split over several lines with backslashes. Read it
from the top.

`register` has a different grammar from most NITorch commands. Options that
apply to the whole run come first: `-v` asks for progress messages, and
`-o reg-affine` names the directory where results are written (it is created if
it does not exist). After that, the command is a sequence of *groups*, each
opened by a tag that starts with `@`, which collect the options of one part of
the problem. A tag that starts with `@@` opens a group nested inside the
preceding `@` group. The full grammar is described in
[The command-line grammar](../reference/registration/register.md#the-command-line-grammar),
and the general mechanism in [Groups](../conventions.md#groups).

- `@loss cc` opens a *matching term*: a measure of how badly the two images
  agree, which the registration tries to reduce. `cc` is the squared
  correlation coefficient, a good choice when both images have the same
  contrast, as here. If you were registering images of different contrasts,
  such as a T1-weighted scan and a FLAIR scan, you would use `mi` (mutual
  information). See [Choosing a loss](../reference/registration/register.md#choosing-a-loss).
- `@@fix` and `@@mov` belong to that loss and name the fixed and moving
  images. The `-r` after the moving image asks `register` to write that image
  resampled onto the grid of the fixed image, so that you get a ready-to-use
  result without a second command.
- `@affine affine` asks for a full 12-parameter affine transformation. The
  name after `@affine` selects the model: `translation`, `rotation`, `rigid`
  (the default, six parameters), `similitude` (rigid plus a single zoom) and
  `affine`. Each can be abbreviated to its first letter.

There is no `@pyramid` group, so the default one is used: the images are
registered at three resolutions, from coarse to fine. Coarse levels are
cheap and find the large movements; fine levels then refine them. With `-v`,
you will see one block of progress lines per level.

On full-size images this takes a while, especially on a CPU. While you are
checking that the command is right, you can drop the most expensive level by
adding `@pyramid -l 1:3` at the end (level 0 is the finest), and run on a GPU
by adding `--gpu` before the first group. Remove these changes for the final
run.

### What was written

When the command finishes, the last lines of its output look like this
(illustrative):

```text
Affine -> reg-affine/affine.lta
Minimal reslice: sub-02_T1w.nii.gz -> reg-affine/sub-02_T1w.moved.nii.gz ... done.
Full reslice: sub-02_T1w.nii.gz -> reg-affine/sub-02_T1w.resliced.nii.gz ... done.
Minimal reslice: sub-01_T1w.nii.gz -> reg-affine/sub-01_T1w.moved.nii.gz ... done.
```

and the `reg-affine` directory contains four files:

| File                           | Content                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| `affine.lta`                   | The transformation: a 4&times;4 world-to-world (RAS-to-RAS) matrix in a FreeSurfer LTA file. The file is named after the model. |
| `sub-02_T1w.resliced.nii.gz`   | The moving image resampled onto the voxel grid of the fixed image (because of `-r`).               |
| `sub-02_T1w.moved.nii.gz`      | The moving image with a modified header, aligned with the fixed image.                             |
| `sub-01_T1w.moved.nii.gz`      | The fixed image with a modified header, aligned with the moving image.                             |

The matrix in `affine.lta` is what you are after. Read it as a map from
**fixed** world coordinates to **moving** world coordinates, in millimetres.
That sounds backwards until you remember how an image is resampled: to fill
each voxel of the output (on the fixed grid), you ask where in the input (the
moving image) that voxel's position lies. The matrix answers exactly that
question. Section
[Spatial transformations](../conventions.md#spatial-transformations) of the
conventions page and the [next tutorial](transforms.md) say more about this
direction.

The two `.moved` files show the other kind of output. They are written by
default, for the fixed image as well as the moving one, and use *minimal
reslicing*: for an affine registration, only the orientation matrix in the
header is changed, and the voxel data are copied untouched. Because the
default affine is *symmetric*, both headers are changed, and each image
receives the full transformation: the moving image is carried into the space
of the fixed image, and the fixed image into the space of the moving image.
Open `sub-02_T1w.moved.nii.gz` together with `sub-01_T1w.nii.gz` and they
overlap, with no interpolation; likewise `sub-01_T1w.moved.nii.gz` with
`sub-02_T1w.nii.gz`. The two `.moved` files do not overlap each other, since
they are in opposite spaces. Each one keeps the voxel grid of its own input,
so neither lies on the grid of the other image. If you do not want them, add
`-o false` after each of `@@fix` and `@@mov`. The `.resliced` image, in
contrast, is always on the grid of the other image.

!!! tip "Check the result by eye"
    Compare the fixed image with the moving image before and after:

    ```shell
    nitorch view sub-01_T1w.nii.gz sub-02_T1w.nii.gz reg-affine/sub-02_T1w.resliced.nii.gz
    ```

    Move the cursor over the edge of the ventricles or the skull: the third
    block should follow the first much more closely than the second does. A
    registration whose loss decreased but whose result looks wrong is the sign
    of a bad starting point (see the warning below) or of an unsuitable loss.

!!! warning "If the registration fails to converge"
    A common cause is that the images start too far apart for the optimiser to
    find its way. Start from an alignment of the centres of mass by adding
    `-i com` to the `@affine` group (`@affine affine -i com`), and consider
    `-g`, which frees the parameters progressively, from translations to the
    full affine.

## Step 2: add a nonlinear component

Two different brains differ by more than a global shift, rotation and stretch,
and an affine registration leaves visible residual differences in the shape of
the gyri and ventricles. A nonlinear transformation corrects those. In
`register`, you add it by appending an `@nonlin` group:

```shell
nitorch register -v -o reg-nonlin \
    @loss cc \
        @@fix sub-01_T1w.nii.gz \
        @@mov sub-02_T1w.nii.gz -r \
    @affine affine \
    @nonlin svf
```

The output directory is different, because the affine file would otherwise be
overwritten with a different matrix. Everything before `@nonlin` is as in the
first command.

`@nonlin svf` asks for a *stationary velocity field*. Instead of storing a
displacement for each voxel directly, the model stores a smooth velocity at
each point and obtains the deformation by letting the image flow along it for
unit time. The advantage is that the resulting deformation is smooth and, in
principle, invertible: it does not fold the brain onto itself, and its inverse
is the flow along the opposite velocity. The other models are `shoot`
(geodesic shooting) and `smalldef` (a plain displacement field, which is not
constrained to be invertible).

Because the nonlinear field has many more degrees of freedom than the affine
matrix, it is *regularised*: the registration is penalised for deformations
that are large, wrinkled or sharply varying. The defaults are a reasonable
starting point. To make the deformation smoother, give a factor larger than one
after the model name (`@nonlin svf 5`); to let it follow finer detail, give a
smaller one. See [Nonlinear: `@nonlin`](../reference/registration/register.md#nonlinear-nonlin)
for all the penalties.

Since there is no `@optim` group, the default joint strategy is used: the
affine matrix and the velocity field are *interleaved*, with a few updates of
one, then of the other, so that each adapts to the changes in the other. In
particular, the affine here is not the one from step 1; it is re-estimated
together with the field.

### What was written

The directory `reg-nonlin` now contains:

| File                           | Content                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------- |
| `affine.lta`                   | The affine part. It is only meaningful together with the velocity field.                            |
| `svf.nii.gz`                   | The velocity field: a volume with three components (the velocity along x, y and z) in its last dimension. |
| `sub-02_T1w.resliced.nii.gz`   | The moving image warped onto the fixed grid, with both transformations applied.                     |
| `sub-01_T1w.moved.nii.gz`, `sub-02_T1w.moved.nii.gz` | Each image carried into the space of the other, as before. Now that there is a nonlinear part, the images are resampled (on their own grid) rather than given only a new header. |

The velocity field has its own lattice, described by the header of
`svf.nii.gz`, which need not be the grid of the images.
Its values are expressed in voxels of that lattice, not in millimetres. You can
inspect the header with `info`:

```shell
nitorch info reg-nonlin/svf.nii.gz
```

Look at the result in the viewer, and compare the quality of the alignment with
that of the affine-only result:

```shell
nitorch view sub-01_T1w.nii.gz reg-affine/sub-02_T1w.resliced.nii.gz reg-nonlin/sub-02_T1w.resliced.nii.gz
```

The sulci and ventricles of the third block should follow those of the first
more closely than those of the second.

!!! tip "Run the two steps one after the other"
    If the images start far apart, the interleaved strategy can be unstable. The
    `@optim sequential` group makes `register` fit the whole affine first and
    the whole nonlinear field second:
    `... @affine affine @nonlin svf @optim sequential`.

## Step 3: apply the result to other images

The `.resliced` image is only one of the images you may want to bring into the
fixed space. A segmentation of the moving image, a mask, or another contrast
acquired in the same session all need the same transformation. For this, use
[`reslice`](../reference/spatial/reslice.md).

`reslice` takes the images to resample, a list of transformations, and a
*target*, the image whose grid the output should have. The image is resampled
by looking up, for each voxel of the target, where it falls in the input; the
transformations describe that lookup, so they are the fixed-to-moving
transformations that `register` wrote.

### After the affine registration

For the affine-only result, a single `-l` (linear) transformation is enough.
Here it resamples the moving T1-weighted image onto the fixed grid, which
reproduces the `.resliced` image:

```shell
nitorch reslice sub-02_T1w.nii.gz -l reg-affine/affine.lta \
    -t sub-01_T1w.nii.gz -o sub-02_T1w.in-sub-01.nii.gz
```

A label map needs a different treatment, because its values are names, not
quantities. If you interpolate labels linearly (the default), the voxels at the
border between label 2 and label 41 get values such as 21.5, which means
nothing. `reslice` has a dedicated mode for labels, selected by `-i l`:

```shell
nitorch reslice sub-02_aseg.nii.gz -l reg-affine/affine.lta \
    -t sub-01_T1w.nii.gz -i l -o sub-02_aseg.in-sub-01.nii.gz
```

In this mode each label is interpolated separately and every output voxel
receives the label with the highest value, so only labels that exist in the
input appear in the output, and the output keeps the data type of the input.
Nearest-neighbour interpolation, `-i 0`, is a simpler alternative that also
introduces no new values. Either way, check the result against the fixed image
with `nitorch view`.

### After the nonlinear registration

With a nonlinear transformation, the single matrix is no longer enough: the
field has to be applied too, and the order matters. `reslice` takes the
transformations as a chain, and applies the last one first, to the position of
each target voxel.

The default affine of `register` is symmetric: the model applies *half* of the
affine on each side of the deformation, which is, for a target position `x` in
the fixed image, `A^(1/2)(φ(A^(1/2) x))`. The file `affine.lta` stores the whole
matrix `A`, so to apply the model you tell `reslice` to apply its square root
with `-l2`, twice, around the velocity field `-v`:

```shell
nitorch reslice sub-02_aseg.nii.gz \
    -l2 reg-nonlin/affine.lta -v reg-nonlin/svf.nii.gz -l2 reg-nonlin/affine.lta \
    -t sub-01_T1w.nii.gz -i l -o sub-02_aseg.in-sub-01.nonlin.nii.gz
```

The same chain resamples an intensity image. Use it to check your command:
applied to `sub-02_T1w.nii.gz`, it must reproduce
`reg-nonlin/sub-02_T1w.resliced.nii.gz` up to interpolation noise.

```shell
nitorch reslice sub-02_T1w.nii.gz \
    -l2 reg-nonlin/affine.lta -v reg-nonlin/svf.nii.gz -l2 reg-nonlin/affine.lta \
    -t sub-01_T1w.nii.gz -o sub-02_T1w.in-sub-01.nonlin.nii.gz
```

!!! warning "The order of the chain depends on the position of the affine"
    Using `-l reg-nonlin/affine.lta -v reg-nonlin/svf.nii.gz` instead of the
    chain above does not give an error. It gives a different warp, which is wrong. The
    chain is `-l2 ... -v ... -l2 ...` for the default, symmetric affine. If you
    register with the affine on the moving image (`@affine affine -p mov`), it is
    `-l ... -v ...`, and with the affine on the fixed image (`-p fix`) it is
    `-v ... -l ...`. The
    [reference](../reference/registration/register.md#applying-the-transformations)
    has the table, including the chains for the opposite direction. The
    [next tutorial](transforms.md) explains how chains work.

## Shortcuts

Two commands wrap `register` with a flatter, more classical syntax, for the
common case of aligning one image to another:

- [`affreg`](../reference/registration/affreg.md) is an affine registration with
  options named `-m` (moving), `-f` (fixed), `-l` (loss), `-r` (resliced
  output), and `-a` for a full affine instead of the default rigid model.
- [`simplereg`](../reference/registration/simplereg.md) has similar options,
  and adds a nonlinear part: a stationary velocity field by default, or none
  with `--nonlin none`. Mind the differences: the affine model is chosen with
  `--affine [TYPE]` (a bare `--affine` means a full affine), and `-a` and `-w`
  name the output files of the affine and of the nonlinear field.

For example, the affine registration of step 1 can be written:

```shell
nitorch affreg -m sub-02_T1w.nii.gz -f sub-01_T1w.nii.gz -a -l cc \
    -r sub-02_T1w.resliced.nii.gz
```

The transformation is written to `affine.lta` in the directory of the fixed
image. Their defaults differ from those of `register`: they use mutual
information (`nmi`) unless you choose another loss with `-l`. They expose only
a subset of the options of `register`, so they are meant for quick jobs.

## Next steps

`register` can do much more than this tutorial has used:

- **Several contrasts at once.** Give several files after `@@fix` and after
  `@@mov`, and they are used as the channels of a single image. A multi-contrast
  registration, for instance of a template that has T1w and T2w images to a
  subject that has both, looks like this:

    ```shell
    nitorch register -v -o reg \
        @loss mi \
            @@fix sub-01_T1w.nii.gz sub-01_T2w.nii.gz -o false \
            @@mov template_T1w.nii.gz template_T2w.nii.gz -o false -r \
        @affine affine -g \
        @nonlin svf
    ```

- **Segmentations in the loss.** A second `@loss` group with `dice` or `cat`
  compares label maps, so that the deformation is driven by anatomy as well as by
  intensities. The labels to use are listed after `-l`, and `-n 0` keeps the
  labels intact when they are interpolated:

    ```shell
    nitorch register -v -o reg \
        @loss cc \
            @@fix sub-01_T1w.nii.gz -o false \
            @@mov sub-02_T1w.nii.gz -o false -r \
        @loss dice 0.5 \
            @@fix sub-01_aseg.nii.gz -l 2 41 3 42 -o false \
            @@mov sub-02_aseg.nii.gz -l 2 41 3 42 -o false -n 0 \
        @affine affine \
        @nonlin svf
    ```

- **More control.** The pyramid, the solvers, the regularisation and the
  initialisation all have their own groups. The
  [`register` reference](../reference/registration/register.md) documents them
  all, with more examples.

Continue with [Working with transformations](transforms.md) to learn how the
files that `register` wrote can be combined, inverted and applied in the other
direction.
