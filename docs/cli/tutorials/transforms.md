---
icon: octicons/book-24
---

# Working with transformations

A registration produces a transformation, and a transformation is only useful
if you can tell what it does: which way it points, what units it is expressed
in, and how to combine it with another one. Most mistakes in registration
pipelines are not failures of the optimiser but misreadings of a transformation
file, such as applying a matrix in the wrong direction or chaining two fields in
the wrong order. This tutorial builds the mental model first, and then uses it
with the commands that manipulate transformations.

## What you will learn

- the kinds of transformation NITorch writes, and how to tell them apart;
- what "fixed to moving" means, and why it is the direction you need to
  resample the *moving* image;
- how voxel units and world (millimetre) units differ, and where each is used;
- how to combine transformations with
  [`compose`](../reference/spatial/compose.md), invert them, and turn a
  velocity field into a displacement field with
  [`vexp`](../reference/spatial/vexp.md);
- how to resample an image through a chain of transformations with
  [`reslice`](../reference/spatial/reslice.md).

The examples build on the files written in
[Registering two images](registration.md): `reg-affine/affine.lta` from the
affine registration, and `reg-nonlin/affine.lta` and `reg-nonlin/svf.nii.gz`
from the registration with a nonlinear part, between a fixed image
`sub-01_T1w.nii.gz` and a moving image `sub-02_T1w.nii.gz`. The rest of the
files are introduced when they are needed. The summary of the conventions is
on [Common conventions](../conventions.md#spatial-transformations).

## The mental model

### Two representations: matrices and fields

A **linear** (affine) transformation is a 4&times;4 matrix that acts on
homogeneous world coordinates `[x, y, z, 1]` in millimetres. It is stored in a
small text file in the FreeSurfer LTA format, usually with the extension `.lta`.

A **dense** transformation assigns a vector to every point of a regular lattice.
It is stored as an ordinary volume with three components in the last dimension,
and the header of that volume says where the lattice sits in world space. The
lattice need not be the grid of any of your images: a smooth deformation needs
far fewer samples than an image, so a field may be defined on a coarser lattice.

### Two kinds of dense fields: displacement and velocity

A **displacement field** $u$ says how far to move. Applying it to a point $x$
gives $x + u(x)$.

A **velocity field** $v$ says how fast to move. The transformation is obtained by
following the velocity for one unit of time, which is written $\exp(v)$ and
computed by the *scaling and squaring* method (the field is divided by $2^N$,
then composed with itself $N$ times). `register` estimates velocity fields by
default, because $\exp(v)$ is smooth and invertible, and the inverse is the
exponential of the opposite velocity, $\exp(-v)$. A displacement field has no
such closed-form inverse: inverting it is a numerical approximation.

Which kind a file is cannot be told from the file alone: both are volumes with
three components. You must know it from where the file comes from (`svf.nii.gz`
is a velocity field), and you tell commands about it with the option you use
(`-v` for a velocity, `-d` for a displacement).

### Direction: what does a transformation map?

Commands that apply transformations, such as `reslice`, work the way any
resampling works: they go through the voxels of the *output* image and, for
each one, look up the value at the corresponding place in the *input* image.
A transformation is therefore a map *from the output space to the input space*.
This map is what NITorch stores and what you give to `reslice`.

For a registration, the output of the resampling is the fixed grid and the input
is the moving image. So the matrix that `register` writes takes **fixed**
coordinates to **moving** coordinates: "fixed to moving". This sounds
backwards, because the image moves the other way. Keep the two readings
separate:

- as a *coordinate* map, the transformation goes from the fixed space to the
  moving space;
- as an *image* operation, it brings the moving image into the fixed space.

Hence the rule of thumb. To resample the moving image onto the fixed grid, use
the transformation as it is written. To resample the fixed image onto the moving
grid, use its inverse. For the same reason, a point (a landmark, a vertex of a
surface) that is expressed in moving space needs the *inverse* to be moved to
the fixed space.

### Units: world space and voxels

A matrix is always expressed in world coordinates, in millimetres. A field is
different. By default, the vectors in a field are expressed in **voxels of the
field's own lattice**: a displacement of `1.0` means "one voxel of the lattice
that the field file defines", whatever the size of that voxel. A velocity field
is always read in the same way. Two consequences follow. First, a field file is
meaningless without its header, since it is the header that gives the size of a
voxel. Second, resampling a field to a finer or coarser lattice does not only
change its shape, it also has to rescale its values, which a generic image
resampler does not do. Leave the lattice of a field alone unless you know how
the values must be adapted.

Fields can be converted to millimetres when you need them for another software,
but be careful with that conversion (see the notes in the steps below).

### Chains

You can apply several transformations in one go. `reslice` and `compose` take
them as a list, and the rule is the one of function composition. If you list
$T_1, T_2, \dots, T_n$ in this order, a point of the output space is first moved
by $T_n$, then by $T_{n-1}$, and last by $T_1$. The *last* transformation on the
command line is the one that touches the output grid first. Writing the list
from left to right, you describe the path from the input image towards the
output space; at the output end, the point lands in input space, where it is
interpolated.

For a few transformations of the same kind this is the familiar rule for
matrices: `-l A.lta -l B.lta` is the matrix product $AB$, which applies $B$
first.

An inverted transformation is written by adding an `i` to the tag: `-il`
(linear), `-id` (displacement), `-iv` (velocity). The inverse of a chain is the
chain in reverse order, with every element inverted.

## Step 1: look at your transformation files

A matrix is a text file you can read:

```shell
cat reg-affine/affine.lta
```

```text
type      = 1  # LINEAR_RAS_TO_RAS
nxforms   = 1
1 4 4
0.9999923 0.0038641477 -0.00063938176 -2.7163801
-0.0038651135 0.99999136 -0.0015160437 4.215938
0.00063351804 0.0015185033 0.9999986 1.6569008
0.0 0.0 0.0 1.0
```

(These numbers are illustrative.) The first line says the matrix maps RAS
coordinates to RAS coordinates, in millimetres. The three rows of four numbers
and the final row `0 0 0 1` are the matrix. Here, it is close to the identity
matrix with a translation of a few millimetres.

A field is a volume, so `info` describes it:

```shell
nitorch info reg-nonlin/svf.nii.gz
```

Look at the shape and at the voxel size. The last dimension, of size three, holds
the components; the first three tell you the lattice on which the field is
defined, which is the lattice in whose voxels the values are expressed. The
lattice of a velocity field need not be the grid of the images it was estimated
from.

## Step 2: compose affine transformations

Suppose you have a second registration, of `sub-01_T1w.nii.gz` to a template,
with the template as the *fixed* image:

```shell
nitorch register -v -o reg-template \
    @loss cc \
        @@fix template_T1w.nii.gz \
        @@mov sub-01_T1w.nii.gz \
    @affine affine
```

It writes `reg-template/affine.lta`, which maps template coordinates to
`sub-01` coordinates. You already have `reg-affine/affine.lta`, which maps
`sub-01` coordinates to `sub-02` coordinates. To resample `sub-02` onto the
template, you need the map from template to `sub-02`, which is the second matrix
followed by the first. With the rule of the previous section, that is:

```shell
nitorch compose -l reg-affine/affine.lta -l reg-template/affine.lta \
    -o sub-02_in_template.lta
```

The tag order mirrors the path from the input image (`sub-02`) to the output
space (the template): `sub-01`-to-`sub-02` next to the input, and the
template-to-`sub-01` map last. When all the transformations are linear, the
result is a single matrix, written as an LTA file. You can use it directly:

```shell
nitorch reslice sub-02_T1w.nii.gz -l sub-02_in_template.lta \
    -t template_T1w.nii.gz -o sub-02_in_template.nii.gz
```

!!! tip "Check a composition by composing it with its inverse"
    A transformation followed by its own inverse must give the identity. Run
    `nitorch compose -l reg-affine/affine.lta -il reg-affine/affine.lta -o id.lta`
    and read `id.lta`: the diagonal should be 1 and the other entries zero, up to
    rounding errors of the order of `1e-7`. If you get something else, you have
    mixed up two files.

## Step 3: invert

To invert a matrix, compose it with the inverse tag and nothing else:

```shell
nitorch compose -il reg-affine/affine.lta -o moving_to_fixed.lta
```

`reg-affine/affine.lta` took fixed coordinates to moving coordinates; the new file
takes moving coordinates to fixed ones. Use it as the transformation to
resample the *fixed* image onto the moving grid, or to carry landmarks from the
moving image to the fixed one. You do not need to compute the inverse when you
only want to use it in `reslice` or in a larger `compose`: write `-il` there.

For a velocity field, the inverse is the exponential of the opposite velocity,
and the `-iv` tag does exactly that. You never need to negate a field yourself.

## Step 4: exponentiate a velocity field

[`vexp`](../reference/spatial/vexp.md) turns a velocity field into the
transformation that it encodes:

```shell
nitorch vexp reg-nonlin/svf.nii.gz -o reg-nonlin/svf.disp.nii.gz
```

The output is a displacement field $u$ such that the transformation is
$x \mapsto x + u(x)$. It has the same lattice as the velocity field, and its
values are in voxels of that lattice. Without `-o`, the name would be
`svf.vexp.nii.gz`. Add `-t transformation` to get the coordinates $x + u(x)$
themselves instead of the shifts, and `-n` to change the number of scaling and
squaring steps, 8 by default.

You will need `vexp` when you want to inspect or measure the deformation, or to
hand it to another program that expects a displacement field. For resampling in
NITorch, you do not need it: `reslice` and `compose` take the velocity field
directly (`-v`) and exponentiate it themselves.

!!! warning "Two things `vexp` does not do (yet)"
    The `-inv` option is accepted but has no effect, so `vexp` always writes the
    forward transformation. To get the inverse, use `-iv` in `compose` or
    `reslice`. And the `-u mm` option, which converts the output to millimetres,
    applies the whole orientation matrix to displacements, translation included;
    it is only reliable for `-t transformation`. To get a displacement in
    millimetres, use `compose -ou mm`, as in the next step.

## Step 5: combine a matrix and a field into one dense field

You might need one dense file for a whole registration, for example to give to
another program. `compose` writes such a file when you list a matrix and a field
together. The chain depends on how the registration was set up, because the
affine part of `register` can be placed on three sides of the deformation:
symmetrically (the default, half on each side), on the moving image, or on the
fixed image. See [Position of the affine](../reference/registration/register.md#position-of-the-affine).
With the default symmetric position, the model is

$$
x \;\mapsto\; A^{1/2}\bigl(\varphi(A^{1/2}x)\bigr),
$$

where $A$ is the matrix in `affine.lta` and $\varphi=\exp(v)$ is the field.
`reslice` can apply a matrix's square root (the `-l2` tag), but `compose` cannot.
To obtain a single dense field, register with the affine on the moving image,
which makes the model $x \mapsto A\,\varphi(x)$:

```shell
nitorch register -v -o reg-mov \
    @loss cc \
        @@fix sub-01_T1w.nii.gz -o false \
        @@mov sub-02_T1w.nii.gz -o false -r \
    @affine affine -p mov \
    @nonlin svf
```

Then the chain that goes from the moving image to the fixed grid is the matrix
(next to the input) followed by the velocity field (next to the output):

```shell
nitorch compose -l reg-mov/affine.lta -v reg-mov/svf.nii.gz \
    -t sub-01_T1w.nii.gz -o reg-mov/to-fixed.nii.gz
```

`-t` names the grid on which the field is written. Without it, `compose` uses the
grid of the last field in the chain, here the (coarse) lattice of
`svf.nii.gz`. The output is a displacement field defined on the grid of
`sub-01_T1w.nii.gz`, in voxels of that grid. It can be given to `reslice` as a
`-d` field, and it reproduces the chain:

```shell
nitorch reslice sub-02_T1w.nii.gz -d reg-mov/to-fixed.nii.gz \
    -t sub-01_T1w.nii.gz -o sub-02_T1w.in-sub-01.nii.gz
```

The result should be the same as `-l reg-mov/affine.lta -v reg-mov/svf.nii.gz`,
and the same as the `sub-02_T1w.resliced.nii.gz` image that `register` wrote
(because of its `-r`), up to interpolation noise.

For the opposite direction, from the fixed image to the moving grid, invert the
chain: reverse the order and invert each element.

```shell
nitorch compose -iv reg-mov/svf.nii.gz -il reg-mov/affine.lta \
    -o reg-mov/to-moving.nii.gz
```

The inverse of each element is well defined, but the composed field is obtained
by interpolating the fields, so, like every resampling of a field, it is an
approximation.

!!! note "Fields in millimetres"
    By default, `compose` writes the displacement in voxels of the target grid.
    With `-ou mm`, it writes it in millimetres of world space, which is what
    other software usually expects. Do not give a millimetre field back to
    NITorch with `-u mm`: in the current version, the conversion of
    millimetre displacements by `reslice` and `compose` is not reliable. Keep
    the voxel-unit field for your own pipeline.

## Step 6: resample through a chain with `reslice`

You can skip the dense file altogether and give the chain to `reslice`. This is
the most accurate way of applying a registration, since the fields are
interpolated once, and it is how you apply a registration to a new image. The
general form is:

```shell
nitorch reslice IMAGE... TRANSFORMATIONS... -t TARGET -o OUTPUT...
```

where `TARGET` is the image whose grid is wanted, and `TRANSFORMATIONS` is the
chain, from the input to the output space. These are the chains, for each
position of the affine, that take the moving image to the fixed grid and the
fixed image to the moving grid. In the table, `aff` stands for `affine.lta` and
`vel` for the velocity field.

| Position of the affine | Moving to fixed (resample `moving` with `-t fixed`) | Fixed to moving (resample `fixed` with `-t moving`) |
| ---------------------- | --------------------------------------------------- | --------------------------------------------------- |
| `sym` (default)        | `-l2 aff -v vel -l2 aff`                            | `-il2 aff -iv vel -il2 aff`                         |
| `mov`                  | `-l aff -v vel`                                     | `-iv vel -il aff`                                   |
| `fix`                  | `-v vel -l aff`                                     | `-il aff -iv vel`                                   |

A fixed-to-moving resampling is the one that brings, for example, a template's
atlas labels to a subject when the *template* is the moving image. For the
default registration of the previous tutorial, resampling the fixed image onto
the moving grid is:

```shell
nitorch reslice sub-01_T1w.nii.gz \
    -il2 reg-nonlin/affine.lta -iv reg-nonlin/svf.nii.gz -il2 reg-nonlin/affine.lta \
    -t sub-02_T1w.nii.gz -o sub-01_T1w.in-sub-02.nii.gz
```

Compare this with the table: the order of the chain is reversed and every
element is inverted. An affine on its own needs no square root: for the
affine-only registration, the fixed-to-moving resampling uses
`-il reg-affine/affine.lta` alone.

A few more things to know about `reslice`:

- **The output grid.** With `-t`, the output has the shape and orientation matrix
  of the target. Without it and with only linear transformations, the image is
  not resampled at all: the voxel data are copied and only the orientation
  matrix in the header changes. With a field in the chain and no `-t`, the output
  stays on the grid of the input. `-vx` changes the voxel size of the output
  grid.
- **Labels.** Use `-i l` for label maps and `-i 1` (the default, linear) for
  intensity images, as in the registration tutorial.
- **Always give `-o`.** The default output name of `reslice` makes the command
  fail for volumes.
- **Streamlines.** `reslice` also accepts streamline files. It applies the
  inverse of the chain to their points, so that the same transformations take
  images and streamlines alike from the moving space to the fixed space. See
  [Streamlines](../reference/spatial/reslice.md#streamlines).

!!! warning "Two common mistakes"
    **Applying a matrix backwards.** If an image comes out displaced by the
    wrong amount (twice the shift, or in the wrong direction), you have probably
    used `-l` where `-il` was needed, or the reverse. Test with a small
    experiment: resample the *fixed* image with the matrix and with its inverse
    and look at which one lines up with the moving image.

    **Reversing the order of a chain.** Chains are not commutative. The model
    of a symmetric registration has the field *between* two copies of the
    matrix. Writing the matrix once, in front of the field, is accepted by
    `reslice` and produces a plausible but wrong image.

## Next steps

- The [`reslice`](../reference/spatial/reslice.md),
  [`compose`](../reference/spatial/compose.md) and
  [`vexp`](../reference/spatial/vexp.md) pages describe every option, including
  the ones this tutorial skipped, and list the known limitations.
- [`register`](../reference/registration/register.md#applying-the-transformations)
  tells which chain to use for each model and position of the affine.
- [Common conventions](../conventions.md#spatial-transformations) summarises
  the conventions for all commands, including the file formats used for
  transformations.
