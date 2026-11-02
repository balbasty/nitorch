---
icon: octicons/book-24
---

# Inspecting and preparing images

Almost every neuroimaging analysis starts with the same chore: before you
register, segment or measure anything, you check what you were given and put it
in a shape that the next tool is happy with. This tutorial walks through that
chore for a single T1-weighted scan, using the small commands of NITorch that
read and rewrite images.

## What you will learn

- how to read an image's shape, voxel size, orientation and data type with
  [`info`](../reference/images/info.md);
- how to change the file format or the data type with
  [`convert`](../reference/images/convert.md), and why the default way of
  changing the data type is dangerous;
- what an orientation *layout* is, how to store an image in a standard layout
  with [`reorient`](../reference/spatial/reorient.md), and when to use
  [`orient`](../reference/spatial/orient.md) instead (rarely);
- how to cut out a region with [`crop`](../reference/images/crop.md) and add a
  margin with [`pad`](../reference/images/pad.md);
- how to change the voxel size with [`resize`](../reference/spatial/resize.md);
- how to blur an image with [`smooth`](../reference/images/smooth.md).

You need one T1-weighted scan. The [tutorial index](index.md#getting-sample-images)
suggests where to find one. Put it in an empty folder, call it `T1w.nii.gz`,
and run the commands from that folder. None of the commands below modifies its
input: each one writes a new file whose name is derived from the input's, with
one exception that is flagged where it comes up.

## Step 1: read the header

Start by asking the file what it contains:

```shell
nitorch info T1w.nii.gz
```

The output looks like this (illustrative; your numbers will differ):

```text
filename        : T1w.nii.gz
shape           : (176, 256, 256)
layout          : LAS
center          : (-1.2, 14.5, 8.9) mm (RAS)
format          : Nifti1
affine          : [[  -1.     0.     0.   87.5 ]
                   [   0.     1.     0.  -113.  ]
                   [   0.     0.     1.  -119.  ]
                   [   0.     0.     0.     1.  ]]
dtype           : int16
voxel_size      : (1.0, 1.0, 1.0)
voxel_size_unit : mm
slope           : 1.0
inter           : 0.0
```

The first four lines (`filename`, `shape`, `layout` and `center`) are computed
by NITorch from the file and its orientation matrix; the rest is read from the
file's header.

- `shape` is the number of voxels along each axis. A fourth number means that
  the file holds several volumes or channels.
- `layout` tells you how the voxel axes are stored on disk. It has three
  letters, one per axis, naming the direction in which the voxel index grows: `R`
  is left-to-right, `L` is right-to-left, `A` is posterior-to-anterior, `P` is
  anterior-to-posterior, `S` is inferior-to-superior and `I` is
  superior-to-inferior. `LAS` therefore means that the first axis runs from the
  right of the head to the left, the second from the back to the front, and the
  third from the bottom to the top. The layout says nothing about the quality of
  the orientation matrix; it is derived from it.
- `center` is the position, in millimetres in world (RAS) space, of the centre of
  the field of view. A value far from the origin is not an error, but a centre
  hundreds of millimetres away is a hint that the header deserves a closer look.
- `affine` is the orientation matrix: the 4&times;4 matrix that maps voxel
  coordinates to world coordinates. See
  [Coordinates and units](../conventions.md#coordinates-and-units).
- `dtype`, `slope` and `inter` describe how the numbers are stored on disk. In a
  NIfTI file, the stored value `v` stands for the intensity `slope * v + inter`.

To see only certain fields, name them with `-m`. The names are the ones in the
output above (`affine`, `dtype`, `voxel_size`, `voxel_size_unit`, `slope`,
`inter`, `format`, and a few others; see the
[reference](../reference/images/info.md)):

```shell
nitorch info T1w.nii.gz -m voxel_size dtype
```

Add `-s` to read the voxel values and print their minimum, maximum and mean.
This is the quickest way to spot an image that is empty, saturated, or stored
with a surprising intensity range:

```shell
nitorch info T1w.nii.gz -s
```

You can pass several files at once, which is handy for checking that a whole
batch of images has the same shape, voxel size and layout:

```shell
nitorch info sub-01_T1w.nii.gz sub-02_T1w.nii.gz -m voxel_size
```

`info` reads numbers; it does not show the image. To look at it, use
[`view`](../reference/viewer/view.md), which draws the three orthogonal slices
in world space and can show several images side by side:

```shell
nitorch view T1w.nii.gz
```

Keep this window handy: it is the best way to confirm, at each later step, that
what you did to the image is what you meant to do.

## Step 2: convert the format or the data type

[`convert`](../reference/images/convert.md) rewrites an image in another file
format, with another data type, or both. The format is taken from the extension
of the output, or from `-f`. For instance, to write a FreeSurfer MGZ file:

```shell
nitorch convert T1w.nii.gz -f mgz
```

This creates `T1w.mgz` next to the input.

!!! warning "The default output name is the input name"
    The default output of `convert` is `{dir}{sep}{base}{ext}`: the same
    directory, name and extension as the input. If you change only the data
    type, and not the format, `convert` overwrites the original file. Always
    pass `-o` when you do not change the extension. See
    [Output filenames](../conventions.md#output-filenames).

Changing the data type is where care is needed. Suppose you want to store the
image as 16-bit integers to save space:

```shell
nitorch convert T1w.nii.gz -dt int16 -c rescale -o T1w_int16.nii.gz
```

The `-c` option sets the *casting* rule, which decides what happens to values
that do not fit in the new type:

| Value          | What it does                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------- |
| `unsafe`       | The default. A plain cast: the values are converted as they are, fractions are dropped, and values outside the range of the new type give meaningless results. |
| `rescale`      | The intensities are rescaled so that the minimum and maximum of the image map to the minimum and maximum of the new type. In a NIfTI file the scaling is recorded in the header, so that readers that honour it recover the original intensities. |
| `rescale_zero` | As `rescale`, but zero stays zero. Use it for images where zero means "nothing" (backgrounds, masks, probability maps).      |

Converting a floating-point image to `uint8` with the default rule is a typical
trap. Fractional values lose their fractions, so a probability map with values
between 0 and 1 becomes an image of zeros and ones, and negative values, which
cannot be represented, come out as unrelated numbers. If the target type is an integer type and the values
are not already integers in its range, choose `rescale` or `rescale_zero`.

Check what you got with `info`:

```shell
nitorch info T1w_int16.nii.gz -m dtype
```

```text
dtype : int16
```

## Step 3: check and standardise the orientation

Medical images are stored in whatever axis order and direction the scanner or
the conversion tool happened to use. Your `info` output told you the layout of
your file. Many tools, and many of your own commands, are easier to reason
about when all images share one layout. `RAS` is the usual choice, because then
increasing a voxel index moves you to the right, forwards and up, which is the
way world coordinates grow.

[`reorient`](../reference/spatial/reorient.md) does that:

```shell
nitorch reorient T1w.nii.gz
```

By default it targets `RAS` and writes `T1w.RAS.nii.gz`; use `-l` to ask for
another layout, such as `-l LAS`. Look at the result:

```shell
nitorch info T1w.RAS.nii.gz
```

```text
filename        : T1w.RAS.nii.gz
shape           : (176, 256, 256)
layout          : RAS
center          : (-1.2, 14.5, 8.9) mm (RAS)
...
```

The layout has changed, but the centre has not, and neither has the voxel
size. This is what `reorient` guarantees: it permutes and flips the voxel
array so that it follows the requested layout, and it rewrites the orientation
matrix to match, so that *every voxel keeps its position in world space*. The
two files describe the same head at the same place. Open them together to
convince yourself:

```shell
nitorch view T1w.nii.gz T1w.RAS.nii.gz
```

The two blocks show the same anatomy at the same cursor position. Only the
numbers on disk have been reshuffled. No interpolation takes place, so no
intensity is altered.

Why bother, if the image is the same? Because everything that is expressed in
*voxels* depends on the storage order. When you cut out 160 voxels along the
first axis in the next step, "first axis" means left-right in a `RAS` image
and could mean something else in the original file. Standardising the layout
first makes such numbers mean the same thing for every image in a study.

### When `orient` is the right tool (and the danger)

NITorch has a second, similarly named command, [`orient`](../reference/spatial/orient.md),
that does something very different. Where `reorient` trusts the orientation
matrix and moves the data, `orient` *distrusts the orientation matrix and
overwrites it*, leaving the data alone. After `orient`, the voxels are exactly
where they were in the file, but they now sit somewhere else in world space.

That is what you want when the header is wrong: an image whose orientation
matrix is missing, was lost by a conversion tool, or does not match the
scanner's coordinate frame, so that it appears flipped, rotated or miles away
from the image it belongs to. A typical case is a mask or segmentation written
by a tool that dropped the header, whose voxel grid is the same as that of a
correctly oriented scan. You then copy the orientation of the good scan onto
it with `-k`, which names the reference file:

```shell
nitorch orient mask.nii.gz -k T1w.nii.gz -o mask.fixed.nii.gz
```

Another is recentring an image whose translation is garbage, so that the centre
of its field of view is at the world origin, keeping everything else:

```shell
nitorch orient T1w.nii.gz -c 0 -o T1w.centred.nii.gz
```

!!! warning "`orient` destroys information"
    Do not run `orient` on an image whose orientation matrix is correct. Doing
    so changes where the image is in world space, and anything that was aligned
    with it, such as a registration, a segmentation or a mask, is no longer
    aligned. The original orientation matrix is lost from the output file.
    Always give an explicit `-o`, never overwrite your only copy, and look at
    the result with `view` next to a trusted image. By default `orient` also
    writes a transformation file, named after the input and the output layout
    (for example `T1w_to_LPS.lta`), recording what it did.

If you are unsure which command you need, the rule is: if the image looks right
in a viewer that reads the header, but is stored in an inconvenient order, use
`reorient`. If it looks wrong, with the right data and the wrong header, use
`orient`.

## Step 4: crop to the region you need

A whole-head T1-weighted scan contains a lot of air and neck. Cropping makes
later steps faster and uses less memory. [`crop`](../reference/images/crop.md)
takes a size and, optionally, a centre, both in voxels by default. To cut a
160 &times; 192 &times; 160 box from the middle of the field of view:

```shell
nitorch crop T1w.RAS.nii.gz -s 160 192 160
```

The command reports what it did (illustrative):

```text
Cropping patch [8:168, 32:224, 48:208] from volume with shape [176, 256, 256]
```

and writes `T1w.RAS.crop.nii.gz`. To cut the same box around a different
centre, give the voxel coordinates of the centre with `-c`. Voxel indices
start at zero:

```shell
nitorch crop T1w.RAS.nii.gz -s 160 192 160 -c 88 140 140
```

Cropping removes voxels and shifts the origin of the image, but it does not
move anything: the cropped image has an orientation matrix that places every
remaining voxel exactly where it was in world space. You can overlay the
cropped image on the original in `view` and they will coincide.

If the requested box extends beyond the image, it is clipped to the image
rather than padded, so the output can be smaller than the size you asked for.
Look at the `Cropping patch` line (or at `info`) if you need an exact size, and
use `pad`, below, to grow an image.

!!! warning "Stay with `-s` and `-c` for now"
    `crop` also has options to crop at the bounding box of the non-zero part of
    the image (`-b`), to copy the region of another file (`-k`) and to write the
    crop as a transformation file (`-t`). In the version of NITorch this
    tutorial was written against, these three options stop with an error, so the
    size and centre options above are the ones to rely on. The
    [`crop` reference](../reference/images/crop.md) says what each option is meant
    to do.

## Step 5: pad to add a margin

Padding is the opposite of cropping. It is useful to give an image room before
a rotation, a convolution or a registration that would otherwise cut off the
edges, or to reach a size that a model or a multiresolution scheme expects.
[`pad`](../reference/images/pad.md) takes a size too, but the size is the
amount added on *each side* of every axis:

```shell
nitorch pad T1w.RAS.crop.nii.gz -s 16 16 16
```

```text
Padding volume with shape [160, 192, 160] by [(16, 16), (16, 16), (16, 16)] voxels
```

The output is `T1w.RAS.crop.pad.nii.gz`, with a shape of 192 &times; 224
&times; 192. As with `crop`, the orientation matrix is updated so that the
original voxels keep their world position, and the new voxels extend the
field of view.

By default the new voxels are zero. Use `-b` to choose the boundary
condition: `replicate` repeats the edge voxels, which is gentler on an image
that does not fade to zero at its border, and a number fills the margin with
that value:

```shell
nitorch pad T1w.RAS.crop.nii.gz -s 16 16 16 -b replicate -o T1w.RAS.crop.padrep.nii.gz
```

Notice how the output names accumulate: `T1w.RAS.crop.pad.nii.gz` records
the history of the file, `reorient`, then `crop`, then `pad`. That is the
default naming scheme, described in
[Output filenames](../conventions.md#output-filenames), and you can replace it
with `-o` whenever you prefer a name of your own.

## Step 6: change the resolution

[`resize`](../reference/spatial/resize.md) resamples an image to a different
voxel size or matrix size. To get 2 mm voxels:

```shell
nitorch resize T1w.RAS.nii.gz -v 2
```

This writes `T1w.RAS.resized.nii.gz`. You specify the result in one of three
ways: a voxel size (`-v`), a resizing factor (`-f`, where 0.5 halves the number
of voxels along each axis, and therefore doubles the voxel size), or an exact
shape (`-s`). Values are linearly interpolated by default; `-i 3` selects cubic
interpolation instead, and `-i 0` nearest-neighbour interpolation, which is the
right choice for label maps.

Check the voxel size you obtained:

```shell
nitorch info T1w.RAS.resized.nii.gz -m voxel_size
```

You may be surprised that the voxel size is not exactly 2 mm (for a 256-voxel
axis it comes out near 2.008 mm). The reason is the *anchor*. By default,
`resize` keeps the field of view and aligns the *centres* of the first and last
voxels of the old and new grids (`-a centers`), so the voxel size has to absorb
any rounding of the matrix size. If you care more about the exact voxel size
than about the exact field of view, anchor the first voxel instead:

```shell
nitorch resize T1w.RAS.nii.gz -v 2 -a first -o T1w.RAS.2mm.nii.gz
```

With `-a first` the voxel size is exactly 2 mm and the matrix size is the
integer part of the ratio, so the field of view can shrink by up to a voxel
at the far end.

You may also notice that `info` now reports a fourth dimension of size one.
That is harmless: `resize` writes the image as a single-channel volume.

!!! tip "Smooth before reducing the resolution a lot"
    `resize` interpolates; it does not low-pass filter the image first. When you
    reduce the resolution by a large factor, fine structures that the coarser grid
    cannot represent can fold into the result as aliasing. Blur the image a little
    first (next step), with a FWHM comparable to the new voxel size.

## Step 7: smooth

[`smooth`](../reference/images/smooth.md) convolves an image with a Gaussian
kernel. You give the *full width at half maximum* (FWHM) of the kernel,
which relates to the standard deviation by FWHM &asymp; 2.355 &times; &sigma;.
The unit is millimetres by default, so the amount of blur does not depend on
the voxel size of the image:

```shell
nitorch smooth T1w.RAS.nii.gz -f 4 -o T1w.RAS.smooth4.nii.gz
```

A single value applies to all three axes. To use different widths per axis,
give three values, and add a unit if you want voxels instead of millimetres:

```shell
nitorch smooth T1w.RAS.nii.gz -f 2 2 2 vox -o T1w.RAS.smooth2vox.nii.gz
```

Without `-o`, the output is `T1w.RAS.smooth.nii.gz`, and without `-f` the width
is 1 mm. `smooth` preserves the orientation matrix and the shape of the image;
only the intensities change.

Smoothing is for intensity images. Do not smooth a label map: averaging
label numbers produces meaningless intermediate values at the borders between
structures.

## Putting it together

A typical preparation for a registration, applied to every scan in a study,
is to standardise the layout, cut the field of view down to the head, and
bring the voxel size to a common value. With the default names, the commands
chain like this:

```shell
nitorch reorient sub-01_T1w.nii.gz
nitorch crop sub-01_T1w.RAS.nii.gz -s 176 224 176
nitorch resize sub-01_T1w.RAS.crop.nii.gz -v 1 -a first -o sub-01_T1w.prep.nii.gz
```

The outputs of the first two commands are intermediate files that you can
delete; `sub-01_T1w.prep.nii.gz` is the image to carry forward. Run
`nitorch info` on all your prepared images at the end to confirm that they
agree.

## Next steps

- Use your prepared images in [Registering two images](registration.md).
- Read the reference pages for the options this tutorial did not need:
  [`info`](../reference/images/info.md),
  [`convert`](../reference/images/convert.md),
  [`reorient`](../reference/spatial/reorient.md),
  [`orient`](../reference/spatial/orient.md),
  [`crop`](../reference/images/crop.md),
  [`pad`](../reference/images/pad.md),
  [`resize`](../reference/spatial/resize.md) and
  [`smooth`](../reference/images/smooth.md).
- Understand how options and file names are handled in
  [Common conventions](../conventions.md#how-options-are-parsed).
