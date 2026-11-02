---
icon: octicons/terminal-24
---

# Command line

The `nitorch` program gathers NITorch's tools into one command-line
interface. Each tool is a subcommand: you inspect an image with `nitorch info`,
crop it with `nitorch crop`, register two images with `nitorch register`, and
so on. The subcommands read and write standard neuroimaging files, so they fit
into shell scripts and batch jobs alongside other software, and none of them
requires you to write Python.

## How the program works

The general form is

```shell
nitorch COMMAND [ARGUMENTS...]
```

where `COMMAND` is the name of one of the [reference](reference/index.md)
pages. Run `nitorch` on its own to list all commands. To learn about one of
them, ask for its help text in either of two equivalent ways:

```shell
nitorch help crop
nitorch crop -h
```

Everything also works through the Python interpreter, which is handy when the
`nitorch` script is not on your `PATH`:

```shell
python -m nitorch crop -h
```

The help text is a compact summary of the syntax. The reference pages go
further: they explain what each command does, list every option with its
default, and show examples.

Commands take their arguments in a consistent style. Input files come first,
then options, each option followed by its values:

```shell
nitorch COMMAND INPUT... [-option VALUE...] [-option VALUE...]
```

Most commands write their results next to the input files under a name derived
from the input name, such as `T1w.crop.nii.gz` for `T1w.nii.gz`, and let you
choose another name with `-o`. The conventions that all commands share (how
options are parsed, which formats are supported, how names are built, how
transformations are stored) are described once, in
[Common conventions](conventions.md).

!!! note "A warning at start-up"
    Until you build the [compiled backend](../install/compiled.md), every
    command prints a warning that NITorch is using its non-compiled backend.
    It is harmless. Set `NI_CHECK_BACKEND=0` to silence it.

## A first session

The following steps need one image, which we call `T1w.nii.gz` here. Replace
it with any NIfTI or MGH volume of your own.

Start by looking at the image. [`info`](reference/images/info.md) prints the
file name, the shape of the volume, its orientation, the position of its
centre in millimetres and the stored metadata:

```shell
nitorch info T1w.nii.gz
```

```text
filename        : T1w.nii.gz
shape           : (176, 240, 256)
layout          : RAS
center          : (0.0, 0.0, 0.0) mm (RAS)
format          : Nifti1
affine          : [[   1.    0.    0.  -88.]
                   [   0.    1.    0. -120.]
                   [   0.    0.    1. -128.]
                   [   0.    0.    0.    1.]]
dtype           : int16
...
```

(The values depend on your image.) Add `-s` to compute intensity statistics
as well.

Next, crop a cube of 64 voxels out of the centre of the field of view with
[`crop`](reference/images/crop.md):

```shell
nitorch crop T1w.nii.gz -s 64 64 64
```

```text
Cropping patch [56:120, 88:152, 96:160] from volume with shape [176, 240, 256]
```

The result is written next to the input as `T1w.crop.nii.gz`. The orientation
matrix of the new file has been updated, so that it describes the same place
in world space as the corresponding part of the original image. Check it:

```shell
nitorch info T1w.crop.nii.gz
```

To choose where the result goes, give the output name yourself, either in full
or as a template that reuses parts of the input name (see
[Output filenames](conventions.md#output-filenames)):

```shell
nitorch crop T1w.nii.gz -s 64 64 64 -o '{dir}{sep}{base}.roi{ext}'
```

Finally, [`convert`](reference/images/convert.md) changes the file format. This
writes `T1w.mgz`:

```shell
nitorch convert T1w.nii.gz -f mgz
```

## Where to go from here

- [Common conventions](conventions.md) explains the rules that every command
  follows. Read it before your first long command line.
- The [Reference](reference/index.md) has one page per command, grouped by
  purpose: images, orientation and resampling, registration, quantitative MRI,
  denoising and segmentation, and visualisation.
- The [Tutorials](tutorials/index.md) walk through complete tasks, such as
  preparing images and registering two images.
