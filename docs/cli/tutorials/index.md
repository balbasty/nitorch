---
icon: octicons/mortar-board-24
---

# Tutorials

The [reference pages](../reference/index.md) describe each command on its own:
what every option does, which values it accepts, which files it writes. That
is what you need when you know which command to run. It is less helpful when
you have a folder of scans and a goal, such as "get these two brains into the
same space", and you have to work out which commands to chain, in which order,
and what to check along the way.

The tutorials fill that gap. Each one follows a realistic task from start to
finish, runs real commands on real images, explains why each step is there,
and points out the mistakes that are easy to make. They are meant to be read
in order, but each one stands alone: you can jump straight to the one that
matches your task.

## What you need

- **A working installation of NITorch**, so that the `nitorch` command runs.
  See the [installation instructions](../../install/index.md). You can check
  with `nitorch info -h`. If the `nitorch` script is not on your `PATH`,
  replace `nitorch` with `python -m nitorch` in every command below; see
  [Running commands](../conventions.md#running-commands).
- **A few NIfTI images.** Any format NITorch reads will do (see
  [File formats](../conventions.md#file-formats)), but the examples use
  `.nii.gz`.
- **A terminal** in a folder that you can write to. Every command in the
  tutorials writes new files and none of them modifies its input, except where a
  warning says otherwise.
- **Patience, or a GPU.** Registration is the only slow step. A
  full-resolution registration of two T1-weighted scans can take a long time on
  a CPU, especially without the compiled backend, and is much faster on a GPU
  (see [Hardware](../conventions.md#hardware)). The registration tutorial shows
  how to keep the first attempts short.

### Getting sample images

You do not need special data. The tutorials use T1-weighted anatomical scans of
the head, which are the most common kind of MRI image and are easy to find
publicly.

[OpenNeuro](https://openneuro.org) hosts a large number of open datasets
organised according to the BIDS standard. In such a dataset, anatomical
images live in `sub-*/anat/` and are named like `sub-01_T1w.nii.gz`. Pick a
dataset whose licence suits you, and download the T1-weighted image of two
different participants. The website lets you download individual files, and
the OpenNeuro command-line tool and DataLad can fetch them from a terminal.
Check the licence of the dataset before you redistribute anything.

How many images you need depends on the tutorial:

| Tutorial                                                           | Images you need                                                                                          |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| [Inspecting and preparing images](preparing.md)                    | One T1-weighted scan.                                                                                    |
| [Registering two images](registration.md)                          | Two T1-weighted scans (two participants, or two sessions of one participant). Optionally, a segmentation (label map) of the second one, from any segmentation tool. |
| [Working with transformations](transforms.md)                      | The files written by the registration tutorial.                                                          |

In the tutorials, images are called `T1w.nii.gz`, `sub-01_T1w.nii.gz`,
`sub-02_T1w.nii.gz` and so on. Replace these names with your own.

### Conventions used in the tutorials

Commands are shown without a shell prompt, so that you can copy them as they
are. A trailing backslash (`\`) continues a command on the next line; it is
only there for readability. Output that was copied from a real run says
nothing; output that is shortened or adapted, because the exact numbers depend
on your images, is labelled *illustrative*.

The first time a command runs in a session, it prints a warning that NITorch is
using its non-compiled backend, unless you have installed the
[compiled backend](../../install/compiled.md). The warning is harmless and is
left out of the examples.

## The tutorials

1. **[Inspecting and preparing images](preparing.md)** walks through a typical
   preparation of an anatomical scan: read its header with `info`, change its
   format or data type with `convert`, put it in a standard orientation with
   `reorient`, trim or pad the field of view with `crop` and `pad`, change its
   resolution with `resize`, and blur it with `smooth`. It also explains when
   `orient`, the dangerous sibling of `reorient`, is the right tool.
2. **[Registering two images](registration.md)** aligns a moving T1-weighted
   scan to a fixed one with `register`, first with an affine transformation and
   then with an added nonlinear component, explains the files that come out,
   and applies the result to the image itself and to a label map with
   `reslice`.
3. **[Working with transformations](transforms.md)** builds the mental model
   behind the transformation files (affine or dense, displacement or velocity,
   fixed-to-moving or moving-to-fixed, voxels or millimetres), then shows how to
   combine transformations with `compose`, invert them, turn a velocity field
   into a displacement field with `vexp`, and resample through a chain with
   `reslice`.

When you have finished, the [command reference](../reference/index.md) is the
place to look up the options that the tutorials leave out, and
[Common conventions](../conventions.md) explains the rules shared by all
commands: how options are parsed, how output files are named, and how
transformations are stored.
