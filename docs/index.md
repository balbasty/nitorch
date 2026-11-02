---
icon: octicons/home-24
---

<div style="text-align: center" markdown>

![NITorch](images/nitorch_logo_v0.1.png){ width="50%" }

**N**euro**I**maging in Py**Torch

</div>

# NITorch

NITorch is a toolbox for **medical image processing and analysis**, with a
focus on neuroimaging. It is written in [PyTorch](https://pytorch.org), which
gives it two properties that are rare in a single package: its algorithms run
unchanged on a CPU or on a GPU, and its building blocks are differentiable, so
they can be used inside neural networks as well as inside classical
optimisation-based pipelines.

Much of NITorch is about **image registration and spatial transformation**. It
implements affine matrices (including their Lie-algebra encoding), dense
displacement fields, B-spline-encoded deformations, stationary velocity fields
and geodesic shooting. On top of these it provides registration tools that
work from the command line, algorithms for quantitative MRI and for inverse
problems in MRI, a reimplementation of SPM's Unified Segmentation, and a set of
small utilities for cropping, padding, resampling and inspecting volumes.

NITorch has been mostly written by Yael Balbastre and Mikael Brudfors in
John Ashburner's group at the Wellcome Centre for Human Neuroimaging (the FIL),
and is conceptually related to [SPM](https://www.fil.ion.ucl.ac.uk/spm/). It is
released under the MIT license.

## Who it is for

NITorch is for people who process brain images and want control over what
happens: researchers who build pipelines in shell scripts, who need a
registration or a resampling step that behaves the same way every time, or who
need to run an algorithm on a cluster without writing any code. It is equally
for people who write Python and want tested, differentiable building blocks
for registration, resampling and image-based losses.

## Two ways to use it

**The command line** is the focus of this site. The `nitorch` program bundles
the tools into subcommands that read and write NIfTI and MGH volumes, so they
fit into a shell pipeline next to FreeSurfer, FSL or SPM. You do not need to
know any Python to use them. Start with the [command-line overview](cli/index.md).

**The Python library** exposes the same algorithms, and the lower-level
functions and neural-network layers they are built from. The Python
documentation is still being written; see the [Python page](python/index.md)
for an overview of what is there.

## What the command line can do

<div class="grid cards" markdown>

-   :octicons-file-media-24: **[Images](cli/reference/images/index.md)**

    Inspect, convert, crop, pad, split, pool, smooth and inpaint volumes, and
    compute overlap metrics between images.

-   :octicons-move-to-end-24: **[Orientation and resampling](cli/reference/spatial/index.md)**

    Fix or change the orientation of a volume, resize it, reslice it through
    linear, dense and velocity transformations, compose transformations and
    compute a mean space.

-   :octicons-git-compare-24: **[Registration](cli/reference/registration/index.md)**

    Pairwise affine and non-linear registration with a choice of similarity
    measures and optimisers, and groupwise normalisation of several scans of a
    subject.

-   :octicons-pulse-24: **[Quantitative MRI](cli/reference/qmri/index.md)**

    Fit T1, T2\* and MTsat maps from multi-echo gradient-echo data, fit
    exponential decays, and correct geometric distortions of echo-planar data.

-   :octicons-sliders-24: **[Denoising and segmentation](cli/reference/denoising/index.md)**

    Total-variation denoising of images and of MRI volumes, and tissue
    segmentation with a reimplementation of SPM's Unified Segmentation.

-   :octicons-eye-24: **[Visualisation](cli/reference/viewer/index.md)**

    View volumetric images in an interactive window.

</div>

The [reference](cli/reference/index.md) lists every command with a one-line
summary.

## Quick start

Install NITorch (this also installs PyTorch if it is missing; see
[Installation](install/index.md) for GPU support), look at an image, and crop
a 64-voxel cube out of its centre:

```shell
pip install "nitorch[io,plot] @ git+https://github.com/balbasty/nitorch"
nitorch info T1w.nii.gz
nitorch crop T1w.nii.gz -s 64 64 64
```

The last command writes `T1w.crop.nii.gz` next to the input.

## Where to go next

<div class="grid cards" markdown>

-   :octicons-download-24: **[Installation](install/index.md)**

    Install NITorch, set up PyTorch for your GPU, and check that it works.

-   :octicons-terminal-24: **[Command line](cli/index.md)**

    How the `nitorch` program works, with a first session you can follow.

-   :octicons-book-24: **[Common conventions](cli/conventions.md)**

    How options are parsed, which file formats are supported, how output files
    are named, and how transformations and coordinates are expressed.

-   :octicons-list-unordered-24: **[Reference](cli/reference/index.md)**

    One page per command, with every option and examples.

-   :octicons-mortar-board-24: **[Tutorials](cli/tutorials/index.md)**

    Worked examples: preparing images, registering two images and working with
    transformations.

-   :fontawesome-brands-python: **[Python](python/index.md)**

    The library behind the commands.

</div>

NITorch is developed on [GitHub](https://github.com/balbasty/nitorch), where
you can report problems and propose changes.
