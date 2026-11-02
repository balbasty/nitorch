---
icon: octicons/git-merge-24
---

# Registration

Registration finds the spatial transformation that brings one image into
alignment with another. NITorch's registration commands cover the whole
range, from a two-line rigid alignment of two scans to a groupwise
normalisation of several sessions of one subject, and they all rest on the
same engine: the [`register`](register.md) command. The other commands are
either simplified front ends to it, or tools that combine its results.

Registering an image pair involves four decisions: *how* the images are
compared (a loss such as mean squared error, correlation or mutual
information), *what* deformations are allowed (translations, rigid, affine,
or a dense diffeomorphic field with a smoothness penalty), *how* the
optimisation proceeds (solver, number of iterations, coarse-to-fine
pyramid), and *which* files you get back. `register` exposes all four;
the wrappers make them for you. If you have never registered images with
NITorch, the [registration tutorial](../../tutorials/registration.md) walks
through a first example.

## Choosing a command

**You want to align two images, and are happy with a rigid or full affine
transformation.** Use [`affreg`](affreg.md). It takes `-m MOVING -f FIXED`,
picks sensible defaults (mutual information, three pyramid levels,
a symmetric model) and writes a transformation file and, on request, the
moving image resampled onto the fixed image.

**You need more control**, for example a different affine model, several
image pairs, a mix of intensity images and segmentations, a nonlinear
deformation, or a different optimiser or regularisation. Use
[`register`](register.md). Its syntax is nested, so its page begins with an
explanation of the grammar; once you have seen it, the options are
uniform. `affreg` is a good way to start, because the page shows the exact
`register` command it translates to.

**You have several scans of one subject** (sessions, time points, contrasts)
and want them all aligned to a common space without choosing a reference.
Use [`optinorm`](optinorm.md), which registers all pairs and computes the
transformation of each image to a mean space.

**You already have pairwise registrations**, produced with `register` or
elsewhere, for several images, and want to convert them into
transformations between each image and a common ("template") space. Use
[`affopt`](affopt.md) for affine transformations and [`vopt`](vopt.md) for
stationary velocity fields. `optinorm` chains them for you in the common
within-subject case, but you can use them directly on many subjects, or on
incomplete sets of pairs.

**You have scripts that use [`autoreg`](autoreg.md).** It is the original
registration tool of NITorch, written on top of automatic differentiation
(PyTorch's autograd, with the Adam optimiser by default), and has a
single-dash syntax of its own. It is still installed and works, but its
source describes its own implementation as "a giant mess", and the
newer `register` offers many more losses, solvers and options.
Consider it for existing pipelines, or for the two things `register` does not
offer: free-form deformations (cubic B-splines) and the joint total variation
loss. For new work, prefer `register`.

[`simplereg`](simplereg.md) was meant as a simplified front end alongside
`affreg`. In this version of NITorch it is an affine-only equivalent of
`affreg`, and its page explains what that means.

Once a registration has been run, the transformations it wrote can be
applied to images with [`reslice`](../spatial/reslice.md), composed with
[`compose`](../spatial/compose.md), and exponentiated with
[`vexp`](../spatial/vexp.md).

## Commands

| Command                       | Summary                                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [`register`](register.md)     | Pairwise registration with an affine and/or nonlinear model, many losses, and full control of the optimisation.      |
| [`simplereg`](simplereg.md)   | Intended as a simplified wrapper of `register`; currently an affine-only equivalent of `affreg`.                    |
| [`affreg`](affreg.md)         | Affine (rigid or full) registration of two images with a simple `-m`/`-f` interface; a wrapper of `register`.        |
| [`optinorm`](optinorm.md)     | Unbiased groupwise alignment of several scans of one subject to their mean space, with affine then nonlinear steps.  |
| [`affopt`](affopt.md)         | Computes optimal template-to-subject affine matrices from pairwise subject-to-subject matrices.                      |
| [`vopt`](vopt.md)             | Computes optimal template-to-subject velocity fields from pairwise subject-to-subject fields.                        |
| [`autoreg`](autoreg.md)       | The older registration tool, based on automatic differentiation, with free-form and diffeomorphic deformations.      |

## Conventions

The registration commands work in world (RAS) space, which means the images
do not need to share a voxel grid. The affine matrices written by `register`
and by the commands built on it (`affreg`, `simplereg`, `optinorm`) are in
RAS-to-RAS convention and map coordinates of the fixed image to coordinates
of the moving image; `autoreg` follows its own convention, described on its
page. These and the other conventions for transformation files are described
in
[Spatial transformations](../../conventions.md#spatial-transformations), and
the rules for output filenames in
[Output filenames](../../conventions.md#output-filenames).
