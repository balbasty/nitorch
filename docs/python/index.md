---
icon: fontawesome/brands/python
---

# Python

NITorch is first a Python library, built on [PyTorch](https://pytorch.org).
The [command-line tools](../cli/index.md) documented on this site are thin
wrappers around it. If you want to go beyond them, to embed a registration in
your own code, to use a transformation model as a layer in a neural network, or
to read and write volumes as PyTorch tensors, you can use the library directly.

!!! note "Documentation in progress"
    This section is a placeholder. API reference pages and Python tutorials are
    planned, but they do not exist yet. Until they do, the docstrings in the
    source code are the reference, and the demo notebooks below show the library
    at work.

## A tour of the packages

`import nitorch` makes the subpackages below available as attributes (for
example `nitorch.spatial`).

| Package          | What it contains                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `nitorch.io`     | Readers and writers for volumes (NIfTI, MGH, TIFF), affine transformations (LTA and `register.dat`) and TrackVis streamlines. It exposes `map`, `load`, `loadf`, `save` and `savef`. |
| `nitorch.spatial`| Differentiable tools for spatial transformations: orientation matrices, grids and displacement fields, resampling (`grid_pull`, `grid_push`), velocity-field exponentiation and geodesic shooting, regularisers and solvers, finite differences and mathematical morphology. |
| `nitorch.nn`     | Neural-network building blocks: activations, convolutional and U-Net style backbones, ResNets, registration networks such as `VoxelMorph`, losses, metrics, data augmentation and image generators, preprocessing, training utilities and datasets. |
| `nitorch.tools`  | High-level algorithms: affine and non-linear registration (`registration`), quantitative MRI (`qmri`), denoising (`denoising`), unified segmentation (`uniseg`) and image preprocessing and statistics. These are what the command-line tools call. |
| `nitorch.core`   | Low-level utilities used everywhere else: boundary conditions, constants, convolution kernels, linear algebra, optimisation, and helpers for PyTorch and for Python itself. |
| `nitorch.plot`   | Plotting utilities, for instance to display volumes. Requires `matplotlib`. |
| `nitorch.vb`     | Probabilistic and variational Bayesian algorithms, such as mixture models and PCA. |
| `nitorch.cli`    | The implementation of the `nitorch` command-line program.                                                          |

A further package, `nitorch.mesh`, exists but is not imported automatically; import it explicitly.

The conventions of `nitorch.spatial` are documented at the top of its source
file. In short, images are ordered as `(batch, channel, x, y, z)`, displacement
fields as `(batch, x, y, z, direction)`, and arguments that refer to spatial
dimensions are given in `(x, y, z)` order. These conventions follow
`nibabel`, not PyTorch's own `grid_sample`.

## A first look

Volumes are loaded with `nitorch.io`. `io.map` opens a file without reading its
data and gives access to its metadata, such as its shape and its orientation
matrix (the voxel-to-world affine, in millimetres). `fdata` reads the data into
a floating-point tensor:

```python
from nitorch import io

f = io.map("T1w.nii.gz")
print(f.shape)    # spatial shape, e.g. [176, 240, 256]
print(f.affine)   # 4x4 voxel-to-world matrix, as a torch tensor
dat = f.fdata()   # the image, as a torch tensor
```

`io.loadf("T1w.nii.gz")` reads a volume in one step, and `io.savef` writes one.
When you give it `like=`, the new file takes its metadata from an existing one.

Tensors can then be used with the rest of the library. This example shifts an
image by five voxels along its first axis, by building a sampling grid from a
translation matrix and resampling the image on it with `spatial.grid_pull`, and
writes the result with the orientation of the input:

```python
import torch
from nitorch import io, spatial

f = io.map("T1w.nii.gz")
dat = f.fdata()

shift = torch.eye(4)
shift[0, 3] = 5                                    # voxels along x
grid = spatial.affine_grid(shift, dat.shape)       # (X, Y, Z, 3)
moved = spatial.grid_pull(dat[None, None], grid[None],
                          interpolation=1, bound="dct2")[0, 0]

io.savef(moved, "T1w.shifted.nii.gz", like=f)
```

Each voxel `i` of the result is read from position `i + 5` of the input, which
is how every transformation in NITorch is applied: it tells each voxel of the
output where to look in the input. The same idea underlies the
[direction of spatial transformations](../cli/conventions.md#spatial-transformations)
on the command line.

## Demo notebooks

The `demo` folder of the repository contains Jupyter notebooks that show some
of NITorch's functionality: `demo_affine_reg.ipynb`, `demo_register.ipynb`,
`demo_mni_align.ipynb` and `demo_spatial.ipynb`. Follow the instructions in the
notebook to get started, or run an installation-free copy in Google Colab:

- **Affine registration:**
  [open in Colab](https://colab.research.google.com/drive/13eSBtEvAp1wIJD0Rlvq5Q9kJWnuEc7WI?usp=sharing)
- **Spatial tools:**
  [open in Colab](https://colab.research.google.com/drive/1-dfCosj9XoesFt7byIhp84p2JMUuHxby?usp=sharing)

The notebooks are in the
[`demo` folder on GitHub](https://github.com/balbasty/nitorch/tree/master/demo).

## Installation

The Python library and the command-line program are installed together; see
[Installation](../install/index.md). The optional [compiled backend](../install/compiled.md)
speeds up the resampling routines of `nitorch.spatial`.
