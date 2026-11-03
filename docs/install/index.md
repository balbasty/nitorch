---
icon: octicons/download-24
---

# Installation

NITorch is a Python package that you install with `pip`. It is **not
published on PyPI** under the name `nitorch`, so you install it from its GitHub
repository. The default installation is pure Python and works on any platform
that PyTorch supports. An optional, faster [compiled backend](compiled.md) can
be built from source.

## Requirements

NITorch needs Python and PyTorch. The package metadata is permissive (it
declares Python 3.6 or later and PyTorch 1.6 or later), but only a few
combinations are actually tested. The continuous-integration tests run on Linux
with the CPU build of PyTorch, with:

| Python | PyTorch | NumPy      |
| ------ | ------- | ---------- |
| 3.9    | 1.13.1  | below 2    |
| 3.12   | 2.5.1   | 2 or later |

PyTorch 1.x does not work with NumPy 2, which is why the older combination pins
NumPy. Newer versions of both usually work, but they are not part of the test
matrix. Windows and macOS are not tested.

The core dependencies, installed automatically with NITorch, are `torch`,
`numpy`, `scipy` and
[`torch-interpol`](https://pypi.org/project/torch-interpol/) (the resampling
routines that NITorch uses by default).

## Install PyTorch first

If you plan to use a GPU, or if you need a specific build of PyTorch, install
PyTorch **before** NITorch. The selector on
[pytorch.org](https://pytorch.org/get-started/locally/) gives the right `pip`
command for your operating system, package manager and CUDA version. For
example:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If a suitable PyTorch is already installed, `pip` leaves it alone when it
installs NITorch. If none is, `pip` installs the default build from PyPI, which
on Linux includes CUDA libraries and is a large download. For a machine
without a GPU, choose the CPU build in the selector to get a much smaller
installation.

You can check what PyTorch sees with:

```shell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Install NITorch

Install the latest version from GitHub:

```shell
pip install "nitorch[io,plot] @ git+https://github.com/balbasty/nitorch"
```

The part in square brackets selects *extras*, described in the next section.
For the command line you want at least `io`. A specific branch or tag can be
selected by appending it to the address, for example
`git+https://github.com/balbasty/nitorch@master`.

!!! tip "Avoid downloading PyTorch twice"
    The build requirements of NITorch (`pyproject.toml`) list PyTorch, so by
    default `pip` builds the package in an isolated environment into which it
    may download a copy of PyTorch, even if you have installed one. If you want
    to avoid this, install `setuptools` and `wheel` yourself and add
    `--no-build-isolation`:

    ```shell
    pip install setuptools wheel
    pip install --no-build-isolation "nitorch[io,plot] @ git+https://github.com/balbasty/nitorch"
    ```

    The `--no-build-isolation` flag is mandatory when you build the
    [compiled backend](compiled.md).

## Optional dependencies

The core installation reads no files by itself: the readers and writers, the
plotting tools and the dataset loaders each rely on an optional package. They
are grouped into *extras* that you add in square brackets:

| Extra    | Installs                                  | Needed for                                                                  |
| -------- | ----------------------------------------- | --------------------------------------------------------------------------- |
| `io`     | `numpy`, `nibabel`, `tifffile`            | Reading and writing volumes: NIfTI and MGH (through `nibabel`) and TIFF (through `tifffile`). Required by essentially every command. |
| `plot`   | `matplotlib`                              | Plotting, including [`view`](../cli/reference/viewer/view.md).              |
| `data`   | `wget`, `appdirs`                         | Downloading pretrained models and datasets from the Python library.         |
| `all`    | `numpy`, `nibabel`, `tifffile`, `wget`, `appdirs`, `matplotlib` | Everything above.                                     |
| `nibabel`| `numpy`, `nibabel`                        | Only the `nibabel`-based formats (NIfTI, MGH).                              |
| `tiff`   | `numpy`, `tifffile`                       | Only TIFF.                                                                  |
| `numpy`  | `numpy`                                   | Nothing beyond the core dependencies.                                       |

Extras can be combined with commas, as in `nitorch[io,plot]`.

Without `nibabel`, NITorch cannot open NIfTI or MGH files. If it finds a
version of `nibabel` that cannot be imported with your version of NumPy
(`nibabel` older than 5.2 with NumPy 2), it warns you and disables the
`nibabel`-based formats; upgrade `nibabel` to fix this.

## Development install

To work on NITorch itself, clone the repository and install it in *editable*
mode, so that changes to the source take effect immediately:

```shell
git clone https://github.com/balbasty/nitorch.git
cd nitorch
pip install -e ".[io,plot]"
```

Install `pytest` to run the tests. The fast suite is what runs on pull requests
and skips the tests marked `slow`:

```shell
pytest --ignore nitorch/tests/test_gradcheck_spatial.py
```

Add `--runslow` to include the slow, end-to-end registration tests. The
gradient checks of the resampling routines are in
`nitorch/tests/test_gradcheck_spatial.py`, which you can run on its own.

## Check the installation

The installation provides a `nitorch` program. Ask it for its list of
commands:

```shell
nitorch --help
```

If the `nitorch` script is not on your `PATH`, the same program runs through
Python:

```shell
python -m nitorch --help
```

To see which version is installed and which implementation of the resampling
routines is in use:

```shell
python -c "import nitorch; print(nitorch.__version__, nitorch.compiled_backend)"
```

The second value is `TS` for the default pure-PyTorch implementation, and `C`
for the [compiled backend](compiled.md). Importing NITorch without the compiled
backend logs a warning that some algorithms may be slow; it is expected, and
you can silence it by setting `NI_CHECK_BACKEND=0`.

## conda

NITorch is not packaged for conda, but you can use conda to manage the
environment and `pip` to install NITorch into it. Create an environment with a
recent Python, install PyTorch with the command given by the
[PyTorch selector](https://pytorch.org/get-started/locally/), then install
NITorch with `pip` as above:

```shell
conda create -n nitorch python=3.12
conda activate nitorch
# install PyTorch here, following pytorch.org
pip install "nitorch[io,plot] @ git+https://github.com/balbasty/nitorch"
```

The repository contains two `conda/*.yml` files (`nitorch.yml` and
`nitorch-demo.yml`). They date from the first versions of NITorch, pin Python
3.6, PyTorch 1.6 and CUDA 10.1, and do not install NITorch itself, so we do
not recommend them for a new installation.

## Docker

The repository contains a `Dockerfile` that builds an image with the
[compiled backend](compiled.md). It is a multi-stage build: the first stage
compiles a NITorch wheel with `NI_COMPILED_BACKEND="C"` in a PyTorch
development image, and the final stage installs that wheel, together with
`nibabel`, `scipy`, `matplotlib`, `appdirs` and `wget`, in a PyTorch runtime
image. Build it from the root of the repository:

```shell
git clone https://github.com/balbasty/nitorch.git
cd nitorch
docker build -t nitorch .
```

Then run commands in it, mounting the folder that holds your images (add
`--gpus all` to use a GPU, which requires the NVIDIA container toolkit):

```shell
docker run --rm -v "$PWD:/data" -w /data nitorch nitorch info T1w.nii.gz
```

No image is published, and the build is not part of the project's automated
tests. The image is based on PyTorch 1.12.1 with CUDA 11.3 (the `PYTORCH_TAG_PREFIX`
argument at the top of the `Dockerfile` selects another base image) and
compiles for the GPU architectures listed in `TORCH_CUDA_ARCH_LIST`, so treat
it as a starting point that you may need to adapt for recent GPUs.

## Next steps

Read the [command-line overview](../cli/index.md) for a first session, or the
page on the [compiled backend](compiled.md) if you need faster resampling.
