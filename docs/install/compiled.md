---
icon: octicons/cpu-24
---

# Compiled backend

By default, NITorch runs entirely on PyTorch: its resampling routines are
written in TorchScript and need no compilation. NITorch can also be built with
C++ (and optionally CUDA) implementations of its most expensive routines. This
is the *compiled backend*. It is optional, aimed at advanced users, and needs a
compiler toolchain, but it makes the heavy commands such as
[`register`](../cli/reference/registration/register.md) and
[`reslice`](../cli/reference/spatial/reslice.md) considerably faster.

## What it replaces

Two implementations of the same routines exist:

- The **TorchScript backend** (`TS`) is the default. It is portable and needs
  nothing beyond the packages that [`pip` installs](index.md) for you.
- The **compiled backend** (`C`) provides C++ code for the CPU and CUDA code
  for NVIDIA GPUs. According to the project's README, it speeds up image
  resampling about tenfold over the default.

The compiled sources (in `nitorch/_C/src` in the repository) cover the
resampling routines that sample, splat and differentiate images on a grid, the
resizing of images and fields, and the regularisation and linear solvers
(relaxation, conjugate gradients, multigrid) that non-linear registration
relies on. Everything else is the same code in both cases.

A third value, `MONAI`, makes NITorch try to use the equivalent routines of the
[MONAI](https://monai.io) package, which were ported from NITorch. It needs
`monai` (version 0.5 or later).

## Trade-offs

Compiling NITorch has a cost, which is why it is not the default.

- **The result is tied to one environment.** The extension is built against
  the version of PyTorch that is installed when you build it, and the wheel
  declares a dependency on exactly that version (`torch==X.Y.Z`). It is also
  specific to an operating system, a Python version and, with CUDA, a CUDA
  version. If you later upgrade or change PyTorch, the compiled extension no
  longer loads, and NITorch silently falls back to the TorchScript backend.
- **You need a toolchain.** A C++ compiler that matches your PyTorch build
  (`gcc` or `clang`, or MSVC on Windows) is required, and for GPU support the
  CUDA toolkit with its compiler `nvcc`, not just the driver.
- **Building takes time**, especially for CUDA with several GPU architectures.
- **Only Linux is exercised by the project.** The build script contains
  support for macOS and Windows, but it is not regularly tested there.

## Check what you are running

To see which backend is loaded, print the value of `nitorch.compiled_backend`:

```shell
python -c "import nitorch; print(nitorch.compiled_backend)"
```

It prints `C` for the compiled backend and `TS` for TorchScript. Setting
`NI_SHOW_COMPILED_BACKEND=1` makes NITorch print the backend when it loads.

## Build and install

1. **Install the prerequisites.** Install the PyTorch version that you want to
   use, as described in [Installation](index.md#install-pytorch-first). For a GPU build, also install the
   [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (the
   *driver* and the toolkit, which holds the compiler, the headers and the
   libraries). Choose the toolkit version that matches the CUDA version of
   your PyTorch (shown by `python -c "import torch; print(torch.version.cuda)"`).

2. **Build NITorch with the environment variable `NI_COMPILED_BACKEND=C`.**

    ```shell
    git clone https://github.com/balbasty/nitorch.git
    cd nitorch
    NI_COMPILED_BACKEND="C" pip install --no-build-isolation ".[io,plot]"
    ```

    The option `--no-build-isolation` is important. Without it, `pip` builds
    in a temporary environment and may fetch a *different* PyTorch there, which
    the extension would then be built against and pinned to. With the option,
    the build uses the PyTorch that you installed. It requires `setuptools` and
    `wheel` to be installed already.

3. **Check** that it worked, as shown above. Use `pip install -v` if you want
   to see the compiler output and the messages about CUDA, described in
   [Troubleshooting](#troubleshooting).

If you do not set `NI_COMPILED_BACKEND`, `setup.py` builds the TorchScript
version and warns that the default differs from earlier releases, which always
compiled the extensions.

### Environment variables

These variables are read when you build (`pip install` or `setup.py`).

| Variable                 | Value                              | Effect                                                                                  |
| ------------------------ | ---------------------------------- | --------------------------------------------------------------------------------------- |
| `NI_COMPILED_BACKEND`    | `TS` (default), `C`, `MONAI`       | Which backend to build for. `C` compiles the extensions; `MONAI` adds a dependency on `monai`. |
| `NI_USE_CUDA`            | `1` (default), `0`                 | Set to `0` to build CPU code only, even if CUDA is available.                           |
| `CUDA_HOME`              | path                               | Location of the CUDA toolkit. `CUDA_PATH` is used if `CUDA_HOME` is not set.            |
| `TORCH_CUDA_ARCH_LIST`   | `mine` (default), `all`, or a list | GPU architectures to compile for. See below.                                            |
| `NI_COMPILE_WORKERS`     | integer                            | Number of source files compiled in parallel (default: the number of CPU cores).         |
| `NI_PARALLEL_BACKEND`    | `AT_PARALLEL_OPENMP`, `AT_PARALLEL_NATIVE`, `AT_PARALLEL_NATIVE_TBB` | Threading library used by PyTorch. It is detected automatically, so you rarely need this. |

If `CUDA_HOME` and `CUDA_PATH` are both unset, the build looks for the directory that
contains `nvcc` on your `PATH`, then for `/usr/local/cuda` (or the standard
installation folder on Windows).

**GPU architectures.** With `TORCH_CUDA_ARCH_LIST` unset or set to `mine`, the
code is compiled for the GPU installed in the machine that builds it. This
requires a GPU to be visible at build time, so it fails on a build machine
without one (for example inside a Docker build); set an explicit list there. A list is
given with compute capabilities (`"7.0 8.0"`, or `"7.0;8.0"`), optionally with
`+PTX` to also embed code that newer GPUs can compile when they run it
(`"7.0 8.0+PTX"`), or with architecture names (`Volta`, `Turing`, `Ampere`,
`Ada`, `Hopper`, ...). The value `all` compiles for the architectures that
your installed PyTorch itself was built for.

### Choosing the backend at run time

The environment variable `NI_COMPILED_BACKEND` also has an effect when NITorch
is *used*. If it is unset, NITorch picks the compiled backend when it was built,
then MONAI's routines if `monai` is installed, then TorchScript. Set it to `TS`
to force the TorchScript implementation even when the extension is installed,
which is useful to compare results, or to work around a problem. Write the value in
upper case.

```shell
NI_COMPILED_BACKEND=TS nitorch reslice T1w.nii.gz -l affine.lta -t fixed.nii.gz
```

## Build your own wheel

To distribute the compiled backend to other machines that share your
environment (the same operating system, Python, PyTorch and CUDA versions),
build a wheel once and install it elsewhere.

```shell
git clone https://github.com/balbasty/nitorch.git
cd nitorch
NI_COMPILED_BACKEND="C" python setup.py bdist_wheel
```

This creates a wheel in the `dist/` directory, whose name carries the Python
and platform tags:

```text
dist/
└── nitorch-[*].whl
```

Install it with `pip`:

```shell
pip install dist/nitorch-*.whl
```

Without `NI_COMPILED_BACKEND="C"`, the same command builds a pure-Python wheel
(TorchScript only), which does not depend on a particular PyTorch version. The
compiled wheel declares `torch==X.Y.Z` for the version it was built with, so
`pip` requires exactly that PyTorch and tries to replace a different one. Pay
attention to the packages that already exist in the target environment.

!!! tip "Building in a container"
    The `Dockerfile` at the root of the repository builds such a wheel in a
    PyTorch development image, then installs it in a runtime image. It shows a
    complete example of an explicit architecture list. See [Docker](index.md#docker).

## Troubleshooting

Build messages are hidden by default. Run `pip install -v` to see them, and
look for the three messages below, which mean that the build continues
**without CUDA**:

- `-- CUDA not found.` The build did not find a CUDA toolkit. Install one,
  or point `CUDA_HOME` to it.
- `PyTorch was not compiled with CUDA. Compiling for CPU only.` The installed
  PyTorch is a CPU build, so a GPU extension would be useless. Install a CUDA
  build of PyTorch first.
- `Your version of CUDA is vX.Y while PyTorch was compiled with CUDA vA.B. NiTorch cannot be compiled with CUDA.`
  The toolkit and PyTorch must agree on the major and minor version of CUDA.
  Install a matching toolkit, or a PyTorch built for the toolkit you have.

A build can therefore succeed, and `nitorch.compiled_backend` can report `C`,
and still be CPU-only. If the compiled backend fails on GPU data, check these
messages first.

**Choosing a toolkit.** Different versions of the CUDA toolkit support
different *compute capabilities* (see the
[list of CUDA GPUs](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)).
Install a toolkit that supports the compute capability of your device.

**No administrator rights.** The CUDA installer installs both the *driver*
and the *toolkit*. The driver needs administrator privileges, but the toolkit
does not. Install only the toolkit, in a folder that you own, with the runfile
installer
([adapted from NVIDIA's forum](https://forums.developer.nvidia.com/t/72087/6)):

```shell
./cuda_<VERSION>_linux.run --silent --toolkit --toolkitpath=<INSTALLPATH> --defaultroot=<INSTALLPATH>
```

**Toolkit in a non-standard location.** If the toolkit is not in
`/usr/local/cuda`, tell the build where to find it with `CUDA_HOME`, and make
sure that `nvcc` on your `PATH` is the matching compiler:

```shell
export CUDA_HOME=<PATH_TO_CUDA>
export PATH="$CUDA_HOME/bin:$PATH"
NI_COMPILED_BACKEND="C" pip install --no-build-isolation .
```

**Non-standard host compiler.** `nvcc` calls a host compiler (`gcc`, `clang`,
...). If you must use a different one than the default (for example `gcc-8`
rather than the system `gcc`, because your CUDA version does not support the
newer one), tell `nvcc` which one to use with its `-ccbin` option. One way is
to alias it in your `~/.bashrc`:

```shell
alias nvcc='nvcc -ccbin <PATH_TO_GCC_BIN>'
```

**"Unknown CUDA arch" error.** An entry of `TORCH_CUDA_ARCH_LIST` is not a
compute capability or architecture name that the build script knows. Use
values such as `7.0`, `8.6` or `8.0+PTX`, or names such as `Ampere`.

**The build fails on a machine without a GPU.** The default
`TORCH_CUDA_ARCH_LIST=mine` queries the installed GPU. Set an explicit list
(for example `TORCH_CUDA_ARCH_LIST="7.0 8.0+PTX"`), or set `NI_USE_CUDA=0` to build
the CPU code only.

**The extension stopped loading after a PyTorch upgrade.** The compiled
libraries are linked against the PyTorch that was used to build them. After
changing PyTorch, rebuild NITorch with the same command as before. In the
meantime NITorch falls back on the TorchScript backend, so
`nitorch.compiled_backend` reports `TS`.

**macOS.** Apple's `clang` does not include OpenMP. The build script looks for
an OpenMP library in the usual places, including the folder where Homebrew
installs `libomp`; install it before building.
