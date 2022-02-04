<p align="center">
  <img align="center" src="docs/images/nitorch_logo_v0.1.png" alt="NITorch" width="50%">
</p>
<p align="center">
<b>N</b>euro<b>I</b>maging in Py<b>Torch</b>
</p>


NITorch is a library written in [PyTorch](https://pytorch.org) aimed at 
**medical image processing and analysis**, with a focus on neuroimaging. 

It is a versatile package that implements low-level tools and high-level 
algorithms for **deep learning** _and_ **optimization-based algorithms**.
It implements low level differentiable functions, layers, backbones,
optimizers, but also high level algorithms for registration and inverse 
problems as well as a set of command line utilities for manipulating 
medical images.

Much of the current implementation targets image registration tasks, and 
many differentiable transformation models are implemented (classic and 
Lie-encoded affine matrices, B-spline-encoded deformation fields, dense 
deformation fields, stationary velocity fields, geodesic shooting). All
of these models can be used as layers in neural networks, and we showcase
them by reimplementing popular registration networks such as 
[VoxelMorph](https://github.com/voxelmorph/voxelmorph). We also provide 
generic augmentation layers and easy-to-use and highly-parameterized 
backbone models (U-Nets, ResNets, _etc._).

We also provide optimization-based registration tools that can be easily
applied from the command line, as well as algorithms and utilities for 
solving inverse problems in Magnetic Resonance Imaging.

## Quick start

Clone and install `nitorch`
```shell
pip install git+https://github.com/balbasty/nitorch

# Or, alternatively
git clone git@github.com:balbasty/nitorch.git
pip install ./nitorch
```

However, this only installs the core dependencies (torch and numpy). 
If you wish to automatically install dependencies used by, _e.g._, 
readers and writers, plotters and/or dataset loaders, you can specify 
the extra tags `io`, `plot`, `data`. Alternatively, the tag `all` 
combines all of these dependencies.

```shell
pip install git+https://github.com/balbasty/nitorch#egg=nitorch[all]

# Or, alternatively
git clone git@github.com:balbasty/nitorch.git
pip install -e "./nitorch[all]"
```

You may then start using NITorch in a Python program:
```python
import nitorch as ni
# my cool script
```

Or use high-level tools from the command line:
```shell
nitorch --help
```

## Demo code

The demo folder contains various Jupyter notebooks that showcase some of NITorch's funtionality. Just follow the instructions in the notebook to get started, or follow the links below to run an installation-free copy instantly in Google Colab:

* **Affine Registration** <br /> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13eSBtEvAp1wIJD0Rlvq5Q9kJWnuEc7WI?usp=sharing "NITorch Affine Registration Demo")
* **Spatial Tools** <br /> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-dfCosj9XoesFt7byIhp84p2JMUuHxby?usp=sharing "NITorch Spatial Tools Demo")


## Compiling C++/CUDA extensions

By default, a pure PyTorch implementation is used. However, we also 
provide a version of our image resampling tools written in C++/CUDA, 
which provide a x10 speedup over the default version. Because it requires
a specific compilation environment, this is for advanced users only.

Note that these extensions are built _against_ PyTorch and therefore pin
the installation to the specific PyTorch version used for compilation. 
To force pip to use a specific version of PyTorch, it is advised to 
install it beforehand and call `pip install` with the option 
`--no-build-isolation`. 

0. (If GPU support needed) Install [cuda](https://developer.nvidia.com/cuda-toolkit-archive) 
    
    - You need the *driver* and the toolkit (*compiler*, *headers* and *libraries*).
    - Follow instructions from the nvidia website based on your OS and the cuda version supported by your device.
    - See also: section **Troubleshooting**.

1. Install NITorch with compilation enabled:
```shell
git clone git@github.com:balbasty/nitorch.git
cd nitorch
NI_COMPILED_BACKEND="C" pip install .

# Or, alternatively (the version used is an arbitrary example)
pip install torch==1.9.0+cu111
NI_COMPILED_BACKEND="C" pip install --no-build-isolation .
```

## Compiling your own wheel

1. Build a wheel file
```{bash}
git clone git@github.com:balbasty/nitorch.git
cd nitorch
./setup.py bdist_wheel
# or alternatively
# NI_COMPILED_BACKEND="C" ./setup.py bdist_wheel
```
This will create a wheel file in a `dist/` directory:
```
.
├── dist
│   ├── nitorch-[*].whl
```

2. Install wheel file using `pip`
```shell
pip install nitorch-[*].whl
```

Note that when `NI_COMPILED_BACKEND="C"` is used, NITorch becomes specific 
to an **OS**, a **Python version** and (if CUDA is enabled) a **CUDA version**. 
Since we link against libtorch, it is also specific to a **PyTorch version**
You must therefore be careful about what packages are present in your 
environment.


## Troubleshooting

### CUDA

- Different versions of the CUDA toolkit support different *compute 
  capability* versions (see: https://en.wikipedia.org/wiki/CUDA#GPUs_supported). 
  You should install a version of the toolkit that is compatible with the   
  compute capability of your device.
- The toolkit installer allows both the *driver* and the *toolkit*
  (compiler, headers, libraries) to be installed. The driver needs admin 
  priviledges to be installed, but the toolkit does not. Here's a way to 
  install the toolkit without admin priviledges (copied from 
  [here](https://forums.developer.nvidia.com/t/72087/6)):
  ```shell
  ./cuda_<VERSION>_linux.run --silent --toolkit --toolkitpath=<INSTALLPATH> --defaultroot=<INSTALLPATH>
  ```
- If your CUDA toolkit is installed in a non-standard location (*i.e.*, 
  different from `/usr/local/cuda`), use the environement 
  variable `CUDA_HOME` to help the setup script locate it:
  ```shell
  CUDA_HOME=<PATH_TO_CUDA> ./setup.py install
  ```
  However, note that `nvcc` should call the correct nvidia compiler. 
  Therefore, setup your path accordingly:
  ```shell
  export PATH="$CUDA_HOME/bin:$PATH"
  ```
- The nvidia compiler (`nvcc`) calls a host compiler (`gcc`, `clang`, ...). 
  If you wish to use a non-standard host compiler (*e.g.*, you are using 
  `gcc-8` instead of the native `gcc`), things might be trickier. 
  A solution could be to alias nvcc so that it uses the `-ccbin` option 
  by default. In your `~/.bashrc`, add:
  ```shell
  alias nvcc='nvcc -ccbin <PATH_TO_GCC_BIN>'
  ```

## Authors

NITorch has been mostly written by Yael Balbastre and Mikael Brudfors, while post-docs 
in John Ashburner's group at the FIL (or *Wellcome Centre for Human Neuroimaging* as it 
is officially known). It is therefore conceptually related to SPM.

All contributions are welcome 
(though no nicely drafted guidelines exist, we'll try to get better).

## License

NITorch is released under the MIT license.
