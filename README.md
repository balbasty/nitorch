# NITorch
NeuroImaging in PyTorch

## DISCLAIMER

NiTorch is currently in a very *alpha* state: its API is far from stable, 
it has been loosely tested, and there is not warranty that it can run on any 
OS or against any version of PyTorch. Up to now, it has only been used 
on linux, compiled from source against PyTorch 1.4, 1.5 and 1.6.

## Quick start

0. (If GPU support needed) Install [cuda](https://developer.nvidia.com/cuda-toolkit-archive) 
    
    - You need the *driver* and the toolkit (*compiler*, *headers* and *libraries*).
    - Follow instructions from the nvidia website based on your OS and the cuda version supported by your device.
    - See also: section **Troubleshooting**.


1. Build conda environement

    - (If GPU support needed) Change the `cudatoolkit` version so that it matches that of your local cuda toolkit.
    - (Else) Remove `cudatoolkit` from the package list

```{bash}
conda env create --file ./conda/nitorch.yml
conda activate nitorch
```

2. Compile C++/CUDA library
    - `install` copies files in the python `site-packages` directory
    - `develop` softlinks files in the python `site-packages` directory, allowing them to be modified without requiring a new `install` step.

```{bash}
./setup.py [install|develop]
```

3. Use nitorch
```{python}
import nitorch as ni
```

## Demo code

The demo folder contains various Jupyter notebooks that showcase some of NITorch's funtionality. Just follow the instructions in the notebook to get started.

## Compiling your own wheel

1. Build conda environement
```{bash}
conda env create --file ./conda/nitorch.yml
conda activate nitorch
```
NiTorch is a **compiled** package and is therefore specific to an **OS**, a **Python version** and (if CUDA is enabled) a **CUDA version**. 
Since we link against libtorch, I *think* it is also specific to a **PyTorch version** (this should be checked).
You must therefore be careful about what packages are present in your environment. It would be good practice to name the created wheel files accordingly.

2. Build a wheel file
```{bash}
./setup.py bdist_wheel
```
This will create a wheel file in a `dist/` directory:
```{bash}
.
├── dist
│   ├── nitorch-[*].whl
```

3. Install wheel file using `pip`
```{bash}
pip install nitorch-[*].whl
```

## Troubleshooting

### CUDA

- Different versions of the CUDA toolkit support different *compute capability* versions 
  (see: https://en.wikipedia.org/wiki/CUDA#GPUs_supported). 
  You should install a version of the toolkit that is compatible with the compute capability of your device.
- The toolkit installer allows both the *driver* and the *toolkit* (compiler, headers, libraries) to be installed.
  The driver needs admin priviledges to be installed, but the toolkit does not. Here's a way to install the toolkit without 
  admin priviledges (copied from [here](https://forums.developer.nvidia.com/t/72087/6)):
  ```{bash}
  ./cuda_<VERSION>_linux.run --silent --toolkit --toolkitpath=<INSTALLPATH> --defaultroot=<INSTALLPATH>
  ```
- If your CUDA toolkit is installed in a non-standard location (*i.e.*, different from `/usr/local/cuda`), use the environement 
  variable `CUDA_HOME` to help the setup script locate it:
  ```{bash}
  CUDA_HOME=<PATH_TO_CUDA> ./setup.py install
  ```
  However, note that `nvcc` should call the correct nvidia compiler. Therefore, setup your path accordingly:
  ```{bash}
  export PATH="$CUDA_HOME/bin:$PATH"
  ```
- The nvidia compiler (`nvcc`) calls a host compiler (`gcc`, `clang`, ...). If you wish to use a non-standard host compiler 
  (*e.g.*, you are using `gcc-8` instead of the native `gcc`), things might be trickier. A solution could be to alias nvcc 
  so that it uses the `-ccbin` option by default. In your `~/.bashrc`, add:
  ```{bash}
  alias nvcc='nvcc -ccbin <PATH_TO_GCC_BIN>'
  ```

## Authors

NiTorch has been mostly written by Yael Balbastre and Mikael Brudfors, while post-docs 
in John Ashburner's group at the FIL (or *Wellcome Centre for Human Neuroimaging* as it 
is officially known). It is therefore conceptually related to SPM.

All contributions are welcome 
(though no nicely drafted guidelines exist, we'll try to get better).

## License

NiTorch is released under the MIT license. However, the `spm.py` module is directly 
based on MATLAB code from the SPM software (https://www.fil.ion.ucl.ac.uk/spm/software/), 
which is copyright and released under the GNU-GPL license (version >= 2).
