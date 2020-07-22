# nitorch
Neuroimaging in PyTorch

## DISCLAIMER

NiTorch is currently in a very *alpha* state: its API is far from stable, 
it has been loosely tested, and there is not warranty that it can run on any 
OS or against any version of PyTorch. Up to now, it has only been used 
on linux, compiled from source against PyTorch 1.4 and 1.5.

## Quick start

1. Build conda environement
```{bash}
conda env create --file ./conda/nitorch.yml
conda activate nitorch
```

2. Compile C++/CUDA library
```{bash}
./setup.py [install|develop]
```

3. Use nitorch
```{python}
import nitorch
```

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
