# nitorch
Neuroimaging in PyTorch

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
