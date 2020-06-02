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
./setup.py build_clib
./setup.py [develop]
```

3. Use nitorch
```{python}
import nitorch
```