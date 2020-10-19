"""Check which optional modules are available."""

# Numpy
try:
    import numpy
except ImportError:
    numpy = None

# Scipy
try:
    import scipy
except ImportError:
    scipy = None

# Matplotlib
try:
    import matplotlib
except ImportError:
    matplotlib = None
