"""Useful constants."""

import math
import torch
from .optionals import numpy as np

pi = math.pi      # pi
inf = math.inf    # infinity
ninf = -math.inf  # negative infinity
nan = math.nan    # not-a-number
tau = math.tau    # 2*pi
e = math.e        # exp(0)


def eps(dtype='float32'):
    """Machine epsilon for different precisions."""
    f16_types = []
    if hasattr(torch, 'float16'):
        f16_types += ['float16', torch.float16]
    if hasattr(torch, 'complex32'):
        f16_types += ['complex32', torch.complex32]
    f32_types = ['float32', torch.float32, 'complex64', torch.complex64] \
                + ([np.float32, np.complex64] if np else [])
    f64_types = ['float64', torch.float64, 'complex128', torch.complex128] \
                + ([np.float64, np.complex128] if np else [])

    if dtype in f16_types:
        return 2 ** -10
    if dtype in f32_types:
        return 2 ** -23
    elif dtype in f64_types:
        return 2 ** -52
    else:
        raise NotImplementedError
