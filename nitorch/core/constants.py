import math
import sys

pi = math.pi      # pi
inf = math.inf    # infinity
ninf = -math.inf  # negative infinity
nan = math.nan    # not-a-number
tau = math.tau    # 2*pi
e = math.e        # exp(0)


def eps(dtype='float32'):
    """Machine epsilon for different precisions."""
    if dtype == 'float16':
        return 2 ** -10
    if dtype == 'float32':
        return 2 ** -23
    elif dtype == 'float64':
        return 2 ** -52
    else:
        raise NotImplementedError
