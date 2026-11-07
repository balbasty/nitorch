"""Spline prefilter, which turns values into spline coefficients.

A spline of order 2 or more does not pass through its coefficients, so
interpolating the samples of an image directly would blur it. The
prefilter computes the coefficients whose spline takes the value of the
image at every sample. Along one axis, sampling the coefficients `c` at
the integer coordinates gives `A @ c`, where the matrix `A` depends on
the order and on the boundary condition, and the prefilter computes
`c = inv(A) @ f` from the values `f`.

For the boundaries `dct1`, `dct2`, `dst1`, `dst2` and `dft`, the
inverse is a recursive filter, with one causal and one anticausal pass
for each pole of the spline. Each pass needs its value at the first
sample that it computes. These boundaries extend the input of every
pass in the same way as they extend the coefficients, so that this
value is a sum over one period of the extended input, which is computed
in closed form. The recursive filter comes from torch-interpol, which
ported it from the C code of John Ashburner in SPM, itself derived from
the code of Philippe Thevenaz, and from SciPy for `dct2`.

The boundaries `zero` and `replicate` have no such rule from order 4,
because the inputs of the passes after the first one do not follow the
extension of the coefficients. At orders 2 and 3, their matrices `A`
are those of `dst1` and `dct2`, whose filters therefore give their
coefficients. From order 4, the coefficients are first computed with
the filter of `dst1` or `dct2`, and then corrected near the edges.
Away from the edges, every row of `A` is the same convolution, so that
the correction is a sum of powers of the poles that decay away from
each edge. Its weights solve a small linear system, written on the
first and last `order // 2` rows of `A`.

torch-interpol (whose prefilter nitorch used before) applied the
`dct1` filter to the zero boundary and the `dct2` filter to the
replicate boundary, so that its coefficients did not interpolate the
values near the edges, at every order for the zero boundary and from
order 4 for the replicate boundary. It did not support the
antisymmetric boundaries (`dst1`, `dst2`) either.

This module is a PyTorch port of the prefilter of brainhops
(`brainhops._ext.interpol`), itself a port of torch-interpol 0.3.0.
It also implements the transposed prefilter `inv(A).T`, used in the
backward pass of the prefilter. `A` is symmetric for the boundaries
`zero`, `dct2`, `dst1`, `dst2` and `dft`, but not for `dct1` and
`replicate`.

References
----------
1.  M. Unser, A. Aldroubi and M. Eden.
    "B-Spline Signal Processing: Part I-Theory,"
    IEEE Transactions on Signal Processing 41(2):821-832 (1993).
2.  M. Unser, A. Aldroubi and M. Eden.
    "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
    IEEE Transactions on Signal Processing 41(2):834-848 (1993).
3.  M. Unser.
    "Splines: A Perfect Fit for Signal and Image Processing,"
    IEEE Signal Processing Magazine 16(6):22-38 (1999).
"""
import fractions
import functools
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# Integer codes of the boundary conditions, as in torch-interpol's
# `BoundType` (and nitorch's `bound_to_nitorch(..., as_type='int')`)
_BOUND_CODES = {0: 'zero', 1: 'replicate', 2: 'dct1', 3: 'dct2',
                4: 'dst1', 5: 'dst2', 6: 'dft'}
_BOUNDS = tuple(_BOUND_CODES.values())

_POLES = {
    0: (),
    1: (),
    2: (-0.171572875253809902396622551581,),
    3: (-0.267949192431122706472553658494,),
    4: (
        -0.361341225900220177092212841326,
        -0.0137254292973391213603312269391,
    ),
    5: (
        -0.430575347099973791851434783494,
        -0.0430962882032646538227123768226,
    ),
    6: (
        -0.488294589303044755130118038884,
        -0.0816792710762375125979377657371,
        -0.00141415180832581775108724397656,
    ),
    7: (
        -0.535280430796438165542403781682,
        -0.122554615192326690515272264359,
        -0.00914869480960827692859302165165,
    ),
}
"""The poles of the prefilter of each spline order.

The spline of order `p` takes the values `b[m]` at the integers `m`
from `-h` to `h`, with `h = p // 2`. The poles are the `h` roots in the
interval (-1, 0) of the polynomial `sum(b[m] * z**(h - m))`, from the
largest in magnitude to the smallest (computed at 60 digits and rounded
to 30 significant digits)."""

_BASE = {'zero': 'dst1', 'replicate': 'dct2'}
"""The boundary whose filter starts the prefilter of `zero` and
`replicate`, and which gives it entirely at orders 2 and 3."""

_LOG_EPS = math.log(1e-17)
"""Logarithm of the size below which a power of a pole is neglected."""


def normalize_bound(bound) -> str:
    """Name of a boundary condition, from its integer code or name"""
    if isinstance(bound, int) or hasattr(bound, 'value'):
        return _BOUND_CODES[int(getattr(bound, 'value', bound))]
    bound = str(bound).lower()
    if bound not in _BOUNDS:
        raise ValueError(f'Unknown boundary condition: {bound!r}.')
    return bound


def prefilter_poles(order: int) -> Tuple[float, ...]:
    """Poles of the prefilter of a spline of order 0 to 7"""
    if order not in _POLES:
        raise NotImplementedError(f'Spline order {order} is not implemented.')
    return _POLES[order]


@functools.lru_cache(maxsize=None)
def _stencil(order: int) -> Tuple[fractions.Fraction, ...]:
    """Values of the centred B-spline of order `p` at the integers
    `-h..h` (`h = p // 2`), as exact fractions: sampling coefficients
    at the integers is the convolution by these values."""
    h = order // 2
    centre = fractions.Fraction(order + 1, 2)
    values = []
    for m in range(-h, h + 1):
        total = fractions.Fraction(0)
        for k in range(order + 2):
            t = m + centre - k
            if t > 0:
                total += (-1) ** k * math.comb(order + 1, k) * t ** order
        values.append(total / math.factorial(order))
    return tuple(values)


def _gain(order: int) -> float:
    # The gain makes the filter return a constant signal unchanged. It
    # is the product of (1 - z) * (1 - 1 / z) over the poles z, which
    # equals 1 / b[h] exactly.
    return float(1 / _stencil(order)[-1])


def pole_horizon(z: float) -> int:
    # The smallest exponent K for which z**K is below 1e-17 in magnitude
    return int(math.ceil(_LOG_EPS / math.log(abs(z))))


# --- initial conditions -----------------------------------------------
#
# A pole z makes a causal pass y[k] = x[k] + z * y[k - 1], then an
# anticausal pass w[k] = z * (w[k + 1] - y[k]). When the input x is
# extended past the axis by the boundary, the exact first values of the
# two passes are
#
#     y[0] = sum(z**j * x[-j] for j >= 0),
#     t = sum(z**j * x[n - 1 + j] for j >= 1),
#     w[n - 1] = -z / (1 - z**2) * (y[n - 1] + t).
#
# The functions below write these sums over one period of the
# extension. They write t with the values of y, since the causal pass
# has overwritten x when t is needed. When n - 1 is at least the
# horizon K of the pole, the powers z**K and above are dropped.
#
# Each function returns terms `(start, stop, weights)`, and the initial
# value is the sum of `weights @ x[start:stop]` over the terms.


@functools.lru_cache(maxsize=256)
def _causal_init(n: int, z: float, bound: str):
    horizon = pole_horizon(z)
    if n - 1 >= horizon:
        powers = z ** np.arange(horizon, dtype=np.float64)
        if bound == 'dct1':
            return ((0, horizon, powers),)
        if bound == 'dft':
            # x[0] + z * x[n - 1] + z**2 * x[n - 2] + ...
            return ((0, 1, np.ones(1)),
                    (n - horizon, n, z * powers[::-1]))
        # x[0] plus a weighted sum of the first samples, which the
        # boundary reflects to the left of the axis, about -1/2 for
        # dct2 and dst2 and about -1 for dst1, with a change of sign
        # for dst1 and dst2.
        factor = {'dct2': z, 'dst2': -z, 'dst1': -z * z}[bound]
        weights = factor * powers
        weights[0] += 1.0
        return ((0, horizon, weights),)
    k = np.arange(n, dtype=np.float64)
    if bound == 'dct1':
        # One period has 2 * n - 2 samples, in which the first and the
        # last sample appear once and the others twice.
        weights = z ** k + z ** (2 * n - 2 - k)
        weights[0] = 1.0
        weights[-1] = z ** (n - 1)
        weights /= 1.0 - z ** (2 * n - 2)
    elif bound == 'dct2':
        weights = z * (z ** k + z ** (2 * n - 1 - k)) / (1.0 - z ** (2 * n))
        weights[0] += 1.0
    elif bound == 'dst2':
        weights = -z * (z ** k - z ** (2 * n - 1 - k)) / (1.0 - z ** (2 * n))
        weights[0] += 1.0
    elif bound == 'dst1':
        # One period has 2 * n + 2 samples, two of which are zero.
        weights = z * z * (z ** (2 * n - k) - z ** k)
        weights /= 1.0 - z ** (2 * n + 2)
        weights[0] += 1.0
    elif bound == 'dft':
        weights = z ** (n - k)
        weights[0] = 1.0
        weights /= 1.0 - z ** n
    else:
        raise ValueError(f'No recursive filter for the boundary {bound!r}.')
    return ((0, n, weights),)


@functools.lru_cache(maxsize=256)
def _anticausal_init(n: int, z: float, bound: str):
    if bound == 'dct1':
        # The samples after the last one are y[n - 2], y[n - 3], ...
        weights = np.asarray([z * z, z]) / (z * z - 1.0)
        return ((n - 2, n, weights),)
    if bound == 'dct2':
        return ((n - 1, n, np.asarray([z / (z - 1.0)])),)
    if bound == 'dst2':
        return ((n - 1, n, np.asarray([-z / (1.0 + z)])),)
    if bound == 'dst1':
        return ((n - 1, n, np.asarray([-z])),)
    if bound != 'dft':
        raise ValueError(f'No recursive filter for the boundary {bound!r}.')
    # The causal pass keeps the input periodic, and the samples after
    # the last one are y[0], y[1], ...
    horizon = pole_horizon(z)
    if n - 1 >= horizon:
        weights = -(z ** np.arange(2, horizon + 1, dtype=np.float64))
        return ((0, horizon - 1, weights), (n - 1, n, np.asarray([-z])))
    weights = -(z ** np.arange(2, n + 2, dtype=np.float64))
    weights[-1] = -z
    return ((0, n, weights / (1.0 - z ** n)),)


def _as_tensor(array, like):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=like.dtype,
                           device=like.device)


def _apply_init(x, terms):
    value = None
    for start, stop, weights in terms:
        if stop - start == 1:
            term = x[start] * float(weights[0])
        else:
            term = torch.tensordot(_as_tensor(weights, x), x[start:stop],
                                   dims=1)
        value = term if value is None else value + term
    return value


# --- recursive filter -------------------------------------------------


def _recursive_filter(x, order: int, bound: str):
    """Apply the recursive prefilter in place along the first axis of a
    (n, m) tensor (boundaries dct1, dct2, dst1, dst2, dft; dct1 needs
    n >= 2)."""
    n = x.shape[0]
    x *= _gain(order)
    for z in prefilter_poles(order):
        x[0] = _apply_init(x, _causal_init(n, z, bound))
        for i in range(1, n):
            x[i].add_(x[i - 1], alpha=z)
        x[n - 1] = _apply_init(x, _anticausal_init(n, z, bound))
        for i in range(n - 2, -1, -1):
            x[i].neg_().add_(x[i + 1]).mul_(z)


# --- edge correction for the zero and replicate boundaries ------------


def _bound_index(i: np.ndarray, n: int, bound: str, weight: np.ndarray):
    """Map indices that may lie outside [0, n-1] inside it, multiplying
    `weight` (in place) by the sign (-1) or factor (0) of the boundary.
    Returns the indices."""
    i = i.copy()
    if bound == 'replicate':
        return np.clip(i, 0, n - 1)
    if bound == 'dft':
        return np.remainder(i, n)
    if bound == 'dct1' and n == 1:
        return np.zeros_like(i)
    if bound == 'zero':
        weight *= (i >= 0) & (i < n)
        return np.clip(i, 0, n - 1)
    if bound == 'dct1':
        n2 = 2 * (n - 1)
        i = np.remainder(np.abs(i), n2)
        return np.minimum(i, n2 - i)
    if bound in ('dct2', 'dst2'):
        n2 = 2 * n
        i = np.remainder(i, n2)
        if bound == 'dst2':
            weight *= np.sign((n2 - 1 - i) - i)
        return np.minimum(i, n2 - 1 - i)
    if bound == 'dst1':
        # One period of the extension is [x, 0, -x reversed, 0]
        n2 = 2 * (n + 1)
        q = np.remainder(i + 1, n2)
        weight *= np.sign(n + 1 - q) * np.sign(q)
        return np.clip(np.minimum(q - 1, n2 - 1 - q), 0, n - 1)
    raise ValueError(f'Unknown boundary condition: {bound!r}.')


def _sampling_rows(n: int, order: int, bound: str, rows: Sequence[int]):
    """Rows of the matrix `A` that samples the coefficients of a spline
    at the integers `0..n-1`, as a (len(rows), n) float64 array."""
    h = order // 2
    rows = np.asarray(rows, dtype=np.intp)
    offsets = np.arange(-h, h + 1, dtype=np.intp)
    index = rows[:, None] - offsets
    weight = np.empty(index.shape)
    weight[...] = [float(b) for b in _stencil(order)]
    index = _bound_index(index, n, bound, weight)
    out = np.zeros((len(rows), n))
    np.add.at(out, (np.arange(len(rows))[:, None], index), weight)
    return out


@functools.lru_cache(maxsize=64)
def _edge_correction(n: int, order: int, bound: str):
    """Terms that correct the filter of the base boundary (`_BASE`).

    The filter of the base boundary gives coefficients `cb` with
    `B @ cb = f`. The rows of `B` and of the matrix `A` of the boundary
    are the same convolution away from the edges, so that `c - cb`
    solves the homogeneous recurrence of the convolution there, and is
    a sum of the sequences `z**k` and `z**(n - 1 - k)` over the poles.
    Its weights make the first and last `h` rows of `A @ c` equal to
    `f`; they solve a system of size `2 * h`, whose right-hand side
    `(B - A) @ cb` only reads a few coefficients at each edge.

    Returns `width` (number of coefficients read at each edge), `left`
    and `right` ((2h, width) matrices that give the weights from the
    first and last `width` coefficients) and `powers` ((length, h)
    matrix of the powers of the poles, beyond which they are
    negligible). The correction is then
        weights = left @ cb[:width] + right @ cb[-width:]
        c[:length] += powers @ weights[:h]
        c[-length:] += powers[::-1] @ weights[h:]
    """
    h = order // 2
    poles = np.asarray(prefilter_poles(order))
    rows = list(range(h)) + list(range(n - h, n))
    exact = _sampling_rows(n, order, bound, rows)
    diff = _sampling_rows(n, order, _BASE[bound], rows) - exact
    band = np.flatnonzero(np.abs(exact).max(axis=0) > 0)
    k = band[:, None].astype(np.float64)
    sequences = np.concatenate([poles ** k, poles ** (n - 1 - k)], axis=1)
    system = exact[:, band] @ sequences
    read = np.flatnonzero(np.abs(diff).max(axis=0) > 0)
    width = 1 + int(np.minimum(read, n - 1 - read).max())
    left = np.linalg.solve(system, diff[:, :width])
    right = np.linalg.solve(system, diff[:, n - width:])
    length = min(n, max(pole_horizon(z) for z in poles))
    powers = poles ** np.arange(length, dtype=np.float64)[:, None]
    return width, left, right, powers


@functools.lru_cache(maxsize=16)
def _small_inverse(n: int, order: int, bound: str):
    # The inverse of the matrix A of an axis too short for the edge
    # correction, which has no rows away from the edges.
    return np.linalg.inv(_sampling_rows(n, order, bound, range(n)))


def _filter_axis0(x, order: int, bound: str, transpose: bool = False):
    """Prefilter a (n, m) tensor in place along its first axis.

    If `transpose`, apply `inv(A).T` instead of `inv(A)` (adjoint of the
    prefilter)."""
    n = x.shape[0]
    h = order // 2
    if n == 1 and bound in ('dct1', 'dct2', 'dft', 'replicate'):
        # The extension of a single sample is constant, and A = [1].
        # (Not for dst1, whose extension of a single sample x0 is
        # `0, x0, 0, -x0` with period 4, as in `grid_pull`: there,
        # A = [b[0] - 2 b[2] + 2 b[4] - ...], which the small inverse
        # below gives.)
        return
    if n < 2 * h or (n == 1 and bound not in _BASE):
        inverse = _small_inverse(n, order, bound)
        if transpose:
            inverse = inverse.T
        x.copy_(torch.matmul(_as_tensor(inverse, x), x))
        return
    if bound not in _BASE or h == 1:
        if transpose and bound == 'dct1':
            # A.T = W A inv(W), with W = diag(1, 2, ..., 2, 1), so that
            # inv(A).T = W inv(A) inv(W). (The matrices of the other
            # boundaries with a recursive filter are symmetric, and so
            # are those of zero and replicate at orders 2 and 3.)
            x[1:n - 1] *= 0.5
            _recursive_filter(x, order, bound)
            x[1:n - 1] *= 2
            return
        _recursive_filter(x, order, _BASE.get(bound, bound))
        return
    width, left, right, powers = _edge_correction(n, order, bound)
    left, right, powers = (_as_tensor(a, x) for a in (left, right, powers))
    length = powers.shape[0]
    if transpose:
        # inv(A) = (I + U M) inv(B), with B symmetric, so that
        # inv(A).T = inv(B) (I + M.T U.T)
        weights = torch.cat([
            torch.matmul(powers.T, x[:length]),
            torch.matmul(powers.flip(0).T, x[n - length:]),
        ])
        x[:width] += torch.matmul(left.T, weights)
        x[n - width:] += torch.matmul(right.T, weights)
        _recursive_filter(x, order, _BASE[bound])
        return
    _recursive_filter(x, order, _BASE[bound])
    weights = torch.matmul(left, x[:width])
    weights += torch.matmul(right, x[n - width:])
    # The sequences z**k start at the first sample, and the sequences
    # z**(n - 1 - k) at the last one.
    x[:length] += torch.matmul(powers, weights[:h])
    x[n - length:] += torch.matmul(powers.flip(0), weights[h:])


# --- public functions -------------------------------------------------


def spline_coeff(inp, bound, order: int, dim: int = -1,
                 inplace: bool = False, transpose: bool = False):
    """Compute the interpolating spline coefficients, for a given
    spline order and boundary condition, along a single dimension.

    The coefficients are those whose spline, sampled at the integer
    coordinates with the same order and boundary condition, gives back
    the input exactly.

    Parameters
    ----------
    inp : tensor
    bound : int or str
        Boundary condition (integer code of `BoundType`, or name):
        zero, replicate, dct1, dct2, dst1, dst2, dft.
    order : {0..7}
    dim : int, default=-1
    inplace : bool, default=False
    transpose : bool, default=False
        Apply the transpose of the prefilter (its adjoint), which is
        its backward pass.

    Returns
    -------
    out : tensor

    """
    bound = normalize_bound(bound)
    order = int(order)
    if order in (0, 1):
        return inp if inplace else inp.clone()
    prefilter_poles(order)  # check order
    x = inp.movedim(dim, 0)
    shape = x.shape
    if not x.dtype.is_floating_point:
        raise TypeError('spline_coeff expects a floating point tensor')
    # The filtered axis is laid out first, so that each step of the
    # recursion reads and writes a contiguous block.
    x = x.reshape([shape[0], -1]).clone(memory_format=torch.contiguous_format)
    _filter_axis0(x, order, bound, transpose)
    x = x.reshape(shape).movedim(0, dim)
    if inplace:
        inp.copy_(x)
        return inp
    return x


def spline_coeff_nd(inp, bound, order, dim: Optional[int] = None,
                    inplace: bool = False, transpose: bool = False):
    """Compute the interpolating spline coefficients, for a given
    spline order and boundary condition, along the last `dim`
    dimensions.

    Parameters
    ----------
    inp : (..., *spatial) tensor
    bound : [list of] int or str
    order : [list of] {0..7}
    dim : int, default=`inp.dim()`
    inplace : bool, default=False
    transpose : bool, default=False
        Apply the transpose of the prefilter (its backward pass).

    Returns
    -------
    out : (..., *spatial) tensor

    """
    if dim is None:
        dim = inp.dim()
    bound = _pad_list(bound, dim)
    order = _pad_list(order, dim)
    out = inp if inplace else inp.clone()
    for d, b, o in zip(range(dim), bound, order):
        out = spline_coeff(out, b, o, dim=-dim + d, inplace=True,
                           transpose=transpose)
    return out


def _pad_list(x, n: int) -> List:
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    return x + x[-1:] * max(0, n - len(x))
