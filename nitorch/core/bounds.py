"""Boundary conditions

There is no common convention to name boundary conditions.
This file lists all possible aliases and provides tool to "convert"
between them. It also defines function that can be used to implement
these boundary conditions.

=========   ===========   ===============================   =======================   =======================
NITorch     SciPy         PyTorch                           Other                     Description
=========   ===========   ===============================   =======================   =======================
replicate   border        nearest                           repeat                    a  a | a b c d |  d  d
zero        constant(0)   zero                              zeros                     0  0 | a b c d |  0  0
dct2        reflect       reflection(align_corners=True)    neumann                   b  a | a b c d |  d  c
dct1        mirror        reflection(align_corners=False)                             c  b | a b c d |  c  b
dft         wrap                                            circular                  c  d | a b c d |  a  b
dst2                                                        antireflect, dirichlet   -b -a | a b c d | -d -c
dst1                                                        antimirror               -a  0 | a b c d |  0 -d
"""
import torch
from nitorch._C.grid import BoundType


nitorch_bounds = ('replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft')
scipy_bounds = ('border', 'constant', 'reflect', 'mirror', 'wrap')
pytorch_bounds = ('nearest', 'zero', 'reflection')
other_bounds = ('repeat', 'zeros', 'neumann', 'circular',
                'antireflect', 'dirichlet', 'antimirror')
all_bounds = (*nitorch_bounds, *scipy_bounds, *pytorch_bounds, *other_bounds)


def to_nitorch(bound, as_enum=False):
    """Convert boundary type to NITorch's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention
    as_enum : bool, default=False
        Return BoundType rather than str

    Returns
    -------
    bound : [list of] str or BoundType
        Boundary condition in NITorch's convention

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower() if isinstance(b, str) else b
        if b in ('replicate', 'repeat', 'border', 'nearest', BoundType.replicate):
            obound.append('replicate')
        elif b in ('zero', 'zeros', 'constant', BoundType.zero):
            obound.append('zero')
        elif b in ('dct2', 'reflect', 'reflection', 'neumann', BoundType.dct2):
            obound.append('dct2')
        elif b in ('dct1', 'mirror', BoundType.dct1):
            obound.append('dct1')
        elif b in ('dft', 'wrap', 'circular', BoundType.dft):
            obound.append('dft')
        elif b in ('dst2', 'antireflect', 'dirichlet', BoundType.dst2):
            obound.append('dst2')
        elif b in ('dst1', 'antimirror', BoundType.dst1):
            obound.append('dst1')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if as_enum:
        obound = list(map(lambda b: getattr(BoundType, b), obound))
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_scipy(bound):
    """Convert boundary type to SciPy's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] str
        Boundary condition in SciPy's convention

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower()
        if b in ('replicate', 'border', 'nearest', BoundType.replicate):
            obound.append('border')
        elif b in ('zero', 'zeros', 'constant', BoundType.zero):
            obound.append('constant')
        elif b in ('dct2', 'reflect', 'reflection', 'neumann', BoundType.dct2):
            obound.append('reflect')
        elif b in ('dct1', 'mirror', BoundType.dct1):
            obound.append('mirror')
        elif b in ('dft', 'wrap', 'circular', BoundType.dft):
            obound.append('wrap')
        elif b in ('dst2', 'antireflect', 'dirichlet', BoundType.dst2):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        elif b in ('dst1', 'antimirror', BoundType.dst1):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_torch(bound):
    """Convert boundary type to PyTorch's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention

    Returns
    -------
    [list of]
        bound : str
            Boundary condition in PyTorchs's convention
        align_corners : bool or None

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower()
        if b in ('replicate', 'border', 'nearest', BoundType.replicate):
            obound.append(('nearest', None))
        elif b in ('zero', 'zeros', 'constant', BoundType.zero):
            obound.append(('zero', None))
        elif b in ('dct2', 'reflect', 'reflection', 'neumann', BoundType.dct2):
            obound.append(('reflection', True))
        elif b in ('dct1', 'mirror', BoundType.dct1):
            obound.append(('reflection', False))
        elif b in ('dft', 'wrap', 'circular', BoundType.dft):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in ('dst2', 'antireflect', 'dirichlet', BoundType.dst2):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in ('dst1', 'antimirror', BoundType.dst1):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def dft(i, n):
    """Apply DFT (circulant/wrap) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dft)

    """
    if isinstance(i, int):
        return i % n, 1
    return i.remainder(n), 1


def dft_(i, n):
    """Apply DFT (circulant/wrap) boundary conditions to an index, in-place

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dft)

    """
    if isinstance(i, int):
        return dft(i, n)
    return i.remainder_(n), 1


def replicate(i, n):
    """Apply replicate (nearest/border) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for replicate)

    """
    if isinstance(i, int):
        return min(max(i, 0), n-1), 1
    return i.clamp(min=0, max=n-1), 1


def replicate_(i, n):
    """Apply replicate (nearest/border) boundary conditions to an index, in-place

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for replicate)

    """
    if isinstance(i, int):
        return replicate(i, n)
    return i.clamp_(min=0, max=n-1), 1


def dct2(i, n):
    """Apply DCT-II (reflect) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct2)

    """
    n2 = n * 2
    if isinstance(i, int):
        i = n2 - 1 - ((-i-1) % n2) if i < 0 else i % n2
        i = n2 - 1 - i if i >= n else i
        return i, 1
    i = torch.where(i < 0, (n2-1) - (-i-1).remainder(n2),
                    i.remainder(n2))
    i = torch.where(i >= n, (n2 - 1) - i, i)
    return i, 1


def dct2_(i, n):
    """Apply DCT-II (reflect) boundary conditions to an index, in-place

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct2)

    """
    if isinstance(i, int):
        return dct2(i, n)
    n2 = n*2
    pre = (i < 0)
    i[pre] = n2 - 1 - ((-i[pre]-1) % n2)
    i[~pre] = (i[~pre] % n2)
    post = (i >= n)
    i[post] = n2 - i[post] - 1
    return i, 1


def dct1(i, n):
    """Apply DCT-I (mirror) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct1)

    """
    if n == 1:
        if isinstance(i, int):
            return 0, 1
        return torch.zeros_like(i), 1
    else:
        n2 = (n - 1) * 2
        if isinstance(i, int):
            i = abs(i) % n2
            i = n2 - i if i >= n else i
            return i, 1
        i = i.abs().remainder(n2)
        i = torch.where(i >= n, -i + n2, i)
        return i, 1


def dct1_(i, n):
    """Apply DCT-I (mirror) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct1)

    """
    if isinstance(i, int):
        return dct1(i, n)
    if n == 1:
        return i.zero_(), 1
    else:
        n2 = (n - 1) * 2
        i = i.abs_().remainder_(n2)
        mask = i >= n
        i[mask] *= -1
        i[mask] += n2
        return i, 1


def dst1(i, n):
    """Apply DST-I (antimirror) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct1)

    """
    n2 = 2 * (n + 1)
    if isinstance(i, int):
        # sign
        ii = n - 1 - i if i < 0 else i
        ii = ii % n2
        x = 0 if ii == 0 else 1
        x = 0 if ii % (n + 1) == n else x
        x = -x if (ii / (n + 1)) % 2 > 0 else x
        # index
        i = -i - 2 if i < 0 else i
        i = i % n2
        i = n2 - 2 - i if i > n else i
        i = min(max(i, 0), n-1)
        return i, x

    one = torch.ones([1], dtype=torch.int8, device=i.device)
    zero = torch.zeros([1], dtype=torch.int8, device=i.device)
    first = torch.full([1], 0, dtype=i.dtype, device=i.device)
    last = torch.full([1], n - 1, dtype=i.dtype, device=i.device)

    i = torch.where(i < 0, -i - 2, i)
    i = i.remainder(n2)

    # sign
    x = torch.where(i.remainder(n + 1) == n, zero, one)
    x = torch.where((i / (n + 1)).remainder(2) > 0, -x, x)

    # index
    i = torch.where(i > n, -i + (n2 - 2), i)
    i = torch.where(i == -1, first, i)
    i = torch.where(i == n, last, i)
    return i, x


def dst1_(i, n):
    """Apply DST-I (antimirror) boundary conditions to an index, in-place

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct1)

    """
    if isinstance(i, int):
        return dst1(i, n)

    one = torch.ones([1], dtype=torch.int8, device=i.device)
    zero = torch.zeros([1], dtype=torch.int8, device=i.device)

    n2 = 2 * (n + 1)

    mask = i < 0
    i[mask] += 2
    i[mask] *= -1
    i = i.remainder_(n2)

    # sign
    x = torch.where(i.remainder(n + 1) == n, zero, one)
    mask = (i / (n + 1)).remainder(2) > 0
    x *= 1 - 2 * mask

    # index
    mask = i > n
    i[mask] *= -1
    i[mask] += n2 - 2
    i.clamp_(0, n-1)
    return i, x


def dst2(i, n):
    """Apply DST-II (antireflect) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct1)

    """
    if n == 1:
        return dct2(i, n)
    else:
        ii = torch.where(i < 0, n - 1 - i, i)
        x = torch.ones([1], dtype=torch.int8, device=i.device)
        x = torch.where((ii / n).remainder(2) > 0, -x, x)
        return dct2(i, n)[0], x


def dst2_(i, n):
    """Apply DST-II (antireflect) boundary conditions to an index, in-place

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct1)

    """
    if n == 1:
        return dct2_(i, n)
    else:
        ii = torch.where(i < 0, n - 1 - i, i)
        x = torch.ones([1], dtype=torch.int8, device=i.device)
        x = torch.where((ii / n).remainder(2) > 0, -x, x)
        return dct2_(i, n)[0], x


nearest = border = replicate
reflect = neumann = dct2
mirror = dct1
antireflect = dirichlet = dst2
antimirror = dst1
wrap = circular = dft
