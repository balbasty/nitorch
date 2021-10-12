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


def dft(i, n, inplace=False):
    """Apply DFT (circulant/wrap) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view
    inplace : bool, default=False

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dft)

    """
    return (i.fmod_(n) if inplace else i.fmod(n)), 1


def replicate(i, n, inplace=False):
    """Apply replicate (nearest/border) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view
    inplace : bool, default=False

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for replicate)

    """
    return i.clamp_(min=0, max=n-1) if inplace else i.clamp(min=0, max=n-1), 1


def dct2(i, n, inplace=False):
    """Apply DCT-II (reflect) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view
    inplace : bool, default=False

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct2)

    """
    if not inplace:
        i = i.clone()
    n2 = n*2
    pre = (i < 0)
    i[pre] = n2 - 1 - ((-i[pre]-1) % n2)
    i[~pre] = (i[~pre] % n2)
    post = (i >= n)
    i[post] = n2 - i[post] - 1
    return i, 1


def dct1(i, n, inplace=False):
    """Apply DCT-I (mirror) boundary conditions to an index

    Parameters
    ----------
    i : int                 Index
    n : int                 Length of the field of view
    inplace : bool, default=False

    Returns
    -------
    i : int                 Index that falls inside the field of view [0, n-1]
    s : {1, -1}             Sign of the transformation (always 1 for dct1)

    """
    if n == 1:
        return i.zero_() if inplace else torch.zeros_like(i), 1
    else:
        if not inplace:
            i = i.clone()
        n2 = (n-1)*2
        pre = (i < 0)
        i[pre].neg_()
        i = i.fmod_(n2)
        post = (i >= n)
        i[post].neg_()
        i[post] += n2
        return i, 1


nearest = border = replicate
reflect = dct2
mirror = dct1
wrap = circular = dft
