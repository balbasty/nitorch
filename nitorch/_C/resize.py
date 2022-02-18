from .grid import COMPILED_BACKEND, bound_to_nitorch, inter_to_nitorch
import torch
import math


def make_list(x, n=None):
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    if n:
        x = x + max(0, n - len(x)) * x[-1:]
    return x


def align_to_nitorch(align, as_type='str'):
    """Convert alignment mode to niTorch's convention.

    Parameters
    ----------
    align : [list of] str or align_like
        Alignment mode in any convention
    as_type : {'str', 'enum', 'int'}, default='str'
        Return GridAlignType or int rather than str

    Returns
    -------
    align : [list of] str or GridAlignType
        Alignment mode in NITorch's convention

    """
    from .spatial import GridAlignType

    intype = type(align)
    align = make_list(align)
    oalign = []
    for b in align:
        b = b.lower() if isinstance(b, str) else b
        if b[0] == 'c' or b == GridAlignType.center:
            oalign.append('center')
        elif b[0] == 'e' or b == GridAlignType.edge:
            oalign.append('edge')
        elif b[0] == 'f' or b == GridAlignType.first:
            oalign.append('first')
        elif b[0] == 'l' or b == GridAlignType.last:
            oalign.append('last')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if as_type in ('enum', 'int', int):
        oalign = list(map(lambda b: getattr(GridAlignType, b), oalign))
    if issubclass(intype, (list, tuple)):
        oalign = intype(oalign)
    else:
        oalign = oalign[0]
    return oalign

if COMPILED_BACKEND == 'C':

    from .spatial import (
        resize as _c_resize,
        prolongation as _c_prolongation,
        restriction as _c_restriction)

    def c_resize(input, factor=None, bound='dct2', interpolation=1, mode='center',
                 shape=None, output=None, adjoint=False, normalize=False):
        """Resize a spatial tensor

        Parameters
        ----------
        input : (N, C, *inshape) tensor
        factor : [sequence of] float, default=1.
        bound : [sequence of] bound_like, default='dct2'
        interpolation : [sequence of] int, default=1
        mode : [sequence of] {'c', 'e', 'f', 'l'}, default='c'
        shape : sequence[int], optional
        output : (N, C, *shape) tensor, optional

        Returns
        -------
        output : (N, C, *shape) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        interpolation = inter_to_nitorch(make_list(interpolation), 'enum')
        mode = align_to_nitorch(make_list(mode), 'enum')
        if output is None:
            if shape:
                output = input.new_zeros([*input.shape[:2], *shape])
            else:
                output = torch.Tensor()
        if torch.is_tensor(factor):
            factor = factor.double().tolist()
        if not factor and not shape:
            raise ValueError('At least one of factor or shape must be provided')
        factor = make_list(factor or [1.])
        return _c_resize(input, output, factor, bound, interpolation,
                         mode, adjoint, normalize)

    def c_prolongation(input, factor=2., bound='dct2', interpolation=2,
                       shape=None, output=None):
        """Prolongation of a spatial tensor

        Parameters
        ----------
        input : (N, C, *inshape) tensor
        factor : [sequence of] float, default=2
        bound : [sequence of] bound_like, default='dct2'
        interpolation : [sequence of] int, default=2
        shape : sequence[int], optional
        output : (N, C, *shape) tensor, optional

        Returns
        -------
        output : (N, C, *shape) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        interpolation = inter_to_nitorch(make_list(interpolation), 'enum')
        if output is None:
            if not shape:
                factor = make_list(factor, 3)
                shape = [int((s*f)//1) for s, f in zip(input.shape[2:], factor)]
            if shape:
                output = input.new_zeros([*input.shape[:2], *shape])
            else:
                output = torch.Tensor()
        return _c_prolongation(input, output, bound, interpolation)

    def c_restriction(input, factor=2., bound='dct2', interpolation=1,
                      shape=None, output=None):
        """Restriction of a spatial tensor

        Parameters
        ----------
        input : (N, C, *inshape) tensor
        factor : [sequence of] float, default=2
        bound : [sequence of] bound_like, default='dct2'
        interpolation : [sequence of] int, default=1
        shape : sequence[int], optional
        output : (N, C, *shape) tensor, optional

        Returns
        -------
        output : (N, C, *shape) tensor

        """
        bound = bound_to_nitorch(make_list(bound), 'enum')
        interpolation = inter_to_nitorch(make_list(interpolation), 'enum')
        if output is None:
            if not shape:
                factor = make_list(factor, 3)
                shape = [int(math.ceil(s/f)) for s, f in zip(input.shape[2:], factor)]
            if shape:
                output = input.new_zeros([*input.shape[:2], *shape])
            else:
                output = torch.Tensor()
        return _c_restriction(input, output, bound, interpolation)
