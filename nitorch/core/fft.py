import torch
from . import py, utils

_torch_has_fft_module = utils.torch_version('>=', (1, 8))
_torch_has_complex = utils.torch_version('>=', (1, 6))


def fftshift(x, dim=None):
    """Move the first value to the center of the tensor.

    Notes
    -----
    .. If the dimension has an even shape, the center is the first
        position *after* the middle of the tensor: `c = s//2`
    .. This function triggers a copy of the data.
    .. If the dimension has an even shape, `fftshift` and `ifftshift`
       are equivalent.

    Parameters
    ----------
    x : tensor
        Input tensor
    dim : [sequence of] int, optional
        Dimensions to shift

    Returns
    -------
    x : tensor
        Shifted tensor

    """
    x = torch.as_tensor(x)
    if dim is None:
        dim = list(range(x.dim()))
    dim = py.make_list(dim)
    if len(dim) > 1:
        x = x.clone()  # clone to get an additional buffer

    y = torch.empty_like(x)
    slicer = [slice(None)] * x.dim()
    for d in dim:
        # move front to back
        pre = list(slicer)
        pre[d] = slice(None, (x.shape[d]+1)//2)
        post = list(slicer)
        post[d] = slice(x.shape[d]//2, None)
        y[post] = x[pre]
        # move back to front
        pre = list(slicer)
        pre[d] = slice(None, x.shape[d]//2)
        post = list(slicer)
        post[d] = slice((x.shape[d]+1)//2, None)
        y[pre] = x[post]
        # exchange buffers
        x, y = y, x

    return x


def ifftshift(x, dim=None):
    """Move the center value to the front of the tensor.

    Notes
    -----
    .. If the dimension has an even shape, the center is the first
        position *after* the middle of the tensor: `c = s//2`
    .. This function triggers a copy of the data.
    .. If the dimension has an even shape, `fftshift` and `ifftshift`
       are equivalent.

    Parameters
    ----------
    x : tensor
        Input tensor
    dim : [sequence of] int, optional
        Dimensions to shift

    Returns
    -------
    x : tensor
        Shifted tensor

    """
    x = torch.as_tensor(x)
    if dim is None:
        dim = list(range(x.dim()))
    dim = py.make_list(dim)
    if len(dim) > 1:
        x = x.clone()  # clone to get an additional buffer

    y = torch.empty_like(x)
    slicer = [slice(None)] * x.dim()
    for d in dim:
        # move back to front
        pre = list(slicer)
        pre[d] = slice(None, (x.shape[d]+1)//2)
        post = list(slicer)
        post[d] = slice(x.shape[d]//2, None)
        y[pre] = x[post]
        # move front to back
        pre = list(slicer)
        pre[d] = slice(None, x.shape[d]//2)
        post = list(slicer)
        post[d] = slice((x.shape[d]+1)//2, None)
        y[post] = x[pre]
        # exchange buffers
        x, y = y, x

    return x
