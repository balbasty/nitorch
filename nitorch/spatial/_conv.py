import math as pymath
import itertools
import torch
from torch.nn import functional as F
from nitorch import core
from nitorch.core import utils, math
from nitorch.core.utils import to_max_backend, unsqueeze, movedim
from nitorch.core.py import make_list
from ._affine import affine_conv


def conv(dim, tensor, kernel, bias=None, stride=1, padding=0, bound='zero',
         dilation=1, groups=1):
    """Perform a convolution

    Parameters
    ----------
    dim : {1, 2, 3}
        Number of spatial dimensions
    tensor : (*batch, [channel_in,] *spatial_in) tensor
        Input tensor
    kernel : ([channel_in, channel_out,] *kernel_size) tensor
        Convolution kernel
    bias : ([channel_out,]) tensor, optional
        Bias tensor
    stride : int or sequence[int], default=1
        Strides between output elements,
    padding : 'auto' or int or sequence[int], default=0
        Padding performed before the convolution.
        If 'auto', the padding is chosen such that the shape of the
        output tensor is `spatial_in // stride`.
    bound : str, default='zero'
        Boundary conditions used in the padding.
    dilation : int or sequence[int], default=1
        Dilation of the kernel.
    groups : int, default=1

    Returns
    -------
    convolved : (*batch, [channel_out], *spatial_out)

    """
    # move everything to the same dtype/device
    tensor, kernel, bias = to_max_backend(tensor, kernel, bias)

    # sanity checks + reshape for torch's conv
    if kernel.dim() not in (dim, dim + 2):
        raise ValueError('Kernel shape should be (*kernel_size) or '
                         '(channel_in, channel_out, *kernel_size) but '
                         'got {}'.format(kernel.shape))
    has_channels = kernel.dim() == dim + 2
    channels_in = kernel.shape[0] if has_channels else 1
    channels_out = kernel.shape[1] if has_channels else 1
    kernel_size = kernel.shape[(2*has_channels):]
    kernel = kernel.reshape([channels_in, channels_out, *kernel_size])
    batch = tensor.shape[:-(dim+has_channels)]
    spatial_in = tensor.shape[(-dim):]
    if has_channels and tensor.shape[-(dim+has_channels)] != channels_in:
        raise ValueError('Number of input channels not consistent: '
                         'Got {} (kernel) and {} (tensor).' .format(
                         channels_in, tensor.shape[-(dim+has_channels)]))
    tensor = tensor.reshape([-1, channels_in, *spatial_in])
    if bias:
        bias = bias.flatten()
        if bias.numel() == 1:
            bias = bias.expand(channels_out)
        elif bias.numel() != channels_out:
            raise ValueError('Number of output channels not consistent: '
                             'Got {} (kernel) and {} (bias).' .format(
                             channels_out, bias.numel()))

    # Perform padding
    padding = make_list(padding, dim)
    dilation = make_list(dilation, dim)
    for i in range(dim):
        if padding[i].lower() == 'auto':
            if kernel_size[i] % 2 == 0:
                raise ValueError('Cannot compute automatic padding '
                                 'for even-sized kernels.')
            padding[i] = dilation[i] * (kernel_size[i] // 2)
    if bound != 'zero' and sum(padding) > 0:
        tensor = core.utils.pad(tensor, padding, bound, side='both')
        padding = 0

    conv_fn = (F.conv1d if dim == 1 else
               F.conv2d if dim == 2 else
               F.conv3d if dim == 3 else None)
    if not conv_fn:
        raise NotImplementedError('Convolution is only implemented in '
                                  'dimension 1, 2 or 3.')
    tensor = conv_fn(tensor, kernel, bias, stride=stride, padding=padding,
                     dilation=dilation, groups=groups)
    spatial_out = tensor.shape[(-dim):]
    channels_out = [channels_out] if has_channels else []
    tensor = tensor.reshape([*batch, *channels_out, *spatial_out])
    return tensor


conv1d = lambda *args, **kwargs: conv(1, *args, **kwargs)
conv2d = lambda *args, **kwargs: conv(2, *args, **kwargs)
conv3d = lambda *args, **kwargs: conv(3, *args, **kwargs)


def pool(dim, tensor, kernel_size=3, stride=None, dilation=1, padding=0,
         bound='zero', reduction='mean', return_indices=False, affine=None):
    """Perform a pooling

    Parameters
    ----------
    dim : {1, 2, 3}
        Number of spatial dimensions
    tensor : (*batch, *spatial_in) tensor
        Input tensor
    kernel_size : int or sequence[int], default=3
        Size of the pooling window
    stride : int or sequence[int], default=`kernel_size`
        Strides between output elements.
    dilation : int or sequece[int], default=1
        Strides between elements of the kernel.
    padding : 'auto' or int or sequence[int], default=0
        Padding performed before the convolution.
        If 'auto', the padding is chosen such that the shape of the
        output tensor is `spatial_in // stride`.
    bound : str, default='zero'
        Boundary conditions used in the padding.
    reduction : {'mean', 'max', 'min', 'median', 'sum'} or callable, default='mean'
        Function to apply to the elements in a window.
    return_indices : bool, default=False
        Return input index of the min/max/median element.
        For other types of reduction, return None.
    affine : (..., D+1, D+1) tensor, optional
        Input orientation matrix

    Returns
    -------
    pooled : (*batch, *spatial_out) tensor
    indices : (*batch, *spatial_out, dim) tensor, if `return_indices`
    affine : (..., D+1, D+1) tensor, if `affine`

    """
    # move everything to the same dtype/device
    tensor = torch.as_tensor(tensor)

    # sanity checks + reshape for torch's conv
    batch = tensor.shape[:-dim]
    spatial_in = tensor.shape[-dim:]
    tensor = tensor.reshape([-1, *spatial_in])

    # Perform padding
    kernel_size = make_list(kernel_size, dim)
    stride = make_list(stride or None, dim)
    stride = [st or ks for st, ks in zip(stride, kernel_size)]
    dilation = make_list(dilation or 1, dim)
    padding = make_list(padding, dim)
    padding0 = padding  # save it to update the affine
    for i in range(dim):
        if isinstance(padding[i], str) and padding[i].lower() == 'auto':
            if kernel_size[i] % 2 == 0:
                raise ValueError('Cannot compute automatic padding '
                                 'for even-sized kernels.')
            padding[i] = ((kernel_size[i]-1) * dilation[i] + 1) // 2

    use_torch = (reduction in ('mean', 'avg', 'max') and 
                 dim in (1, 2, 3) and
                 dilation == [1] * dim)

    if (not use_torch) or bound != 'zero' and sum(padding) > 0:
        # torch implementation -> handles zero-padding
        # our implementation -> needs explicit padding
        tensor = utils.pad(tensor, padding, bound, side='both')
        padding = [0] * dim

    return_indices0 = False
    pool_fn = reduction if callable(reduction) else None

    if use_torch:
        if reduction in ('mean', 'avg'):
            return_indices0 = return_indices
            return_indices = False
            pool_fn = (F.avg_pool1d if dim == 1 else
                       F.avg_pool2d if dim == 2 else
                       F.avg_pool3d if dim == 3 else None)
            if pool_fn:
                pool_fn0 = pool_fn
                pool_fn = lambda x, *a, **k: pool_fn0(x[:, None], *a, **k,
                                                      padding=padding)[:, 0]
        elif reduction == 'max':
            pool_fn = (F.max_pool1d if dim == 1 else
                       F.max_pool2d if dim == 2 else
                       F.max_pool3d if dim == 3 else None)
            if pool_fn:
                pool_fn0 = pool_fn
                pool_fn = lambda x, *a, **k: pool_fn0(x[:, None], *a, **k,
                                                      padding=padding)[:, 0]

    if not pool_fn:
        if reduction not in ('min', 'max', 'median'):
            return_indices0 = return_indices
            return_indices = False
        if reduction == 'mean':
            reduction = lambda x: math.mean(x, dim=-1)
        elif reduction == 'sum':
            reduction = lambda x: math.sum(x, dim=-1)
        elif reduction == 'min':
            reduction = lambda x: math.min(x, dim=-1)
        elif reduction == 'max':
            reduction = lambda x: math.max(x, dim=-1)
        elif reduction == 'median':
            reduction = lambda x: math.median(x, dim=-1)
        elif not callable(reduction):
            raise ValueError(f'Unknown reduction {reduction}')
        pool_fn = lambda *a, **k: _pool(*a, **k, dilation=dilation, reduction=reduction)

    outputs = []
    if return_indices:
        tensor, ind = pool_fn(tensor, kernel_size, stride=stride)
        ind = utils.ind2sub(ind, stride)
        ind = utils.movedim(ind, 0, -1)
        outputs.append(ind)
    else:
        tensor = pool_fn(tensor, kernel_size, stride=stride)
        if return_indices0:
            outputs.append(None)

    spatial_out = tensor.shape[-dim:]
    tensor = tensor.reshape([*batch, *spatial_out])
    outputs = [tensor, *outputs]

    if affine is not None:
        affine, _ = affine_conv(affine, spatial_in,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding0, dilation=dilation)
        outputs.append(affine)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _pool(x, kernel_size, stride, dilation, reduction, return_indices=False):
    """Implement pooling by "manually" extracting patches using `unfold`

    Parameters
    ----------
    x : (batch, *spatial)
    kernel_size : (dim,) int
    stride : (dim,) int
    dilation : (dim,) int
    reduction : callable
        This function should collapse the last dimension of a tensor
        (..., K) tensor -> (...)
    return_indices : bool

    Returns
    -------
    x : (batch, *spatial_out)

    """
    kernel_size = [(sz-1)*dl + 1 for sz, dl in zip(kernel_size, dilation)]
    for d, (sz, st, dl) in enumerate(zip(kernel_size, stride, dilation)):
        x = x.unfold(dimension=d + 1, size=sz, step=st)
        if dl != 1:
            x = x[..., ::dl]
    dim = len(kernel_size)
    patch_shape = x.shape[dim+1:]
    x = x.reshape((*x.shape[:dim+1], -1))
    if return_indices:
        x, ind = reduction(x, return_indices=True)
        return x, ind
    else:
        x = reduction(x)
        return x


pool1d = lambda *args, **kwargs: pool(1, *args, **kwargs)
pool2d = lambda *args, **kwargs: pool(2, *args, **kwargs)
pool3d = lambda *args, **kwargs: pool(3, *args, **kwargs)


def smooth(tensor, type='gauss', fwhm=1, basis=1, bound='dct2', dim=None):
    """Smooth a tensor.

    Parameters
    ----------
    tensor : (..., *spatial) tensor
        Input tensor. Its last `dim` dimensions
        will be smoothed.
    type : {'gauss', 'tri', 'rect'}, default='gauss'
        Smoothing function:
        - 'gauss' : Gaussian
        - 'tri'   : Triangular
        - 'rect'  : Rectangular
    fwhm : float or sequence[float], default=1
        Full-Width at Half-Maximum of the smoothing function.
    basis : {0, 1}, default=1
        Encoding basis.
        The smoothing kernel is the convolution of a smoothing
        function and an encoding basis. This ensures that the
        resulting kernel integrates to one.
    bound : str, default='dct2'
        Boundary condition.
    dim : int, default=tensor.dim()
        Dimension of the convolution.
        The last `dim` dimensions of `tensor` will
        be smoothed.

    Returns
    -------
    tensor : (..., *spatial) tensor
        The resulting tensor has the same shape as the input tensor.
        This differs from the behaviour of torch's `conv*d`.
    """

    dim = dim or tensor.dim()
    batch = tensor.shape[:-dim]
    shape = tensor.shape[-dim:]
    tensor = tensor.reshape([-1, 1, *shape])
    backend = dict(dtype=tensor.dtype, device=tensor.device)
    fwhm = make_list(fwhm, dim)
    kernels = core.kernels.smooth(type, fwhm, basis, **backend)
    pad_size = [kernels[i].shape[i + 2] // 2 for i in range(len(kernels))]
    pad_size = [0, 0] + pad_size
    bound = ('reflect2' if bound == 'dct2' else
             'reflect1' if bound == 'dct1' else
             'circular' if bound == 'dft' else
             'reflect2')
    tensor = core.utils.pad(tensor, pad_size, mode=bound, side='both')
    if dim == 1:
        conv = torch.nn.functional.conv1d
    elif dim == 2:
        conv = torch.nn.functional.conv2d
    elif dim == 3:
        conv = torch.nn.functional.conv3d
    else:
        raise NotImplementedError
    for kernel in kernels:
        tensor = conv(tensor, kernel)
    tensor = tensor.reshape([*batch, *shape])
    return tensor

