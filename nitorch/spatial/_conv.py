import math as pymath
import torch
from torch.nn import functional as F
from nitorch import core
from nitorch.core import utils, math, py
from nitorch.core.py import make_list
from ._affine import affine_conv


def _same_padding(in_size, kernel_size, stride, ceil):
    if not ceil:
        # This is equivalent to the formula below, but with floor
        # instead of ceil. I find this more readable though.
        padding = (kernel_size - stride - in_size % stride)
    else:
        out_size = pymath.ceil(float(in_size) / float(stride))
        padding = max((out_size - 1) * stride + kernel_size - in_size, 0)
    padding = max(0, padding)
    if padding % 2 == 0:
        return padding // 2
    else:
        return (padding // 2, padding - padding // 2)


def compute_conv_shape(input_size, kernel_size, padding=0, dilation=1,
                       stride=1, ceil=False):
    """Compute the amount of padding to apply

    Parameters
    ----------
    input_size : sequence of int
        Spatial shape of the input tensor.
    kernel_size : [sequence of] int
    padding : [sequence of] {'valid', 'same'} or int or (int, int)
        Padding type (if str) or symmetric amount (if int) or
        low/high amount (if [int, int]).
    dilation : [sequence of] int, default=1
        The effective size of the kernel is
        `(kernel_size - 1) * dilation + 1`
    stride : [sequence of] int, default=1
    ceil : bool, default=False
        Ceil mode used to compute output shape
        (tensorflow uses True, pytorch uses False by default)

    Returns
    -------
    output_size : tuple of int

    """
    trunc_fn = pymath.ceil if ceil else pymath.floor

    dim = len(input_size)
    kernel_size = py.ensure_list(kernel_size, dim)
    dilation = py.ensure_list(dilation, dim)
    stride = py.ensure_list(stride, dim)

    padding = compute_conv_padding(input_size, padding, kernel_size,
                                   dilation, stride, ceil)
    padding = _normalize_padding(padding)
    kernel_size = [(k - 1) * d + 1 for (k, d) in zip(kernel_size, dilation)]

    out_size = []
    for L, S, P, K in zip(input_size, stride, padding, kernel_size):
        if isinstance(P, (list, tuple)):
            P = sum(P)
        out_size += [trunc_fn((L + P - K) / S + 1)]
    return out_size


def compute_conv_padding(input_size, kernel_size, padding, dilation=1,
                         stride=1, ceil=False):
    """Compute the amount of padding to apply

    Parameters
    ----------
    input_size : sequence of int
        Spatial shape of the input tensor.
    kernel_size : [sequence of] int
    padding : [sequence of] {'valid', 'same'} or int or (int, int)
        Padding type (if str) or symmetric amount (if int) or
        low/high amount (if [int, int]).
    dilation : [sequence of] int, default=1
        The effective size of the kernel is
        `(kernel_size - 1) * dilation + 1`
    stride : [sequence of] int, default=1
    ceil : bool, default=False
        Ceil mode used to compute output shape
        (tensorflow uses True, pytorch uses False by default)

    Returns
    -------
    padding : tuple of int or (int, int)

    """
    # https://stackoverflow.com/questions/37674306/ (Answer by Vaibhav Dixit)

    dim = len(input_size)
    kernel_size = py.ensure_list(kernel_size, dim)
    dilation = py.ensure_list(dilation, dim)
    stride = py.ensure_list(stride, dim)
    kernel_size = [(k-1) * d + 1 for (k, d) in zip(kernel_size, dilation)]
    padding = py.ensure_list(padding, dim)

    padding = [0 if p == 'valid'
               else _same_padding(i, k, s, ceil) if p in ('same', 'auto')
               else p if isinstance(p, int)
               else tuple(py.ensure_list(p))
               for p, i, k, s in zip(padding, input_size, kernel_size, stride)]
    if not all(isinstance(p, int) or
               (isinstance(p, tuple) and len(p) == 2
                and all(isinstance(pp, int) for pp in p)) for p in padding):
        raise ValueError('Invalid padding', padding)
    return padding


def _normalize_padding(padding):
    """Ensure that padding has format (left, right, top, bottom, ...)"""
    if all(isinstance(p, int) for p in padding):
        return padding
    else:
        npadding = []
        for p in padding:
            if isinstance(p, (list, tuple)):
                npadding.extend(p)
            else:
                npadding.append(p)
                npadding.append(p)
        return npadding


def pad_same(dim, tensor, kernel_size, dilation=1, bound='zero', value=0):
    """Applies a padding that preserves the input dimensions when
    followed by a convolution-like (i.e. moving window) operation.

    Parameters
    ----------
    dim : int
    tensor : (..., *spatial) tensor
    kernel_size : [sequence of] int
    dilation : [sequence f] int, default=1
    bound : {'constant', 'dft', 'dct1', 'dct2', ...}, default='constant'
    value : float, default=0

    Returns
    -------
    padded : (..., *spatial_out) tensor

    """
    kernel_size = make_list(kernel_size, dim)
    dilation = make_list(dilation, dim)
    input_shape = tensor.shape[-dim:]
    padding = compute_conv_padding(input_shape, kernel_size, 'same', dilation)
    padding = _normalize_padding(padding)
    padding = [0] * (2*tensor.dim()-dim) + padding
    return utils.pad(tensor, padding, mode=bound, value=value)


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
    padding : 'same' or int or sequence[int], default=0
        Padding performed before the convolution.
        If 'same', the padding is chosen such that the shape of the
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
    tensor, kernel, bias = utils.to_max_backend(tensor, kernel, bias)

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
    dilation = make_list(dilation, dim)
    padding = make_list(padding, dim)
    padding = [0 if p == 'valid' else 'same' if p == 'auto' else p
               for p in padding]
    for i in range(dim):
        if isinstance(padding[i], str):
            assert padding[i].lower() == 'same'
            if kernel_size[i] % 2 == 0:
                raise ValueError('Cannot compute "same" padding '
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


def _pad_for_ceil(input_size, kernel_size, padding, stride, dilation):
    new_padding = []
    for i in range(len(input_size)):
        L = input_size[i]
        S = stride[i]
        P = padding[i]
        K = kernel_size[i]
        D = dilation[i]
        K = D * (K - 1) + 1
        sumP = P
        if isinstance(P, (list, tuple)):
            sumP = sum(P)
        extra_pad = (L + sumP - K) % S
        if extra_pad:
            extra_pad = S - extra_pad
        if isinstance(P, (list, tuple)):
            pad = (padding[i][0], padding[i][1] + extra_pad)
        else:
            pad = padding[i] + extra_pad
        new_padding.append(pad)
    return new_padding


def _fill_value(reduction, tensor):
    if reduction == 'max':
        if tensor.dtype.is_floating_point:
            fill_value = -float('inf')
        else:
            fill_value = tensor.min()
    elif reduction == 'min':
        if tensor.dtype.is_floating_point:
            fill_value = float('inf')
        else:
            fill_value = tensor.max()
    else:
        fill_value = 0
    return fill_value


def pool(dim, tensor, kernel_size=3, stride=None, dilation=1, padding=0,
         bound='constant', reduction='mean', ceil=False, return_indices=False,
         affine=None):
    """Perform a pooling

    Parameters
    ----------
    dim : {1, 2, 3}
        Number of spatial dimensions
    tensor : (*batch, *spatial_in) tensor
        Input tensor
    kernel_size : int or sequence[int], default=3
        Size of the pooling window. If <= 0, pool the entire dimension.
    stride : int or sequence[int], default=`kernel_size`
        Strides between output elements.
    dilation : int or sequence[int], default=1
        Strides between elements of the kernel.
    padding : 'same' or int or sequence[int], default=0
        Padding performed before the convolution.
        If 'same', the padding is chosen such that the shape of the
        output tensor is `floor(spatial_in / stride)` (or
        `ceil(spatial_in / stride)` if `ceil` is True).
    bound : str, default='constant'
        Boundary conditions used in the padding.
    reduction : {'mean', 'max', 'min', 'median', 'sum', 'ssq'} or callable, default='mean'
        Function to apply to the elements in a window.
    ceil : bool, default=False
        Use ceil instead of floor to compute output shape
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

    # compute padding
    kernel_size = make_list(kernel_size, dim)
    kernel_size = [s if ks <= 0 else ks
                   for s, ks in zip(spatial_in, kernel_size)]
    stride = make_list(stride or None, dim)
    stride = [st or ks for st, ks in zip(stride, kernel_size)]
    dilation = make_list(dilation or 1, dim)
    padding = compute_conv_padding(spatial_in, kernel_size, padding,
                                   dilation, stride, ceil)
    if ceil:
        # ceil mode cannot be obtained using unfold. we may need to
        # pad the input a bit more
        padding = _pad_for_ceil(spatial_in, kernel_size, padding, stride, dilation)

    use_torch = (reduction in ('mean', 'avg', 'max') and 
                 dim in (1, 2, 3) and
                 dilation == [1] * dim)

    padding0 = padding
    sum_padding = sum([sum(p) if isinstance(p, (list, tuple)) else p
                       for p in padding])
    if ((not use_torch) or (bound != 'zero' and sum_padding > 0)
            or any(isinstance(p, (list, tuple)) for p in padding)):
        # torch implementation -> handles zero-padding
        # our implementation -> needs explicit padding
        padding = _normalize_padding(padding)
        tensor = utils.pad(tensor, padding, bound, side='both',
                           value=_fill_value(reduction, tensor))
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
        if reduction == 'ssq':
            reduction = lambda x: math.mean(x.square(), dim=-1).sqrt()
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


def smooth(tensor, type='gauss', fwhm=1, basis=1, bound='dct2', dim=None,
           stride=1, padding='same', fn=None, kernel=None):
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
    stride : [sequence of] int, default=1
        Stride between output elements.
    padding : [sequence of] int or 'same', default='same'
        Amount of padding applied to the input volume.
        'same' ensures that the output dimensions are the same as the
        input dimensions.

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
    if kernel is None:
        fwhm = utils.make_vector(fwhm, dim)
        kernel = core.kernels.smooth(type, fwhm, basis, **backend)

    stride = make_list(stride, dim)
    padding = make_list(padding, dim)
    if torch.is_tensor(kernel):
        if fn:
            kernel = fn(kernel)
        tensor = conv(dim, tensor, kernel, bound=bound,
                      stride=stride, padding=padding)
    else:
        for d, ker in enumerate(kernel):
            substride = [1] * dim
            substride[d] = stride[d]
            subpadding = [0] * dim
            subpadding[d] = padding[d]
            if fn:
                ker = fn(ker)
            tensor = conv(dim, tensor, ker, bound=bound,
                          stride=substride, padding=subpadding)
    # stride = make_list(stride, dim)
    # slicer = [Ellipsis] + [slice(None, None, s) for s in stride]
    # tensor = tensor[tuple(slicer)]
    tensor = tensor.reshape([*batch, *tensor.shape[-dim:]])
    return tensor

