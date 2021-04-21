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

    use_torch = reduction in ('mean', 'avg', 'max') and dim in (1, 2, 3)

    if (not use_torch) or bound != 'zero' and sum(padding) > 0:
        # torch implementation -> handles zero-padding
        # our implementation -> needs explicit padding
        tensor = utils.pad(tensor, padding, bound, side='both')
        padding = [0] * dim

    return_indices0 = False
    pool_fn = reduction if callable(reduction) else None

    if reduction in ('mean', 'avg'):
        return_indices0 = True
        return_indices = False
        pool_fn = (F.avg_pool1d if dim == 1 else
                   F.avg_pool2d if dim == 2 else
                   F.avg_pool3d if dim == 3 else None)
        if pool_fn:
            pool_fn0 = pool_fn
            pool_fn = lambda x, *a, **k: pool_fn0(x[:, None], *a, **k,
                                                  padding=padding,
                                                  dilation=dilation)[:, 0]
    elif reduction == 'max':
        pool_fn = (F.max_pool1d if dim == 1 else
                   F.max_pool2d if dim == 2 else
                   F.max_pool3d if dim == 3 else None)
        if pool_fn:
            pool_fn0 = pool_fn
            pool_fn = lambda x, *a, **k: pool_fn0(x[:, None], *a, **k,
                                                  padding=padding,
                                                  dilation=dilation)[:, 0]

    if not pool_fn:
        if reduction not in ('min', 'max', 'median'):
            return_indices0 = True
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
        pool_fn = lambda *a, **k: _pool(*a, **k, reduction=reduction)

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


pool1d = lambda *args, **kwargs: pool(1, **kwargs)
pool2d = lambda *args, **kwargs: pool(2, **kwargs)
pool3d = lambda *args, **kwargs: pool(3, **kwargs)


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


def spconv(input, kernel, bound='dct2', dim=None):
    """Convolution with a sparse kernel.

    Notes
    -----
    .. This convolution does not support strides, padding, dilation.
    .. The output spatial shape is the same as the input spatial shape.
    .. The output batch shape is the same as the input batch shape.
    .. Data outside the field-of-view is extrapolated according to `bound`
    .. It is implemented as a linear combination of views into the input
       tensor and should therefore be relatively memory-efficient.

    Parameters
    ----------
    input : (..., [channel_in], *spatial) tensor
        Input tensor, to convolve.
    kernel : ([channel_in, [channel_out]], *kernel_size) sparse tensor
        Convolution kernel.
    bound : [sequence of] str, default='dct2'
        Boundary condition (per spatial dimension).
    dim : int, default=kernel.dim()
        Number of spatial dimensions.

    Returns
    -------
    output : (..., [channel_out or channel_in], *spatial) tensor

        * If the kernel shape is (channel_in, channel_out, *kernel_size),
          the output shape is (..., channel_out, *spatial) and cross-channel
          convolution happens:
            out[co] = \sum_{ci} conv(inp[ci], ker[ci, co])
        * If the kernel_shape is (channel_in, *kernel_size), independent
          single-channel convolutions are applied to each channels::
            out[c] = conv(inp[c], ker[c])
        * If the kernel shape is (*kernel_size), the same convolution
          is applied to all input channels:
            out[c] = conv(inp[c], ker)

    """
    # get kernel dimensions
    dim = dim or kernel.dim()
    if kernel.dim() == dim + 2:
        channel_in, channel_out, *kernel_size = kernel.shape
    elif kernel.dim() == dim + 1:
        channel_in, *kernel_size = kernel.shape
        channel_out = None
    elif kernel.dim() == dim:
        kernel_size = kernel.shape
        channel_in = channel_out = None
    else:
        raise ValueError('Incompatible kernel shape: too many dimensions')

    import functools
    def lambda_flip(x, d):
        if x.shape[d] > 1:
            x = x.flip(d)
        return x

    def lambda_iflip(x, d):
        if x.shape[d] > 1:
            x = x.flip(d)
            return x.flip(d).neg()

    def get_lambda_flip(d): return functools.partial(lambda_flip, d=d)
    def get_lambda_iflip(d): return functools.partial(lambda_iflip, d=d)

    # check input dimensions
    added_dims = max(0, dim + 1 - input.dim())
    input = unsqueeze(input, 0, added_dims)
    if channel_in is not None:
        if input.shape[-dim-1] not in (1, channel_in):
            raise ValueError('Incompatible kernel shape: input channels')
        spatial_shape = input.shape[-dim:]
        batch_shape = input.shape[:-dim-1]
        output_shape = tuple([*batch_shape, channel_out, *spatial_shape])
    else:
        # add a fake channel dimension
        spatial_shape = input.shape[-dim:]
        batch_shape = input.shape[:-dim]
        input = input.reshape([*batch_shape, 1, *spatial_shape])
        output_shape = input.shape
    output = input.new_zeros(output_shape)

    # move channel + spatial dimensions to the front
    spdim = list(range(input.dim()-dim-1, input.dim()))
    input = movedim(input, spdim, list(range(dim+1)))
    output = movedim(output, spdim, list(range(dim+1)))

    # prepare other stuff
    bound = make_list(bound, dim)
    shift = torch.LongTensor([int(pymath.floor(k/2)) for k in kernel_size])

    for idx, weight in zip(kernel._indices().t(), kernel._values()):
        if kernel.dim() == dim + 2:
            ci, co, *idx = idx
        elif kernel.dim() == dim + 1:
            ci, *idx = idx
            co = ci
        else:
            ci = co = 0
        idx = idx - shift

        inp = input[ci]
        out = output[co]

        # Prepare slicers for the out-of-bound bits
        input_side_slice = []
        output_side_slice = []
        transfo_side = []
        for d, (i, s, b) in enumerate(zip(idx, spatial_shape, bound)):
            if i < 0:
                if b == 'dst1':
                    if i < -1:
                        output_side_slice.append(slice(i+1, None))
                        input_side_slice.append(slice(None, -i-1))
                        transfo_side.append(get_lambda_iflip(d))
                    else:
                        output_side_slice.append(None)
                        input_side_slice.append(None)
                        transfo_side.append(None)
                    continue
                output_side_slice.append(slice(None, -i))
                if b == 'dct1':
                    input_side_slice.append(slice(1, -i+1))
                    transfo_side.append(get_lambda_flip(d))
                elif b == 'dft':
                    input_side_slice.append(slice(i, None))
                    transfo_side.append(None)
                elif b == 'replicate':
                    input_side_slice.append(slice(None, 1))
                    transfo_side.append(None)
                elif b == 'zeros':
                    input_side_slice.append(None)
                    transfo_side.append(None)
                else:
                    input_side_slice.append(slice(None, -i))
                    if b == 'dct2':
                        transfo_side.append(get_lambda_flip(d))
                    elif b == 'dst2':
                        transfo_side.append(get_lambda_iflip(d))
            elif i > 0:
                if b == 'dst1':
                    if i > 1:
                        output_side_slice.append(slice(None, i-1))
                        input_side_slice.append(slice(-i+1, None))
                        transfo_side.append(get_lambda_iflip(d))
                    else:
                        output_side_slice.append(None)
                        input_side_slice.append(None)
                        transfo_side.append(None)
                    continue
                output_side_slice.append(slice(-i, None))
                if b == 'dct1':
                    input_side_slice.append(slice(-i-1, -1))
                    transfo_side.append(get_lambda_flip(d))
                elif b == 'dft':
                    input_side_slice.append(slice(None, i))
                    transfo_side.append(None)
                elif b == 'replicate':
                    input_side_slice.append(slice(-1, None))
                    transfo_side.append(None)
                elif b == 'zeros':
                    input_side_slice.append(None)
                    transfo_side.append(None)
                else:
                    input_side_slice.append(slice(-i, None))
                    if b == 'dct2':
                        transfo_side.append(get_lambda_flip(d))
                    elif b == 'dst2':
                        transfo_side.append(get_lambda_iflip(d))
            else:
                output_side_slice.append(None)
                input_side_slice.append(None)
                transfo_side.append(None)

        # Prepare slicers for the in-bound bits
        input_center_slice = [slice(max(0, i), min(s + i, s))
                              for s, i in zip(spatial_shape, idx)]
        output_center_slice = [slice(max(0, -i), min(s - i, s))
                               for s, i in zip(spatial_shape, idx)]

        # Iterate all combinations of in/out of bounds
        sides = itertools.product([True, False], repeat=dim)
        for side in sides:
            input_slicer = [input_center_slice[d] if inside
                            else input_side_slice[d]
                            for d, inside in enumerate(side)]
            output_slicer = [output_center_slice[d] if inside
                             else output_side_slice[d]
                             for d, inside in enumerate(side)]
            transfo = [None if inside else transfo_side[d]
                       for d, inside in enumerate(side)]

            if any(sl is None for sl in input_slicer):
                continue

            # slice + apply boundary condition + accumulate
            dat = inp[input_slicer]
            for trf in transfo:
                if trf:
                    dat = trf(dat)
                if dat is None:
                    break
            if dat is not None:
                out[output_slicer].addcmul_(dat, weight)

    # move spatial dimensions to the back
    output = movedim(output, list(range(dim+1)), spdim)
    # remove fake channels
    if channel_in is None:
        output = output.squeeze(len(batch_shape))
    # remove added dimensions
    for _ in range(added_dims):
        output = output.squeeze(-dim-1)
    return output
