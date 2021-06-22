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


def spconv(input, kernel, step=1, start=0, stop=None, inplace=False, bound='dct2', dim=None):
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
    start : [sequence of] int, default=0
    stop : [sequence of] int, default=None
    step : [sequence of] int, default=1
        Equivalent to spconv(x)[start:stop:step]
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
    start = core.py.ensure_list(start or 0, dim)
    stop = core.py.ensure_list(stop, dim)
    step = core.py.ensure_list(step, dim)

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
    output_spatial_shape = spatial_shape
    start = [0 if not str else str + sz if str < 0 else str
             for str, sz in zip(start, spatial_shape)]
    stop = [sz if stp is None else stp + sz if stp < 0 else stp
            for stp, sz in zip(stop, spatial_shape)]
    stop = [stp - 1 for stp in stop]  # we use an inclusive stop in the rest of the code
    step = [st or 1 for st in step]
    if step:
        output_spatial_shape = [int(pymath.floor((stp-str)/float(st) + 1))
                                for stp, st, str in
                                zip(stop, step, start)]
        output_shape = [*output_shape[:-dim], *output_spatial_shape]
    slicer = [slice(str, stp+1, st) for str, stp, st in zip(start, stop, step)]
    slicer = tuple([Ellipsis, *slicer])
    identity = input[slicer]
    assert identity.shape[-dim:] == tuple(output_shape[-dim:]), "oops"
    if inplace:
        output = identity
        identity = identity.clone()
        output.zero_()
    else:
        output = input.new_zeros(output_shape)

    # move channel + spatial dimensions to the front
    spdim = list(range(input.dim()-dim-1, input.dim()))
    input = movedim(input, spdim, list(range(dim+1)))
    output = movedim(output, spdim, list(range(dim+1)))
    identity = movedim(identity, spdim, list(range(dim+1)))

    # prepare other stuff
    bound = make_list(bound, dim)
    shift = torch.as_tensor([int(pymath.floor(k/2)) for k in kernel_size],
                            dtype=torch.long, device=kernel.device)

    # Numeric magic to (hopefully) avoid floating point inaccuracy
    subw0 = True
    if subw0:
        w0 = kernel[tuple([Ellipsis, *[s//2 for s in kernel.shape[-dim:]]])]
        if w0.dim():
            w0 = torch.stack([w0[d, d] for d in range(dim)])
            diagonal = kernel._indices()[0] == kernel._indices()[1]
            center = (kernel._indices()[2:] == kernel.shape[-1]//2).all(0)
            keep = ~(diagonal & center)
            kernel = torch.sparse_coo_tensor(
                kernel._indices()[:, keep],
                kernel._values()[keep],
                kernel.shape)
        else:
            center = (kernel._indices()[2:] == kernel.shape[-1]//2).all(-1)
            kernel = torch.sparse_coo_tensor(
                kernel._indices()[:, ~center],
                kernel._values()[~center],
                kernel.shape)
        for d in range(dim):
            w0[d] += kernel.to_dense()[d, d].sum()

    # loop across weights in the sparse kernel
    for idx, weight in zip(kernel._indices().t(), kernel._values()):

        # map input and output channels
        if kernel.dim() == dim + 2:
            ci = idx[0]
            co = idx[1]
            idx = idx[2:]
        elif kernel.dim() == dim + 1:
            ci = idx[0]
            idx = idx[1:]
            co = ci
        else:
            ci = co = 0
        idx = idx - shift

        inp = input[ci]
        out = output[co]
        idt = identity[co]

        # Bounds of inbounds/out-of-bounds regions
        #
        # Let j encode output voxels and i encode input voxels, the
        # number of output voxels is `floor[(stop - start)//step] + 1`
        # (i.e., j takes value in [0 .. floor[(stop - start)//step]])
        # The index of a corresponding voxel in the input volume is
        # `i = start + j * step`, and the convolution that we perform
        # can be written as:
        # for all k, y[j] += w[k] x[i + k]
        # x is sampled inbounds if i + k >= 0 and i + k <= stop, which
        # give us
        #             j >= -(k + offset)/stride
        #       => j_low = ceil[-(k + offset)/stride]
        #             j < (stop - offset - k)/stride
        #       => j_up = floor[(stop - offset - k)/stride]


        # out_lower = [int(pymath.ceil((-i-str)/float(st)))
        #              for i, str, st in zip(idx, start, step)]
        # has_center = [l >=0 for l in out_lower]
        # out_lower = [max(0, l) for l in out_lower]
        # out_upper = [min(s-1, int(pymath.floor((stp-i-str)//float(st))))
        #              for stp, s, i, str, st in
        #              zip(stop, output_spatial_shape, idx, start, step)]
        # print('out', out_lower, out_upper)
        # # lower and upper bound of voxels to extract in the input volume
        # # (convert to input coordinates + add kernel index)
        # inp_lower = [o + l * st + i for i, o, l, st
        #              in zip(idx, start, out_lower, step)]
        # inp_upper = [o + u * st + i for i, o, u, st
        #              in zip(idx, start, out_upper, step)]
        # print('inp', inp_lower, inp_upper)
        #
        # # Prepare slicers for the in-bound bits
        # input_center_slice = [slice(l, u+1, st) for l, u, st, s
        #                       in zip(inp_lower, inp_upper, step, spatial_shape)]
        # output_center_slice = [slice(l, u+1) for l, u, s in
        #                        zip(out_lower, out_upper, output_spatial_shape)]

        # last left out index that is out of bounds
        out_lower = [int(pymath.ceil((-i-str)/float(st))) - 1
                     for i, str, st in zip(idx, start, step)]
        # last right out index that is out of bounds
        out_upper = [int(pymath.floor((stp-i-str)//float(st))) + 1
                     for stp, s, i, str, st in
                     zip(stop, output_spatial_shape, idx, start, step)]

        # last left inp index that is out of bound
        inp_lower = [o + l * st + i for i, o, l, st
                     in zip(idx, start, out_lower, step)]
        # last right inp index that is out of bound
        inp_upper = [o + u * st + i for i, o, u, st
                     in zip(idx, start, out_upper, step)]

        # Prepare slicers for the out-of-bound bits
        input_side_slice = []
        output_side_slice = []
        transfo_side = []
        all_params = zip(idx, spatial_shape, bound, start, step)
        for d, (i, s, b, o, st) in enumerate(all_params):
            convert = getattr(core.bounds, b, None)
            # if i < -o:  # do we fall outside of the FOV on the left?
            if out_lower[d] >= 0:

                if b.startswith('zero'):
                    output_side_slice.append(None)
                    input_side_slice.append(None)
                    transfo_side.append(None)
                    continue

                i = i + o
                # bounds
                i_first = inp_lower[d] - st * out_lower[d]
                i_last = inp_lower[d]
                i_first, _ = convert(i_first, s)
                if i_first < 0:
                    i_first += s
                i_last, _ = convert(i_last, s)
                if i_last < 0:
                    i_last += s
                i_first, i_last = ((i_first, i_last) if i_first <= i_last
                                   else (i_last, i_first))
                if b == 'dst1':
                    # FIXME
                    if i < -1:
                        output_side_slice.append(slice(i+1, None))
                        input_side_slice.append(slice(None, -i-1, st))
                        transfo_side.append(get_lambda_iflip(d))
                    else:
                        output_side_slice.append(None)
                        input_side_slice.append(None)
                        transfo_side.append(None)
                    continue
                output_side_slice.append(slice(None, out_lower[d]+1))
                input_side_slice.append(slice(i_first, i_last+1, st))
                if b == 'dct1':
                    transfo_side.append(get_lambda_flip(d))
                elif b == 'dft':
                    transfo_side.append(None)
                elif b == 'replicate':
                    transfo_side.append(None)
                else:
                    if b == 'dct2':
                        transfo_side.append(get_lambda_flip(d))
                    elif b == 'dst2':
                        transfo_side.append(get_lambda_iflip(d))
            # elif i > (stop[d] - o) % st:  # do we fall outside of the FOV on the right?
            elif out_upper[d] < output_spatial_shape[d]:

                if b.startswith('zero'):
                    output_side_slice.append(None)
                    input_side_slice.append(None)
                    transfo_side.append(None)
                    continue

                # bounds
                i_first = inp_upper[d] + st * (output_spatial_shape[d] - 1 - out_upper[d])
                i_last = inp_upper[d]
                i_first, _ = convert(i_first, s)
                i_last, _ = convert(i_last, s)
                if i_first < 0:
                    i_first += s
                i_last, _ = convert(i_last, s)
                if i_last < 0:
                    i_last += s
                i_first, i_last = ((i_first, i_last) if i_first <= i_last
                                   else (i_last, i_first))
                if b == 'dst1':
                    # FIXME
                    if i > 1:
                        output_side_slice.append(slice(None, i-1))
                        input_side_slice.append(slice(-i+1, None, st))
                        transfo_side.append(get_lambda_iflip(d))
                    else:
                        output_side_slice.append(None)
                        input_side_slice.append(None)
                        transfo_side.append(None)
                    continue
                output_side_slice.append(slice(out_upper[d], None))
                input_side_slice.append(slice(i_first, i_last+1, st))
                if b == 'dct1':
                    transfo_side.append(get_lambda_flip(d))
                elif b == 'dft':
                    transfo_side.append(None)
                elif b == 'replicate':
                    transfo_side.append(None)
                else:
                    if b == 'dct2':
                        transfo_side.append(get_lambda_flip(d))
                    elif b == 'dst2':
                        transfo_side.append(get_lambda_iflip(d))
            else:
                output_side_slice.append(None)
                input_side_slice.append(None)
                transfo_side.append(None)

        # inbounds bits
        # print(out_lower, out_upper)
        out_lower = [max(0, l+1) for l in out_lower]
        out_upper = [min(s-1, u-1)
                     for s, u in zip(output_spatial_shape, out_upper)]
        inp_lower = [o + l * st + i for i, o, l, st
                     in zip(idx, start, out_lower, step)]
        inp_upper = [o + u * st + i for i, o, u, st
                     in zip(idx, start, out_upper, step)]

        output_center_slice = [slice(l, u+1) for l, u, s in
                               zip(out_lower, out_upper, output_spatial_shape)]
        input_center_slice = [slice(l, u+1, st) for l, u, st, s
                              in zip(inp_lower, inp_upper, step, spatial_shape)]

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
            if any(sl is None for sl in output_slicer):
                continue

            # slice + apply boundary condition + accumulate
            # print(start, stop, step, spatial_shape, output_spatial_shape)
            # print(idx.tolist(), side, output_slicer, input_slicer)
            # print(inp.shape, out.shape)
            dat = inp[input_slicer]
            for trf in transfo:
                if trf:
                    dat = trf(dat)
                if dat is None:
                    break
            if dat is None:
                continue
            dout = out[output_slicer]
            # if dat.shape != dout.shape:
            #     print(start, stop, step, spatial_shape, output_spatial_shape)
            #     print(idx.tolist(), side, output_slicer, input_slicer)
            #     print(dat.shape, dout.shape)
            #     raise ValueError
            if dout.numel() == 0 or dat.numel() == 0:
                continue
            # print(dat.shape, out[output_slicer].shape)
            if subw0 and ci == co:
                dat = dat - idt[output_slicer]
            out[output_slicer].add_(dat, alpha=weight)
            # out[output_slicer] += 1  ## TEST

    # add weighted identity
    if subw0:
        w0 = core.utils.unsqueeze(w0, -1, output.dim() - 1)
        output.addcmul_(identity, w0)

    # move spatial dimensions to the back
    output = movedim(output, list(range(dim+1)), spdim)

    # remove fake channels
    if channel_in is None:
        output = output.squeeze(len(batch_shape))
    # remove added dimensions
    for _ in range(added_dims):
        output = output.squeeze(-dim-1)
    return output
