import math
import itertools
import torch
from torch.nn import functional as F
from nitorch import core
from nitorch.core.utils import to_max_backend, unsqueeze, movedim
from nitorch.core.pyutils import make_list


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
    spatial_in = tensor.shape[:(-dim)]
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
    spatial_out = tensor.shape[:(-dim)]
    channels_out = [channels_out] if has_channels else []
    tensor = tensor.reshape([*batch, *channels_out, *spatial_out])
    return tensor


conv1d = lambda *args, **kwargs: conv(1, *args, **kwargs)
conv2d = lambda *args, **kwargs: conv(2, *args, **kwargs)
conv3d = lambda *args, **kwargs: conv(3, *args, **kwargs)


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
    .. Data outside the field-of-view is extrapolated according to `bound`
    .. It is implemented as a linear combination of views into the input
       tensor and should therefore be relatively memory-efficient.

    Parameters
    ----------
    input : (..., [channel_in], *spatial) tensor
        Input tensor, to convolve.
    kernel : (..., [channel_in, [channel_out]], *kernel_size) sparse tensor
        Convolution kernel.
    bound : [sequence of] str, default='dct2'
    dim : int, default=kernel.dim()

    Returns
    -------
    output : (..., [channel_out], *spatial) tensor

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
        spatial_shape = input.shape[-dim:]
        batch_shape = input.shape[:-dim-1]
        output_shape = input.shape
    output = input.new_zeros(output_shape)

    # move spatial dimensions to the front
    spdim = list(range(input.dim()-dim-1, input.dim()))
    input = movedim(input, spdim, list(range(dim+1)))
    output = movedim(output, spdim, list(range(dim+1)))

    # prepare other stuff
    bound = make_list(bound, dim)
    shift = torch.LongTensor([int(math.floor(k/2)) for k in kernel_size])

    for idx, weight in zip(kernel._indices().t(), kernel._values()):
        if kernel.dim() == dim + 2:
            ci, co, *idx = idx
        elif kernel.dim() == dim + 1:
            ci, *idx = idx
            co = ci
        else:
            ci = co = 0
        idx = idx - shift

        # def zpadl(x, d):
        #     pad = [0] * x.dim()
        #     pad[d] = 1
        #     return core.utils.pad(x, pad, side='left')
        #
        # def zpadr(x, d):
        #     pad = [0] * x.dim()
        #     pad[d] = 1
        #     return core.utils.pad(x, pad, side='right')

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
                        transfo_side.append(lambda x: x.flip(d).neg())
                    else:
                        output_side_slice.append(None)
                        input_side_slice.append(None)
                        transfo_side.append(lambda x: None)
                    continue
                output_side_slice.append(slice(i, None))
                if b == 'dct1':
                    input_side_slice.append(slice(1, -i+1))
                    transfo_side.append(lambda x: x.flip(d))
                elif b == 'dft':
                    input_side_slice.append(slice(i, None))
                    transfo_side.append(lambda x: x)
                elif b == 'replicate':
                    input_side_slice.append(slice(None, 1))
                    transfo_side.append(lambda x: x)
                elif b == 'zeros':
                    input_side_slice.append(None)
                    transfo_side.append(lambda x: None)
                else:
                    input_side_slice.append(slice(None, -i))
                    if b == 'dct2':
                        transfo_side.append(lambda x: x.flip(d))
                    elif b == 'dst2':
                        transfo_side.append(lambda x: x.flip(d).neg())
            elif i > 0:
                if b == 'dst1':
                    if i > 1:
                        output_side_slice.append(slice(None, i-1))
                        input_side_slice.append(slice(-i+1, None))
                        transfo_side.append(lambda x: x.flip(d).neg())
                    else:
                        output_side_slice.append(None)
                        input_side_slice.append(None)
                        transfo_side.append(lambda x: None)
                    continue
                output_side_slice.append(slice(None, i))
                if b == 'dct1':
                    input_side_slice.append(slice(-i-1, -1))
                    transfo_side.append(lambda x: x.flip(d))
                elif b == 'dft':
                    input_side_slice.append(slice(None, i))
                    transfo_side.append(lambda x: x)
                elif b == 'replicate':
                    input_side_slice.append(slice(-1, None))
                    transfo_side.append(lambda x: x)
                elif b == 'zeros':
                    input_side_slice.append(None)
                    transfo_side.append(lambda x: None)
                else:
                    input_side_slice.append(slice(-i, None))
                    if b == 'dct2':
                        transfo_side.append(lambda x: x.flip(d))
                    elif b == 'dst2':
                        transfo_side.append(lambda x: x.flip(d).neg())
            else:
                output_side_slice.append(None)
                input_side_slice.append(None)
                transfo_side.append(None)

        # Prepare slicers for the in-bound bits
        input_center_slice = [slice(max(0, -i), min(s - i, s))
                              for s, i in zip(spatial_shape, idx)]
        output_center_slice = [slice(max(0, i), min(s + i, s))
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
            transfo = [(lambda x: x) if inside else transfo_side[d]
                       for d, inside in enumerate(side)]

            if any(sl is None for sl in input_slicer):
                continue

            # slice + apply boundary condition + accumulate
            dat = inp[input_slicer]
            for trf in transfo:
                if dat is not None and trf is not None:
                    dat = trf(dat)
            if dat is not None:
                out[output_slicer] += weight * dat
            del dat

    # move spatial dimensions to the back
    output = movedim(output, list(range(dim+1)), spdim)
    for _ in range(added_dims):
        output = output.squeeze(-dim-1)
    return output
