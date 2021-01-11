import torch
from torch.nn import functional as F
from nitorch import core


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
    tensor, kernel, bias = core.utils.to_max_backend(tensor, kernel, bias)

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
    padding = core.pyutils.make_list(padding, dim)
    dilation = core.pyutils.make_list(dilation, dim)
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
    fwhm = core.pyutils.make_list(fwhm, dim)
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
