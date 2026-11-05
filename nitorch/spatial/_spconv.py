import torch
from nitorch import core
from nitorch.core.utils import unsqueeze


def spconv(input, kernel, step=1, start=0, stop=None, inplace=False, bound='dct2', dim=None):
    r"""Convolution with a sparse kernel.

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

    # check input dimensions
    added_dims = max(0, dim + 1 - input.dim())
    input = unsqueeze(input, 0, added_dims)
    if channel_in is not None:
        if input.shape[-dim-1] not in (1, channel_in):
            raise ValueError('Incompatible kernel shape: input channels')
        spatial_shape = input.shape[-dim:]
        batch_shape = input.shape[:-dim-1]
        nb_out = channel_out or channel_in
    else:
        # add a fake channel dimension
        spatial_shape = input.shape[-dim:]
        batch_shape = input.shape[:-dim]
        input = input.reshape([*batch_shape, 1, *spatial_shape])
        nb_out = 1
    start = [0 if not str else str + sz if str < 0 else str
             for str, sz in zip(start, spatial_shape)]
    stop = [sz if stp is None else stp + sz if stp < 0 else stp
            for stp, sz in zip(stop, spatial_shape)]
    step = [st or 1 for st in step]
    slicer = tuple([Ellipsis, *[slice(str, stp, st) for str, stp, st
                                in zip(start, stop, step)]])
    identity = input[slicer]
    output_spatial_shape = identity.shape[-dim:]

    # Extend the input by the radius of the kernel, according to its
    # boundary condition (each extended sample copies a voxel, flips its
    # sign, or is zero), and convolve the extended input.
    radius = [k // 2 for k in kernel_size]
    bound = core.py.ensure_list(bound, dim)
    extended = core.utils.pad(input, radius, mode=bound, side='both')

    # Numeric magic to (hopefully) avoid floating point inaccuracy:
    # the central weights of the kernels that map a channel onto itself
    # are applied as `w0 * x`, where `w0` is the sum of their weights,
    # and the other weights as `w * (x[shifted] - x)`.
    kernel, w0 = _split_kernel(kernel, dim)
    split_idx = _get_idx_split(kernel.dim(), dim)

    output = None
    outputs = [0] * nb_out
    indices = kernel._indices().t().tolist()
    values = kernel._values()
    for idx, weight in zip(indices, values):
        ci, co, idx = split_idx(idx)
        ci = 0 if ci is None else ci
        co = 0 if co is None else co
        if input.shape[-dim-1] == 1:
            ci = 0
        shifted = extended.select(-dim-1, ci)
        shifted = shifted[(Ellipsis, *[
            slice(str + i, stp + i, st)
            for str, stp, st, i in zip(start, stop, step, idx)])]
        if ci == co:
            shifted = shifted - identity.select(-dim-1, ci)
        outputs[co] = outputs[co] + shifted * weight

    # add weighted identity
    w0 = torch.as_tensor(w0, dtype=input.dtype, device=input.device)
    for co in range(nb_out):
        ci = 0 if input.shape[-dim-1] == 1 else co
        if ci >= identity.shape[-dim-1]:
            continue
        w = w0[co] if w0.dim() else w0
        out = outputs[co] + identity.select(-dim-1, ci) * w
        if not torch.is_tensor(out):
            out = identity.select(-dim-1, ci).new_zeros(output_spatial_shape)
        outputs[co] = out.expand([*batch_shape, *output_spatial_shape])
    output = torch.stack(outputs, -dim-1)

    if inplace:
        input[slicer].copy_(output)
        output = input[slicer]

    # remove fake channels
    if channel_in is None:
        output = output.squeeze(len(batch_shape))
    # remove added dimensions
    for _ in range(added_dims):
        output = output.squeeze(-dim-1)
    return output


def _split_kernel(kernel, dim):
    """Split the kernel into central and non central weights
     (for improved numerical accuracy)

    Parameters
    ----------
    kernel : ([ci, co], *spatial) sparse tensor
    dim : int

    Returns
    -------
    kernel : ([ci, co], *spatial) sparse tensor
        Kernel without (diagonal) central weights
    w0 : ([d,]) tensor
        Sum of weights of each (diagonal) kernel

    """
    kernel = kernel.coalesce()
    indices, values = kernel.indices(), kernel.values()
    nb_chan = kernel.dim() - dim  # 0: scalar, 1: diagonal, 2: matrix
    kcenter = [s // 2 for s in kernel.shape[-dim:]]
    kcenter = torch.as_tensor(kcenter, dtype=indices.dtype,
                              device=indices.device)
    center = indices[nb_chan:].eq(kcenter[:, None]).all(0)
    if nb_chan == 2:
        # only kernels that map a channel onto itself are "diagonal"
        diagonal = indices[0].eq(indices[1])
    else:
        diagonal = torch.ones_like(center)
    channel = indices[0] if nb_chan else torch.zeros_like(indices[0])
    nb_out = kernel.shape[0] if nb_chan else 1
    # w0[c] = sum of all weights of the (diagonal) kernel c
    w0 = values.new_zeros([nb_out])
    w0.index_add_(0, channel[diagonal], values[diagonal])
    if not nb_chan:
        w0 = w0[0]
    # remove (diagonal) central weights
    keep = diagonal.bitwise_and(center).bitwise_not_()
    kernel = torch.sparse_coo_tensor(indices[:, keep], values[keep],
                                     kernel.shape)
    return kernel, w0


def _get_idx_split(kdim, dim):

    def split_matrix(idx):
        ci = idx[0]
        co = idx[1]
        idx = idx[2:]
        return ci, co, idx

    def split_diag(idx):
        ci = idx[0]
        idx = idx[1:]
        co = ci
        return ci, co, idx

    def split_scalar(idx):
        ci = co = 0
        return ci, co, idx

    if kdim == dim + 2:
        return split_matrix
    elif kdim == dim + 1:
        return split_diag
    else:
        return split_scalar
