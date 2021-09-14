import math as pymath
import itertools
import torch
from nitorch import core
from nitorch.core import utils, math
from nitorch.core.utils import unsqueeze
import functools


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

    # check input dimensions
    added_dims = max(0, dim + 1 - input.dim())
    input = unsqueeze(input, 0, added_dims)
    if channel_in is not None:
        if input.shape[-dim-1] not in (1, channel_in):
            raise ValueError('Incompatible kernel shape: input channels')
        spatial_shape = input.shape[-dim:]
        batch_shape = input.shape[:-dim-1]
        output_shape = tuple([*batch_shape, channel_out or channel_in, *spatial_shape])
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
                                for stp, st, str in zip(stop, step, start)]
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
    for d in range(dim+1):  # +1 for channel dim
        input = core.utils.fast_movedim(input, -1, 0)
        output = core.utils.fast_movedim(output, -1, 0)
        identity = core.utils.fast_movedim(identity, -1, 0)

    # prepare other stuff
    bound = core.py.ensure_list(bound, dim)
    bound = [getattr(_bounds, b, None) for b in bound]
    # shift = torch.as_tensor([int(pymath.floor(k/2)) for k in kernel_size],
    #                         dtype=torch.long, device=kernel.device)
    shift = [int(pymath.floor(k/2)) for k in kernel_size]
    sides = list(itertools.product([True, False], repeat=dim))

    # Numeric magic to (hopefully) avoid floating point inaccuracy
    subw0 = True
    if subw0:
        kernel, w0 = _split_kernel(kernel, dim)
    else:
        identity = None

    split_idx = _get_idx_split(kernel.dim(), dim)

    # loop across weights in the sparse kernel
    indices = kernel._indices().t().tolist()
    values = kernel._values()
    for idx, weight in zip(indices, values):

        # map input and output channels
        ci, co, idx = split_idx(idx)
        idx = [i - s for i, s in zip(idx, shift)]

        inp = input[ci]
        out = output[co]
        if identity is not None:
            idt = identity[co]
        else:
            idt = None

        # generate slicers
        (input_center_slice, input_side_slice,
         output_center_slice, output_side_slice, transfo_side) = \
            _make_slicers(idx, start, stop, step,
                          output_spatial_shape, spatial_shape, bound)

        # Iterate all combinations of in/out of bounds
        for side in sides:
            input_slicer = tuple(input_center_slice[d] if inside
                                 else input_side_slice[d]
                                 for d, inside in enumerate(side))
            output_slicer = tuple(output_center_slice[d] if inside
                                  else output_side_slice[d]
                                  for d, inside in enumerate(side))
            transfo = tuple(None if inside else transfo_side[d]
                            for d, inside in enumerate(side))

            if any(sl is None for sl in input_slicer):
                continue
            if any(sl is None for sl in output_slicer):
                continue

            _accumulate(out, inp, output_slicer, input_slicer, transfo,
                        weight, idt=idt, diag=(ci == co))

    # add weighted identity
    if subw0:
        w0 = core.utils.unsqueeze(w0, -1, output.dim() - 1)
        output.addcmul_(identity, w0)

    # move spatial dimensions to the back
    for d in range(dim + 1):
        output = core.utils.fast_movedim(output, 0, -1)

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
    w0 = kernel[tuple([Ellipsis, *[s // 2 for s in kernel.shape[-dim:]]])]
    if w0.dim() == 2:
        w0 = torch.stack([w0[d, d] for d in range(dim)])
        diagonal = kernel._indices()[0].eq(kernel._indices()[1])
        center = (kernel._indices()[2:].eq(kernel.shape[-1] // 2).all(0))
        keep = diagonal.bitwise_and_(center).bitwise_not_()
        kernel = torch.sparse_coo_tensor(
            kernel._indices()[:, keep],
            kernel._values()[keep],
            kernel.shape)
        for d in range(dim):
            w0[d].add_(kernel.to_dense()[d, d].sum())
    else:
        center = kernel._indices()[2:].eq(kernel.shape[-1] // 2).all(0)
        keep = center.bitwise_not_()
        kernel = torch.sparse_coo_tensor(
            kernel._indices()[:, keep],
            kernel._values()[keep],
            kernel.shape)
        for d in range(dim):
            w0.add_(kernel.to_dense().sum())
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


def _accumulate(out, inp, output_slicer, input_slicer, transfo, weight, idt=None, diag=False):
    """Accumulate inp into out using slicers

    Parameters
    ----------
    out : (*spatial) tensor
        Output volume
    inp : (*spatial) tensor
        Input volume
    idt : (*spatial) tensor
        Identity volume
    output_slicer : tuple[slice]
    input_slicer : tuple[slice]
    transfo : tuple[callable]
        Tranformations to apply to inp
    weight : () tensor or float
        Convolutin weight

    """
    dat = inp[input_slicer]
    if dat.numel() == 0:
        return
    for trf in transfo:
        if trf:
            dat = trf(dat)
        if dat is None:
            return
    dout = out[output_slicer]
    if dout.numel() == 0:
        return
    if idt is not None and diag:
        dat = dat - idt[output_slicer]
    out[output_slicer].add_(dat, alpha=weight)


def _make_slicers(idx, start, stop, step, oshape, ishape, bound):
    """Generate slicers into input and output volumes

    Parameters
    ----------
    idx : (d,) tensor
        Kernel index (can be negative or positive)
    start : (d,) tuple[int]   \
    stop : (d,) tuple[int]    |> virtual slicing of ouutput volume
    step : (d,) tuple[int]   /
    oshape :  (d,) tuple[int]
        Output shape
    ishape : (d,) tuple[int]
        Input shape
    bound : (d,) tuple[str]
        Boundary conditions

    Returns
    -------
    input_center_slice : (d,) tuple[slice]
    input_side_slice : (d,) tuple[slice]
    output_center_slice : (d,) tuple[slice]
    output_side_slice : (d,) tuple[slice]
    transfo_side : (d,) tuple[callable]

    """

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

    input_center_slice = []
    output_center_slice = []
    input_side_slice = []
    output_side_slice = []
    transfo_side = []

    for d in range(len(idx)):
        inpc, inps, outc, outs, t =  \
            _make_slicer(d, idx[d], start[d], stop[d], step[d],
                           oshape[d], ishape[d], bound[d])
        input_center_slice.append(inpc)
        output_center_slice.append(outc)
        input_side_slice.append(inps)
        output_side_slice.append(outs)
        transfo_side.append(t)

    return (input_center_slice, input_side_slice,
            output_center_slice, output_side_slice, transfo_side)


def _make_slicer(dim, idx, start, stop, step, oshape, ishape, bound):

    # idx = idx.item()  # converted earlier

    # last left out index that is out of bounds
    out_lower = int(pymath.ceil((-idx - start) / float(step))) - 1
    # last right out index that is out of bounds
    out_upper = int(pymath.floor((stop - idx - start) / float(step))) + 1

    # last left inp index that is out of bound
    inp_lower = start + out_lower * step + idx
    # last right inp index that is out of bound
    inp_upper = start + out_upper * step + idx

    # Prepare slicers for the out-of-bound bits
    if out_lower >= 0:
        # out-of-bounds bit is on the left

        if bound is None:
            output_side_slice = None
            input_side_slice = None
            transfo_side = None

        else:

            # bounds
            i_first = inp_lower - step * out_lower
            i_last = inp_lower
            i_first = bound.convert(i_first, ishape)
            if i_first < 0:
                i_first += ishape
            i_last = bound.convert(i_last, ishape)
            if i_last < 0:
                i_last += ishape
            i_first, i_last = ((i_first, i_last) if i_first <= i_last
                               else (i_last, i_first))

            output_side_slice = slice(None, out_lower + 1)
            input_side_slice = slice(i_first, i_last + 1, step)
            transfo_side = bound.transform(dim)

            # if bound == 'dst1':
            #     # FIXME
            #     raise ValueError
            #     # if i < -1:
            #     #     output_side_slice.append(slice(i + 1, None))
            #     #     input_side_slice.append(slice(None, -i - 1, st))
            #     #     transfo_side.append(get_lambda_iflip(d))
            #     # else:
            #     #     output_side_slice.append(None)
            #     #     input_side_slice.append(None)
            #     #     transfo_side.append(None)
            #     # continue

    elif out_upper < oshape:
        # out-of-bounds bit is on the right

        if bound is None:
            output_side_slice = None
            input_side_slice = None
            transfo_side = None

        else:

            # bounds
            i_first = inp_upper + step * (oshape - 1 - out_upper)
            i_last = inp_upper
            i_first = bound.convert(i_first, ishape)
            if i_first < 0:
                i_first += ishape
            i_last = bound.convert(i_last, ishape)
            if i_last < 0:
                i_last += ishape
            i_first, i_last = ((i_first, i_last) if i_first <= i_last
                               else (i_last, i_first))

            output_side_slice = slice(out_upper, None)
            input_side_slice = slice(i_first, i_last + 1, step)
            transfo_side = bound.transform(dim)

            # if bound == 'dst1':
            #     # FIXME
            #     raise ValueError
            #     # if i > 1:
            #     #     output_side_slice.append(slice(None, i - 1))
            #     #     input_side_slice.append(slice(-i + 1, None, st))
            #     #     transfo_side.append(get_lambda_iflip(d))
            #     # else:
            #     #     output_side_slice.append(None)
            #     #     input_side_slice.append(None)
            #     #     transfo_side.append(None)
            #     # continue

    else:
        output_side_slice = None
        input_side_slice = None
        transfo_side = None

    # inbounds bits
    # print(out_lower, out_upper)
    out_lower = max(0, out_lower + 1)
    out_upper = min(oshape - 1, out_upper - 1)
    inp_lower = start + out_lower * step + idx
    inp_upper = start + out_upper * step + idx

    output_center_slice = slice(out_lower, out_upper + 1)
    input_center_slice = slice(inp_lower, inp_upper + 1, step)

    return (input_center_slice, input_side_slice,
            output_center_slice, output_side_slice, transfo_side)


def lambda_flip(x, d):
    if x.shape[d] > 1:
        x = x.flip(d)
    return x


def lambda_iflip(x, d):
    if x.shape[d] > 1:
        x = x.flip(d)
        return x.flip(d).neg()


def get_lambda_flip(d):
    return functools.partial(lambda_flip, d=d)


def get_lambda_iflip(d):
    return functools.partial(lambda_iflip, d=d)


class _bounds:
    # Boundary conditions implemented for python scalars

    class dft:
        @staticmethod
        def convert(i, n):
            """Apply DFT (circulant/wrap) boundary conditions to an index

            Parameters
            ----------
            i : int                 Index
            n : int                 Length of the field of view
            inplace : bool, default=False

            Returns
            -------
            i : int                 Index that falls inside the field of view [0, n-1]

            """
            return i % n

        @staticmethod
        def transform(*a, **k):
            return None

    class replicate:
        @staticmethod
        def convert(i, n):
            """Apply replicate (nearest/border) boundary conditions to an index

            Parameters
            ----------
            i : int                 Index
            n : int                 Length of the field of view

            Returns
            -------
            i : int                 Index that falls inside the field of view [0, n-1]

            """
            return 0 if i < 0 else n-1 if i > n-1 else i

        @staticmethod
        def transform(*a, **k):
            return None

    class dct2:
        @staticmethod
        def convert(i, n):
            """Apply DCT-II (reflect) boundary conditions to an index

            Parameters
            ----------
            i : int                 Index
            n : int                 Length of the field of view

            Returns
            -------
            i : int                 Index that falls inside the field of view [0, n-1]

            """
            n2 = n*2
            if i < 0:
                i = n2 - 1 - ((-i-1) % n2)
            else:
                i = (i % n2)
            if i >= n:
                i = n2 - i - 1
            return i

        @staticmethod
        def transform(dim):
            return get_lambda_flip(dim)

    class dst2(dct2):

        @staticmethod
        def transform(dim):
            return get_lambda_iflip(dim)

    class dct1:
        @staticmethod
        def convert(i, n):
            """Apply DCT-I (mirror) boundary conditions to an index

            Parameters
            ----------
            i : int                 Index
            n : int                 Length of the field of view

            Returns
            -------
            i : int                 Index that falls inside the field of view [0, n-1]

            """
            if n == 1:
                return 0, 1
            else:
                n2 = (n-1)*2
                if i < 0:
                    i = -i
                i = i % n2
                if i >= n:
                    i = n2 - i
                return i

        @staticmethod
        def transform(dim):
            return get_lambda_flip(dim)

    nearest = border = replicate
    reflect = dct2
    mirror = dct1
    wrap = circular = dft
