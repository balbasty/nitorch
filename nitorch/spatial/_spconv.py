import math as pymath
import itertools
import torch
from nitorch import core
from nitorch.core import utils, math
from nitorch.core.utils import to_max_backend, unsqueeze, movedim
from nitorch.core.py import make_list


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
