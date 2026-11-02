"""
Utility to apply (weighted) convolutions.
It is mostly use to compute local averages in LCC/LGMM.
It is implemented in TorchScript for performance.
"""

import torch
from torch.nn import functional as F
from typing import Optional, List, Dict, Tuple
import math
Tensor = torch.Tensor


__all__ = ['local_mean', 'local_mean_norm', 'cache']



# ----------------------------------------------------------------------
#                           JIT UTILS
# ----------------------------------------------------------------------
USE_JIT = False
jit_script = torch.jit.script if USE_JIT else (lambda f: f)


@jit_script
def pad_list_int(x: List[int], length: int) -> List[int]:
    """Pad a List[int] using its last value until it has length `length`."""
    x = x + x[-1:] * max(0, length-len(x))
    x = x[:length]
    return x


@jit_script
def pad_list_float(x: List[float], length: int) -> List[float]:
    """Pad a List[float] using its last value until it has length `length`."""
    x = x + x[-1:] * max(0, length-len(x))
    x = x[:length]
    return x


@jit_script
def list_any(x: List[bool]) -> bool:
    for elem in x:
        if elem:
            return True
    return False


@jit_script
def list_all(x: List[bool]) -> bool:
    for elem in x:
        if not elem:
            return False
    return True


@jit_script
def prod(x: List[int]) -> int:
    if len(x) == 0:
        return 1
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 * x1
    return x0


# ----------------------------------------------------------------------
#                               CONV
# ----------------------------------------------------------------------
# We only use odd kernels, which simplifies some things when computing shapes


@jit_script
def conv_input_padding(kernel_size: List[int]):
    """Input padding -- mode 'same' """
    # assert k % 2 == 1
    pad: List[int] = [(k - 1)//2 for k in kernel_size]
    return pad


@jit_script
def convt_output_shape(input_shape: List[int],
                       kernel_size: List[int],
                       stride: List[int]) -> List[int]:
    """Output shape of transposed conv -- mode 'same'"""
    pad = conv_input_padding(kernel_size)
    oshape: List[int] = [(l - 1) * s - 2 * p + (k - 1) + 1 for k, s, l, p
                         in zip(kernel_size, stride, input_shape, pad)]
    return oshape


@jit_script
def convt_output_padding(output_shape: List[int],
                         input_shape: List[int],
                         kernel_size: List[int],
                         stride: List[int]) -> List[int]:
    """Input padding -- mode 'same' """
    output_shape0 = convt_output_shape(input_shape, kernel_size, stride)
    opad: List[int] = [l - l0 for l, l0 in zip(output_shape, output_shape0)]
    return opad


@jit_script
def conv(x: Tensor, kernel: Tensor, stride: List[int]) -> Tensor:
    """ND convolution (padding = 'same')
    x : (B, Ci, *inspatial) tensor
    kernel : (Ci, Co, *kernel_size) tensor
    stride : List{dim}[int]
    returns : (B, Co, *outspatial) tensor
    """
    dim = x.dim() - 2
    pad = conv_input_padding(kernel.shape[-dim:])
    out_channels = kernel.shape[-dim-1]
    inp_channels = kernel.shape[-dim-2]
    if inp_channels == out_channels == 1:
        groups = x.shape[-dim-1]
        kernel = kernel.expand(torch.Size([groups, 1]) + kernel.shape[2:])
    else:
        groups = 1
    if dim == 1:
        return F.conv1d(x, kernel, stride=stride, groups=groups, padding=pad)
    elif dim == 2:
        return F.conv2d(x, kernel, stride=stride, groups=groups, padding=pad)
    else:
        return F.conv3d(x, kernel, stride=stride, groups=groups, padding=pad)


@jit_script
def conv_transpose(x: Tensor, kernel: Tensor, stride: List[int],
                   oshape: List[int]) -> Tensor:
    """ND transposed convolution (padding = 'same')
    x : (B, Ci, *inspatial) tensor
    kernel : (Ci, Co, *kernel_size) tensor
    stride : List{dim}[int]
    oshape : List{dim}[int]
    returns : (B, Co, *outspatial) tensor
    """
    dim = x.dim() - 2
    ishape = x.shape[-dim:]
    kernel_size = kernel.shape[-dim:]
    out_channels = kernel.shape[-dim-1]
    inp_channels = kernel.shape[-dim-2]
    if inp_channels == out_channels == 1:
        groups = x.shape[-dim-1]
        kernel = kernel.expand(torch.Size([groups, 1]) + kernel.shape[2:])
    else:
        groups = 1
    ipad = conv_input_padding(kernel_size)
    opad = convt_output_padding(oshape, ishape, kernel_size, stride)
    if dim == 1:
        x = F.conv_transpose1d(x, kernel, stride=stride, output_padding=opad,
                               padding=ipad, groups=groups)
    elif dim == 2:
        x = F.conv_transpose2d(x, kernel, stride=stride, output_padding=opad,
                               padding=ipad, groups=groups)
    else:
        x = F.conv_transpose3d(x, kernel, stride=stride, output_padding=opad,
                               padding=ipad, groups=groups)
    return x


@jit_script
def do_conv(x: Tensor, kernel: List[Tensor], stride: List[int]) -> Tensor:
    """Apply a [separable] convolution
    x : (B, C, *inspatial) tensor
    kernel : List[(C, C, *kernel_size) tensor]
    stride : List{dim}[int]
    returns : (B, C, *outspatial) tensor
    """
    dim = x.dim() - 2
    if len(kernel) == 1:
        x = conv(x, kernel[0], stride)
    else:
        for d, (k, s) in enumerate(zip(kernel, stride)):
            stride1: List[int] = [1] * dim
            stride1[d] = s
            x = conv(x, k, stride1)
    return x


@jit_script
def do_convt(x: Tensor, kernel: List[Tensor], stride: List[int],
             oshape: List[int]) -> Tensor:
    """Apply a [separable] transposed convolution
    x : (B, C, *inspatial) tensor
    kernel : List[(C, C, *kernel_size) tensor]
    stride : List{dim}[int]
    oshape : List{dim}[int]
    returns : (B, C, *outspatial) tensor
    """
    dim = x.dim() - 2
    if len(kernel) == 1:
        x = conv_transpose(x, kernel[0], stride, oshape)
    else:
        for d, (k, s, z) in enumerate(zip(kernel, stride, oshape)):
            stride1: List[int] = [1] * dim
            stride1[d] = s
            shape1: List[int] = list(x.shape[-dim:])
            if s == 1 and z == shape1[d]:
                # Without stride, the transposed 'same' convolution is a
                # 'same' convolution with the flipped kernel, which is
                # faster than conv_transpose on most backends.
                x = conv(x, k.flip([d+2]), stride1)
                continue
            shape1[d] = z
            x = conv_transpose(x, k, stride1, shape1)
    return x


# ----------------------------------------------------------------------
#                               PATCH
# ----------------------------------------------------------------------
# We don't use padding with square patches

@jit_script
def patcht_output_shape(input_shape: List[int],
                        kernel_size: List[int],
                        stride: List[int]) -> List[int]:
    oshape: List[int] = [(l - 1) * s + (k - 1) + 1 for k, s, l
                         in zip(kernel_size, stride, input_shape)]
    return oshape


@jit_script
def do_patch(x: Tensor, kernel_size: List[int], stride: List[int]) -> Tensor:
    """Convolution by a constant square kernel of shape `kernel_size`

    The box kernel is separable: we extract (strided, possibly overlapping)
    windows along one dimension at a time and average them. This uses
    views and reductions only (no convolution), and costs sum(kernel_size)
    rather than prod(kernel_size) operations per voxel.

    x : (B, C, *inspatial) tensor
    kernel_size : List{dim}[int]
    stride : List{dim}[int]
    returns : (B, C, *outspatial) tensor
    """
    dim = x.dim() - 2
    for d in range(dim):
        if kernel_size[d] == 1 and stride[d] == 1:
            continue
        x = x.unfold(d+2, kernel_size[d], stride[d]).mean(-1)
    return x


@jit_script
def do_patcht(x: Tensor, kernel_size: List[int], stride: List[int],
              oshape: List[int]) -> Tensor:
    """Transposed convolution by a constant square kernel of shape `kernel_size`

    This is the exact adjoint of `do_patch`: overlapping patches (when
    `stride < kernel_size`) accumulate their contributions, and voxels
    that are not covered by any patch (trailing voxels, when
    `(oshape - kernel_size) % stride != 0`) are zero.

    Like `do_patch`, it is applied separably and uses views only
    (no convolution).

    x : (B, C, *inspatial) tensor
    kernel_size : List{dim}[int]
    stride : List{dim}[int]
    oshape : List{dim}[int]
    returns : (B, C, *oshape) tensor
    """
    dim = x.dim() - 2
    for d in range(dim):
        k = kernel_size[d]
        s = stride[d]
        n = x.shape[d+2]
        if k == 1 and s == 1 and n == oshape[d]:
            continue
        yshape: List[int] = list(x.shape)
        yshape[d+2] = oshape[d]
        y = torch.zeros(yshape, dtype=x.dtype, device=x.device)
        # (B, C, ..., n, ..., k) view of the output: window i covers
        # output voxels [i*s, i*s + k)
        z = y.narrow(d+2, 0, (n - 1) * s + k).unfold(d+2, k, s)
        x = x.div(k)
        if s >= k:
            # non-overlapping windows: broadcast each value to its window
            z.copy_(x.unsqueeze(-1))
        else:
            # overlapping windows: accumulate one window offset at a time.
            # Each `z[..., j]` is a strided view without self-overlap.
            for j in range(k):
                z.select(-1, j).add_(x)
        x = y
    return x

# ----------------------------------------------------------------------
#                              LOCAL MEAN
# ----------------------------------------------------------------------


@jit_script
def _local_mean_patch(
        x: Tensor,
        kernel_size: List[int],
        stride: List[int],
        backward: bool = False,
        shape: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
        norm: Optional[Tensor] = None) -> Tensor:
    """Compute a local average by extracting patches"""
    dim = x.dim() - 2
    nrm = norm
    if mask is not None and nrm is None:
        nrm = do_patch(mask, kernel_size, stride).clamp_min(1e-5)

    # --- backward pass ------------------------------------------------
    if backward:
        if shape is None:
            shape = patcht_output_shape(x.shape[-dim:], kernel_size, stride)
        if mask is not None:
            assert nrm is not None
            x = x.div(nrm)
        x = do_patcht(x, kernel_size, stride, shape)
        if mask is not None:
            x = x.mul(mask)
    # --- forward pass -------------------------------------------------
    else:
        if mask is not None:
            x = x.mul(mask)
        x = do_patch(x, kernel_size, stride)
        if mask is not None:
            assert nrm is not None
            x = x.div(nrm)

    return x


cache: Dict[str, Tensor] = {}


@jit_script
def _gaussian_kernel(fwhm: List[float], dim: int, dtype: torch.dtype,
                     device: torch.device) -> Tuple[List[Tensor], List[int]]:
    """Separable Gaussian kernel, normalised so that it sums to one.

    Returns
    -------
    kernel : List[(1, 1, *kernel_size) tensor]
        One tensor per smoothed axis (for `do_conv`/`do_convt`)
    kernel_size : List{dim}[int]
        Kernel length along each axis (0 = not smoothed)
    """
    kernel_size = [int(math.ceil(k*3)) for k in fwhm]
    kernel_size = [k + 1 - (k % 2) if k else 0 for k in kernel_size]  # ensure odd
    sigma2 = [(f/2.355)**2 for f in fwhm]
    kernel: List[Tensor] = []
    norm: Optional[Tensor] = None
    for d in range(dim):
        if not kernel_size[d]:
            continue
        k = torch.arange(kernel_size[d], dtype=dtype, device=device)
        k = k - kernel_size[d]//2
        k = k.mul(k).div(-2*sigma2[d]).exp()
        for dd in range(d):
            k = k.unsqueeze(0)
        for dd in range(dim-d-1):
            k = k.unsqueeze(-1)
        k = k[None, None]
        if norm is None:
            norm = k
        else:
            norm = norm * k
        kernel.append(k)
    if norm is not None:
        norm = norm.sum().pow(1/len(kernel))
        for k in kernel:
            k.div_(norm)
    return kernel, kernel_size


@jit_script
def _conv_norm(shape: List[int], kernel: List[Tensor], kernel_size: List[int],
               stride: List[int], dtype: torch.dtype, device: torch.device,
               mask: Optional[Tensor] = None,
               cache: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """Denominator of a Gaussian local mean: conv(mask), or conv(ones)
    (cached when `cache` is provided and there is no mask)."""
    if mask is not None:
        return do_conv(mask, kernel, stride).clamp_min(1e-5)
    key = str((list(shape), list(kernel_size), list(stride), dtype, device))
    if cache is not None:
        if key in cache:
            return cache[key]
    convmask = torch.ones(shape, dtype=dtype, device=device)[None, None]
    convmask = do_conv(convmask, kernel, stride).clamp_min(1e-5)
    if cache is not None:
        cache[key] = convmask
    return convmask


@jit_script
def _local_mean_conv(
        x: Tensor,
        fwhm: List[float],
        stride: List[int],
        backward: bool = False,
        shape: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
        cache: Optional[Dict[str, Tensor]] = None,
        norm: Optional[Tensor] = None) -> Tensor:
    """Compute a local average by convolving with a (normalized) Gaussian."""
    dim = x.dim() - 2
    kernel, kernel_size = _gaussian_kernel(fwhm, dim, x.dtype, x.device)

    # --- original spatial shape ---------------------------------------
    if shape is None:
        if backward:
            shape = convt_output_shape(x.shape[-dim:], kernel_size, stride)
        else:
            shape = x.shape[-dim:]

    # --- conv(mask) or cached conv(ones) ------------------------------
    if norm is None:
        nrm = _conv_norm(list(shape), kernel, kernel_size, stride,
                         x.dtype, x.device, mask, cache)
    else:
        nrm = norm

    # --- backward pass ------------------------------------------------
    if backward:
        x = x.div(nrm)
        x = do_convt(x, kernel, stride, shape)
        if mask is not None:
            x = x.mul(mask)
    # --- forward pass -------------------------------------------------
    else:
        if mask is not None:
            x = x.mul(mask)
        x = do_conv(x, kernel, stride)
        x = x.div(nrm)

    return x


@jit_script
def pre_reshape(x, dim: int):
    nb_batch = x.dim() - (dim + 2)
    batch: List[int] = []
    if nb_batch > 0:
        batch = x.shape[:-dim-1]
        x = x.reshape(torch.Size([-1]) + x.shape[-dim-1:])
    if nb_batch < 0:
        x = x[None]
    if nb_batch < -1:
        x = x[None]
    return x, nb_batch, batch


@jit_script
def post_reshape(x, nb_batch: int, batch: List[int]):
    if len(batch) > 0:
        x = x.reshape(batch + x.shape[1:])
    else:
        for _ in range(-nb_batch):
            x = x[0]
    return x


@jit_script
def local_mean(x: Tensor,
               kernel_fwhm: List[float],
               stride: Optional[List[int]] = None,
               mode: str = 'g',
               dim: Optional[int] = None,
               backward: bool = False,
               shape: Optional[List[int]] = None,
               mask: Optional[Tensor] = None,
               cache: Optional[Dict[str, Tensor]] = None,
               norm: Optional[Tensor] = None) -> Tensor:
    """Compute a local average by convolution

    Parameters
    ----------
    x : ([*batch, channels], *spatial) tensor
        Input tensor
    kernel_fwhm : List{1+}[float]
        Kernel size, will be padded to length `dim`
    stride : List{1+}[int], default=[1]
        Strides, will be padded to length `dim`
    mode : {'c', 'g'}, default='g'
        'constant' or 'gaussian'
    dim : int, default=`x.dim() - 1`
        Number of spatial dimensions.
    backward : bool, default=False
        Whether we are in a backward (transposed) pass
    shape : List[int], default=`spatial`
        Output shape (if backward).
    mask : ([*batch, channels], *spatial) tensor, optional
        A mask (if bool) or weight map (if float) used to weight the
        contribution of each voxel.
    cache : dict, optional
        Cache for the normaliser of unmasked local means.
    norm : tensor, optional
        Normaliser of the masked local mean, as returned by
        `local_mean_norm` with the same mask and options. It depends
        on the mask only, so calls (forward and backward) that share a
        mask can compute it once.

    Returns
    -------
    x : ([*batch, channels], *outspatial) tensor

    """
    mode = mode[0].lower()
    if dim is None:
        dim = x.dim() - 1
    x, nb_batch, extra_batch = pre_reshape(x, dim)
    if mask is not None:
        mask = mask.to(x.device, x.dtype)
        mask, *_ = pre_reshape(mask, dim)

    # Patches cannot be larger than the (forward) input image.
    # In the backward pass, the forward input shape is `shape`.
    fwd_shape: List[int] = list(x.shape[-dim:])
    if backward and shape is not None:
        fwd_shape = shape
    fwhm1, kernel_size, stride1 = _prepare_kernel(
        kernel_fwhm, stride, mode, dim, fwd_shape)

    if mode in ('c', 's'):  # const/square
        x = _local_mean_patch(x, kernel_size, stride1, backward, shape, mask,
                              norm)
    elif mode == 'g':        # gauss
        x = _local_mean_conv(x, fwhm1, stride1, backward, shape, mask,
                             cache, norm)
    else:
        raise ValueError(f'Unknown mode {mode}')

    x = post_reshape(x, nb_batch, extra_batch)
    return x


@jit_script
def _prepare_kernel(kernel_fwhm: List[float], stride: Optional[List[int]],
                    mode: str, dim: int, fwd_shape: List[int]
                    ) -> Tuple[List[float], List[int], List[int]]:
    """Pad kernel sizes and strides to `dim` (shared by `local_mean` and
    `local_mean_norm`)"""
    kernel_fwhm = pad_list_float(kernel_fwhm, dim)
    if mode == 'g':
        kernel_size = [int(math.ceil(k*3)) for k in kernel_fwhm]
    else:
        kernel_size = [min(int(math.ceil(k)), d)
                       for k, d in zip(kernel_fwhm, fwd_shape)]
    stride1: List[int] = [1]
    if stride is not None:
        stride1 = stride
    stride1 = pad_list_int(stride1, dim)
    stride1 = [s if s > 0 else k for s, k in zip(stride1, kernel_size)]
    return kernel_fwhm, kernel_size, stride1


def local_mean_norm(mask: Tensor,
                    kernel_fwhm: List[float],
                    stride: Optional[List[int]] = None,
                    mode: str = 'g',
                    dim: Optional[int] = None,
                    dtype: Optional[torch.dtype] = None) -> Tensor:
    """Normaliser of a masked local mean (the local sum of the mask).

    `local_mean(x, ..., mask=mask)` recomputes it at each call: compute
    it once with this function and pass it as `norm=` to all calls
    (forward and backward) that use the same mask and options.

    Parameters
    ----------
    mask : ([*batch, channels], *spatial) tensor
    kernel_fwhm, stride, mode, dim
        Same as in `local_mean` (forward-pass geometry).
    dtype : torch.dtype, optional
        Data type of the images the local means will be applied to
        (default: the mask's floating point type).

    Returns
    -------
    norm : tensor
        In the internal layout of `local_mean`: only meant to be passed
        to `local_mean(..., norm=norm)`.
    """
    mode = mode[0].lower()
    if dim is None:
        dim = mask.dim() - 1
    if dtype is None:
        dtype = mask.dtype if mask.dtype.is_floating_point else \
            torch.get_default_dtype()
    mask = mask.to(dtype)
    mask, *_ = pre_reshape(mask, dim)
    shape = list(mask.shape[-dim:])
    fwhm1, kernel_size, stride1 = _prepare_kernel(
        kernel_fwhm, stride, mode, dim, shape)
    if mode in ('c', 's'):
        return do_patch(mask, kernel_size, stride1).clamp_min(1e-5)
    elif mode == 'g':
        kernel, kernel_size = _gaussian_kernel(
            fwhm1, dim, mask.dtype, mask.device)
        return _conv_norm(shape, kernel, kernel_size, stride1,
                          mask.dtype, mask.device, mask)
    else:
        raise ValueError(f'Unknown mode {mode}')
