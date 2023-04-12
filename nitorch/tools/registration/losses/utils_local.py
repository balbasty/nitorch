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


__all__ = ['local_mean', 'cache']


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
def patcht_output_padding(output_shape: List[int],
                          input_shape: List[int],
                          kernel_size: List[int],
                          stride: List[int]) -> List[int]:
    output_shape_nopad = patcht_output_shape(input_shape, kernel_size, stride)
    opad: List[int] = [l - l0 for l, l0
                       in zip(output_shape, output_shape_nopad)]
    return opad


@jit_script
def do_patch(x: Tensor, kernel_size: List[int], stride: List[int]) -> Tensor:
    """Convolution by a constant square kernel of shape `kernel_size`"""
    dim = x.dim() - 2
    for d in range(dim):
        x = x.unfold(d+2, kernel_size[d], stride[d])
    dims = [d for d in range(-dim, 0)]
    x = x.mean(dim=dims)
    return x


@jit_script
def do_patcht(x: Tensor, kernel_size: List[int], stride: List[int],
              oshape: List[int]) -> Tensor:
    """Transposed convolution by a constant square kernel of shape `kernel_size`"""
    dim = x.dim() - 2
    # allocate output
    ishape = x.shape[-dim:]
    opad = patcht_output_padding(oshape, ishape, kernel_size, stride)
    if list_all([p == 0 for p in opad]):
        y = torch.empty(oshape, dtype=x.dtype, device=x.device)
    else:
        y = torch.zeros(oshape, dtype=x.dtype, device=x.device)
    # unfold output
    z = y
    for d in range(dim):
        z = z.unfold(d+2, kernel_size[d], stride[d])
        x = x.unsqueeze(-1)
    # copy values
    z = z.copy_(x)
    z = z.div_(prod(kernel_size))
    return y

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
        mask: Optional[Tensor] = None) -> Tensor:
    """Compute a local average by extracting patches"""
    dim = x.dim() - 2

    # --- backward pass ------------------------------------------------
    if backward:
        if shape is None:
            shape = patcht_output_shape(x.shape[-dim:], kernel_size, stride)
        if mask is not None:
            convmask = do_patch(mask, kernel_size, stride).clamp_min(1e-5)
            x = x.div(convmask)
        x = do_patcht(x, kernel_size, stride, shape)
        if mask is not None:
            x = x.mul(mask)
    # --- forward pass -------------------------------------------------
    else:
        if mask is not None:
            x = x.mul(mask)
        x = do_patch(x, kernel_size, stride)
        if mask is not None:
            mask = do_patch(mask, kernel_size, stride).clamp_min(1e-5)
            x = x.div(mask)

    return x


cache: Dict[str, Tensor] = {}


@jit_script
def _local_mean_conv(
        x: Tensor,
        fwhm: List[float],
        stride: List[int],
        backward: bool = False,
        shape: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
        cache: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """Compute a local average by convolving with a (normalized) Gaussian."""
    dim = x.dim() - 2

    # build kernel
    kernel: List[Tensor] = []
    # Gaussian kernel (weighted mean)
    kernel_size = [int(math.ceil(k*3)) for k in fwhm]
    kernel_size = [k + 1 - (k % 2) if k else 0 for k in kernel_size]  # ensure odd
    sigma2 = [(f/2.355)**2 for f in fwhm]
    norm: Optional[Tensor] = None
    for d in range(dim):
        if not kernel_size[d]:
            continue
        k = torch.arange(kernel_size[d], dtype=x.dtype, device=x.device)
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

    # --- original spatial shape ---------------------------------------
    if shape is None:
        if backward:
            shape = convt_output_shape(x.shape[-dim:], kernel_size, stride)
        else:
            shape = x.shape[-dim:]

    # --- cached conv(ones) --------------------------------------------
    if mask is None:
        key = str((list(shape), list(kernel_size), list(stride)))
        if cache is not None:
            if key not in cache:
                convmask = torch.ones(shape, dtype=x.dtype, device=x.device)
                convmask = convmask[None, None]
                cache[key] = do_conv(convmask, kernel, stride).clamp_min(1e-5)
            convmask = cache[key]
        else:
            convmask = torch.ones(shape, dtype=x.dtype, device=x.device)
            convmask = convmask[None, None]
            convmask = do_conv(convmask, kernel, stride).clamp_min(1e-5)
    else:
        convmask = do_conv(mask, kernel, stride).clamp_min(1e-5)

    # --- backward pass ------------------------------------------------
    if backward:
        x = x.div(convmask)
        x = do_convt(x, kernel, stride, shape)
        if mask is not None:
            x = x.mul(mask)
    # --- forward pass -------------------------------------------------
    else:
        if mask is not None:
            x = x.mul(mask)
        x = do_conv(x, kernel, stride)
        x = x.div(convmask)

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
               cache: Optional[Dict[str, Tensor]] = None) -> Tensor:
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

    kernel_fwhm = pad_list_float(kernel_fwhm, dim)
    if mode == 'g':
        kernel_size = [int(math.ceil(k*3)) for k in kernel_fwhm]
    else:
        kernel_size = [min(int(math.ceil(k)), d)
                       for k, d in zip(kernel_fwhm, x.shape[-dim:])]
    if stride is None:
        stride = [1]
    stride = pad_list_int(stride, dim)
    stride = [s if s > 0 else k for s, k in zip(stride, kernel_size)]

    if mode in ('c', 's'):  # const/square
        x = _local_mean_patch(x, kernel_size, stride, backward, shape, mask)
    elif mode == 'g':        # gauss
        x = _local_mean_conv(x, kernel_fwhm, stride, backward, shape, mask, cache)
    else:
        raise ValueError(f'Unknown mode {mode}')

    x = post_reshape(x, nb_batch, extra_batch)
    return x
