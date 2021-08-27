"""Utility to apply (weighted) convolutions.
It is mostly use to compute local averages in LCC/LGMM.
It is implemented in TorchScript for performance.
"""

import torch
from torch.nn import functional as F
from typing import Optional, List
Tensor = torch.Tensor


__all__ = ['local_mean']


@torch.jit.script
def _pad_list_int(x: List[int], length: int) -> List[int]:
    """Pad a List[int] using its last value until it has length `length`."""
    x = x + x[-1:] * max(0, length-len(x))
    x = x[:length]
    return x


@torch.jit.script
def _any(x: List[bool]) -> bool:
    for elem in x:
        if elem:
            return True
    return False


@torch.jit.script
def _all(x: List[bool]) -> bool:
    for elem in x:
        if not elem:
            return False
    return True


@torch.jit.script
def _guess_output_shape(inshape: List[int],
                        dim: int,
                        kernel_size: List[int],
                        stride: Optional[List[int]] = None,
                        transposed: bool = False) -> List[int]:
    """Guess the output shape of a convolution"""
    # assumes dilation=1, padding=0, output_padding=0
    kernel_size = _pad_list_int(kernel_size, dim)
    if stride is None:
        stride = [1]
    stride = _pad_list_int(stride, dim)

    N = inshape[0]
    C = inshape[1]
    shape = [N, C]
    for L, S, K in zip(inshape[2:], stride, kernel_size):
        Pi = 0  # padding
        Po = 0  # output padding
        D = 1   # dilation
        if transposed:
            shape.append((L - 1) * S - 2 * Pi + D * (K - 1) + Po + 1)
        else:
            shape.append(int(((L + 2 * Pi - D * (K - 1) - 1) / S + 1) // 1))
    return shape


@torch.jit.script
def conv(dim: int, x: Tensor, kernel: Tensor, stride: List[int]) -> Tensor:
    """ND convolution
    dim : {1, 2, 3}
    x : (B, Ci, *inspatial) tensor
    kernel : (Ci, Co, *kernel_size) tensor
    stride : List{dim}[int]
    returns : (B, Co, *outspatial) tensor
    """
    if kernel.shape[-dim-1] == kernel.shape[-dim-2] == 1:
        groups = x.shape[-dim-1]
        kernel = kernel.expand([x.shape[-dim-1], 1] + kernel.shape[2:])
    else:
        groups = 1
    if dim == 1:
        return F.conv1d(x, kernel, stride=stride, groups=groups)
    elif dim == 2:
        return F.conv2d(x, kernel, stride=stride, groups=groups)
    else:
        return F.conv3d(x, kernel, stride=stride, groups=groups)


@torch.jit.script
def conv_transpose(dim: int, x: Tensor, kernel: Tensor, stride: List[int],
                   opad: List[int]) -> Tensor:
    """ND transposed convolution
    dim : {1, 2, 3}
    x : (B, Ci, *inspatial) tensor
    kernel : (Ci, Co, *kernel_size) tensor
    stride : List{dim}[int]
    opad : List{dim}[int]
    returns : (B, Co, *outspatial) tensor
    """
    if kernel.shape[-dim-1] == kernel.shape[-dim-2] == 1:
        groups = x.shape[-dim-1]
        kernel = kernel.expand([x.shape[-dim-1], 1] + kernel.shape[2:])
    else:
        groups = 1
    tpad: Optional[List[int]] = None
    if _any([p > s for p, s in zip(opad, stride)]):
        tpad = opad
        opad = [0] * dim
    if dim == 1:
        x = F.conv_transpose1d(x, kernel, stride=stride, output_padding=opad,
                               groups=groups)
    elif dim == 2:
        x = F.conv_transpose2d(x, kernel, stride=stride, output_padding=opad,
                               groups=groups)
    else:
        x = F.conv_transpose3d(x, kernel, stride=stride, output_padding=opad,
                               groups=groups)
    if tpad is not None:
        y = x
        oshape = [s+p for s, p in zip(x.shape[-dim:], tpad)]
        oshape = x.shape[:-dim] + oshape
        x = torch.zeros(oshape, dtype=y.dtype, device=y.device)
        if dim == 1:
            x[:, :, :y.shape[2]] = y
        elif dim == 2:
            x[:, :, :y.shape[2], :y.shape[3]] = y
        else:
            x[:, :, :y.shape[2], :y.shape[3], :y.shape[4]] = y
    return x


@torch.jit.script
def do_conv(x: Tensor, kernel: List[Tensor], stride: List[int], dim: int) -> Tensor:
    """Apply a [separable] convolution
    x : (B, C, *inspatial) tensor
    kernel : List[(C, C, *kernel_size) tensor]
    stride : List{dim}[int]
    dim : {1, 2, 3}
    returns : (B, C, *outspatial) tensor
    """
    if len(kernel) == 1:
        x = conv(dim, x, kernel[0], stride)
    else:
        for d, (k, s) in enumerate(zip(kernel, stride)):
            ss: List[int] = [1] * d + [s] + [1] * (dim-d-1)
            x = conv(dim, x, k, ss)
    return x


@torch.jit.script
def do_convt(x: Tensor, kernel: List[Tensor], stride: List[int],
             opad: List[int], dim: int) -> Tensor:
    """Apply a [separable] transposed convolution
    x : (B, C, *inspatial) tensor
    kernel : List[(C, C, *kernel_size) tensor]
    stride : List{dim}[int]
    opad : List{dim}[int]
    dim : {1, 2, 3}
    returns : (B, C, *outspatial) tensor
    """
    if len(kernel) == 1:
        x = conv_transpose(dim, x, kernel[0], stride, opad)
    else:
        for d, (k, s, p) in enumerate(zip(kernel, stride, opad)):
            ss: List[int] = [1] * d + [s] + [1] * (dim - d - 1)
            pp: List[int] = [0] * d + [p] + [0] * (dim - d - 1)
            x = conv_transpose(dim, x, k, ss, pp)
    return x


@torch.jit.script
def do_patch(x: Tensor, kernel_size: List[int], stride: List[int]) -> Tensor:
    """Convolution by a constant square kernel of shape `kernel_size`"""
    dim = x.dim() - 2
    for d in range(dim):
        x = x.unfold(d+2, kernel_size[d], stride[d])
    dims = [d for d in range(-dim, 0)]
    x = x.mean(dim=dims)
    return x


@torch.jit.script
def prod(x: List[int]) -> int:
    if len(x) == 0:
        return 1
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 * x1
    return x0


@torch.jit.script
def do_patcht(x: Tensor, kernel_size: List[int], stride: List[int],
              opad: List[int]) -> Tensor:
    """Transposed convolution by a constant square kernel of shape `kernel_size`"""
    dim = x.dim() - 2
    # allocate output
    oshape = [s*k + p for s, p, k in zip(x.shape[-dim:], opad, kernel_size)]
    if all([p == 0 for p in opad]):
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


@torch.jit.script
def _local_mean_patch(
        x: Tensor,
        kernel_size: List[int],
        stride: List[int],
        backward: bool = False,
        shape: Optional[List[int]] = None,
        mask: Optional[Tensor] = None) -> Tensor:
    """Compute a local average by extracting patches"""
    dim = x.dim() - 2

    if shape is not None:  # estimate output padding
        ishape = [1, 1] + list(x.shape[-dim:])
        oshape = _guess_output_shape(ishape, dim, kernel_size, stride, transposed=True)
        oshape = oshape[2:]
        opad = [s - os for s, os in zip(shape, oshape)]
    else:
        opad = [0] * dim

    # conv
    if backward:
        if mask is not None:
            convmask = do_patch(mask, kernel_size, stride).clamp_min_(1e-5)
            x = x / convmask
        x = do_patcht(x, kernel_size, stride, opad)
        if mask is not None:
            x = x.mul_(mask)
    else:  # forward pass
        if mask is not None:
            x = x * mask
        x = do_patch(x, kernel_size, stride)
        if mask is not None:
            mask = do_patch(mask, kernel_size, stride).clamp_min_(1e-5)
            x = x.div_(mask)

    return x


@torch.jit.script
def _local_mean_conv(
        x: Tensor,
        kernel_size: List[int],
        stride: List[int],
        backward: bool = False,
        shape: Optional[List[int]] = None,
        mask: Optional[Tensor] = None) -> Tensor:
    """Compute a local average by convolving with a (normalized) Gaussian."""
    dim = x.dim() - 2

    # build kernel
    kernel: List[Tensor] = []
    # Gaussian kernel (weighted mean)
    fwhm = [float(k)/3. for k in kernel_size]
    sigma2 = [(f/2.355)**2 for f in fwhm]
    norm: Optional[Tensor] = None
    for d in range(dim):
        k = torch.arange(kernel_size[d], dtype=x.dtype, device=x.device)
        k -= kernel_size[d]//2
        k = k.square_().div_(-2*sigma2[d]).exp_()
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
        norm = norm.sum().pow(1/dim)
        for k in kernel:
            k.div_(norm)

    if shape is not None:  # estimate output padding
        ishape = [1, 1] + list(x.shape[-dim:])
        oshape = _guess_output_shape(ishape, dim, kernel_size, stride, transposed=True)
        oshape = oshape[2:]
        opad = [s - os for s, os in zip(shape, oshape)]
    else:
        opad = [0] * dim

    # conv
    if backward:
        if mask is not None:
            convmask = do_conv(mask, kernel, stride, dim).clamp_min_(1e-5)
            x = x / convmask
        x = do_convt(x, kernel, stride, opad, dim)
        if mask is not None:
            x = x.mul_(mask)
    else:  # forward pass
        if mask is not None:
            x = x * mask
        x = do_conv(x, kernel, stride, dim)
        if mask is not None:
            mask = do_conv(mask, kernel, stride, dim).clamp_min_(1e-5)
            x = x.div_(mask)

    return x


@torch.jit.script
def local_mean(x: Tensor,
               kernel_size: List[int],
               stride: Optional[List[int]] = None,
               mode: str = 'g',
               dim: Optional[int] = None,
               backward: bool = False,
               shape: Optional[List[int]] = None,
               mask: Optional[Tensor] = None) -> Tensor:
    """Compute a local average by convolution

    Parameters
    ----------
    x : ([[B], C], *spatial) tensor
        Input tensor
    kernel_size : List{1+}[int]
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
    mask : ([[B], C], *spatial) tensor, optional
        A mask (if bool) or weight map (if float) used to weight the
        contribution of each voxel.

    Returns
    -------
    x : ([[B], C], *outspatial) tensor

    """
    if mask is not None:
        mask = mask.to(x.device, x.dtype)
    if dim is None:
        dim = x.dim() - 1
    extra_batch = x.dim() > dim + 2
    batch: List[int] = []
    if extra_batch:
        batch = x.shape[:-dim-1]
        x = x.reshape([-1] + x.shape[-dim-1:])
    virtual_channel = x.dim() <= dim
    virtual_batch = x.dim() <= dim + 1
    if virtual_channel:
        x = x[None]
    if virtual_batch:
        x = x[None]
    if mask is not None:
        for d in range(max(x.dim() - mask.dim(), 0)):
            mask = mask[None]

    kernel_size = kernel_size + kernel_size[-1:] * max(0, dim - len(kernel_size))
    if stride is None:
        stride = [1]
    stride = stride + stride[-1:] * max(0, dim - len(stride))
    stride = [s if s > 0 else k for s, k in zip(stride, kernel_size)]

    if mode[0].lower() in ('c', 's'):  # const/square
        x = _local_mean_patch(x, kernel_size, stride, backward, shape, mask)
    elif mode[0].lower() == 'g':  # gauss
        x = _local_mean_conv(x, kernel_size, stride, backward, shape, mask)
    else:
        raise ValueError(f'Unknown mode {mode}')

    if virtual_batch:
        x = x[0]
    if virtual_channel:
        x = x[0]
    if extra_batch:
        x = x.reshape(batch + x.shape[1:])
    return x
