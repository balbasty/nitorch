import torch
from nitorch.core import utils
from nitorch._C._ts.utils import movedim1, list_reverse_int
from typing import List


@torch.jit.script
def _l1dt_1d(f, dim: int = -1, w: float = 1.):
    """Algorithm 2 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if f.shape[dim] == 1:
        return f

    f = movedim1(f, dim, 0)
    d = f.clone()

    for q in range(1, len(f)):
        d[q] = torch.min(d[q], d[q-1] + w)
    rng: List[int] = [e for e in range(len(f)-1)]
    for q in list_reverse_int(rng):
        d[q] = torch.min(d[q], d[q+1] + w)

    d = movedim1(d, 0, dim)
    return d


@torch.jit.script
def _edt_1d(f, dim: int = -1, w: float = 1.):
    """Algorithm 1 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """

    if f.shape[dim] == 1:
        return f

    w = w*w

    dtype: torch.dtype = f.dtype
    if dtype not in (torch.half, torch.float, torch.double):
        dtype = torch.float

    f = movedim1(f, dim, 0)                                              # input function
    k = f.new_zeros(f.shape[1:], dtype=torch.long)                       # index of rightmost parabola in lower envelope
    v = f.new_zeros(f.shape, dtype=torch.long)                           # locations of parabolas in lower envelope
    z = f.new_empty([len(f)+1] + f.shape[1:], dtype=dtype)               # location of boundaries between parabolas
    s = f.new_zeros(f.shape[1:], dtype=dtype)                            # intersection between two parabolas

    # compute lower envelope
    z[0] = -float('inf')
    for q in range(1, len(f)):
        mask = f.new_full(f.shape[1:], 1, dtype=torch.bool)
        while mask.any():
            kmask = k[mask]
            vk = v[:, mask].gather(0, kmask[None])[0]
            fvk = f[:, mask].gather(0, vk[None])[0]
            fq = f[q, mask]
            s[mask] = (fq - fvk + w * (q*q - vk*vk)).true_divide(2*w*(q - vk))
            zk = z.gather(0, k[None])[0]
            mask = (k > 0) & (s <= zk) # & (zk > -float('inf'))
            k[mask] -= 1
        k += 1
        v.scatter_(0, k[None], q)
        z.scatter_(0, k[None], s[None])
        z.scatter_(0, k[None] + 1, float('inf'))
    z[torch.isnan(z)] = -float('inf')

    # fill in values of distance transform
    k.zero_()
    d = torch.empty_like(f)
    for q in range(len(f)):
        mask = f.new_full(f.shape[1:], 1, dtype=torch.bool)
        while mask.any():
            zk = z.gather(0, k[None] + 1)[0]
            mask = zk < q
            k[mask] += 1
        vk = v.gather(0, k[None])[0]
        fvk = f.gather(0, vk[None])[0]
        d[q] = w * (q - vk).square() + fvk

    d = movedim1(d, 0, dim)
    return d


def euclidean_distance_transform(x, dim=None, vx=1):
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x[x > 0] = float('inf')
    dim = dim or x.dim()
    vx = utils.make_vector(vx, dim, dtype=torch.float).tolist()
    x = _l1dt_1d(x, -dim, vx[0]).square_()
    for d, w in zip(range(1, dim), vx[1:]):
        x = _edt_1d(x, d-dim, w)
    x.sqrt_()
    return x


def l1_distance_transform(x, dim=None, vx=1):
    """Compute the L1 distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x[x > 0] = float('inf')
    dim = dim or x.dim()
    vx = utils.make_vector(vx, dim, dtype=torch.float).tolist()
    for d, w in enumerate(vx):
        x = _l1dt_1d(x, d-dim, w)
    return x
