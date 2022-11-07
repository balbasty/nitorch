import torch
from nitorch.core import utils
from nitorch._C._ts.utils import movedim1, list_reverse_int
from typing import List
Tensor = torch.Tensor


@torch.jit.script
def _l1dt_1d_(f, dim: int = -1, w: float = 1.):
    """Algorithm 2 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if f.shape[dim] == 1:
        return f

    f = movedim1(f, dim, 0)

    for q in range(1, len(f)):
        f[q] = torch.min(f[q], f[q-1] + w)
    rng: List[int] = [e for e in range(len(f)-1)]
    for q in list_reverse_int(rng):
        f[q] = torch.min(f[q], f[q+1] + w)

    f = movedim1(f, 0, dim)
    return f


@torch.jit.script
def _l1dt_1d(f, dim: int = -1, w: float = 1.):
    """Algorithm 2 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    return _l1dt_1d_(f.clone(), dim, w)


if hasattr(torch, 'true_divide'):
    _true_div = torch.true_divide
else:
    _true_div = torch.div


@torch.jit.script
def _square(x):
    return x*x


@torch.jit.script
def _edt_1d_fillin(f, v, z, w2: float = 1.):
    # process along the first dimension
    #
    # f: input function
    # v: locations of parabolas in lower envelope
    # z: location of boundaries between parabolas

    k = f.new_zeros(f.shape[1:], dtype=torch.long)
    d = torch.empty_like(f)
    for q in range(len(f)):

        zk = z.gather(0, k[None] + 1)[0]
        mask = zk < q

        while mask.any():
            k = k.add_(mask)
            zk = z.gather(0, k[None] + 1)[0]
            mask = zk < q

        vk = v.gather(0, k[None])[0]
        fvk = f.gather(0, vk[None])[0]
        d[q] = w2 * _square(q - vk) + fvk

    return d


@torch.jit.script
def _edt_1d_intersection(f, v, z, k, q: int, w2: float = 1.):
    vk = v.gather(0, k[None])[0]
    fvk = f.gather(0, vk[None])[0]
    fq = f[q]
    a, b = w2 * (q - vk), q + vk
    s = _true_div((fq - fvk) + a * b, 2 * a)
    zk = z.gather(0, k[None])[0]
    mask = (k > 0) & (s <= zk)
    return s, mask


@torch.jit.script
def _edt_1d(f, dim: int = -1, w: float = 1.):
    """Algorithm 1 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """

    if f.shape[dim] == 1:
        return f

    w = w * w                                        # unit length (squared)
    f = movedim1(f, dim, 0)                          # input function
    k = f.new_zeros(f.shape[1:], dtype=torch.long)   # index of rightmost parabola in lower envelope
    v = f.new_zeros(f.shape, dtype=torch.long)       # locations of parabolas in lower envelope
    z = f.new_empty([len(f)+1] + list(f.shape[1:]))  # location of boundaries between parabolas

    # compute lower envelope
    z[0] = -float('inf')
    z[1] = float('inf')
    for q in range(1, len(f)):

        s, mask = _edt_1d_intersection(f, v, z, k, q, w)
        while mask.any():
            k.add_(mask, alpha=-1)
            s, mask = _edt_1d_intersection(f, v, z, k, q, w)

        s.masked_fill_(torch.isnan(s), -float('inf'))  # is this correct?

        k.add_(1)
        v.scatter_(0, k[None], q)
        z.scatter_(0, k[None], s[None])
        z.scatter_(0, k[None] + 1, float('inf'))

    # fill in values of distance transform
    d = _edt_1d_fillin(f, v, z, w)
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
    x.masked_fill_(x > 0, float('inf'))
    dim = dim or x.dim()
    vx = utils.make_vector(vx, dim, dtype=torch.float).tolist()
    x = _l1dt_1d_(x, -dim, vx[0]).square_()
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
    x.masked_fill_(x > 0, float('inf'))
    dim = dim or x.dim()
    vx = utils.make_vector(vx, dim, dtype=torch.float).tolist()
    for d, w in enumerate(vx):
        x = _l1dt_1d_(x, d-dim, w)
    return x
