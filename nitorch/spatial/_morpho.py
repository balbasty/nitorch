import torch
from nitorch.core import utils
from ._conv import conv
import itertools


def connectivity_kernel(dim, conn=1, **backend):
    """Build a connectivity kernel

    Parameters
    ----------
    dim : int
        Number of spatial dimensions
    conn : int, default=1
        Order of the connectivity
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    kernel : (*spatial) tensor

    """
    kernel = torch.zeros((3,)*dim, **backend)
    for coord in itertools.product([0, 1], repeat=dim):
        if sum(coord) > conn:
            continue
        for sgn in itertools.product([-1, 1], repeat=dim):
            coord1 = [1 + c*s for c, s in zip(coord, sgn)]
            kernel[tuple(coord1)] = 1
    return kernel


def _morpho(mode, x, conn, nb_iter, dim):
    """Common worker for binary operations

    Notes
    -----
    .. Adapted from Neurite-Sandbox (author: B Fischl)

    Parameters
    ----------
    mode : {'dilate', 'erode', 'dist'}
    x : (..., *spatial) tensor
    conn : tensor or int
    nb_iter : int
    dim : int

    Returns
    -------
    y : (..., *spatial) tensor

    """
    in_dtype = x.dtype
    if in_dtype is not torch.bool:
        x = x > 0
    x = x.to(torch.uint8)
    backend = utils.backend(x)

    dim = dim or x.dim()
    if isinstance(conn, int):
        conn = connectivity_kernel(dim, conn, **backend)
    else:
        conn = conn.to(**backend)

    ix = dist = None
    if mode == 'dist':
        dist = torch.full_like(x, nb_iter+1, dtype=torch.int)
        dist.masked_fill_(x > 0, -(nb_iter+1))
        ix = 1 - x
    if mode == 'erode':
        ix = 1 - x
        x = None

    @torch.jit.script
    def xor(x, y):
        return (x + y) == 1

    for n_iter in range(1, nb_iter+1):
        if dist is not None:
            ox, oix = x, ix
        if x is not None:
            x = conv(dim, x, conn, padding='same').clamp_max_(1)
        if ix is not None:
            ix = conv(dim, ix, conn, padding='same').clamp_max_(1)
        if dist is not None:
            dist.masked_fill_(xor(x, ox), n_iter)
            dist.masked_fill_(xor(ix, oix), -n_iter)

    if mode == 'dilate':
        return x.to(in_dtype)
    if mode == 'erode':
        return ix.neg_().add_(1).to(in_dtype)
    return dist


def _soft_morpho(mode, x, conn, nb_iter, dim):
    """Common worker for soft operations

    Parameters
    ----------
    mode : {'dilate', 'erode'}
    x : (..., *spatial) tensor
    conn : tensor or int
    nb_iter : int
    dim : int

    Returns
    -------
    y : (..., *spatial) tensor

    """
    backend = utils.backend(x)

    dim = dim or x.dim()
    if isinstance(conn, int):
        conn = connectivity_kernel(dim, conn, **backend)
    else:
        conn = conn.to(**backend)

    x = x.clone()
    if mode == 'dilate':
        x.neg_().add_(1)
    x = x.clamp_(0.001, 0.999).log_()

    for n_iter in range(1, nb_iter+1):
        x = conv(dim, x, conn, padding='same')

    x = x.exp_()
    if mode == 'dilate':
        x.neg_().add_(1)
    return x


def erode(x, conn=1, nb_iter=1, dim=None, soft=False):
    """Binary erosion

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor (will be binarized, if not `soft`)
    conn : int or tensor, default=1
        If a tensor, the connectivity kernel to use.
        If an int, the connectivity order
            1 : 4-connectivity (2D) //  6-connectivity (3D)
            2 : 8-connectivity (2D) // 18-connectivity (3D)
            3 :                     // 26-connectivity (3D)
    nb_iter : int, default=1
        Number of iterations
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    soft : bool, default=False
        Assume input are probabilities and use a soft operator.

    Returns
    -------
    y : (..., *spatial) tensor
        Eroded tensor

    """
    fn = _soft_morpho if soft else _morpho
    return fn('erode', x, conn, nb_iter, dim)


def dilate(x, conn=1, nb_iter=1, dim=None, soft=False):
    """Binary dilation

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor (will be binarized if not `soft`)
    conn : int or tensor, default=1
        If a tensor, the connectivity kernel to use.
        If an int, the connectivity order
            1 : 4-connectivity (2D) //  6-connectivity (3D)
            2 : 8-connectivity (2D) // 18-connectivity (3D)
            3 :                     // 26-connectivity (3D)
    nb_iter : int, default=1
        Number of iterations
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    soft : bool, default=False
        Assume input are probabilities and use a soft operator.

    Returns
    -------
    y : (..., *spatial) tensor
        Dilated tensor

    """
    fn = _soft_morpho if soft else _morpho
    return fn('dilate', x, conn, nb_iter, dim)


def bounded_distance(x, conn=1, nb_iter=1, dim=None):
    """Bounded signed (city block) distance to a binary object.

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor (will be binarized)
    conn : int or tensor, default=1
        If a tensor, the connectivity kernel to use.
        If an int, the connectivity order
            1 : 4-connectivity (2D) //  6-connectivity (3D)
            2 : 8-connectivity (2D) // 18-connectivity (3D)
            3 :                     // 26-connectivity (3D)
    nb_iter : int, default=1
        Number of iterations. All voxels farther from the object than
        `nb_iter` will be given the distance `nb_iter + 1`
    dim : int, default=`x.dim()`
        Number of spatial dimensions

    Returns
    -------
    y : (..., *spatial) tensor
        Dilated tensor

    """
    return _morpho('dist', x, conn, nb_iter, dim)


def dilate_likely_voxels(labels, intensity, label=None, nb_iter=1,
                         dist_ratio=1, half_patch=3, conn=1, dim=None):
    """Dilate labels into voxels with a similar intensity.

    Notes
    -----
    .. Voxels get switched if their intensity is closer to the foreground
       intensity than to the background intensity, in terms of Gaussian
       distance (abs(intensity - class_mean)/class_std) computed in a local
       patch.
    .. Adapted from neurite-sandbox (author: B Fischl)

    Parameters
    ----------
    labels : (..., *spatial) tensor
        Tensor of labels
    intensity : (..., *spatial) tensor
        Tensor of intensities
    label : int, optional
        Label to dilate. Default: binarize input labels.
    nb_iter : int, default=1
        Number of iterations
    dist_ratio : float, default=1
        Value that decides how much closer from the foreground intensity
        than the background intensity a voxel must be to be flipped.
        Smaller == easier to switch.
    half_patch : int, default=3
        Half-size of the window used to compute intensity statistics.
    conn : int, default=1
        Connectivity order
    dim : int, default=`labels.dim()`
        Number of spatial dimensions

    Returns
    -------
    labels : (..., *spatial) tensor
        Dilated labels

    """
    in_dtype = labels.dtype
    foreground = (labels > 0) if label is None else (labels == label)
    dim = dim or labels.dim()

    patch = [2*half_patch + 1] * dim
    unfold = lambda x: utils.unfold(x, patch, stride=1)
    intensity = unfold(intensity)

    def mean_var(intensity, fg):
        """Compute mean and variance"""
        sum_fg = fg.sum(list(range(-dim, 0)))
        mean_fg = (intensity * fg).sum(list(range(-dim, 0)))
        var_fg = (intensity.square() * fg).sum(list(range(-dim, 0)))
        mean_fg /= sum_fg
        var_fg /= sum_fg
        var_fg -= mean_fg.square()
        return mean_fg, var_fg

    if isinstance(conn, int):
        conn = connectivity_kernel(dim, conn, device=foreground.device,
                                   dtype=torch.uint8)

    for n_iter in range(nb_iter):
        dilated = dilate(foreground, conn=conn, dim=dim)
        dilated = dilated.bitwise_xor_(foreground)

        # Extract patches0
        center = (Ellipsis, *([half_patch]*dim))
        win_dilated = unfold(dilated)
        msk_dilated = win_dilated[center]
        win_dilated = win_dilated[msk_dilated, ...]
        win_intensity = intensity[msk_dilated, ...]
        win_fg = unfold(foreground)[msk_dilated, ...]
        win_bg = ~(win_fg | win_dilated)

        # compute statistics
        mean_fg, var_fg = mean_var(win_intensity, win_fg)
        mean_bg, var_bg = mean_var(win_intensity, win_bg)

        # compute criterion
        crit = dist_ratio * mean_fg < mean_bg
        win_intensity = win_intensity[center]
        mean_fg.sub_(win_intensity).abs_().div_(var_fg.sqrt_())
        mean_bg.sub_(win_intensity).abs_().div_(var_bg.sqrt_())

        # set value
        win_fg[center].masked_fill_(crit, 1)
        unfold(foreground)[msk_dilated, ...] = win_fg

    if label is None:
        labels = foreground.to(in_dtype)
    else:
        labels = labels.clone()
        labels[foreground] = label
    return labels


def geodesic_dist(x, w, conn=1, nb_iter=1, dim=None):
    """Geodesic distance to a label

    Parameters
    ----------
    x : (..., *spatial) tensor
    w : (..., *spatial) tensor
    conn : int
    nb_iter : int
    dim : int

    Returns
    -------
    y : (..., *spatial) tensor

    """
    in_dtype = x.dtype
    if in_dtype is not torch.bool:
        x = x > 0
    x = x.to(torch.uint8)
    dim = dim or x.dim()

    d = torch.full(x.shape, float('inf'), **utils.backend(w))
    d[x > 0] = 0
    crop = (Ellipsis,  *([slice(1, -1)]*dim))
    dcrop = utils.unfold(d, [3]*dim, stride=1)
    w = utils.unfold(w, [3]*dim, stride=1)
    for n_iter in range(1, nb_iter+1):
        w0 = w[(Ellipsis, *([1]*dim))]
        for coord in itertools.product([0, 1], repeat=dim):
            if sum(coord) == 0 or sum(coord) > conn:
                continue
            mini_dist = sum(c*c for c in coord) ** 0.5
            coords = set()
            for sgn in itertools.product([-1, 1], repeat=dim):
                coord1 = [1 + c*s for c, s in zip(coord, sgn)]
                if tuple(coord1) in coords:
                    continue
                coords.add(tuple(coord1))
                coord1 = (Ellipsis, *coord1)
                new_dist = (w[coord1] - w0).abs() * (dcrop[coord1] + mini_dist)
                new_dist.masked_fill_(torch.isfinite(new_dist).bitwise_not_(), float('inf'))
                msk = new_dist < d[crop]
                d[crop][msk] = new_dist[msk]
                print(d[crop].isfinite().sum())

    msk = torch.isfinite(d).bitwise_not_()
    d[msk] = d[~msk].max()
    return d
