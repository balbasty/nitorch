import torch
from nitorch.core import py, utils, constants
from nitorch import spatial


def dist_map(shape, dtype=None, device=None):
    """Return the squared distance between all pairs in a FOV.

    Parameters
    ----------
    shape : sequence[int]
    dtype : optional
    device : optional

    Returns
    -------
    dist : (prod(shape), proD(shape) tensor
        Squared distance map

    """
    backend = dict(dtype=dtype, device=device)
    shape = py.make_tuple(shape)
    dim = len(shape)
    g = spatial.identity_grid(shape, **backend)
    g = g.reshape([-1, dim])
    g = (g[:, None, :] - g[None, :, :]).square_().sum(-1)
    return g


def build_se(sqdist, sigma, lam, **backend):
    """Build squared-exponential covariance matrix

    Parameters
    ----------
    sqdist : sequence[int] or (vox, vox) tensor
        If a tensor -> it is the pre-computed squared distance map
        If a tuple -> it is the shape and we build the distance map
    sigma : (*batch) tensor_like
        Amplitude
    lam : (*batch) tensor_like
        Length-scale

    Returns
    -------
    cov : (*batch, vox, vox) tensor
        Covariance matrix

    """
    lam, sigma = utils.to_max_backend(lam, sigma, **backend, force_float=True)
    backend = utils.backend(lam)

    # Build SE covariance matrix
    if not torch.is_tensor(sqdist):
        shape = sqdist
        e = dist_map(shape, **backend)
    else:
        e = sqdist.to(**backend)
    del sqdist
    lam = lam[..., None, None]
    sigma = sigma[..., None, None]
    e = e.mul(-0.5 / (lam ** 2)).exp_().mul_(sigma ** 2)
    return e


def build_se_thermal(sqdist, sigma, lam, dim=None, **backend):
    """Build squared-exponential covariance matrix

    Parameters
    ----------
    sqdist : sequence[int] or (vox, vox) tensor
        If a tensor -> it is the pre-computed squared distance map
        If a tuple -> it is the shape and we build the distance map
    sigma : (*batch) tensor_like
        Amplitude of the thermal noise
    lam : (*batch) tensor_like
        Length-scale of the recon
    dim : int, optional
        Number of spatial dimensions.
        Mandatory if a distance map is provided.

    Returns
    -------
    cov : (*batch, vox, vox) tensor
        Covariance matrix

    """
    dim = dim or len(sqdist)
    sigma = sigma * (4. * constants.pi * lam ** 2) ** (-dim / 4)
    lam = (2. ** 0.5) * lam
    return build_se(sqdist, sigma, lam)


def build_se_physio(sqdist, sigma, lam, lam_r=None, dim=None, **backend):
    """Build squared-exponential covariance matrix

    Parameters
    ----------
    sqdist : sequence[int] or (vox, vox) tensor
        If a tensor -> it is the pre-computed squared distance map
        If a tuple -> it is the shape and we build the distance map
    sigma : (*batch) tensor_like
        Amplitude of the physio noise
    lam : (*batch) tensor_like
        Length-scale of the physio noise
    lam_r : (*batch) tensor_like, optional
        Length-scale of the recon
    dim : int, optional
        Number of spatial dimensions.
        Mandatory if a distance map is provided.

    Returns
    -------
    cov : (*batch, vox, vox) tensor
        Covariance matrix

    """
    dim = dim or len(sqdist)
    if lam_r:
        sigma = sigma * (1 + 2 * (lam_r ** 2) / (lam ** 2)) ** (-dim / 4)
        lam = (lam ** 2 + 2. * lam_r ** 2) ** 0.5
    else:
        raise NotImplementedError
    return build_se(sqdist, sigma, lam)


def get_patches(volume, patch=3, stride=1):
    """Extract patches from an image/volume.

    Parameters
    ----------
    volume : (batch, *shape) tensor_like
    patch : int, default=3
    stride : int, default=1

    Returns
    -------
    patched_volume : (nb_patches, batch, *patch_shape)

    """
    dim = len(volume.shape) - 1
    patch = utils.make_list(patch, dim)
    patch = utils.make_list(patch, dim)

    volume = utils.unfold(volume, patch, stride, True)
    volume = volume.transpose(0, 1)
    return volume


def patch_and_cov(series, patch=3, return_mean=False):
    """Extract patches from a series and compute their empirical covariance.

    Parameters
    ----------
    series : (replicates, *input_shape) tensor_like
        Series.
    patch : int, default=3
        Patch size. Should be odd.

    Returns
    -------
    patch_cov : (*output_shape, N, N)
        Covariance for each patch. `N = prod(input_shape)`.
    mean : (*output_shape, N), if `return_mean`
        Voxel wise mean of each patch

    Notes
    -----
    The output maps only contain voxels for which full patches could be
    extracted in the input volume. Therefore, the output shape is
    `output_shape = input_shape - patch + 1`.

    """
    series = torch.as_tensor(series)

    # Compute output shape
    dim = len(series.shape) - 1
    shape = series.shape[1:]
    output_shape = [s - patch + 1 for s in shape]

    # Extract patches
    series = get_patches(series, patch)
    patch = [patch] * dim
    series = series.reshape([-1, series.shape[1], *patch])

    # Compute covariance
    return empirical_cov(series, dim=1, nb_dim=dim, flatten=True,
                         return_mean=return_mean)


def empirical_cov(series, nb_dim=1, dim=None, subtract_mean=True,
                  flatten=False, keepdim=False, return_mean=False):
    """Compute an empirical covariance

    Parameters
    ----------
    series : (..., *dims) tensor_like
        Sample series
    nb_dim : int, default=1
        Number of spatial dimensions.
    dim : [sequence of] int, default=None
        Dimensions that are reduced when computing the covariance.
        If None: all but the last `nb_dim`.
    subtract_mean : bool, default=True
        Subtract empirical mean before computing the covariance.
    flatten : bool, default=False
        If True, flatten the 'covariance' dimensions.
    keepdim : bool, default=False
        Keep reduced dimensions.

    Returns
    -------
    cov : (..., *dims, *dims) or (..., prod(dims), prod(dims)) tensor
        Covariance.
    mean : (..., *dims) or (..., prod(dims)) tensor, if `return_mean`
        Mean.

    """

    # Convert to tensor
    series = torch.as_tensor(series)
    prespatial = series.shape[:-nb_dim]
    spatial = series.shape[-nb_dim:]

    if dim is None:
        dim = range(series.dim() - nb_dim)
    dim = py.make_tuple(dim)
    dim = [series.dim() + d if d < 0 else d for d in dim]

    reduced = [prespatial[d] for d in dim]
    batch = [prespatial[d] for d in range(series.dim() - nb_dim)
             if d not in dim]

    # Subtract mean
    if subtract_mean:
        mean = series.mean(dim=dim, keepdim=True)
        series = series - mean

    # Compute empirical covariance.
    series = series.reshape([*series.shape[:-nb_dim], -1])
    series = utils.movedim(series, dim, -2)
    series = series.reshape([*batch, -1, series.shape[-1]])
    n_reduced = series.shape[-2]
    n_vox = series.shape[-1]
    # (*batch, reduced, spatial)

    # Torch's matmul just uses too much memory
    # We don't expect to have more than about 100 time frames,
    # so it is better to unroll the loop in python.
    # cov = torch.matmul(series.transpose(-1, -2), series)
    cov = None
    buf = series.new_empty([*batch, n_vox, n_vox])
    for i in range(n_reduced):
        buf = torch.mul(series.transpose(-1, -2)[..., :, i, None],
                        series[..., i, None, :], out=buf)
        if cov is None:
            cov = buf.clone()
        else:
            cov += buf
    cov /= py.prod(reduced)

    if keepdim:
        outshape = [1 if d in dim else s for d, s in enumerate(prespatial)]
    else:
        outshape = list(batch)
    if flatten:
        outshape_mean = outshape + [py.prod(spatial)]
        outshape += [py.prod(spatial)] * 2
    else:
        outshape_mean = outshape + list(spatial)
        outshape += list(spatial) * 2

    cov = cov.reshape(outshape)
    if return_mean:
        mean = mean.reshape(outshape_mean)
        return cov, mean
    return cov
