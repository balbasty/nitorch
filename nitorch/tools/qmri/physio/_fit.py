import torch
from nitorch.core import py, utils, constants, linalg
from ._utils import dist_map, build_se_thermal, patch_and_cov, get_patches


def fit_se(cov, sqdist, mode='gn', **gnprm):
    """Fit the amplitude and length-scale of a squared-exponential kernel

    Parameters
    ----------
    cov : (*batch, vox, vox)
        Empirical covariance matrix
    sqdist : tuple[int] or (vox, vox) tensor
        If a tensor -> it is the pre-computed squared distance map
        If a tuple -> it is the shape and we build the distance map
    mode : {'gn', 'log'}, default='gn'
        'gn' : non-linear fit with Gauss-Newton optimisation
        'log' : log-linear fit

    Other Parameters
    ----------------
    max_iter : int, default=100
    tol : float, default=1e-5
    verbose : {0, 1, 2}, default=0

    Returns
    -------
    sig : (*batch,) tensor
        Amplitude of the kernel
    lam : (*batch,) tensor
        Length-scale of the kernel

    """
    if mode == 'log':
        cov = cov.log()
        fit_fn = fit_se_log
    else:
        fit_fn = lambda *a: fit_se_gn(*a, **gnprm)
    return fit_fn(cov, sqdist)


def fit_se_log(log_cov, sqdist):
    """Fit the amplitude and length-scale of a squared-exponential kernel

    Parameters
    ----------
    log_cov : (*batch, vox, vox)
        Log of the empirical covariance matrix
    sqdist : tuple[int] or (vox, vox) tensor
        If a tensor -> it is the pre-computed squared distance map
        If a tuple -> it is the shape and we build the distance map

    Returns
    -------
    sig : (*batch,) tensor
        Amplitude of the kernel
    lam : (*batch,) tensor
        Length-scale of the kernel

    """
    log_cov = torch.as_tensor(log_cov).clone()
    backend = utils.backend(log_cov)
    if not torch.is_tensor(sqdist):
        shape = sqdist
        sqdist = dist_map(shape, **backend)
    else:
        sqdist = sqdist.to(**backend).clone()

    # linear regression
    eps = constants.eps(log_cov.dtype)
    y = log_cov.reshape([-1, py.prod(sqdist.shape)])
    msk = torch.isfinite(y)
    y[~msk] = 0
    y0 = y.sum(-1, keepdim=True) / msk.sum(-1, keepdim=True)
    y -= y0
    x = sqdist.flatten() * msk
    x0 = x.sum(-1, keepdim=True) / msk.sum(-1, keepdim=True)
    x -= x0
    b = (x*y).sum(-1)/x.square().sum(-1).clamp_min_(eps)
    a = y0 - b * x0
    a = a[..., 0]

    lam = b.reciprocal_().mul_(-0.5).sqrt_()
    sig = a.div_(2).exp_()
    return sig, lam


def fit_se_gn(cov, sqdist, max_iter=10000, tol=1e-8, verbose=False):
    """Fit the amplitude and length-scale of a squared-exponential kernel

    This function minimises the Frobenius norm of the difference
    between the experimental and fitted covariance matrices
    (i.e., it is a least-squares between the elements of the matrices).

    It performs a non-linear least-squares fit that uses the robust
    Hessian from Balbastre et al. (2021).

    Parameters
    ----------
    cov : (*batch, vox, vox)
        Log of the empirical covariance matrix
    sqdist : tuple[int] or (vox, vox) tensor
        If a tensor -> it is the pre-computed squared distance map
        If a tuple -> it is the shape and we build the distance map
    max_iter : int, default=100
    tol : float, default=1e-5
    verbose : {0, 1, 2}, default=0

    Returns
    -------
    sig : (*batch,) tensor
        Amplitude of the kernel
    lam : (*batch,) tensor
        Length-scale of the kernel

    References
    ----------
    ..[1] "Model-based multi-parameter mapping"
          Yael Balbastre, Mikael Brudfors, Michaela Azzarito,
          Christian Lambert, Martina F. Callaghan and John Ashburner
          Preprint, 2021

    """
    cov = torch.as_tensor(cov).clone()
    backend = utils.backend(cov)
    if not torch.is_tensor(sqdist):
        shape = sqdist
        sqdist = dist_map(shape, **backend)
    else:
        sqdist = sqdist.to(**backend).clone()

    sqdist = sqdist.flatten()

    # exponential fit
    a = cov.diagonal(0, -1, -2).abs().mean(-1).log()
    cov = cov.reshape([-1, py.prod(sqdist.shape)])
    b = torch.ones_like(a).div_(-2)
    ll0 = None
    ll1 = None
    for it in range(max_iter):

        # compute objective
        e = sqdist.mul(b[:, None]).add_(a[:, None]).exp_()
        ll = (e-cov).square().sum() * 0.5
        if ll0 is None:
            ll0 = ll
            gain = constants.inf
        else:
            gain = (ll1 - ll)/ll0
        ll1 = ll
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{it+1:3d} | ll = {ll:12.6g} | gain = {gain:12.6g} '
                  f'| a = {a.mean():12.6g} | b = {b.mean():12.6g}', end=end)
            if it == 0:
                print('')
        if abs(gain) < tol:
            break

        # compute gradient
        r = (e - cov).abs_().mul_(sqdist + 1)
        ed = e * sqdist
        ha = (e.square() + e * r).sum(-1)
        hb = (ed.square() + ed * r).sum(-1)
        hab = (e * ed).sum(-1)
        h = torch.stack([ha, hb, hab], -1)
        del ha, hb, hab, ed, r
        ga = e * (e - cov)
        gb = (sqdist * ga).sum(-1)
        ga = ga.sum(-1)
        g = torch.stack([ga, gb], -1)
        del ga, gb

        # udpate
        h[..., :2] += 1e-3
        delta = linalg.sym_solve(h, g)
        del g, h
        a -= delta[..., 0]
        b -= delta[..., 1]
        del delta
    if verbose == 1:
        print('')

    lam = b.reciprocal_().mul_(-0.5).sqrt_()
    sig = a.div_(2).exp_()
    return sig, lam


def fit_thermal(cov, sqdist, dim=None, mode='gn', verbose=0):
    """Fit the thermal noise amplitude and recon smoothness

    Parameters
    ----------
    cov : (*batch, vox, vox)
        Empirical covariance matrix ("replicates" dataset)
    sqdist : tuple[int] or (vox, vox) tensor
        If a tensor -> it is the pre-computed squared distance map
        If a tuple -> it is the shape and we build the distance map
    dim : int, optional if shape provided
        Number of spatial dimensions.

    Returns
    -------
    sig : (*batch,) tensor
        Amplitude of the thermal noise
    lam : (*batch,) tensor
        Length-scale of the recon

    """
    dim = dim or len(sqdist)
    sig, lam = fit_se(cov, sqdist, mode=mode, verbose=verbose)
    lam /= (2. ** 0.5)
    sig *= lam.pow(dim/2).mul_((4*constants.pi) ** (dim/4))
    return sig, lam


def fit_physio(cov, lam, sqdist, dim=None, mode='gn', verbose=0):
    """Fit the physiological noise amplitude and smoothness

    Parameters
    ----------
    cov : (*batch, vox, vox) tensor
        Empirical covariance matrix ("time series" dataset)
        minus the fitted thermal covariance.
    lam : (*batch) tensor
        Length-sale of the recon
    sqdist : tuple[int] or (vox, vox) tensor
        If a tensor -> it is the pre-computed squared distance map
        If a tuple -> it is the shape and we build the distance map
    dim : int, optional if shape provided
        Number of spatial dimensions.

    Returns
    -------
    sig : (*batch,) tensor
        Amplitude of the thermal noise
    lam : (*batch,) tensor
        Length-scale of the recon

    """
    cov = torch.as_tensor(cov)
    backend = utils.backend(cov)
    lam = torch.as_tensor(lam, **backend)

    dim = dim or len(sqdist)
    sigp, lamp = fit_se(cov, sqdist, mode=mode, verbose=verbose)

    lam = lam.square().mul_(2)
    lamp = lamp.square_().sub_(lam)
    sigp *= lam.div_(lamp).add_(1).pow(dim/4)
    lamp = lamp.sqrt_()

    return sigp, lamp


def patch_and_fit_physio(time_series, replicates, patch=3, mask=None,
                         mode='gn', verbose=0):
    """Extract patches from an fMRI time + replicate series and fit parameters.

    Parameters
    ----------
    time_series : (replicates, *input_shape) tensor_like
        fMRI time series.
    replicates : (replicates, *input_shape) tensor_like
        fMRI replicate series.
    patch : int, default=3
        Patch size. Should be odd.

    Returns
    -------
    sig_p : (*output_shape) tensor
        Fitted physiological noise amplitude.
    lam_p : (*output_shape) tensor
        Fitted physiological noise length-scale.
    sig_0 : (*output_shape) tensor
        Fitted thermal noise amplitude.
    lam_0 : (*output_shape) tensor
        Fitted intrinsic smoothness length-scale.

    Notes
    -----
    The output maps only contain voxels for which full patches could be
    extracted in the input volume. Therefore, the output shape is
    `output_shape = input_shape - patch + 1`.

    """
    # Compute output shape
    time_series = torch.as_tensor(time_series)
    replicates = torch.as_tensor(replicates)
    backend = utils.backend(time_series)
    dim = time_series.dim() - 1
    shape = time_series.shape[1:]
    output_shape = [s - patch + 1 for s in shape]

    has_mask = mask is not None
    if has_mask:
        mask = get_patches(mask[None], patch)
        mask = mask.reshape(mask, [len(mask), -1])
        mask = mask.mean(dim=-1) > 0.75

    sqdist = dist_map([patch]*dim, **backend)

    # Estimate thermal/intrinsic from replicate series
    replicates = patch_and_cov(replicates, patch)
    if has_mask:
        replicates = replicates[mask, :, :]
    sig_0, lam_0 = fit_thermal(replicates, sqdist, dim=dim,
                               mode=mode, verbose=verbose)
    se_thermal = build_se_thermal(sqdist, sig_0, lam_0, dim=dim)
    del replicates

    # Estimate physio from time series
    time_series, mean = patch_and_cov(time_series, patch, return_mean=True)
    if has_mask:
        time_series = time_series[mask, :, :]
    time_series -= se_thermal
    del se_thermal
    mean = mean.mean(-1)[..., None, None].square_()
    time_series /= mean
    del mean
    sig_p, lam_p = fit_physio(time_series, lam_0, sqdist, dim=dim,
                              mode=mode, verbose=verbose)
    del time_series

    # Reshape as maps
    if has_mask:
        sig_p0, lam_p0, sig_00, lam_00 = sig_p, lam_p, sig_0, lam_0
        sig_p = sig_p0.new_zeros(len(mask))
        sig_p[mask] = sig_p0
        lam_p = lam_p0.new_zeros(len(mask))
        lam_p[mask] = lam_p0
        sig_0 = sig_00.new_zeros(len(mask))
        sig_0[mask] = sig_00
        lam_0 = lam_00.new_zeros(len(mask))
        lam_0[mask] = lam_00
        del sig_p0, lam_p0, sig_00, lam_00
    sig_p = sig_p.reshape(output_shape)
    lam_p = lam_p.reshape(output_shape)
    sig_0 = sig_0.reshape(output_shape)
    lam_0 = lam_0.reshape(output_shape)

    return sig_p, lam_p, sig_0, lam_0


def patch_and_fit_thermal(replicates, patch=3, mask=None, mode='gn', verbose=0):
    """Extract patches from an fMRI time + replicate series and fit parameters.

    Parameters
    ----------
    replicates : (replicates, *input_shape) tensor_like
        fMRI replicate series.
    patch : int, default=3
        Patch size. Should be odd.

    Returns
    -------
    sig : (*output_shape,) tensor
        Amplitude of the thermal noise
    lam : (*output_shape,) tensor
        Length-scale of the recon

    Notes
    -----
    The output maps only contain voxels for which full patches could be
    extracted in the input volume. Therefore, the output shape is
    `output_shape = input_shape - patch + 1`.

    """
    replicates = torch.as_tensor(replicates)

    # Compute output shape
    dim = replicates.dim() - 1
    shape = replicates.shape[1:]
    output_shape = [s - patch + 1 for s in shape]

    has_mask = mask is not None
    if has_mask:
        mask = get_patches(mask[None], patch)
        mask = mask.reshape(mask, [len(mask), -1])
        mask = mask.mean(dim=-1) > 0.75

    # Extract patches
    replicates = patch_and_cov(replicates, patch)
    replicates = replicates[mask]

    # Estimate ML parameters of the squared-exponential kernel
    sig, lam = fit_thermal(replicates, [patch]*dim, mode=mode, verbose=verbose)

    if has_mask:
        sig0, lam0 = sig, lam
        sig = sig0.new_zeros(len(mask))
        sig[mask] = sig0
        lam = lam0.new_zeros(len(mask))
        lam[mask] = lam0
    sig = sig.reshape(output_shape)
    lam = lam.reshape(output_shape)
    return sig, lam
