import torch
from nitorch.core import py, utils, constants, linalg
from nitorch import spatial
from ._utils import dist_map
import math


def physio_sample(shape=None, sigma_p=0.008, lam_p=0.4, sigma_0=2.,
                  sigma_r=1., lam_r=0.2, signal=100.,
                  repeats=100, sampler='svd', **backend):
    """Sample from the fMRI physiological model.

    Parameters
    ----------
    shape : list[int], default=[32, 32]
        Shape of the field of view
    sigma_p : float, default=0.008
        Amplitude of the physiological noise.
    lam_p : float, default=0.4
        Length-scale of the physiological noise (i.e., smoothness).
    sigma_0 : float, default=2.0
        Amplitude of the thermal noise.
    sigma_r : float, default=1.
        Amplitude of the reconstruction filter.
    lam_r : float, default=0.4
        Length-scale of the reconstruction filter.
    signal : float,default=100.
        Mean signal.
    repeats : int, default=100
        Number of repeats in the time series

    Returns
    -------
    time_series : (repeats, *shape) tensor[dtype]
        fMRI time series. Forward model: ReconFilter(Signal * Physio + Thermal)
    replicate_series : (repeats, *shape) tensor[dtype]
        Replicate series. Forward model: ReconFilter(Signal + Thermal)

    """
    shape = [32, 32] if shape is None else shape
    dim = len(shape)
    sigma_p_recon = sigma_p * sigma_r * (1 + 2 * (lam_r ** 2) / (lam_p ** 2)) ** (-dim / 4)
    sigma_0_recon = sigma_0 * sigma_r * (4. * constants.pi * lam_r ** 2) ** (-dim / 4)
    lam_p_recon = (lam_p ** 2 + 2. * lam_r ** 2) ** 0.5
    lam_0_recon = (2. ** 0.5) * lam_r

    param = sigma_p_recon, sigma_0_recon, lam_p_recon, lam_0_recon
    param = utils.to_max_backend(*param, **backend)
    sigma_p_recon, sigma_0_recon, lam_p_recon, lam_0_recon = param
    backend = utils.backend(sigma_p_recon)

    # thermal noise (*) recon
    tr = lambda: se_sample(shape, sigma_0_recon, lam_0_recon,
                           repeats=repeats, sampler=sampler, **backend)
    # physio noise (*) recon
    pr = lambda: se_sample(shape, sigma_p_recon, lam_p_recon,
                           repeats=repeats, sampler=sampler, **backend)

    time_series = signal * (1. + pr()) + tr()
    replicate_series = signal + tr()

    return time_series, replicate_series


def se_sample(shape, sigma, lam, mu=None, repeats=1, sampler='svd',**backend):
    """Sample random fields with a squared exponential kernel.

    This function computes the square root of the covariance matrix by SVD.

    Parameters
    ----------
    shape : sequence[int]
        Shape of the image / volume.å
    sigma : float
        SE amplitude.
    lam : float
        SE length-scale.
    mu : () or (*shape) tensor_like
        SE mean
    repeats : int, default=1
        Number of sampled fields.
    sampler : {'svd', 'smooth'}, default='svd'
        Sample random field by SVD or by smoothing.

    Returns
    -------
    field : (repeats, *shape) tensor
        Sampled random fields.

    """
    if sampler == 'svd':
        sample_fn = se_sample_svd
    else:
        sample_fn = se_sample_smooth
    return sample_fn(shape, sigma, lam, mu, repeats, **backend)


def se_sample_svd(shape, sigma, lam, mu=None, repeats=1, **backend):
    """Sample random fields with a squared exponential kernel.

    This function computes the square root of the covariance matrix by SVD.

    Parameters
    ----------
    shape : sequence[int]
        Shape of the image / volume.å
    sigma : float
        SE amplitude.
    lam : float
        SE length-scale.
    mu : () or (*shape) tensor_like
        SE mean
    repeats : int, default=1
        Number of sampled fields.

    Returns
    -------
    field : (repeats, *shape) tensor
        Sampled random fields.

    """
    # Build SE covariance matrix
    e = dist_map(shape, **backend)
    backend = utils.backend(e)
    e.mul_(-0.5 / (lam ** 2)).exp_().mul_(sigma ** 2)

    # import matplotlib.pyplot as plt
    # plt.imshow(e)
    # plt.colorbar()
    # plt.title('true cov')
    # plt.show()

    # SVD of covariance
    u, s, _ = torch.svd(e)
    s = s.sqrt_()

    # Sample white noise and apply transform
    full_shape = (repeats, *shape)
    field = torch.randn(full_shape, **backend).reshape([repeats, -1])
    field = linalg.matvec(u, field.mul_(s))
    field = field.reshape(full_shape)

    # Add mean
    if mu is not None:
        mu = torch.as_tensor(mu, **backend)
        field += mu

    return field


def se_sample_smooth(shape, sigma, lam, mu=None, repeats=1, **backend):
    """Sample random fields with a squared exponential kernel.

    This function uses Gaussian smoothing of white noise.

    Parameters
    ----------
    shape : sequence[int]
        Shape of the image / volume.å
    sigma : float
        SE amplitude.
    lam : float
        SE length-scale.
    mu : () or (*shape) tensor_like
        SE mean
    repeats : int, default=1
        Number of sampled fields.

    Returns
    -------
    field : (repeats, *shape) tensor
        Sampled random fields.

    """
    # Convert SE parameters to Gaussian parameters
    dim = len(shape)
    sigma = sigma * ((2*constants.pi) ** (dim/4)) * (lam ** (dim/2))
    lam = lam / (2 ** 0.5)
    fwhm = lam * 2 * ((2 * math.log(2)) ** 0.5)

    # Sample white noise
    mul = 4
    pad = int(math.ceil(2 * fwhm))
    shape2 = [s*mul + 2*pad for s in shape]
    field = torch.randn([repeats, *shape2], **backend)

    # Gaussian smoothing
    field = spatial.smooth(field, fwhm=fwhm*mul, dim=dim, basis=0)
    sub = (slice(None),) + (slice(pad + mul//2, -pad, mul),) * dim
    field = field[sub]
    field *= sigma * (mul ** (dim/2))

    # Add mean
    if mu is not None:
        mu = torch.as_tensor(mu, **backend)
        field += mu

    return field


def cc_sample(shape, sigma, alpha, mu=None, repeats=1, **backend):
    """Sample random fields with a constant correlation.

    This function computes the square root of the covariance matrix by SVD.

    Parameters
    ----------
    shape : sequence[int]
        Shape of the image / volume.å
    sigma : float
        Variance.
    alpha : float
        Correlation.
    mu : () or (*shape) tensor_like
        SE mean
    repeats : int, default=1
        Number of sampled fields.

    Returns
    -------
    field : (repeats, *shape) tensor
        Sampled random fields.

    """
    # Build SE covariance matrix
    n = py.prod(shape)
    e = torch.full([n, n], alpha, **backend)
    e.diagonal(0, -1, -2).add_(1-alpha)
    backend = utils.backend(e)

    # SVD of covariance
    u, s, _ = torch.svd(e)
    s = s.sqrt_()

    # Sample white noise and apply transform
    full_shape = (repeats, *shape)
    field = torch.randn(full_shape, **backend).reshape([repeats, -1])
    field = linalg.matvec(u, field.mul_(s))
    field.mul_(sigma)
    field = field.reshape(full_shape)

    # Add mean
    if mu is not None:
        mu = torch.as_tensor(mu, **backend)
        field += mu

    return field
