import torch
from nitorch.core import py, utils, fft
from nitorch import spatial
from nitorch.tools.img_statistics import estimate_noise
from typing import Optional
from torch import Tensor
import math as pymath


def _laplacian_freq(shape, **backend):
    """
    Compute Fourier squared frequency on the lattice and its inverse.
    """
    dim = len(shape)
    shape = torch.as_tensor(shape, **backend)
    g = spatial.identity_grid(shape, **backend)
    g -= shape // 2
    g /= shape
    g = g.square_().sum(-1)
    if fft._torch_has_old_fft:
        g = g.unsqueeze(-1)
    g = fft.ifftshift(g, dim=list(range(dim)))
    ig = g.reciprocal()
    ig[(0,) * dim] = 0
    return g, ig


def _laplacian_filter(phase, freq, dims):
    """
    Assumes ifftshift has been applied to phase and freq.
    Eq (5) from Schofield and Zhu.
    """
    g, ig = freq

    s = phase.sin()
    c = phase.cos()

    fft_ = lambda x: fft.fftn(x, dim=dims)
    ifft_ = lambda x: fft.ifftn(x, dim=dims)

    phase = ifft_(fft_(s).mul_(g)).mul_(c)
    phase -= ifft_(fft_(c).mul_(g)).mul_(s)
    phase = ifft_(fft_(phase).mul_(ig))
    phase = fft.real(phase)

    return phase


def unwrap(phase, dim=None, bound='dct2', max_iter=0, tol=1e-5):
    """Laplacian unwrapping of the phase

    Parameters
    ----------
    phase : tensor
        Wrapped phase, in radian
    dim : int, default=phase.dim()
        Number of spatial dimensions
    max_iter : int, default=0
        Maximum number of unwrapping iterations.
        If 0, return the Laplacian filtered phase, which is not exactly
        equal to the input phase modulo 2 pi.
    tol : float, default=1e-5
        Tolerance for early stopping


    Returns
    -------
    unwrapped : tensor

    References
    ----------
    .. "Fast phase unwrapping algorithm for interferometric applications"
       Marvin A. Schofield and Yimei Zhu
       Optics Letters (2003)

    """
    # TODO: would be nice to use DCT/DST rather than padding once they
    #       are available in PyTorch.

    dim = dim or phase.dim()
    dims = list(range(-dim, 0))
    shape = bigshape = phase.shape[-dim:]

    if bound not in ('dct', 'circulant'):
        phase = utils.pad(phase, [d//2 for d in shape], side='both', mode=bound)
        bigshape = phase.shape[-dim:]

    freq = _laplacian_freq(bigshape, **utils.backend(phase))
    phase = fft.ifftshift(phase, dim=dims)
    twopi = 2 * pymath.pi

    if max_iter == 0:
        phase = _laplacian_filter(phase, freq, dims)
    else:
        for n_iter in range(1, max_iter+1):
            filtered_phase = _laplacian_filter(phase, freq, dims)
            filtered_phase.sub_(phase).div_(twopi).round_().mul_(twopi)
            phase += filtered_phase

            if n_iter < max_iter and filtered_phase.mean() < tol:
                break

    phase = fft.fftshift(phase, dim=dims)

    if bound not in ('dct', 'circulant'):
        slicer = [slice(d//2, d+d//2) for d in shape]
        phase = phase[(Ellipsis, *slicer)]
    return phase


def mean_phase(phase, weight=None):
    """Compute the average phase using the circular mean

    Parameters
    ----------
    phase : tensor
    weight : tensor, optional

    Returns
    -------
    mean_phase : scalar tensor

    References
    ----------
    https://en.wikipedia.org/wiki/Circular_mean
    """
    if weight is not None:
        sumw = weight.sum()
        mean_cos = phase.cos().mul_(weight).sum() / sumw
        mean_sin = phase.sin().mul_(weight).sum() / sumw
    else:
        mean_cos = phase.cos().mean()
        mean_sin = phase.sin().mean()
    return torch.atan2(mean_sin, mean_cos)


@torch.jit.script
def derivatives(magnitude, phase, fit_log_magnitude, fit_phase,
                g:Optional[Tensor] = None, h: Optional[Tensor] = None):
    """
    Derivatives of the complex MSE wrt log parameters

    Parameters
    ----------
    magnitude : (*spatial) tensor
        Observed magnitude image
    phase : (*spatial) tensor
        Observed phase image
    fit_log_magnitude : (*spatial) tensor
        Current log magnitude fit
    fit_phase : (*spatial) tensor
        Current phase fit
    g, h : tensors, optional
        Output placeholders

    Returns
    -------
    nll : scalar tensor
        Negative log-likelihood
    g : (2, *spatial) tensor
        Gradient wrt to log_magnitude and phase
    h : (2, *spatial) tensor
        Approximate (diagonal) Hessian wrt to log_magnitude and phase

    """

    if g is None:
        g = magnitude.new_empty([2] + magnitude.shape)
    if h is None:
        h = magnitude.new_empty([2] + magnitude.shape)

    fit_magnitude = fit_log_magnitude.exp()

    prod_phase = phase.cos() * fit_phase.cos() + phase.sin() * fit_phase.sin()
    prod_phase = prod_phase * magnitude
    prod_phase_grad = phase.cos() * fit_phase.sin() - phase.sin() * fit_phase.cos()
    prod_phase_grad = prod_phase_grad * magnitude

    g[0] = fit_magnitude * (fit_magnitude - prod_phase)
    g[1] = fit_magnitude * prod_phase_grad
    h[0] = fit_magnitude * fit_magnitude + g[0].abs()
    h[1] = fit_magnitude * fit_magnitude

    ll = 0.5 * (magnitude * magnitude + fit_magnitude * fit_magnitude) - fit_magnitude * prod_phase
    ll = ll.sum()

    return ll, g, h


@torch.jit.script
def nll(magnitude, phase, fit_log_magnitude, fit_phase):
    """Negative log-likelihood of the Complex MSE, in each voxel

    Parameters
    ----------
    magnitude : (*spatial) tensor
        Observed magnitude image
    phase : (*spatial) tensor
        Observed phase image
    fit_log_magnitude : (*spatial) tensor
        Current log magnitude fit
    fit_phase : (*spatial) tensor
        Current phase fit

    Returns
    -------
    nll : (*spatial) tensor
    """
    fit_magnitude = fit_log_magnitude.exp()
    prod_phase = phase.cos() * fit_phase.cos() + phase.sin() * fit_phase.sin()
    prod_phase = prod_phase * magnitude

    ll = 0.5 * (magnitude * magnitude + fit_magnitude * fit_magnitude) - fit_magnitude * prod_phase
    return ll


def dot(x, y):
    """Dot product"""
    return x.flatten().dot(y.flatten())


def plot_fit(magnitude, phase, fit):
    import matplotlib.pyplot as plt
    plt.subplot(2, 3, 1)
    plt.imshow(magnitude[:, :, magnitude.shape[-1] // 2])
    plt.colorbar()
    plt.subplot(2, 3, 2)
    plt.imshow(fit[0, :, :, magnitude.shape[-1] // 2].exp())
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.imshow(phase[:, :, magnitude.shape[-1] // 2], vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.subplot(2, 3, 5)
    plt.imshow(fit[1, :, :, magnitude.shape[-1] // 2], vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.subplot(2, 3, 6)
    diff = nll(
        magnitude[:, :, magnitude.shape[-1] // 2],
        phase[:, :, magnitude.shape[-1] // 2],
        fit[0, :, :, magnitude.shape[-1] // 2],
        fit[1, :, :, magnitude.shape[-1] // 2])
    plt.imshow(diff)
    plt.colorbar()
    plt.show()


def phase_fit(magnitude, phase, lam=(0, 1e1), penalty=('membrane', 'bending')):
    """Fit a complex image using a decreasing phase regularization

    Parameters
    ----------
    magnitude : tensor
    phase : tensor
    lam : (float, float)
    penalty : (str, str)

    Returns
    -------
    magnitude : tensor
    phase : tensor

    """

    # estimate noise precision
    sd = estimate_noise(magnitude)[0]['sd']
    prec = 1 / (sd * sd)

    # initialize fit
    fit = magnitude.new_empty([2, *magnitude.shape])
    fit[0] = magnitude
    fit[0].clamp_min_(1e-8).log_()
    fit[1] = mean_phase(phase, magnitude)

    # allocate placeholders
    g = magnitude.new_empty([2, *magnitude.shape])
    h = magnitude.new_empty([2, *magnitude.shape])
    n = magnitude.numel()

    # prepare regularizer options
    prm = dict(
        membrane=[lam[0] * int(penalty[0] == 'membrane'),
                  lam[1] * int(penalty[1] == 'membrane')],
        bending=[lam[0] * int(penalty[0] == 'bending'),
                 lam[1] * int(penalty[1] == 'bending')],
        bound='dct2')

    lam0 = dict(membrane=prm['membrane'][-1], bending=prm['bending'][-1])
    ll0 = lr0 = factor = float('inf')
    for n_iter in range(20):

        # decrease regularization
        factor, factor_prev = 1 + 10 ** (5 - n_iter), factor
        factor_ratio = factor / factor_prev if n_iter else float('inf')
        prm['membrane'][-1] = lam0['membrane'] * factor
        prm['bending'][-1] = lam0['bending'] * factor

        # compute derivatives
        ll, g, h = derivatives(magnitude, phase, fit[0], fit[1], g, h)
        ll *= prec
        g *= prec
        h *= prec

        # compute regularization
        reg = spatial.regulariser(fit, **prm)
        lr = 0.5 * dot(fit, reg)
        g += reg

        # Gauss-Newton step
        fit -= spatial.solve_field_fmg(h, g, **prm)

        # Compute progress
        l0 = ll0 + factor_ratio * lr0
        l = ll + lr
        gain = l0 - l
        print(f'{n_iter:3d} | {ll/n:12.6g} + {lr/n:6.3g} = {l/n:12.6g} | gain = {gain/n:6.3}')
        if abs(gain) < n * 1e-4:
            break

        ll0, lr0 = ll, lr
        # plot_fit(magnitude, phase, fit)

    return fit[0].exp_(), fit[1]


def b0_rad_to_hz(phase, delta_te, dwell, delta_frequency=0):
    """Convert a phase-difference image to a fieldmap in Hz

    Parameters
    ----------
    phase : tensor
        Phase difference image
    delta_te : float
        Echo time difference of the original images, in sec
    dwell : float
        Dwell time of the fieldmap images, in sec
        Bandwidth = 1 / (dwell * nb_pixels)
    delta_frequency : float
        Difference of center frequencies between the target acquisition
        and fieldmap acquisition, in Hz

    Returns
    -------
    fmap : tensor
        Fieldmap in Hz

    """
    fmap = phase / (delta_te * dwell * 2 * pymath.pi)
    if delta_frequency:
        fmap += delta_frequency
    return fmap


def b0_hz_to_vox(fmap, bandwidth):
    """Convert fieldmap in Hz to displacement map in voxel

    Parameters
    ----------
    fmap : tensor
        Fieldmap, in Hz
    bandwidth : float, Hz/pixel

    Returns
    -------
    disp : tensor
        displacement map

    """
    return fmap / bandwidth