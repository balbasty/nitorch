"""Functions similar of inspired by the SPM package:"""

import torch
from nitorch.core import py, utils
from nitorch.core.kernels import smooth
from nitorch.vb.mixtures import GMM, RMM, CMM
from nitorch.spatial import diff, voxel_size as get_voxel_size
from nitorch.core.constants import inf
from nitorch.plot import show_slices
from nitorch import io
from nitorch.core.dtypes import dtype as dtype_info
import math as pymath


def estimate_fwhm(dat, vx=None, verbose=0, mn=-inf, mx=inf):
    """Estimates full width at half maximum (FWHM) and noise standard
    deviation (sd) of a 2D or 3D image.

    It is assumed that the image has been generated as:
        dat = Ky + n,
    where K is Gaussian smoothing with some FWHM and n is
    additive Gaussian noise. FWHM and n are estimated.

    Parameters
    ----------
    dat : str or (*spatial) tensor
        Image data or path to nifti file
    vx : [sequence of] float, default=1
        Voxel size
    verbose : {0, 1, 2}, default=0
        Verbosity level:
            * 0: No verbosity
            * 1: Print FWHM and sd to screen
            * 2: 1 + show mask
    mn : float, optional
        Exclude values below
    mx : float, optional
        Exclude values above

    Returns
    -------
    fwhm : (dim,) tensor
        Estimated FWHM
    sd : scalar tensor
        Estimated noise standard deviation.

    References
    ----------
    ..[1] "Linked independent component analysis for multimodal data fusion."
          Appendix A
          Groves AR, Beckmann CF, Smith SM, Woolrich MW.
          Neuroimage. 2011 Feb 1;54(3):2198-217.

    """
    if isinstance(dat, str):
        dat = io.map(dat)
    if isinstance(dat, io.MappedArray):
        if vx is None:
            vx = get_voxel_size(dat.affine)
        dat = dat.fdata(rand=True, missing=0)
    dat = torch.as_tensor(dat)

    dim = dat.dim()
    if vx is None:
        vx = 1
    vx = utils.make_vector(vx, dim)
    backend = utils.backend(dat)
    # Make mask
    msk = (dat > mn).bitwise_and_(dat <= mx)
    dat = dat.masked_fill(~msk, 0)
    # TODO: we should erode the mask so that only voxels whose neighbours
    #       are in the mask are considered when computing gradients.
    if verbose >= 2:
        show_slices(msk)
    # Compute image gradient
    g = diff(dat, dim=range(dim), side='central', voxel_size=vx, bound='dft').abs_()
    slicer = (slice(1, -1),) * dim
    g = g[(*slicer, None)]
    g[msk[slicer], :] = 0
    g = g.reshape([-1, dim]).sum(0, dtype=torch.double)
    # Make dat have zero mean
    dat = dat[slicer]
    dat = dat[msk[slicer]]
    x0 = dat - dat.mean()
    # Compute FWHM
    fwhm = pymath.sqrt(4 * pymath.log(2)) * x0.abs().sum(dtype=torch.double)
    fwhm = fwhm / g
    if verbose >= 1:
        print(f'FWHM={fwhm.tolist()}')
    # Compute noise standard deviation
    sx = smooth('gauss', fwhm[0], x=0, **backend)[0][0, 0, 0]
    sy = smooth('gauss', fwhm[1], x=0, **backend)[0][0, 0, 0]
    sz = 1.0
    if dim == 3:
        sz = smooth('gauss', fwhm[2], x=0, **backend)[0][0, 0, 0]
    sc = (sx * sy * sz) / dim
    sc.clamp_min_(1)
    sd = torch.sqrt(x0.square().sum(dtype=torch.double) / (x0.numel() * sc))
    if verbose >= 1:
        print(f'sd={sd.tolist()}')
    return fwhm, sd


def estimate_noise(dat, show_fit=False, fig_num=1, num_class=2,
                   mu_noise=None, max_iter=10000, verbose=0,
                   bins=1024, chi=False):
    """Estimate the noise distribution in an image by fitting either a
    Gaussian, Rician or noncentral Chi mixture model to the image's
    intensity histogram.

    The Gaussian model is only used if negative values are found in the
    image (e.g., if it is a CT scan).

    Parameters
    ----------
    dat : str or tensor
        Tensor or path to nifti file.
    show_fit : bool, default=False
        Show a plot of the histogram fit at the end
    fig_num : int, default=1
        ID of matplotlib figure to use
    num_class : int, default=2
        Number of mixture classes (only for GMM).
    mu_noise : float, optional
        Mean of noise class. If provided, the class with mean value closest
        to `mu_noise` is assumed to be the background class. Otherwise, it is
        the class with smallest standard deviation.
    max_iter : int, default=10000
        Maximum number of EM iterations.
    verbose int, defualt=0:
        Display progress. Defaults to 0.
            * 0: None.
            * 1: Print summary when finished.
            * 2: 1 + Log-likelihood plot.
            * 3: 1 + 2 + print convergence.
    bins : int, default=1024
        Number of histogram bins.
    chi : bool, default=False
        Fit a noncentral Chi rather than a Rice model.

    Returns
    -------
    prm_noise : dict
        Parameters of the distribution of the background (noise) class
        With fields 'sd', 'mean' and (if `chi`) 'dof'
    prm_not_noise : dict
        Parameters of the distribution of the foreground (tissue) class
        With fields 'sd', 'mean' and (if `chi`) 'dof'

    """
    DTYPE = torch.double  # use double for accuracy (maybe single would work?)

    slope = None
    if isinstance(dat, str):
        dat = io.map(dat)
    if isinstance(dat, io.MappedArray):
        slope = dat.slope
        if not slope and not dtype_info(dat.dtype).if_floating_point:
            slope = 1
        dat = dat.fdata(rand=True, missing=0, dtype=DTYPE)
    dat = torch.as_tensor(dat, dtype=DTYPE).flatten()
    device = dat.device
    if not slope and not dat.dtype.is_floating_point:
        slope = 1

    # exclude missing values
    dat = dat[torch.isfinite(dat)]
    dat = dat[dat != 0]

    # Mask and get min/max
    mn = dat.min()
    mx = dat.max()
    dat = dat[dat != mn]
    dat = dat[dat != mx]
    mn = mn.round()
    mx = mx.round()
    if slope:
        # ensure bin width aligns with integer width
        width = (mx - mn) / bins
        width = (width / slope).ceil() * slope
        mx = mn + bins * width

    # Histogram bin data
    dat = torch.histc(dat, bins=bins, min=mn, max=mx).to(DTYPE)
    x = torch.linspace(mn, mx, steps=bins, device=device, dtype=DTYPE)

    # fit mixture model
    if mn < 0:  # Make GMM model
        model = GMM(num_class=num_class)
    elif chi:
        model = CMM(num_class=num_class)
    else:  # Make RMM model
        model = RMM(num_class=num_class)

    # Fit GMM/RMM/CMM using Numpy
    model.fit(x, W=dat, verbose=verbose, max_iter=max_iter,
              show_fit=show_fit, fig_num=fig_num)

    # Get means and mixing proportions
    mu, _ = model.get_means_variances()
    mu = mu.squeeze()
    mp = model.mp
    if mn < 0:  # GMM
        sd = torch.sqrt(model.Cov).squeeze()
    else:  # RMM/CMM
        sd = model.sig.squeeze()

    # Get std and mean of noise class
    if mu_noise:
        # Closest to mu_bg
        _, ix_noise = torch.min(torch.abs(mu - mu_noise), dim=0)
    else:
        # With smallest sd
        _, ix_noise = torch.min(sd, dim=0)
    mu_noise = mu[ix_noise]
    sd_noise = sd[ix_noise]
    if chi:
        dof_noise = model.dof[ix_noise]

    # Get std and mean of other classes (means and sds weighted by mps)
    rng = list(range(num_class))
    del rng[ix_noise]
    mu = mu[rng]
    sd = sd[rng]
    w = mp[rng]
    w = w / torch.sum(w)
    mu_not_noise = sum(w * mu)
    sd_not_noise = sum(w * sd)
    if chi:
        dof = model.dof[rng]
        dof_not_noise = sum(w * dof)

    # return dictionaries of parameters
    prm_noise = dict(sd=sd_noise, mean=mu_noise)
    prm_not_noise = dict(sd=sd_not_noise, mean=mu_not_noise)
    if chi:
        prm_noise['dof'] = dof_noise
        prm_not_noise['dof'] = dof_not_noise
    return prm_noise, prm_not_noise
