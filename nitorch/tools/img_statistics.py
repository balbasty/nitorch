"""Functions simillar of inspired by the SPM package:

"""


import nibabel as nib
import torch
from ..core.kernels import smooth
from ..vb.mixtures import GMM
from ..vb.mixtures import RMM
from ..spatial import im_gradient
from ..core.constants import inf
from ..plot import show_slices


def estimate_fwhm(dat, vx=None, verbose=0, mn=-inf, mx=inf):
    """Estimates full width at half maximum (FWHM) and noise standard
    deviation (sd) of a 2D or 3D image.

    It is assumed that the image has been generated as:
        dat = Ky + n,
    where K is Gaussian smoothing with some FWHM and n is
    additive Gaussian noise. FWHM and n are estimated.

    Args:
        dat (torch.tensor): Image data (X, Y) | (X, Y, Z).
        vx (float, optional): Voxel size. Defaults to (1, 1, 1).
        verbose (int, optional): Verbosity level (0|1|2):
            0: No verbosity
            1: Print FWHM and sd to screen
            2: 1 + show mask
            Defaults to 0.
        mn (float, optional): Exclude values in dat below mn, default=-inf.
        mx (float, optional): Exclude values in dat above mx, default=-inf.

    Returns:
        fwhm (torch.tensor): Estimated FWHM (2,) | (3,).
        sd (torch.tensor): Estimated noise standard deviation.

    Reference:
        Appendix A of:
        Groves AR, Beckmann CF, Smith SM, Woolrich MW.
        Linked independent component analysis for multimodal data fusion.
        Neuroimage. 2011 Feb 1;54(3):2198-217.

    """
    if vx is None:
        vx = (1.0,) * 3
    # Parameters
    device = dat.device
    dtype = dat.dtype
    logtwo = torch.tensor(2.0, device=device, dtype=dtype).log()
    one = torch.tensor(1.0, device=device, dtype=dtype)
    ndim = len(dat.shape)
    # Make mask
    msk = (dat > mn) & (dat <= mx)
    if verbose >= 2:
        show_slices(msk)
    # Compute image gradient
    g = im_gradient(dat, which='central', vx=vx, bound='circular')
    g[~msk.repeat((ndim, 1, 1, 1))] = 0
    g = g.abs()
    if ndim == 3:
        g = g.sum(dim=3, dtype=torch.float64)
    g = g.sum(dim=2, dtype=torch.float64).sum(dim=1, dtype=torch.float64)
    # Make dat have zero mean
    x0 = dat[msk] - dat[msk].mean()
    # Compute FWHM
    fwhm = torch.sqrt(4.0 * logtwo) * torch.sum(x0.abs(), dtype=torch.float64)
    fwhm = fwhm / g
    if verbose >= 1:
        print('FWHM={}'.format(fwhm))
    # Compute noise standard deviation
    sx = smooth('gauss', fwhm[0], x=0, dtype=dtype, device=device)[0][0, 0, 0]
    sy = smooth('gauss', fwhm[1], x=0, dtype=dtype, device=device)[0][0, 0, 0]
    sz = 1.0
    if ndim == 3:
        sz = smooth('gauss', fwhm[2], x=0, dtype=dtype, device=device)[0][0, 0, 0]
    sc = (sx * sy * sz) / ndim
    sc = torch.min(sc, one)
    sd = torch.sqrt(torch.sum(x0 ** 2, dtype=torch.float64) / (x0.numel() * sc))
    if verbose >= 1:
        print('sd={}'.format(sd))
        
    return fwhm, sd


def estimate_noise(pth, show_fit=False, fig_num=1, num_class=2,
                   mu_noise=None, max_iter=10000, verbose=0):
    """Estimate noise from a nifti image by fitting either a GMM or an RMM 
    to the image's intensity histogram.

    Args:
        pth (string): Path to nifti file.
        show_fit (bool, optional): Defaults to False.
        fig_num (bool, optional): Defaults to 1.
        num_class (int, optional): Number of mixture classes (only for GMM).
            Defaults to 2.
        mu_noise (float, optional): Mean of noise class, defaults to None,
            in which case the class with the smallest sd is assumed the noise
            class.
        max_iter (int, optional) Maxmimum number of algorithm iterations.
                Defaults to 10000.
        verbose (int, optional) Display progress. Defaults to 0.
            0: None.
            1: Print summary when finished.
            2: 1 + Log-likelihood plot.
            3: 1 + 2 + print convergence.

    Returns:
        sd_noise (torch.Tensor): Standard deviation of background class.
        sd_not_noise (torch.Tensor): Standard deviation of foreground class.
        mu_noise (torch.Tensor): Mean of background class.
        mu_not_noise (torch.Tensor): Mean of foreground class.

    """
    if isinstance(pth, torch.Tensor):
        dat = pth
    else:  # Load data from nifti
        nii = nib.load(pth)
        dat = torch.tensor(nii.get_fdata())
        dat = dat.flatten()
    device = dat.device
    dat = dat.double()

    # Mask and get min/max
    dat = dat[(dat != 0) & (torch.isfinite(dat)) & (dat != dat.min()) &
              (dat != dat.max())]
    mn = torch.min(dat).round()
    mx = torch.max(dat).round()
    bins = (mx - mn).int()
    if bins < 1024:
        bins = 1024

    # Histogram bin data
    W = torch.histc(dat, bins=bins).double()
    x = torch.linspace(mn, mx, steps=bins, device=device).double()

    # mn = -1
    if mn < 0:  # Make GMM model
        model = GMM(num_class=num_class)
    else:  # Make RMM model
        model = RMM(num_class=num_class)

    # Fit GMM using Numpy
    model.fit(x, W=W, verbose=verbose, max_iter=max_iter, show_fit=show_fit, fig_num=fig_num)

    # Get means and mixing proportions
    mu, _ = model.get_means_variances()
    mu = mu.squeeze()
    mp = model.mp
    if mn < 0:  # GMM
        sd = torch.sqrt(model.Cov).squeeze()
    else:  # RMM
        sd = model.sig

    # Get std and mean of noise class
    if mu_noise:
        # Closest to mu_bg
        _, ix_noise = torch.min(torch.abs(mu - mu_noise), dim=0)
        mu_noise = mu[ix_noise]
        sd_noise = sd[ix_noise]
    else:
        # With smallest sd
        sd_noise, ix_noise = torch.min(sd, dim=0)
        mu_noise = mu[ix_noise]
    # Get std and mean of other classes (means and sds weighted by mps)
    rng = torch.arange(0, num_class, device=device)
    rng = torch.cat([rng[0:ix_noise], rng[ix_noise + 1:]])
    mu1 = mu[rng]
    sd1 = sd[rng]
    w = mp[rng]
    w = w / torch.sum(w)
    mu_not_noise = sum(w * mu1)
    sd_not_noise = sum(w * sd1)

    return sd_noise, sd_not_noise, mu_noise, mu_not_noise
