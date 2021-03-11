import math
from ...core.constants import eps
from ...core.optim import get_gain
from ...io import map
from ...spatial import diff
from ...tools.img_statistics import estimate_noise
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _add_rician_noise(dat, noise_prct=0.1):
    """Adds rician noise to a tensor as:
    dat_rice = dat + n_real + i*n_img, where n_real, n_img ~ N(0, std_rice**2)
    dat_rice = magnitude(dat_rice)

    OBS: Done using Numpy, as torch only recently got support for complex numbers (1.7)

    Parameters
    ----------
    dat : tensor
        Input data
    noise_prct : float, default=0.1
        Amount of noise to add, as a percentage of input max value

    Returns
    ----------
    dat : tensor
        Noisy data

    """
    device = dat.device
    dtype = dat.dtype
    shape = dat.shape
    dat_rice = dat.detach().cpu().numpy()
    std_rice = noise_prct * np.percentile(dat_rice.squeeze(), 99.99)
    dat_rice = dat_rice + np.random.normal(0, std_rice, shape) + 1j * np.random.normal(0, std_rice, shape)
    dat_rice = np.absolute(dat_rice)
    dat_rice = torch.from_numpy(dat_rice).type(dtype).to(device)

    return dat_rice, std_rice


def denoise_mri(dat_x, lam_scl=4.0, lr=1e1, max_iter=10000, tolerance=1e-8, verbose=True):
    """Denoises a multi-channel MR image by solving:
    dat_y_hat = 0.5*sum_c(tau_c*sum_i((dat_x_ci - dat_y_ci)^2)) + jtv(dat_y_1, ..., dat_y_C; lam)
    using PyTorch's auto-diff.

    Reference:
    Brudfors, Mikael, et al. "MRI super-resolution using multi-channel total variation."
    Annual Conference on Medical Image Understanding and Analysis. Springer, Cham, 2018.

    Parameters
    ----------
    dat_x : (dmx, dmy, dmz, nchannels) tensor
        Input noisy image data
    lam_scl : float, default=4.0
        Scaling of regularisation values
    lr : float, default=1e1
        Optimiser learning rate
    max_iter : int, default=10000
        Maximum number of fitting iterations
    tolerance : float, default=1e-8
        Convergence threshold (when to stop iterating)
    verbose : bool, default=True
        Print to screen?

    Returns
    ----------
    dat_y_hat : (dmx, dmy, dmz, nchannels) tensor
        Denoised image data

    """
    device = dat_x.device
    dtype = dat_x.dtype
    # estimate hyper-parameters
    tau = torch.zeros(dat_x.shape[-1], device=device, dtype=dtype)
    lam = torch.zeros(dat_x.shape[-1], device=device, dtype=dtype)
    for i in range(dat_x.shape[-1]):
        sd_bg, _, _, mean_fg = estimate_noise(dat_x[..., i])
        tau[i] = 1 / sd_bg.float() ** 2
        lam[i] = math.sqrt(1 / dat_x.shape[-1]) / mean_fg.float()  # modulates with number of channels (as in JTV reg)
    if verbose:
        print("tau={:}".format(tau))
        print("lam={:}".format(lam))
    # scale regularisation
    lam = lam_scl * lam
    # initial estimate of reconstruction
    dat_y_hat = torch.zeros_like(dat_x)
    dat_y_hat = torch.nn.Parameter(dat_y_hat, requires_grad=True)
    # prepare optimiser and scheduler
    optim = torch.optim.Adam([dat_y_hat], lr=lr)  # Adam
    # optim = torch.optim.SGD([dat_y_hat], lr=lr, momentum=0.9)  # SGD
    scheduler = ReduceLROnPlateau(optim)
    # optimisation loop
    loss_vals = torch.zeros(max_iter + 1, dtype=torch.float64)
    for n_iter in range(1, max_iter + 1):
        # set gradients to zero (PyTorch accumulates the gradients on subsequent backward passes)
        optim.zero_grad()
        # compute reconstruction loss
        loss_val = _loss_ssqd_jtv(dat_x, dat_y_hat, tau, lam)
        # differentiate reconstruction loss w.r.t. dat_y_hat
        loss_val.backward()
        # store loss
        loss_vals[n_iter] = loss_val.item()
        # update reconstruction
        optim.step()
        # compute gain
        gain = get_gain(loss_vals[:n_iter + 1], monotonicity='decreasing')
        if n_iter > 10 and gain < tolerance:
            # finished
            break
        # incorporate scheduler
        if scheduler is not None and n_iter % 10 == 0:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss_val)
            else:
                scheduler.step()
        if verbose:
            # print to screen
            with torch.no_grad():
                if n_iter % 100 == 0:
                    print('n_iter={:4d}, loss={:12.6f}, gain={:0.10}, lr={:g}'.format(n_iter, loss_val.item(), gain, optim.param_groups[0]['lr']), end='\r')

    return dat_y_hat


def _get_image_data(pths, device=None, dtype=None):
    """Formats image data (on disk) to tensor compatible with the denoise_mri function

    OBS: Assumes that all input images are 3D volumes with the same dimensions.

    Parameters
    ----------
    pths : [nchannels] sequence
        Paths to image data
    device : torch.device, optional
        Torch device
    dtype : torch.dtype, optional
        Torch data type

    Returns
    ----------
    dat : (dmx, dmy, dmz, nchannels) tensor
        Image tensor

    """
    for i, p in enumerate(pths):
        # read nifti
        nii = map(p)
        # get data
        if i == 0:
            dat = nii.fdata(dtype=dtype, device=device, rand=False)[..., None]
        else:
            dat = torch.cat((dat, nii.fdata(dtype=dtype, device=device, rand=False)[..., None]), dim=-1)

    return dat


def _loss_ssqd_jtv(dat_x, dat_y, tau, lam, voxel_size=1, side='f', bound='dct2'):
    """Computes an image denoising loss function, where:
    * fidelity term: sum-of-squared differences (SSQD)
    * regularisation term: joint total variation (JTV)
    * hyper-parameters: tau, lambda

    Parameters
    ----------
    dat_x : (dmx, dmy, dmz, nchannels) tensor
        Input image
    dat_y : (dmx, dmy, dmz, nchannels) tensor
        Reconstruction image
    tau : (nchannels) tensor
        Channel-specific noise precisions
    lam : (nchannels) tensor
        Channel-specific regularisation values
    voxel_size : float or sequence[float], default=1
        Unit size used in the denominator of the gradient.
    side : {'c', 'f', 'b'}, default='f'
        * 'c': central finite differences
        * 'f': forward finite differences
        * 'b': backward finite differences
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'repeat', 'zero'}, default='dct2'
        Boundary condition.

    Returns
    ----------
    nll_yx : tensor
        Loss function value (negative log-posterior)

    """
    # compute negative log-likelihood (SSQD fidelity term)
    nll_xy = 0.5 * torch.sum(tau * torch.sum((dat_x - dat_y) ** 2, dim=(0, 1, 2)))
    # compute gradients of reconstruction, shape=(dmx, dmy, dmz, nchannels, dmgr)
    nll_y = diff(dat_y, order=1, dim=(0, 1, 2), voxel_size=voxel_size, side=side, bound=bound)
    # modulate channels with regularisation
    nll_y = lam[None, None, None, :, None] * nll_y
    # compute negative log-prior (JTV regularisation term)
    nll_y = torch.sum(nll_y ** 2 + eps(), dim=-1)  # to gradient magnitudes (sum over gradient directions)
    nll_y = torch.sum(nll_y, dim=-1)  # sum over reconstruction channels
    nll_y = torch.sqrt(nll_y)
    nll_y = torch.sum(nll_y)  # sum over voxels
    # compute negative log-posterior (loss function)
    nll_yx = nll_xy + nll_y

    return nll_yx
