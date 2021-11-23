import math
from nitorch.core.constants import eps
from nitorch.core.optim import get_gain
from nitorch.core.py import file_replace
from nitorch.io import (map, savef)
from nitorch.spatial import (diff, voxel_size)
from nitorch.tools.img_statistics import estimate_noise
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .parser import DenoiseMRI as opt
import os


def denoise_mri(*dat_x, affine_x=None, lam_scl=opt.lam_scl, lr=opt.learning_rate,
                max_iter=opt.max_iter, tolerance=opt.tolerance, verbose=opt.verbose,
                device=opt.device, do_write=opt.do_write, dir_out=opt.dir_out):
    """Denoises a multi-channel MR image by solving:

    dat_y_hat = 0.5*sum_c(tau_c*sum_i((dat_x_ci - dat_y_ci)^2)) + jtv(dat_y_1, ..., dat_y_C; lam)

    using PyTorch's auto-diff.

    If input is given as paths to files, outputs prefixed 'den_' is written based on options
    'dir_out' and 'do_write'.

    Reference:
    Brudfors, Mikael, et al. "MRI super-resolution using multi-channel total variation."
    Annual Conference on Medical Image Understanding and Analysis. Springer, Cham, 2018.

    Parameters
    ----------
    dat_x : (nchannels, dmx, dmy, dmz) tensor or sequence[str]
        Input noisy image data
    affine_x : (4, 4) tensor, optional
        Input images' affine matrix. If not given, assumes identity.
    lam_scl : float, default=10.0
        Scaling of regularisation values
    lr : float, default=1e1
        Optimiser learning rate
    max_iter : int, default=10000
        Maximum number of fitting iterations
    tolerance : float, default=1e-8
        Convergence threshold (when to stop iterating)
    verbose : bool, default=True
        Print to terminal?
    device : torch.device, default='cuda'
        Torch device
    do_write : bool, default=True
        If input is given as paths to files, output is written to disk,
        prefixed 'den_' to 'dir_out'
    dir_out : str, optional
        Directory where to write output, default is same as input.

    Returns
    ----------
    dat_y_hat : (nchannels, dmx, dmy, dmz) tensor
        Denoised image data

    """
    # read data from disk
    if isinstance(dat_x,(list, tuple)) and \
        sum(isinstance(dat_x[i],str) for i in range(len(dat_x))) == len(dat_x):
        dat_x, affine_x, nii = _get_image_data(dat_x, device=device)
    else:
        do_write = False  # input is tensor, do not write to disk
    # backend
    device = dat_x.device
    dtype = dat_x.dtype
    # estimate hyper-parameters
    tau = torch.zeros(dat_x.shape[0], device=device, dtype=dtype)
    lam = torch.zeros(dat_x.shape[0], device=device, dtype=dtype)
    for i in range(dat_x.shape[0]):
        prm0, prm1 = estimate_noise(dat_x[i, ...], show_fit=False)
        sd_bg = prm0['sd']
        mean_fg = prm1['mean']
        tau[i] = 1 / sd_bg.float() ** 2
        lam[i] = math.sqrt(1 / dat_x.shape[0]) / mean_fg.float()  # modulates with number of channels (as in JTV reg)
    # print("tau={:}".format(tau))
    # print("lam={:}".format(lam))
    # affine matrices
    if affine_x is None:
        affine_x = torch.eye(4, device=device, dtype=dtype)
    # voxel size
    vx = voxel_size(affine_x)
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
    cnt_conv = 0
    for n_iter in range(1, max_iter + 1):
        # set gradients to zero (PyTorch accumulates the gradients on subsequent backward passes)
        optim.zero_grad()
        # compute reconstruction loss
        loss_val = _loss_ssqd_jtv(dat_x, dat_y_hat, tau, lam, vx=vx)
        # differentiate reconstruction loss w.r.t. dat_y_hat
        loss_val.backward()
        # store loss
        loss_vals[n_iter] = loss_val.item()
        # update reconstruction
        optim.step()
        # compute gain
        gain = get_gain(loss_vals[:n_iter + 1], monotonicity='decreasing')
        if verbose:
            # print to screen
            with torch.no_grad():
                if n_iter % 25 == 0:
                    print('n_iter={:4d}, loss={:12.6f}, gain={:0.10}, lr={:g}'. \
                        format(n_iter, loss_val.item(), gain, optim.param_groups[0]['lr']), end='\n')  # end='\r'
        if n_iter > 10 and gain.abs() < tolerance:
            cnt_conv += 1
            if cnt_conv == 5:
                # finished
                break
        else:
            cnt_conv = 0
        # incorporate scheduler
        if scheduler is not None and n_iter % 10 == 0:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss_val)
            else:
                scheduler.step()
    if do_write:
        # write output to disk
        if dir_out is not None:
            os.makedirs(dir_out, exist_ok=True)
        for i in range(dat_y_hat.shape[0]):
            fname = file_replace(nii.fname, prefix='den_', dir=dir_out, suffix='_' + str(i))
            savef(dat_y_hat[i, ...], fname, like=nii)

    return dat_y_hat


def _add_rician_noise(dat, noise_prct=0.1):
    """Adds rician noise to a tensor as:
    dat = dat + n_real + i*n_img, where n_real, n_img ~ N(0, std**2)
    dat = magnitude(dat)

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
	std : float
		Noise standard deviation

    """
    std = noise_prct * dat.max()
    dat = ((dat + std*torch.randn_like(dat))**2 + (std*torch.randn_like(dat))**2).sqrt()

    return dat, std


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
    dat : (nchannels, dmx, dmy, dmz) tensor
        Image tensor
    affine : (4, 4) tensor
        Affine matrix (assumed the same across channels)
    nii : nitorch.BabelArray
        BabelArray object.

    """
    for i, p in enumerate(pths):
        # read nifti
        nii = map(p)
        # get data
        if i == 0:
            dat = nii.fdata(dtype=dtype, device=device, rand=False)[None]
            affine = nii.affine.type(dat.dtype).to(dat.device)
            shape = nii.shape
        else:
            if not torch.equal(torch.as_tensor(shape), torch.as_tensor(nii.shape)):
                raise ValueError('All images must have same dimensions!')
            dat = torch.cat((dat, nii.fdata(dtype=dtype, device=device, rand=False)[None]), dim=0)

    return dat, affine, nii


def _loss_ssqd_jtv(dat_x, dat_y, tau, lam, vx=1, side='f', bound='dct2'):
    """Computes an image denoising loss function, where:
    * fidelity term: sum-of-squared differences (SSQD)
    * regularisation term: joint total variation (JTV)
    * hyper-parameters: tau, lambda

    Parameters
    ----------
    dat_x : (nchannels, dmx, dmy, dmz) tensor
        Input image
    dat_y : (nchannels, dmx, dmy, dmz) tensor
        Reconstruction image
    tau : (nchannels) tensor
        Channel-specific noise precisions
    lam : (nchannels) tensor
        Channel-specific regularisation values
    vx : float or sequence[float], default=1
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
    # exponentiate
    # dat_y = dat_y.exp()
    # compute negative log-likelihood (SSQD fidelity term)
    nll_xy = 0.5 * torch.sum(tau * torch.sum((dat_x - dat_y) ** 2, dim=(1, 2, 3)))
    # compute gradients of reconstruction, shape=(dmx, dmy, dmz, nchannels, dmgr)
    nll_y = diff(dat_y, order=1, dim=(1, 2, 3), voxel_size=vx, side=side, bound=bound)
    # modulate channels with regularisation
    nll_y = lam[..., None, None, None, None] * nll_y
    # compute negative log-prior (JTV regularisation term)
    nll_y = torch.sum(nll_y ** 2 + eps(), dim=-1)  # to gradient magnitudes (sum over gradient directions)
    nll_y = torch.sum(nll_y, dim=0)  # sum over reconstruction channels
    nll_y = torch.sqrt(nll_y)
    nll_y = torch.sum(nll_y)  # sum over voxels
    # compute negative log-posterior (loss function)
    nll_yx = nll_xy + nll_y

    return nll_yx
