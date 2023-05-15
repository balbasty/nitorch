"""Core functions for affine image registration.

"""
import numpy as np
from scipy.optimize import minimize, Bounds
import torch
from torch.nn import functional as F
from nitorch.plot import show_slices
from nitorch.core import linalg
from nitorch.core.kernels import smooth
from nitorch.core.utils import pad
from nitorch.spatial import identity_grid, im_gradient, voxel_size, mean_space
from .._preproc_utils import _mean_space
from ..img_statistics import estimate_noise
from ._costs import (_compute_cost, _costs_edge, _costs_hist)


def _data_loader(dat, mat, opt):
    """Load image data, affine matrices, image dimensions.

    Parameters
    ----------
    dat : [N,] tensor_like
        List of image data.
    mat : [N,] tensor_like
        List of affine matrices.

    Returns
    ----------
    dat : [N,] tensor_like
        List of preprocessed image data.

    """
    if opt['raw']:
        return dat

    for n in range(len(dat)):  # loop over input images
        # Mask
        dat[n][~torch.isfinite(dat[n])] = 0.0
        if opt['cost_fun'] in _costs_edge:
            # Get gradient scaling values
            prm0, prm1 = estimate_noise(dat[n], show_fit=False)
            mu_bg = prm0['mean']
            mu_fg = prm1['mean']
            scl = 1.0 / torch.abs(mu_fg.float() - mu_bg.float())
            if not torch.isfinite(scl):
                scl = 1.0
        if opt['cost_fun'] in _costs_hist:
            # Rescale
            dat[n] = _rescale(dat[n], mn_out=0, mx_out=opt['mx_int'])
        # Smooth a bit
        dat[n] = _smooth_for_reg(dat[n], mat[n], samp=opt['samp'][-1])
        if opt['cost_fun'] in _costs_edge:
            # Compute gradient magnitudes
            dat[n] = _to_gradient_magnitudes(dat[n], mat[n], scl)

    return dat


def _rescale(dat, mn_out=0, mx_out=511):
    """ Rescales image intensities between mn_out and mx_out.

    """
    backend = dict(dtype=dat.dtype, device=dat.device)
    msk = torch.isfinite(dat).bitwise_not_()
    msk = msk.bitwise_or_(dat == dat.min()).bitwise_or_(dat == dat.max())
    dat = dat.masked_fill_(msk, 0)
    # Make scaling to set image intensities between mn_out and mx_out
    mnmx_in = torch.as_tensor([[dat.min(), 1], [dat.max(), 1]], **backend)
    mnmx_out = torch.as_tensor([mn_out, mx_out], **backend)
    sf = linalg.lmdiv(mnmx_in, mnmx_out.unsqueeze(-1)).squeeze(-1)
    # Rescale
    dat = dat.mul_(sf[0]).add_(sf[1])
    # Clamp
    dat = dat.clamp_(mn_out, mx_out)

    return dat


def _do_optimisation(q, args, s, opt, dim):
    """Run optimisation routine to fit q.

    Parameters
    ----------
    q : (N, Nq) tensor_like
        Lie algebra of affine registration fit.
    args : tuple
        All arguments for optimiser (except the parameters to be fitted)
    opt : dict
        See run_affine_reg.
    s : float
        Current sub-sampling level.
    dim : int
        Number of dimensions.

    Returns
    ----------
    q : (N, Nq) tensor_like
        Optimised Lie algebra of affine registration fit.

    """
    if len(q.shape) == 1:
        # Pairwise registration
        N = 1
        Nq = q.shape[0]
    else:
        # Groupwise registration
        N = q.shape[0]
        Nq = q.shape[1]
    # Optimisers require CPU Numpy arrays
    device = q.device
    q = q.cpu().numpy()
    q_shape = q.shape
    q = q.flatten()
    if opt['optimiser'] == 'powell':
        # Powell optimisation
        # ----------
        # Feasible initial search directions
        # [translation, rotation, scaling, skew]
        direc = np.concatenate([0.5*np.ones(dim),   0.025*np.ones(dim),
                                0.005*np.ones(dim), 0.005*np.ones(dim)])
        direc = direc[:Nq]
        direc = np.tile(direc, N)
        direc = np.diag(direc)
        # Feasible upper and lower bounds
        # [translation, rotation, scaling, skew]
        ub = np.concatenate([100*np.ones(dim),   0.5*np.ones(dim),
                             0.15*np.ones(dim), 0.15*np.ones(dim)])
        ub = ub[:Nq]
        ub = np.tile(ub, N)
        lb = -ub
        bounds = Bounds(lb=lb, ub=ub)
        # Callback
        callback = None
        if opt['mean_space']:
            # Ensure that the parameters have zero mean, across images.
            callback = lambda x: _zero_mean(x, q_shape)
        # Do optimisation
        res = minimize(_compute_cost, q, args, method='Powell',
                       callback=callback, bounds=bounds,
                       options=dict(disp=False, direc=direc))
        q = res['x']
    # Cast back to tensor
    q = q.reshape(q_shape)
    q = torch.from_numpy(q).to(device)
    return q


def _fit_q(q, dat_fix, grid, mat_fix, dat, mat, mov, B, s, opt):
    """Fit q, either by pairwise or groupwise optimisation.

    Parameters
    ----------
     q : (N, Nq) tensor_like
        Affine parameters.
    dat_fix : (X1, Y1, Z1) tensor_like
        Fixed image data.
    grid : (X1, Y1, Z1) tensor_like
        Sub-sampled image data's resampling grid.
    mat_fix : (4, 4) tensor_like
        Fixed affine matrix.
    dat : [N,] tensor_like
        List of input images.
    mat : [N,] tensor_like
        List of affine matrices.
    mov : [N,] int
        Indices of moving images.
    B : (Nq, N, N) tensor_like
        Affine basis.
    s : float
        Current sub-sampling level.
    opt : dict
        See run_affine_reg.

    Returns
    ----------
    q : (N, Nq) tensor_like
        Fitted affine parameters.
    args : tuple
        All arguments for optimiser (except the parameters to be fitted (q))

    """
    dim = mat[0].shape[0] - 1
    if opt['cost_fun'] in _costs_edge:  # Groupwise optimisation
        # Arguments to _compute_cost
        m = mov
        args = (grid, dat_fix, mat_fix, dat, mat, m, opt['cost_fun'], B,
                opt['mx_int'], opt['fwhm'])
        # Run groupwise optimisation
        q[m, ...] = _do_optimisation(q[m, ...], args, s, opt, dim)
        # Show results
        _show_results(q[m, ...], args, opt)
    else:  # Pairwise optimisation
        for m in mov:
            # Arguments to _compute_cost
            args = (grid, dat_fix, mat_fix, dat, mat, [m],
                    opt['cost_fun'], B, opt['mx_int'], opt['fwhm'])
            # Run pairwise optimisation
            q[m, ...] = _do_optimisation(q[m, ...], args, s, opt, dim)
            # Show results
            _show_results(q[m, ...], args, opt)

    return q, args


def _get_dat_grid(dat, vx, samp, jitter=True, device='cpu'):
    """Get sub-sampled image data, and resampling grid.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensor_like
        Fixed image data.
    vx : (2|3,) tensor_like.
        Fixed voxel size.
    samp : int|float
        Sub-sampling level.
    jitter : bool, default=True
        Add random jittering to identity grid.

    Returns
    ----------
    dat_samp : (X1, Y1, Z1) tensor_like
        Sub-sampled fixed image data.
    grid : (X1, Y1, Z1) tensor_like
        Sub-sampled image data's resampling grid.

    """
    if isinstance(dat, (list, tuple)):
        dat = torch.zeros(dat, dtype=torch.float32, device=device)
    # Modulate samp with voxel size
    device = dat.device
    samp = torch.tensor((samp,) * vx.numel()).float().to(device)
    samp = torch.clamp(samp / vx, 1)
    # Create grid of fixed image, possibly sub-sampled
    grid = identity_grid(dat.shape,
        dtype=torch.float32, device=device)
    if jitter:
        torch.manual_seed(0)
        grid += torch.rand_like(grid)*samp
    # Sub-sampled
    samp = samp.round().int().tolist()
    if vx.numel() == 3:
        grid = grid[::samp[0], ::samp[1], ::samp[2], ...]
        dat_samp = dat[::samp[0], ::samp[1], ::samp[2]]
    else:
        grid = grid[::samp[0], ::samp[1], ...]
        dat_samp = dat[::samp[0], ::samp[1]]

    return dat_samp, grid


def _get_mean_space(dat, mat):
    """Compute mean space from images' field of views.

    Parameters
    ----------
    dat :  [N,] tensor_like
        List of images.
    mat :  [N,] tensor_like
        List of affines.

    Returns
    ----------
    mat : (4, 4) tensor_like
        Mean affine matrix.
    dims : (3,) tuple
        Mean image dimensions.

    """
    # Copy matrices and dimensions to torch tensors
    dim = mat[0].shape[0] - 1
    dtype = mat[0].dtype
    device = mat[0].device
    N = len(mat)
    all_mat = torch.zeros((N, dim + 1, dim + 1), dtype=dtype, device=device)
    all_dim = torch.zeros((N, dim), dtype=dtype, device=device)
    for n in range(N):
        all_mat[n, ...] = mat[n]
        all_dim[n, ...] = torch.tensor(dat[n].shape,
                                       dtype=dtype, device=device)
    # Compute mean-space
    mat, dim = mean_space(all_mat, all_dim)

    return mat, dim


def _show_results(q, args, opt):
    """ Simple function for visualising some results during algorithm execution.
    """
    if opt['verbose']:
        c, res = _compute_cost(q, *args, return_res=True)
        print('_compute_cost({})={}'.format(opt['cost_fun'], c))
        _ = show_slices(res, fig_num=1, cmap='coolwarm')
        # nii = nib.Nifti1Image(res.cpu().numpy(), mat_fix.cpu().numpy())
        # dir_out = os.path.split(pths[0])[0]
        # nib.save(nii, os.path.join(dir_out, 'res_coreg.nii'))


def _smooth_for_reg(dat, mat, samp):
    """Smoothing for image registration. FWHM is computed from voxel size
       and sub-sampling amount.

    Parameters
    ----------
    dat : (X, Y, Z) tensor_like
        3D image volume.
    mat : (4, 4) tensor_like
        Affine matrix.
    samp : float
        Amount of sub-sampling (in mm).

    Returns
    -------
    dat : (Nx, Ny, Nz) tensor_like
        Smoothed 3D image volume.

    """
    if samp <= 0:
        return dat    
    # Make smoothing kernel
    vx = voxel_size(mat).to(dat.device).type(dat.dtype)
    samp = torch.tensor((samp,) * vx.numel(), dtype=dat.dtype, device=dat.device)
    fwhm = torch.sqrt(torch.max(samp ** 2 - vx ** 2,
        torch.zeros(vx.numel(), device=dat.device, dtype=dat.dtype))) / vx
    smo = smooth(('gauss',) * vx.numel(),
        fwhm=fwhm, device=dat.device, dtype=dat.dtype, sep=True)
    # Padding amount for subsequent convolution
    size_pad = [s.shape[2 + i] for i, s in enumerate(smo)]
    size_pad = tuple(map(lambda x: (x - 1)//2, size_pad))
    # Smooth deformation with Gaussian kernel (by separable convolution)
    dat = pad(dat, size_pad, side='both')
    dat = dat[None, None]
    if vx.numel() == 3:
        dat = F.conv3d(dat, smo[0])
        dat = F.conv3d(dat, smo[1])
        dat = F.conv3d(dat, smo[2])[0, 0]
    else:
        dat = F.conv2d(dat, smo[0])
        dat = F.conv2d(dat, smo[1])[0, 0]

    return dat


def _to_gradient_magnitudes(dat, mat, scl):
    """ Compute squared gradient magnitudes (modulated with scaling and voxel size).

    OBS: Replaces the image data in dat.

    Parameters
    ----------
    dat : (X, Y, Z) tensor_like
        Image data.
    mat : (4, 4) tensor_like
        Affine matrix.
    scl : (N, ) tensor_like
        Gradient scaling parameter.

    Returns
    ----------
    dat : (X, Y, Z) tensor_like
        Squared gradient magnitudes.

    """
    # Get voxel size
    vx = voxel_size(mat)
    gr = scl*im_gradient(dat, vx=vx, which='forward', bound='zero')
    # Square gradients
    gr = torch.sum(gr**2, dim=0)
    dat = gr

    return dat


def _zero_mean(q, q_shape):
    """Make q have zero mean across input images.

    Parameters
    ----------
    q : array_like | tensor_like
        Lie algebra of affine registration fit.
    q_shape : (N, Nq)
        Original shape of q.

    Returns
    ----------
    q : tensor_like
        Lie algebra of affine registration fit.

    """
    # Reshape
    q_shape0 = q.shape
    q = q.reshape(q_shape)
    # Make zero mean
    if isinstance(q, np.ndarray):
        # Numpy array
        q -= q.mean(axis=0)
    else:
        # PyTorch tensor
        q -= torch.mean(q, dim=0)
    # Reshape back
    q = q.reshape(q_shape0)

    return q
