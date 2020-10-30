"""Module for affine image registration.

TODO (primary):
* Comment
* demo_affine_reg.jupy
* Work on 2D as well
* Re-work reslice_dat so can use image data, not only paths.
  Maybe make run_affine_reg return image data and matrices (so to not load twice..)?

TODO (secondary):
* Use Bayesian optimisation
* More affine bases (YB's implementation)

"""


import math
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.nn import functional as F
from scipy.optimize import fmin_powell
from timeit import default_timer as timer
from ..tools.preproc import load_3d
from ..plot import show_slices
from ..core.kernels import smooth
from ..core.utils import pad
from ..tools.preproc import (modify_affine, reslice_dat, write_img)
from ..tools.spm import (affine, identity, noise_estimate, affine_basis, dexpm, mean_space)
from ..spatial import (grid_pull, im_gradient, voxel_size)


# Histogram-based cost functions
costs_hist = ['mi', 'ecc', 'nmi', 'ncc']


def apply2affine(pths, q, B, prefix='a_', dir_out=None):
    """ Apply registration result (q) to input scans header.

    OBS: If prefix='' and dir_out=None, overwrites input path.

    Parameters
    ----------
    pths : list[string]
        List of nibabel compatible paths.

    q : (C, Nq), tensor_like
        Lie parameterisation of registration results (C=channels, Nq=Number of bases).

    B : (4, 4, Nq), tensor_like
        Lie basis.

    prefix : str, default='a_'
        Filename prefix.

    dir_out : str, default=None
        Full path to directory where to write image.

    Returns
    ----------
    pths_out : list[string]
        List of paths after applying registration results.

    Notes
    ----------
    Good reference on nibabel i/o:
    https://www.nipreps.org/niworkflows/1.0.0rc2/_modules/niworkflows/utils/images.html

    """
    pths_out = []
    N = len(pths)
    for n in range(N):
        # Read image
        nii = nib.load(pths[n])
        affine = nii.affine
        M = torch.from_numpy(affine).type(q.dtype).to(q.device)
        # Get registration transform
        R = dexpm(q[n, ...], B)[0]
        # Apply registration transform
        M = M.solve(R)[0]  # R\M
        # Modify affine
        pth = modify_affine(pths[n], M, prefix=prefix, dir_out=dir_out)
        pths_out.append(pth)

    return pths_out


def get_affine_basis(basis='SE', dim=3, dtype=torch.float64, device='cpu'):
    """Get the affine basis used to parameterise the registration transform.

    Parameters
    ----------
    basis : str, default='SE'
        Basis type.

    dim : int, default=3
        Basis dimensions [2|3]

    dtype : torch.dtype, default=torch.float64
        Data type of function output.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    Returns
    ----------
    B : (4, 4, Nq), tensor_like
        Lie basis.

    Nq : int
        Number of Lie basis.

    """
    B = affine_basis(basis=basis, dim=dim, dtype=dtype, device=device)
    Nq = B.shape[-1]

    return B, Nq


def get_resliced_data(pths, q, mat_fix, dim_fix, B, dtype=torch.float32, device='cpu'):
    """Get tensor of image data, resliced to the fixed image.

    Parameters
    ----------

    pths : list[string]
        List of nibabel compatible paths.

    q : (C, Nq), tensor_like
        Lie parameterisation of registration results (C=channels, Nq=Number of bases).

    mat_fix : (4, 4), tensor_like
        Affine matrix of fixed image.

    dim_fix : (3,), tuple, list, tensor_like
        Image dimensions of fixed image.

    B : (4, 4, Nq), tensor_like
        Lie basis.

    dtype : torch.dtype, default=torch.float32
        Data type of function output.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    Returns
    ----------
    rdat : (X, Y, Z, C), tensor_like
        Resliced image data.

    """
    N = len(pths)
    mat_fix = mat_fix.type(torch.float64)
    # Reslice moving images
    rdat = torch.zeros(dim_fix + (N,), dtype=dtype, device=device)
    for n in range(N):
        # Read image
        nii = nib.load(pths[n])
        dat = torch.tensor(nii.get_fdata(), device=device, dtype=dtype)
        if torch.all(q[n, ...] == 0):
            rdat[..., n] = dat
        else:
            affine = nii.affine
            mat_mov = torch.from_numpy(affine).type(mat_fix.dtype).to(device)
            # Get coreg matrix
            R = dexpm(q[n, ...], B)[0]
            R = R.type(mat_fix.dtype)
            # Apply registration transform
            M = mat_mov.solve(R)[0].solve(mat_fix)[0]  # mat_fix\R\mat_mov
            # Do reslice
            rdat[..., n] = reslice_dat(dat[None, None, ...], M, dim_fix)

    return rdat


def reslice2fix(pths, q, mat_fix, dim_fix, B, prefix='ra_', device='cpu', dir_out=None):
    """Reslice the input files to the grid of the fixed image, using the registration transform (q).

    OBS: If prefix='' and dir_out=None, overwrites input path.

    Parameters
    ----------

    pths : list[string]
        List of nibabel compatible paths.

    q : (C, Nq), tensor_like
        Lie parameterisation of registration results (C=channels, Nq=Number of bases).

    mat_fix : (4, 4), tensor_like
        Affine matrix of fixed image.

    dim_fix : (3,), tuple, list, tensor_like
        Image dimensions of fixed image.

    B : (4, 4, Nq), tensor_like
        Lie basis.

    prefix : str, default='ra_'
        Filename prefix.

    device : torch.device or str, default='cpu'
        PyTorch device type.

    dir_out : str, default=None
        Full path to directory where to write image.

    Returns
    ----------
    pths_out : list[string]
        List of paths after applying reslice.

    """
    # Reslice moving images
    pths_out = []
    N = len(pths)
    mat_fix = mat_fix.type(torch.float64)
    for n in range(N):  # Loop over moving images
        nii = nib.load(pths[n])
        affine = nii.affine
        mat_mov = torch.from_numpy(affine).type(mat_fix.dtype).to(device)
        dat = torch.tensor(nii.get_fdata(), device=device)
        if not torch.all(q[n, ...] == 0):
            # Get coreg matrix
            R = dexpm(q[n, ...], B)[0]
            R = R.type(mat_fix.dtype)
            # Do reslice
            M = R.mm(mat_fix).solve(mat_mov)[0]
            dat = reslice_dat(dat[None, None, ...], M, dim_fix)
        # Write resliced data
        pth = write_img(pths[n], dat=dat, prefix=prefix, affine=mat_fix, dir_out=dir_out)
        pths_out.append(pth)

    return pths_out


def run_affine_reg(imgs, optimiser='powell', cost_fun='nmi', device='cpu',
                   samp=(4, 2), fix=0, mean_space=True, verbose=False):
    """Affinely align images.

    Parameters
    ----------
    imgs : [str, ...] | [(X, Y, Z), tensor_like, ...] | [((X, Y, Z), tensor_like, (4 4), tensor_like), ...]
        Can be:
        - List of paths to nibabel compatible files
        - List of 3D image volume ass tensors
        - List of tuples, where each tuple has two tensors: 3D image volume and affine matrix.

    """
    # Set cost_fun function
    if cost_fun in costs_hist:
        mean_space = False
        mx_int = 255
    elif cost_fun == 'njtv':
        mx_int = None
    dtype = torch.float32

    # Torch stuff
    device = _init_torch(device)

    # Load data
    dat, mats, dims = _data_loader(imgs, device, samp[-1], mx_int, dtype=dtype)
    N = len(dat) # Number of input scans
    if N < 2:
        raise ValueError('At least two images needed!')

    if cost_fun == 'njtv':
        # Compute gradient magnitudes
        dat = _to_gradient_magnitudes(dat, mats)

    # Rigid registration basis
    B, Nq = get_affine_basis(device=device, dtype=dtype)

    # Initial guess
    q = torch.zeros((N, Nq), dtype=dtype).numpy()  # Because scipy.optimize.fmin_powell works on ndarrays

    #
    mov = list(range(N))

    if mean_space:
        # TODO
        M_fix, dim_fix, vx_fix = _get_mean_space(mats, dims)
        arg = dim_fix
    else:
        M_fix = mats[fix]
        vx_fix = voxel_size(M_fix)
        dim_fix = dat[fix].shape[:3]
        mov.remove(fix)
        arg = dat[fix]

    # Do registration
    fig_ax = None
    for s in samp:

        dat_fix, grid = _get_dat_grid(arg, vx_fix, s, device, dtype)

        if cost_fun == 'njtv':
            # Arguments to _compute_cost
            m = mov
            args = (grid, dat_fix, M_fix, dat, mats, m, cost_fun, B)
            # Run groupwise optimisation
            q[m, ...] = _optimise(q[m, ...], optimiser, Nq, args=args, mean_space=mean_space)
            # Show results (if verbose=True)
            fig_ax = _show_results(verbose, q[m, ...], args, cost_fun, fig_ax)
        else:
            for m in mov:
                # Arguments to _compute_cost
                args = (grid, dat_fix, M_fix, dat, mats, [m], cost_fun, B)
                # Run pairwise optimisation
                q[m, ...] = _optimise(q[m, ...], optimiser, Nq, args=args)
                # Show results (if verbose=True)
                fig_ax = _show_results(verbose, q[m, ...], args, cost_fun, fig_ax)

    # Blabla
    q = torch.from_numpy(q).to(device)

    return q, M_fix, dim_fix, B


def _affine2grid(grid, M):
    """

    """
    dm = grid.shape[:3]
    grid = torch.reshape(grid, (dm[0] * dm[1] * dm[2], 3))
    grid = torch.mm(grid, torch.t(M[:3, :3])) + torch.t(M[:3, 3])
    grid = torch.reshape(grid, (dm[0], dm[1], dm[2], 3))

    return grid


def _compute_cost(q, grid0, dat_fix, M_fix, dat, mats, mov, cost_fun, B, return_res=False):
    """

    """
    device = grid0.device
    dtype = grid0.dtype
    q = q.flatten()
    q = torch.from_numpy(q).to(device).type(dtype)  # To torch tensor
    N = torch.tensor(len(dat), device=device, dtype=dtype)
    dm_fix = dat_fix.shape
    Nq = B.shape[-1]

    if cost_fun == 'njtv':
        # Compute squared gradient magnitude for fixed
        njtv = -dat_fix.sqrt()
        mtv = dat_fix.clone()

    for i, m in enumerate(mov):
        R = dexpm(q[torch.arange(i*Nq,i*Nq + Nq)], B)[0]
        M = R.mm(M_fix).solve(mats[m])[0].type(dtype)  # M_mov\R*M_fix
        grid = _affine2grid(grid0, M)
        dat_new = grid_pull(dat[m][None, None, ...], grid[None, ...],
            bound='dft', extrapolate=True, interpolation=1)[0, 0, ...]

        if cost_fun == 'njtv':
            mtv += dat_new
            njtv =- dat_new.sqrt()

    # Compute the cost function
    # ----------
    res = None
    if cost_fun in costs_hist:
        # Histogram based costs
        # ----------
        # Compute joint histogram
        H = _hist_2d(dat_fix, dat_new)
        res = H

        # Get probabilities
        pxy = H / H.sum()
        px = pxy.sum(dim=0, keepdim=True)
        py = pxy.sum(dim=1, keepdim=True)

        # Compute cost
        if cost_fun == 'mi':
            # Mutual information
            mi = torch.sum(pxy * torch.log2(pxy / py.mm(px)))
            c = -mi
        elif cost_fun == 'ecc':
            # Entropy Correlation Coefficient
            mi = torch.sum(pxy * torch.log2(pxy / py.mm(px)))
            ecc = -2*mi/(torch.sum(px*px.log2()) + torch.sum(py*py.log2()))
            c = -ecc
        elif cost_fun == 'nmi':
            # Normalised Mutual Information
            nmi = (torch.sum(px*px.log2()) + torch.sum(py*py.log2()))/torch.sum(pxy*pxy.log2())
            c = -nmi
        elif cost_fun == 'ncc':
            # Normalised Cross Correlation
            i = torch.arange(1, pxy.shape[0] + 1, device=device, dtype=dtype)
            j = torch.arange(1, pxy.shape[1] + 1, device=device, dtype=dtype)
            m1 = torch.sum(py*i[..., None])
            m2 = torch.sum(px*j[None, ...])
            sig1 = torch.sqrt(torch.sum(py[..., 0]*(i - m1)**2))
            sig2 = torch.sqrt(torch.sum(px[0, ...]*(j - m2)**2))
            i, j = torch.meshgrid(i - m1, j - m2)
            ncc = torch.sum(torch.sum(pxy*i*j))/(sig1*sig2)
            c = -ncc
    elif cost_fun == 'njtv':
        # Total variation based cost
        njtv += torch.sqrt(N)*mtv.sqrt()
        res = njtv
        c = torch.sum(njtv)

    # To numpy array
    c = c.cpu().numpy()

    if return_res:
        return c, res
    else:
        return c


def _data_loader(imgs, device, samp, mx_int, do_smooth=True, dtype=torch.float32):
    """

    """
    dat = []
    mats = []
    dims = []
    for img in imgs:
        # Load data
        dat_p, M_p, _, _, _ = load_3d(img,
            samp=0, do_mask=False, truncate=True, mx_out=mx_int,
            device=device, dtype=dtype, do_smooth=do_smooth)
        dim_p =  torch.tensor(dat_p.shape[:3], dtype=dtype)

        # Set affine data type
        M_p = M_p.type(dtype)

        # Smooth
        dat_p = _smooth_for_reg(dat_p, voxel_size(M_p), samp)

        # Append
        dat.append(dat_p)
        mats.append(M_p)
        dims.append(dim_p)

    return dat, mats, dims


def _get_dat_grid(dat, vx, samp, device, dtype, jitter=True):
    """ Blabla.

    """

    # Modulate samp with voxel size
    samp = torch.tensor((samp,) * 3).float().to(device)
    samp = tuple((samp / vx).int().tolist())

    # Create grid of fixed image, possibly sub-sampled
    if isinstance(dat, torch.Tensor):
        dm = dat.shape  # tensor
        grid = identity(dm,
            dtype=dtype, device=device, jitter=jitter, step=samp)
    else:
        dm = dat  # tuple
        grid = identity(dm,
            dtype=dtype, device=device, jitter=jitter, step=samp)
        dat = torch.zeros(dm, dtype=dtype, device=device)

    # Sub-sampled fixed image
    dat_samp = dat[::samp[0], ::samp[1], ::samp[2]]

    return dat_samp, grid


def _get_mean_space(mats, dims):
    """

    """
    dtype = mats[0].dtype
    device = mats[0].device
    N = len(mats)
    Mat = torch.zeros((4, 4, N), dtype=dtype, device=device)
    Dim = torch.zeros((3, N), dtype=dtype, device=device)

    for i in range(N):
        Mat[..., i] = mats[i].clone()
        Dim[..., i] = dims[i].clone()

    dim, mat, vx = mean_space(Mat, Dim)
    dim = tuple(dim.cpu().int().tolist())
    mat = mat.type(dtype)

    return mat, dim, vx


def _hist_2d(img0, img1, fwhm=7, fig_num=0, mx_int=255):
    """ Make 2D histogram

        Prerequisites:
        * Images same size
        * Images same min and max intensities

        # Naive method (iterate)
        dat_mov = dat_mov.flatten().int()
        dat_fix = dat_fix.flatten().int()
        h = torch.zeros((mx_int + 1, mx_int + 1),
                        device=device, dtype=torch.float32)
        for n in range(num_vox):
            i1 = dat_mov[n]
            i2 = dat_fix[n]
            h[i1, i2] += 1

    """
    # if not (img0.dtype == torch.int64 and img1.dtype == torch.int64):
        # raise ValueError('Input should be torch.LongTensor')

    fwhm = (fwhm,) * 2

    # Convert each 'coordinate' of intensities to an index
    # (replicates the sub2ind function of MATLAB)
    img0 = img0.flatten().floor()
    img1 = img1.flatten().floor()
    sub = torch.stack((img0, img1),
                      dim=1)  # (num_vox, 2)
    to_ind = torch.tensor((1, mx_int + 1),
                          dtype=sub.dtype, device=img0.device)[..., None]  # (2, 1)
    ind = torch.tensordot(sub, to_ind, dims=([1], [0]))  # (nvox, 1)

    # Build histogram H by adding up counts according to the indicies in ind
    H = torch.zeros(mx_int + 1, mx_int + 1, device=img0.device, dtype=ind.dtype)
    H.put_(ind.long(), torch.ones(1, device=img0.device, dtype=ind.dtype).expand_as(ind), accumulate=True)

    # Smooth the histogram
    smo = smooth(('gauss',) * 2, fwhm=fwhm, device=img0.device, dtype=torch.float32, sep=True)
    # Padding amount for subsequent convolution
    size_pad = (smo[0].shape[2], smo[1].shape[3])
    size_pad = (torch.tensor(size_pad) - 1) // 2
    size_pad = tuple(size_pad.int().cpu().tolist())
    # Smooth deformation with Gaussian kernel (by convolving)
    H = pad(H, size_pad, side='both')
    H = H[None, None, ...]
    H = F.conv1d(H, smo[0])
    H = F.conv1d(H, smo[1])[0, 0, ...]

    # Add eps
    H = H + 1e-7

    if fig_num > 0:
        # Visualise histogram
        plt.figure(num=fig_num)
        plt.imshow(H,
            cmap='coolwarm', interpolation='nearest',
            aspect='equal', vmax=0.05*H.max())
        plt.axis('off')
        plt.show()

    return H


def _init_torch(device='cpu'):
    """


    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    return device


def _optimise(q, optimiser, Nq, args, mean_space=False):
    """

    """
    # Run optimisation
    if optimiser == 'powell':
        # N = len(q)
        # tol = np.concatenate([0.02*np.ones(3), 0.001*np.ones(9)])
        # tol = tol[:Nq]
        # C = N // Nq
        # tol = np.tile(tol, C)
        # direc = np.diag(20*tol)
        direc = None

        callback = None
        if mean_space:
            # TODO: zero mean
            callback = lambda x: _zero_mean(x, Nq)

        s = q.shape
        q = fmin_powell(_compute_cost, q, args=args, disp=False, callback=callback, direc=direc)
        q = q.reshape(s)

    return q


def _show_results(verbose, q, args, cost_fun, fig_ax):
    """

    """
    if verbose:
        # TODO: use function here, and have it inside gw/pw
        c, res = _compute_cost(q, *args, return_res=True)
        print('_compute_cost({})={}'.format(cost_fun, c))
        fig_ax = show_slices(res, fig_ax=fig_ax, fig_num=1, cmap='coolwarm')
        # nii = nib.Nifti1Image(res.cpu().numpy(), M_fix.cpu().numpy())
        # dir_out = os.path.split(pths[0])[0]
        # nib.save(nii, os.path.join(dir_out, 'res_coreg.nii'))

    return fig_ax


def _smooth_for_reg(dat, vx, samp=1):
    """Smoothing for image registration. FWHM is computed from voxel size
       and sub-sampling amount.

        Parameters
        ----------
        dat : (Nx, Ny, Nz) tensor_like
            3D image volume.
        vx : (3,) tensor_like
            Voxel size.
        samp : int, default=1
            Sub-sampling.

        Returns
        -------
        dat : (Nx, Ny, Nz) tensor_like
            Smoothed 3D image volume.

        """
    if samp < 1:
        samp = 1
    samp = torch.tensor((samp,) * 3, dtype=dat.dtype, device=dat.device)
    vx = vx.type(dat.dtype)
    # Make smoothing kernel
    fwhm = torch.sqrt(torch.max(samp ** 2 - vx ** 2, torch.zeros(3, device=dat.device, dtype=dat.dtype))) / vx
    smo = smooth(('gauss',) * 3, fwhm=fwhm, device=dat.device, dtype=dat.dtype, sep=True)
    # Padding amount for subsequent convolution
    size_pad = (smo[0].shape[2], smo[1].shape[3], smo[2].shape[4])
    size_pad = (torch.tensor(size_pad) - 1) // 2
    size_pad = tuple(size_pad.int().cpu().tolist())
    # Smooth deformation with Gaussian kernel (by separable convolution)
    dat = pad(dat, size_pad, side='both')
    dat = dat[None, None, ...]
    dat = F.conv1d(dat, smo[0])
    dat = F.conv1d(dat, smo[1])
    dat = F.conv1d(dat, smo[2])[0, 0, ...]

    return dat


def _to_gradient_magnitudes(dat, M):
    """ Compute squared gradient magnitudes (inc. scaling and voxel size)

    """
    for i in range(len(dat)):
        vx = voxel_size(M[i])
        _, _, mu_bg, mu_fg = noise_estimate(dat[i], show_fit=False)
        scl = 1/torch.abs(mu_fg.float() - mu_bg.float())
        gr = scl*im_gradient(dat[i], vx=vx, which='forward', bound='circular')
        gr = torch.sum(gr**2, dim=0)
        dat[i] = gr

    return dat


def _verify_cost_function(pths, fix=0, cost_fun='nmi', samp=2,
    device='cpu', ix_par=0, mean_space=False, jitter=True, verbose=False,
    x_step=1, x_mn_mx=20):
    """Blabla.

    """
    N = len(pths)
    if N != 2:
        raise ValueError('Two input files required!')

    # Set cost_fun function
    if cost_fun in costs_hist:
        mean_space = False
        mx_int = 255
    elif cost_fun == 'njtv':
        mx_int = None
    dtype = torch.float32

    # Set device
    device = _init_torch(device)

    # Load data
    dat, mats, dims = _data_loader(pths, device, samp, mx_int, dtype=dtype)

    if cost_fun == 'njtv':
        # Compute gradient magnitudes
        dat = _to_gradient_magnitudes(dat, mats)

    # Get affine basis
    B, Nq = get_affine_basis(device=device, dtype=dtype)

    # Initial guess
    q = torch.zeros((N, Nq), dtype=dtype)
    q = q.numpy()  # Because scipy.optimize.fmin_powell works on ndarrays
    q = q.flatten()

    mov = list(range(N))

    x = np.arange(-x_mn_mx, x_mn_mx, x_step)

    mat_mn = torch.zeros(Nq, dtype=dtype, device=device)
    mat_mx = torch.zeros(Nq, dtype=dtype, device=device)
    mat_mn[ix_par] = -x_mn_mx
    mat_mx[ix_par] = x_mn_mx
    mats1 = [dexpm(mat_mn, B)[0].mm(mats[mov[0]]), dexpm(mat_mx, B)[0].mm(mats[mov[0]])]
    dims1 = [dims[mov[0]], dims[mov[0]]]

    if mean_space:
        # TODO
        M_fix, dim_fix, vx_fix = _get_mean_space(mats + mats1, dims + dims1)
        arg = dim_fix
    else:
        M_fix = mats[fix]
        vx_fix = voxel_size(M_fix)
        dim_fix = dat[fix].shape[:3]
        mov.remove(fix)
        arg = dat[fix]

    dat_fix, grid = _get_dat_grid(arg, vx_fix, samp, device, dtype, jitter=jitter)

    # Iterate over a range of values
    costs = np.zeros(len(x))
    fig_ax = None
    for i, xi in enumerate(x):
        # Compute cost
        q[ix_par] = xi
        costs[i], res = _compute_cost(q, grid, dat_fix, M_fix, dat, mats, mov, cost_fun, B, return_res=True)
        if verbose:
            fig_ax = show_slices(res, fig_ax=fig_ax, fig_num=2, cmap='coolwarm', title='x=' + str(x))

    # Plot all costs
    plt.clf()
    fig, ax = plt.subplots(num=1)
    ax.plot(x, costs)
    ax.set(xlabel='Value q[' + str(ix_par) + ']', ylabel='Cost',
           title=cost_fun.upper() + ' cost function (mean_space= ' + str(mean_space) + ')')
    ax.grid()
    plt.show()


def _zero_mean(q, Nq):
    """

    """
    N = len(q)
    C = N//Nq
    q = q.reshape((C, Nq))
    q -= q.mean(axis=0)
    q = q.flatten()

    return q