"""Module for affine image registration.

TODO
* Use Bayesian optimisation
* More affine bases (YB's implementation?)
* YB's MI implementation? Other YB code that could be re-used?
* Work on 2D as well?

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
from ..tools.preproc import (get_origin_mat, modify_affine, reslice_dat, write_img)
from ..tools.spm import (affine, identity, noise_estimate, mean_space)
from ..spatial import (affine_basis, affine_matvec, grid_pull, im_gradient, voxel_size)
from ..core._linalg_expm import expm


# Histogram-based cost functions
costs_hist = ['mi', 'ecc', 'nmi', 'ncc']


def run_affine_reg(imgs, cost_fun='nmi', group='SE', mean_space=False,
                   samp=(4, 2), optimiser='powell', fix=0, verbose=False,
                   fov=None, device='cpu'):
    """Affinely align images.

    This function aligns N images affinely, either pairwise or groupwise,
    by non-gradient based optimisation. An affine transformation has maximum
    12 parameters that control: translation, rotation, scaling and shearing.
    It is a linear registartion.

    The transformation model is:
        M_mov\A*M_fix,
    where M_mov is the affine matrix of the source/moving image, M_fix is the affine matrix
    of the fixed/target image, and A is the affine matrix that we optimise to align voxels
    in the source and target image. The affine matrix is represented by its Lie algebra (q),
    which a vector with as many parameters as the affine transformation.

    When running the algorithm in a pair-wise setting (i.e., the cost-function takes as input two images),
    one image is set as fixed, and all other images are registered to this fixed image. When running the
    algorithm in a groupwese setting (i.e., the cost-function takes as input all images) two options are
    available:
        1. One of the input images are used as the fixed image and all others are aligned to this image.
        2. A mean-space is defined, containing all of the input images, and all images are optimised
           to aling in this mean-space.

    At the end the affine transformation is returned, together with the defintion of the fixed space, and
    the affine group used. For getting the data resliced to the fixed space, look at the function reslice2fix().
    To adjust the affine in the image header, look at apply2affine().

    The registration methods used here are described in:
    A Collignon, F Maes, D Delaere, D Vandermeulen, P Suetens & G Marchal (1995)
    "Automated Multi-modality Image Registration Based On Information Theory", IPMI
    and:
    M Brudfors, Y Balbastre, J Ashburner (2020) "Groupwise Multimodal Image
    Registration using Joint Total Variation", MIUA

    Parameters
    ----------
    imgs : [N,] str, List of paths to nibabel compatible files
           [N,] tensor_like, List of 3D image volume ass tensors
           [N,] (tensor_like, tensor_like) List of tuples, where
            each tuple has two tensors: 3D image volume and affine matrix.
    cost_fun : str, default='nmi'
        * 'nmi' : Normalised Mutual Information (pairwise method)
        * 'mi' : Mutual Information (pairwise method)
        * 'ncc' : Normalised Cross Correlation (pairwise method)
        * 'ecc' : Entropy Correlation Coefficient (pairwise method)
        * 'njtv' : Normalised Joint Total variation (groupwise method)
    group : str, default='SE'
        * 'T'   : Translations
        * 'SO'  : Special Orthogonal (rotations)
        * 'SE'  : Special Euclidean (translations + rotations)
        * 'D'   : Dilations (translations + isotropic scalings)
        * 'CSO' : Conformal Special Orthogonal
                  (translations + rotations + isotropic scalings)
        * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
        * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
        * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
    mean_space : bool, default=False
        Optimise a mean-space fit, only available if cost_fun='njtv' .
    samp : (3,) float, default=(4, 2)
        Optimisation sampling steps (mm).
    optimiser : str, default='powell'
        'powell' : Optimisation method.
    fix : int, default=0
        Index of image to used as fixed image, not used if mean_space=True.
    fov : (2,) tuple, default=None
        A tuple with affine matrix (tensor_like) and dimensions (tuple) of mean space.
    verbose : bool, default=False
        Show registration results.
    device : torch.device or str, default='cpu'
        PyTorch device type.

    Returns
    ----------
    q : (N, Nq) tensor_like (N=number of input images, Nq=number affine bases)
        Lie algebra of affine registration fit. Get matrix representation by:
            A = nitorch.core.linalg.expm(q[n, ...], B)
    M_fix : (4, 4), tensor_like
        Affine matrix of fixed image.
    dim_fix : (3,), tuple, list, tensor_like
        Image dimensions of fixed image.

    """
    # Parse algorithm options
    opt = {'optimiser': optimiser,
           'cost_fun': cost_fun,
           'device': device,
           'samp': samp,
           'fix': fix,
           'mean_space': mean_space,
           'verbose': verbose,
           'fov': fov,
           'group' : group}

    # Some very basic sanity checks
    N = len(imgs) # Number of input scans
    mov = list(range(N))  # Indices of images
    if N < 2:
        raise ValueError('At least two images needed!')
    if opt['cost_fun'] in costs_hist and opt['mean_space']:
        raise ValueError('Option mean_space=True not defined for {} cost!'.format(opt['cost_fun']))

    # Get affine basis
    B = affine_basis(group=opt['group'], device=opt['device'])
    Nq = B.shape[0]

    # Load data
    dat, mats, dims = _data_loader(imgs, opt)

    # Define fixed image space (M_fix, dim_fix, vx_fix)
    if opt['mean_space']:
        # Use a mean-space
        if opt['fov']:
            # Mean-space given
            M_fix = opt['fov'][0]
            dim_fix = opt['fov'][1]
        else:
            # Compute mean-space
            M_fix, dim_fix = _get_mean_space(mats, dims)
        arg_dat_grid = dim_fix
    else:
        # Use one of the input images
        M_fix = mats[opt['fix']]
        dim_fix = dat[opt['fix']].shape[:3]
        mov.remove(opt['fix'])
        arg_dat_grid = dat[opt['fix']]
    # Get voxel size of fixed image
    vx_fix = voxel_size(M_fix)

    # Initial guess for registration parameter
    q = torch.zeros((N, Nq), dtype=torch.float32, device=opt['device'])

    # Do registration
    for s in opt['samp']:  # Loop over sub-sampling level
        # Get possibly sub-sampled fixed image, and its resampling grid
        dat_fix, grid = _get_dat_grid(arg_dat_grid, vx_fix, s, opt)
        # Do optimisation
        q, args = _optimise(q, dat_fix, grid, M_fix, dat, mats, mov, B, opt)

    return q, M_fix, dim_fix


def apply2affine(pths, q, group='SE', prefix='a_', dir_out=None):
    """ Apply registration result (q) to affine in input scans header.

    OBS: If prefix='' and dir_out=None, overwrites input data.

    Parameters
    ----------
    pths : [N,] str
        List of nibabel compatible paths.
    q : (N, Nq), tensor_like
        Lie parameterisation of registration results (N=channels, Nq=Number of bases).
    group : str, default='SE'
        OBS: Should be same that was used in call to run_affine_reg().
    prefix : str, default='a_'
        Filename prefix.
    dir_out : str, default=None
        Full path to directory where to write image.

    Returns
    ----------
    pths_out : [N,] str
        List of paths after applying registration results.

    Notes
    ----------
    Good reference on nibabel i/o:
    https://www.nipreps.org/niworkflows/1.0.0rc2/_modules/niworkflows/utils/images.html

    """
    # Sanity check
    if (not isinstance(pths, list) and not isinstance(pths[0], str)) \
        or (isinstance(pths, list) and not isinstance(pths[0], str)):
        raise ValueError('Input should be list of strings!')
    # Get affine basis
    B = affine_basis(group=group, device=q.device)
    # Incorporate registration result
    pths_out = []
    N = len(pths)
    for n in range(N):  # Loop over input images
        # Read image
        nii = nib.load(pths[n])
        affine = nii.affine
        M = torch.from_numpy(affine).type(q.dtype).to(q.device)
        # Get registration transform
        A = expm(q[n, ...], B)
        # Apply registration transform
        M = M.solve(A)[0]  # A\M
        # Modify affine
        pth = modify_affine(pths[n], M, prefix=prefix, dir_out=dir_out)
        pths_out.append(pth)

    return pths_out


def reslice2fix(imgs, q, M_fix, dim_fix, group='SE', prefix='ra_', dtype=torch.float32,
                device='cpu', dir_out=None, write=False):
    """Reslice the input files to the grid of the fixed image,
    using the registration transform (q).

    OBS: If write=True, prefix='' and dir_out=None, overwrites input data.

    Parameters
    ----------
    imgs : [N,] str, List of paths to nibabel compatible files
           [N,] (tensor_like, tensor_like) List of tuples, where
            each tuple has two tensors: 3D image volume and affine matrix.
    q : (N, Nq), tensor_like
        Lie parameterisation of registration results (N=channels, Nq=Number of bases).
    M_fix : (4, 4), tensor_like
        Affine matrix of fixed image.
    dim_fix : (3,), tuple, list, tensor_like
        Image dimensions of fixed image.
    group : str, default='SE'
        OBS: Should be same that was used in call to run_affine_reg().
    prefix : str, default='ra_'
        Filename prefix.
    dtype : torch.dtype, default=torch.float32
        Data type of function output.
    device : torch.device or str, default='cpu'
        PyTorch device type.
    dir_out : str, default=None
        Full path to directory where to write image.
    write : bool, default=False
        Write resliced data to disk, if inputs are given as paths to nibabel data.

    Returns
    ----------
    rdat : (X, Y, Z, N), tensor_like
        Resliced image data.
    pths_out : [N,] str, default=[]
        List of paths after applying reslice, if write=True.

    """
    pths_out = []
    N = len(imgs)
    M_fix = M_fix.type(torch.float64)  # affine of resliced data
    rdat = torch.zeros(dim_fix + (N,), dtype=dtype, device=device)  # tensor for resliced data
    # Get affine basis
    B = affine_basis(group=group, device=device)
    for n in range(N):  # Loop over input images
        if isinstance(imgs[n], str):
            # Input is nibabel path
            nii = nib.load(imgs[n])
            dat = torch.tensor(nii.get_fdata(), device=device)
            M_mov = torch.from_numpy(nii.affine).type(M_fix.dtype).to(device)
        else:
            # Input is image data, and possibly affine matrix
            write = False  # Do not write to disk
            if isinstance(imgs[n], tuple):
                dat = imgs[n][0]
                M_mov = imgs[n][1]
            else:
                dat = imgs[n]
                M_mov = _get_origin_mat(imgs[n].shape, device=device, dtype=dtype)
        if not torch.all(q[n, ...] == 0):
            # Get coreg matrix
            A = expm(q[n, ...], B)
            A = A.type(M_fix.dtype)
            # Do reslice
            M = A.mm(M_fix).solve(M_mov)[0]  # M_mov\A*M_fix
            rdat[..., n] = reslice_dat(dat[None, None, ...], M, dim_fix)
        else:
            # Fixed image was among input images, do not reslice
            rdat[..., n] = dat
        if write:
            # Write resliced data
            pth = write_img(imgs[n], dat=rdat[..., n], prefix=prefix, affine=M_fix, dir_out=dir_out)
            pths_out.append(pth)

    return rdat, pths_out


def test_cost_function(pths, cost_fun='nmi', group='SE', mean_space=False, samp=2, ix_par=0, jitter=True,
    x_step=0.1, x_mn_mx=30, verbose=False, device='cpu'):
    """Check cost function behaviour by keeping one image fixed and re-aligning
       a second image by modifying one of the affine parameters. Plots cost vs.
       aligment when finished.

    Parameters
    ----------
    pths :[2,] str
        Paths to two nibabel compatible image volumes.
    cost_fun : str, default='nmi'
        * 'nmi' : Normalised Mutual Information (pairwise method)
        * 'mi' : Mutual Information (pairwise method)
        * 'ncc' : Normalised Cross Correlation (pairwise method)
        * 'ecc' : Entropy Correlation Coefficient (pairwise method)
        * 'njtv' : Normalised Joint Total variation (groupwise method)
    mean_space : bool, default=False
        Optimise a mean-space fit, only available if cost_fun='njtv' .
    samp : float, default=2
        Otimisation sampling steps (mm).
    ix_par : int, default=0
        Affine parameter to modify (0 <= ix_par <= 11.)
    jitter : bool, default=True
        Add random jittering to resampling grid.
    x_step : int, default=1
        Step-size when changing the parameter value.
    x_mn_mx : float, default=30
        Min/max value of parameter.
    verbose : bool, default=False
        Show registration results.
    device : torch.device or str, default='cpu'
        PyTorch device type.

    """
    # Parse algorithm options
    opt = {'cost_fun': cost_fun,
           'device': device,
           'samp': samp,
           'mean_space': mean_space,
           'verbose': verbose}

    # Some very basic sanity checks
    N = len(pths)
    mov = list(range(N))  # Indices of images
    fix_img = 0
    mov_img = 1
    if N != 2:
        raise ValueError('Two input files required!')
    if opt['cost_fun'] in costs_hist and opt['mean_space']:
        raise ValueError('Option mean_space=True not defined for {} cost!'.format(opt['cost_fun']))

    # Load data
    dat, mats, dims = _data_loader(pths, opt)

    # Get full 12 parameter affine basis
    B = affine_basis(group='Aff+', device=opt['device'])
    Nq = B.shape[0]

    # Range of parameter
    x = torch.arange(start=-x_mn_mx, end=x_mn_mx, step=x_step, dtype=torch.float32)

    if opt['mean_space']:
        # Use mean-space, so make sure that maximum misalignment is represented
        # in the input to _get_mean_space()
        mat_mn = torch.zeros(Nq, dtype=torch.float32, device=opt['device'])
        mat_mx = torch.zeros(Nq, dtype=torch.float32, device=opt['device'])
        mat_mn[ix_par] = -x_mn_mx
        mat_mx[ix_par] = x_mn_mx
        mats1 = [expm(mat_mn, B).mm(mats[mov_img]), expm(mat_mx, B).mm(mats[mov_img])]
        dims1 = [dims[mov_img], dims[mov_img]]
        # Compute mean-space
        M_fix, dim_fix = _get_mean_space(mats + mats1, dims + dims1)
        arg_dat_grid = dim_fix
    else:
        M_fix = mats[fix_img]
        dim_fix = dat[fix_img].shape[:3]
        mov.remove(fix_img)
        arg_dat_grid = dat[fix_img]
    # Get voxel size of fixed image
    vx_fix = voxel_size(M_fix)

    # Initial guess
    q = torch.zeros((N, Nq), dtype=torch.float32)

    # Get subsampled fixed image and its resampling grid
    dat_fix, grid = _get_dat_grid(arg_dat_grid, vx_fix, opt['samp'], opt, jitter=jitter)

    # Iterate over a range of values
    costs = np.zeros(len(x))
    fig_ax = None  # Used for visualisation
    for i, xi in enumerate(x):
        # Change affine matrix a little bit
        q[fix_img, ix_par] = xi
        # Compute cost
        costs[i], res = _compute_cost(
            q, grid, dat_fix, M_fix, dat, mats, mov, opt['cost_fun'], B, return_res=True)
        if opt['verbose']:
            fig_ax = show_slices(res, fig_ax=fig_ax, fig_num=1, cmap='coolwarm', title='x=' + str(xi))

    # Plot results
    fig, ax = plt.subplots(num=2)
    ax.plot(x, costs)
    ax.set(xlabel='Value q[' + str(ix_par) + ']', ylabel='Cost',
           title=opt['cost_fun'].upper() + ' cost function (mean_space=' + str(opt['mean_space']) + ')')
    ax.grid()
    plt.show()


def _compute_cost(q, grid0, dat_fix, M_fix, dat, mats, mov, cost_fun, B, return_res=False):
    """Compute registration cost function.

    Parameters
    ----------
    q : (N, Nq) tensor_like
        Lie algebra of affine registration fit.
    grid0 : (X1, Y1, Z1) tensor_like
        Sub-sampled image data's resampling grid.
    dat_fix : (X1, Y1, Z1) tensor_like
        Fixed image data.
    M_fix : (4, 4) tensor_like
        Fixed affine matrix.
    dat : [N,] tensor_like
        List of input images.
    mats : [N,] tensor_like
        List of affine matrices.
    mov : [N,] int
        Indices of moving images.
    cost_fun : str
        Cost function to compute (see run_affine_reg).
    B : (Nq, N, N) tensor_like
        Affine basis.
    return_res : bool, default=False
        Return registration results for plotting.

    Returns
    ----------
    c : float
        Cost of aligning images with current estimate of q. If optimiser='powell', array_like,
        else tensor_like.
    res : tensor_like
        Registration results, for visualisation (only if return_res=True).

    """
    # Init
    device = grid0.device
    dtype = grid0.dtype
    q = q.flatten()
    was_numpy = False
    if isinstance(q, np.ndarray):
        was_numpy = True
        q = torch.from_numpy(q).to(device).type(dtype)  # To torch tensor
    dm_fix = dat_fix.shape
    Nq = B.shape[0]

    if cost_fun == 'njtv':
        # Compute NJTV cost for fixed image
        N = torch.tensor(len(dat), device=device, dtype=dtype)  # For modulating cost later on
        njtv = -dat_fix.sqrt()
        mtv = dat_fix.clone()

    for i, m in enumerate(mov):  # Loop over moving images
        # Get affine matrix
        A = expm(q[torch.arange(i*Nq,i*Nq + Nq)], B)
        # Compose matrices
        M = A.mm(M_fix).solve(mats[m])[0].type(dtype)  # M_mov\A*M_fix
        # Transform fixed grid
        grid = affine_matvec(M, grid0)
        # Resample to fixed grid
        dat_new = grid_pull(dat[m][None, None, ...], grid[None, ...],
            bound='dft', extrapolate=True, interpolation=1)[0, 0, ...]
        if cost_fun == 'njtv':
            # Add to NJTV cost for moving image
            mtv += dat_new
            njtv -= dat_new.sqrt()

    # Compute the cost function
    res = None
    if cost_fun in costs_hist:  # Histogram based costs
        # Compute joint histogram
        # OBS: This function expects both images to have the same max and min intesities,
        # this is ensured by the _data_loader() function.
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
    elif cost_fun == 'njtv':  # Total variation based cost
        njtv += torch.sqrt(N)*mtv.sqrt()
        res = njtv
        c = torch.sum(njtv)

    if was_numpy:
        # Back to numpy array
        c = c.cpu().numpy()

    if return_res:
        return c, res
    else:
        return c


def _data_loader(imgs, opt):
    """Load image data, affine matrices, image dimensions.

    Parameters
    ----------
    imgs : list
        See run_affine_reg.
    opt : dict
        See run_affine_reg.

    Returns
    ----------
    dat : [N,] tensor_like
        List of image data.
    mats : [N,] tensor_like
        List of affine matrices.
    dims : [N,] tensor_like
        List of image dimensions.

    """
    if opt['cost_fun'] in costs_hist:
        # This parameter sets the max intensity in the images, which
        # decides how many bins to use in the joint image histograms
        # (e.g, mx_int=511 -> H.shape = (512, 512))
        mx_int = 511
    elif opt['cost_fun'] == 'njtv':
        mx_int = None

    dat = []
    mats = []
    dims = []
    for img in imgs:
        # Load data
        dat_p, M_p, _, _, _ = load_3d(img,
            samp=0, do_mask=False, truncate=True, mx_out=mx_int,
            device=opt['device'], dtype=torch.float32, do_smooth=True)
        dim_p =  torch.tensor(dat_p.shape[:3], dtype=torch.float32)

        # Set affine data type
        M_p = M_p.type(torch.float32)

        # Smooth
        dat_p = _smooth_for_reg(dat_p, voxel_size(M_p), opt)

        # Append
        dat.append(dat_p)
        mats.append(M_p)
        dims.append(dim_p)

    if opt['cost_fun'] == 'njtv':
        # Compute gradient magnitudes
        dat = _to_gradient_magnitudes(dat, mats)

    return dat, mats, dims


def _do_optimisation(q, Nq, args, opt):
    """Run optimisation routine to fit q.

    Parameters
    ----------
    q : (N, Nq) tensor_like
        Lie algebra of affine registration fit.
    Nq : int
        Number of Lie groups.
    args : tuple
        All arguments for optimiser (except the parameters to be fitted)
    opt : dict
        See run_affine_reg.

    Returns
    ----------
    q : (N, Nq) tensor_like
        Optimised Lie algebra of affine registration fit.

    """
    if opt['optimiser'] == 'powell':
        # Powell optimisation
        dtype = q.dtype
        q = q.cpu().numpy()  # SciPy's powell optimiser requires CPU Numpy arrays
        # Callback
        callback = None
        if opt['mean_space']:
            # Ensure that the paramters have zero mean, across images.
            callback = lambda x: _zero_mean(x, Nq)
        s = q.shape
        q = fmin_powell(_compute_cost, q, args=args, disp=False, callback=callback)
        q = q.reshape(s)
        q = torch.from_numpy(q).type(dtype).to(opt['device'])  # Cast back to tensor

    return q


def _get_dat_grid(dat, vx, samp, opt, jitter=True):
    """Get sub-sampled image data, and resampling grid.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensor_like
        Fixed image data.
    vx : (3,) tensor_like.
        Fixed voxel size.
    samp : int|float
        Sub-sampling level.
    opt : dict
        See run_affine_reg.
    jitter : bool, default=True
        Add random jittering to identity grid.

    Returns
    ----------
    dat_samp : (X1, Y1, Z1) tensor_like
        Sub-sampled fixed image data.
    grid : (X1, Y1, Z1) tensor_like
        Sub-sampled image data's resampling grid.

    """
    # Modulate samp with voxel size
    samp = torch.tensor((samp,) * 3).float().to(opt['device'])
    samp = tuple((torch.clamp(samp / vx,1)).int().tolist())
    # Create grid of fixed image, possibly sub-sampled
    if isinstance(dat, torch.Tensor):
        # Use one of the input images as fixed.
        dm = dat.shape  # tensor
        grid = identity(dm,
            dtype=torch.float32, device=opt['device'], jitter=jitter, step=samp)
    else:
        # Use mean-space, dat_samp is an image with zeros.
        dm = dat  # tuple
        grid = identity(dm,
            dtype=torch.float32, device=opt['device'], jitter=jitter, step=samp)
        dat = torch.zeros(dm, dtype=torch.float32, device=opt['device'])
    # Sub-sampled fixed image
    dat_samp = dat[::samp[0], ::samp[1], ::samp[2]]

    return dat_samp, grid


def _get_mean_space(mats, dims):
    """Compute mean space from images' field of views.

    Parameters
    ----------
    mats :  [N,] tensor_like
        List of affine matrices.
    dims :  [N,] tensor_like
        List of image dimensions.

    Returns
    ----------
    mats : (4, 4) tensor_like
        Mean affine matrix.
    dims : (3,) tuple
        Mean image dimensions.

    """
    # Copy matrices and dimensions to torch tensors
    dtype = mats[0].dtype
    device = mats[0].device
    N = len(mats)
    Mat = torch.zeros((4, 4, N), dtype=dtype, device=device)
    Dim = torch.zeros((3, N), dtype=dtype, device=device)
    for i in range(N):
        Mat[..., i] = mats[i].clone()
        Dim[..., i] = dims[i].clone()
    # Compute mean-space
    dim, mat, _ = mean_space(Mat, Dim)
    dim = tuple(dim.cpu().int().tolist())
    mat = mat.type(dtype)

    return mat, dim


def _hist_2d(img0, img1, fwhm=7, mx_int=511):
    """Make 2D histogram.

    Pre-requisites:
    * Images same size.
    * Images same min and max intensities.

    # Naive method for computing a 2D histogram
    h = torch.zeros((mx_int + 1, mx_int + 1))
    for n in range(num_vox):
        h[img0[n], mg1[n]] += 1

    Parameters
    ----------
    img0 : (X, Y, Z) tensor_like
        First image volume.
    img1 : (X, Y, Z) tensor_like
        Second image volume.
    fwhm : float, default=7
        Full-width at half max of Gaussian kernel, for smoothing
        histogram.
    mx_int : float, default=255
        Max intensity of input images.

    Returns
    ----------
    H : (mx_int + 1, mx_int + 1) tensor_like
        Joint intensity histogram.

    """
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
    H = F.conv2d(H, smo[0])
    H = F.conv2d(H, smo[1])[0, 0, ...]

    # Add eps
    H = H + 1e-7

    # # Visualise histogram
    # plt.figure(num=fig_num)
    # plt.imshow(H,
    #     cmap='coolwarm', interpolation='nearest',
    #     aspect='equal', vmax=0.05*H.max())
    # plt.axis('off')
    # plt.show()

    return H


def _optimise(q, dat_fix, grid, M_fix, dat, mats, mov, B, opt):
    '''Optimise q, either pairwise or groupwise.

    Parameters
    ----------
     q : (N, Nq) tensor_like
        Affine parameters.
    dat_fix : (X1, Y1, Z1) tensor_like
        Fixed image data.
    grid : (X1, Y1, Z1) tensor_like
        Sub-sampled image data's resampling grid.
    M_fix : (4, 4) tensor_like
        Fixed affine matrix.
    dat : [N,] tensor_like
        List of input images.
    mats : [N,] tensor_like
        List of affine matrices.
    mov : [N,] int
        Indices of moving images.
    B : (Nq, N, N) tensor_like
        Affine basis.
    opt : dict
        See run_affine_reg.

    Returns
    ----------
    q : (N, Nq) tensor_like
        Fitted affine parameters.
    args : tuple
        All arguments for optimiser (except the parameters to be fitted (q))

    '''
    Nq = B.shape[0]
    if opt['cost_fun'] == 'njtv':  # Groupwise optimisation
        # Arguments to _compute_cost
        m = mov
        args = (grid, dat_fix, M_fix, dat, mats, m, opt['cost_fun'], B)
        # Run groupwise optimisation
        q[m, ...] = _do_optimisation(q[m, ...], Nq, args, opt)
        # Show results
        _show_results(q[m, ...], args, opt)
    else:  # Pairwise optimisation
        for m in mov:
            # Arguments to _compute_cost
            args = (grid, dat_fix, M_fix, dat, mats, [m], opt['cost_fun'], B)
            # Run pairwise optimisation
            q[m, ...] = _do_optimisation(q[m, ...], Nq, args, opt)
            # Show results
            _show_results(q[m, ...], args, opt)

    return q, args


def _show_results(q, args, opt):
    """ Simple function for visualising some results during algorithm execution.
    """
    if opt['verbose']:
        c, res = _compute_cost(q, *args, return_res=True)
        print('_compute_cost({})={}'.format(opt['cost_fun'], c))
        _ = show_slices(res, fig_num=1, cmap='coolwarm')
        # nii = nib.Nifti1Image(res.cpu().numpy(), M_fix.cpu().numpy())
        # dir_out = os.path.split(pths[0])[0]
        # nib.save(nii, os.path.join(dir_out, 'res_coreg.nii'))


def _smooth_for_reg(dat, vx, opt):
    """Smoothing for image registration. FWHM is computed from voxel size
       and sub-sampling amount.

    Parameters
    ----------
    dat : (X, Y, Z) tensor_like
        3D image volume.
    vx : (3,) tensor_like
        Voxel size.
    opt : dict
        See run_affine_reg.

    Returns
    -------
    dat : (Nx, Ny, Nz) tensor_like
        Smoothed 3D image volume.

    """
    # Get final sub-sampling level
    samp = opt['samp']
    if not isinstance(samp, tuple):
        samp = (samp,)
    samp = samp[-1]
    if samp < 1:
        samp = 1
    samp = torch.tensor((samp,) * 3, dtype=dat.dtype, device=dat.device)
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
    dat = F.conv3d(dat, smo[0])
    dat = F.conv3d(dat, smo[1])
    dat = F.conv3d(dat, smo[2])[0, 0, ...]

    return dat


def _to_gradient_magnitudes(dat, M):
    """ Compute squared gradient magnitudes (modulated with scaling and voxel size).

    OBS: Replaces the image data in dat.

    Parameters
    ----------
    dat : [N,] tensor_like
        List of image volumes.
    M : [N,] tensor_like
        List of corresponding affine matrices.

    Returns
    ----------
    dat :  [N,] tensor_like
        List of squared gradient magnitudes.

    """
    for i in range(len(dat)):  # Loop over input images
        # Get voxel size
        vx = voxel_size(M[i])
        # Compute mean background and foreground intensitues
        _, _, mu_bg, mu_fg = noise_estimate(dat[i], show_fit=False)
        # Get scaling
        scl = 1/torch.abs(mu_fg.float() - mu_bg.float())
        # Compute gradients, multiplied by scale factor.
        gr = scl*im_gradient(dat[i], vx=vx, which='forward', bound='circular')
        # Square gradients
        gr = torch.sum(gr**2, dim=0)
        dat[i] = gr

    return dat


def _zero_mean(q, Nq):
    """Make q have zero mean across channels.

    Parameters
    ----------
    q : tensor_like
        Lie algebra of affine registration fit.
    Nq : int
        Number of Lie groups.

    Returns
    ----------
    q : tensor_like
        Lie algebra of affine registration fit.

    """
    Npar = len(q) # Total number of parameters
    N = Npar//Nq # Number of input images
    # Reshape
    q = q.reshape((N, Nq))
    # Make zero mean
    q -= q.mean(axis=0)
    # Reshape back
    q = q.flatten()

    return q
