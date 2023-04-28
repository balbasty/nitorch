"""Module for affine image registration.

"""

import numpy as np
import torch
from nitorch.plot import show_slices
from nitorch.core.datasets import fetch_data
from nitorch.core.datasets import data as available_atlases
from nitorch.spatial import (affine_basis, voxel_size)
from nitorch.core.linalg import expm
from ._costs import (_costs_hist, _compute_cost)
from ._core import (_data_loader, _get_dat_grid, _get_mean_space, _fit_q)
from .._preproc_utils import _format_input
# Try import matplotlib.pyplot
from nitorch.core.optionals import try_import
plt = try_import('matplotlib.pyplot', _as=True)


def _affine_align(dat, mat, cost_fun='nmi', group='SE', mean_space=False,
                  samp=(3, 1.5), optimiser='powell', fix=0, verbose=False,
                  fov=None, mx_int=1023, raw=False, jitter=True, fwhm=7.0):
    """Affine registration of a collection of images.

    Parameters
    ----------
    dat : [N, ...], tensor_like
        List of image volumes.
    mat : [N, ...], tensor_like
        List of affine matrices.
    cost_fun : str, default='nmi'
        Pairwise methods:
            * 'nmi'  : Normalised Mutual Information
            * 'mi'   : Mutual Information
            * 'ncc'  : Normalised Cross Correlation
            * 'ecc'  : Entropy Correlation Coefficient
        Groupwise methods:
            * 'njtv' : Normalised Joint Total variation
            * 'jtv'  : Joint Total variation
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
        Optimise a mean-space fit, only available if cost_fun='njtv'.
    samp : (float, ), default=(3, 1.5)
        Optimisation sampling steps (mm).
    optimiser : str, default='powell'
        'powell' : Optimisation method.
    fix : int, default=0
        Index of image to used as fixed image, not used if mean_space=True.
    verbose : bool, default=False
        Show registration results.
    fov : (2,) tuple, default=None
        A tuple with affine matrix (tensor_like) and dimensions (tuple) of mean space.
    mx_int : int, default=1023
        This parameter sets the max intensity in the images, which decides
        how many bins to use in the joint image histograms
        (e.g, mx_int=1023 -> H.shape = (1024, 1024)). This is only done if
        cost_fun is histogram-based.
    raw : bool, default=False
        Do no processing of input images -> work on raw data.
    jitter : bool, default=False
        Add random jittering to resampling grid.
    fwhm : float, default=7
        Full-width at half max of Gaussian kernel, for smoothing
        histogram.

    Returns
    ----------
    mat_a : (N, 4, 4), tensor_like
        Affine alignment matrices.
    mat_fix : (4, 4), tensor_like
        Affine matrix of fixed image.
    dim_fix : (3,), tuple, list, tensor_like
        Image dimensions of fixed image.
    q : (N, Nq), tensor_like
        Lie parameters.

    """
    with torch.no_grad():
        device = dat[0].device
        # Parse algorithm options
        opt = {'optimiser': optimiser,
               'cost_fun': cost_fun,
               'samp': samp,
               'fix': fix,
               'mean_space': mean_space,
               'verbose': verbose,
               'fov': fov,
               'group' : group,
               'raw': raw,
               'jitter': jitter,
               'mx_int' : mx_int,
               'fwhm' : fwhm}
        if not isinstance(opt['samp'], (list, tuple)):
            opt['samp'] = (opt['samp'], )
        # Some very basic sanity checks
        N = len(dat) # Number of input scans
        mov = list(range(N))  # Indices of images
        if opt['cost_fun'] in _costs_hist and opt['mean_space']:
            raise ValueError('Option mean_space=True not defined for {} cost!'.format(opt['cost_fun']))
        # Get affine basis
        dim = mat[0].shape[0] - 1
        B = affine_basis(group=opt['group'], device=device, dim=dim)
        Nq = B.shape[0]
        # Load data
        dat = _data_loader(dat, mat, opt)
        # Define fixed image space (mat_fix, dim_fix, vx_fix)
        if opt['mean_space']:
            # Use a mean-space
            if opt['fov']:
                # Mean-space given
                mat_fix = opt['fov'][0]
                dim_fix = opt['fov'][1]
            else:
                # Compute mean-space
                mat_fix, dim_fix = _get_mean_space(dat, mat)
            arg_grid = dim_fix
        else:
            # Use one of the input images
            mat_fix = mat[opt['fix']]
            dim_fix = dat[opt['fix']].shape[:3]
            mov.remove(opt['fix'])
            arg_grid = dat[opt['fix']]
        # Get voxel size of fixed image
        vx_fix = voxel_size(mat_fix)
        # Initial guess for registration parameter
        q = torch.zeros((N, Nq), dtype=torch.float64, device=device)
        if N < 2:
            # Return identity
            mat_a = torch.zeros((N, dim + 1, dim + 1),
                                dtype=torch.float64, device=device)
            for n in range(N):
                mat_a[m, ...] = expm(q[n, ...], basis=B)

            return mat_a, mat_fix, dim_fix
        # Do registration
        for s in opt['samp']:  # Loop over sub-sampling level
            # Get possibly sub-sampled fixed image, and its resampling grid
            dat_fix, grid = _get_dat_grid(
                arg_grid, vx_fix, s, jitter=opt['jitter'], device=device)
            # Do optimisation
            q, args = _fit_q(q, dat_fix, grid, mat_fix, dat, mat, mov,
                             B, s, opt)
    # To matrix form
    mat_a = torch.zeros((N, dim + 1, dim + 1),
                        dtype=torch.float64, device=device)
    for n in range(N):
        mat_a[n, ...] = expm(q[n, ...], basis=B)

    return mat_a, mat_fix, dim_fix, q


def _atlas_align(dat, mat, rigid=True, pth_atlas=None, default_atlas='atlas_t1'):
    """Affinely align image to some atlas space.

    Parameters
    ----------
    dat : [N, ...], tensor_like
        List of image volumes.
    mat : [N, ...], tensor_like
        List of affine matrices.
    rigid = bool, default=True
        Do rigid alignment, else does rigid+isotropic scaling.
    pth_atlas : str, optional
        Path to atlas image to match to. Uses Brain T1w atlas by default.
    default_atlas : str, optional
        Name of an atlas available in the collection of default nitorch atlases.

    Returns
    ----------
    mat_a : (N, 4, 4) tensor_like
        Transformation aligning to MNI space as M_mni\M_mov.
    mat_mni : (4, 4), tensor_like
        Affine matrix of MNI image.
    dim_mni : (3,), tuple, list, tensor_like
        Image dimensions of MNI image.
    mat_cso : (N, 4, 4) tensor_like
        CSO transformation.

    """
    if default_atlas not in available_atlases:
        raise ValueError(
            "Default atlas {:} not in available atlases: {:}". \
            format(default_atlas, list(available_atlases.keys()))
        )
    if pth_atlas is None:
        # Get path to nitorch's T1w intensity atlas
        pth_atlas = fetch_data(default_atlas)
    # Get number of input images
    N = len(dat)
    # Append atlas at the end of input data
    dat_mni, mat_mni, _ = _format_input(pth_atlas, device=dat[0].device,
                                        rand=True, cutoff=(0.0005, 0.9995))
    dat.append(dat_mni[0])
    mat.append(mat_mni[0])
    # Align to MNI atlas.
    group = 'CSO'
    _, mat_mni, dim_mni, q = _affine_align(dat, mat,
         group=group, samp=(3, 1.5), cost_fun='nmi', fix=N,
         verbose=False, mean_space=False)
    # Remove atlas
    q = q[:N, ...]
    dat = dat[:N]
    mat = mat[:N]
    # Get matrix representation
    mat_cso = expm(q, affine_basis(group=group))
    if rigid:
        # Extract only rigid part
        group = 'SE'
        q = q[..., :6]
    # Get matrix representation
    mat_a = expm(q, affine_basis(group=group))

    return mat_a, mat_mni, dim_mni, mat_cso


def _test_cost(dat, mat, cost_fun='nmi', group='SE', mean_space=False,
               samp=2, ix_par=0, jitter=False, x_step=0.1, x_mn_mx=30,
               verbose=False, mx_int=1023, raw=False, fwhm=7.0):
    """Check cost function behaviour by keeping one image fixed and re-aligning
    a second image by modifying one of the affine parameters. Plots cost vs.
    aligment when finished.

    """
    with torch.no_grad():
        device = dat[0].device
        # Parse algorithm options
        opt = {'cost_fun': cost_fun,
               'samp': samp,
               'mean_space': mean_space,
               'verbose': verbose,
               'raw': raw,
               'mx_int' : mx_int,
               'fwhm' : fwhm}
        if not isinstance(opt['samp'], (list, tuple)):
            opt['samp'] = (opt['samp'], )
        # Some very basic sanity checks
        N = len(dat)
        if N != 2:
            raise ValueError('N != 2')
        mov = list(range(N))  # Indices of images
        fix_img = 0
        mov_img = 1
        if opt['cost_fun'] in _costs_hist and opt['mean_space']:
            raise ValueError('Option mean_space=True not defined for {} cost!'.format(opt['cost_fun']))
        # Load data
        dat = _data_loader(dat, mat, opt)
        # Get full 12 parameter affine basis
        B = affine_basis(group='Aff+', device=device)
        Nq = B.shape[0]
        # Range of parameter
        x = torch.arange(start=-x_mn_mx, end=x_mn_mx, step=x_step, dtype=torch.float32)
        if opt['mean_space']:
            # Use mean-space, so make sure that maximum misalignment is represented
            # in the input to _get_mean_space()
            mat_mn = torch.zeros(Nq,
                dtype=torch.float64, device=device)
            mat_mx = torch.zeros(Nq,
                dtype=torch.float64, device=device)
            mat_mn[ix_par] = -x_mn_mx
            mat_mx[ix_par] = x_mn_mx
            mat1 = [expm(mat_mn, B).mm(mat[mov_img]),
                    expm(mat_mx, B).mm(mat[mov_img])]
            # Compute mean-space
            dat.append(torch.tensor(dat[mov_img].shape,
                       dtype=torch.float32, device=device))
            dat.append(torch.tensor(dat[mov_img].shape,
                       dtype=torch.float32, device=device))
            mat_fix, dim_fix = _get_mean_space(dat, mat + mat1)
            dat = dat[:2]
            arg_grid = dim_fix
        else:
            mat_fix = mat[fix_img]
            dim_fix = dat[fix_img].shape[:3]
            mov.remove(fix_img)
            arg_grid = dat[fix_img]
        # Get voxel size of fixed image
        vx_fix = voxel_size(mat_fix)
        # Initial guess
        q = torch.zeros((N, Nq), dtype=torch.float64, device=device)
        # Get subsampled fixed image and its resampling grid
        dat_fix, grid = _get_dat_grid(arg_grid,
            vx_fix, samp=opt['samp'][-1], jitter=jitter, device=device)
        # Iterate over a range of values
        costs = np.zeros(len(x))
        fig_ax = None  # Used for visualisation
        for i, xi in enumerate(x):
            # Change affine matrix a little bit
            q[fix_img, ix_par] = xi
            # Compute cost
            costs[i], res = _compute_cost(
                q, grid, dat_fix, mat_fix, dat, mat, mov, opt['cost_fun'], B, opt['mx_int'], opt['fwhm'], return_res=True)
            if opt['verbose']:
                fig_ax = show_slices(res, fig_ax=fig_ax, fig_num=1, cmap='coolwarm', title='x=' + str(xi))
            # print(costs[i])
        # Plot results
        if plt is None:
            return
        fig, ax = plt.subplots(num=2)
        ax.plot(x, costs)
        ax.set(xlabel='Value q[' + str(ix_par) + ']', ylabel='Cost',
               title=opt['cost_fun'].upper() + ' cost function (mean_space=' + str(opt['mean_space']) + ')')
        ax.grid()
        plt.show()
