"""Cost functions for affine image registration.

"""

import numpy as np
import torch
from torch.nn import functional as F
from nitorch.core.kernels import smooth
from nitorch.core.utils import pad, meshgrid_ij
from nitorch.spatial import (affine_matvec, grid_pull)
from nitorch.core.linalg import expm, lmdiv
from nitorch.plot import show_slices


_costs_edge = ['jtv', 'njtv']              # Edge-based cost functions
_costs_hist = ['mi', 'ecc', 'nmi', 'ncc']  # Histogram-based cost functions


def _compute_cost(q, grid0, dat_fix, mat_fix, dat, mat, mov, cost_fun, B,
                  mx_int, fwhm, return_res=False):
    """Compute registration cost function.

    Parameters
    ----------
    q : (N, Nq) tensor_like
        Lie algebra of affine registration fit.
    grid0 : (X1, Y1, Z1) tensor_like
        Sub-sampled image data's resampling grid.
    dat_fix : (X1, Y1, Z1) tensor_like
        Fixed image data.
    mat_fix : (4, 4) tensor_like
        Fixed affine matrix.
    dat : [N,] tensor_like
        List of input images.
    mat : [N,] tensor_like
        List of affine matrices.
    mov : [N,] int
        Indices of moving images.
    cost_fun : str
        Cost function to compute (see run_affine_reg).
    B : (Nq, N, N) tensor_like
        Affine basis.
    mx_int : int
        This parameter sets the max intensity in the images, which decides
        how many bins to use in the joint image histograms
        (e.g, mx_int=511 -> H.shape = (512, 512)).
    fwhm : float
        Full-width at half max of Gaussian kernel, for smoothing
        histogram.
    return_res : bool, default=False
        Return registration results for plotting.

    Returns
    ----------
    c : float
        Cost of aligning images with current estimate of q. If
        optimiser='powell', array_like, else tensor_like.
    res : tensor_like
        Registration results, for visualisation (only if return_res=True).

    """
    # Init
    device = grid0.device
    q = q.flatten()
    was_numpy = False
    if isinstance(q, np.ndarray):
        was_numpy = True
        q = torch.from_numpy(q).to(device)  # To torch tensor
    dm_fix = dat_fix.shape
    Nq = B.shape[0]
    N = torch.tensor(len(dat), device=device, dtype=torch.float32)  # For modulating NJTV cost

    if cost_fun in _costs_edge:
        jtv = dat_fix.clone()
        if cost_fun == 'njtv':
            njtv = -dat_fix.sqrt()

    for i, m in enumerate(mov):  # Loop over moving images
        # Get affine matrix
        mat_a = expm(q[torch.arange(i*Nq, i*Nq + Nq)], B)
        # Compose matrices
        M = lmdiv(mat[m], mat_a.mm(mat_fix)).to(grid0.dtype)  # mat_mov\mat_a*mat_fix
        # Transform fixed grid
        grid = affine_matvec(M, grid0)
        # Resample to fixed grid
        dat_new = grid_pull(dat[m], grid, bound='dft', extrapolate=True, interpolation=1)
        if cost_fun in _costs_edge:
            jtv += dat_new
            if cost_fun == 'njtv':
                njtv -= dat_new.sqrt()

    # Compute the cost function
    res = None
    if cost_fun in _costs_hist:
        # Histogram based costs
        # ----------
        # Compute joint histogram
        # OBS: This function expects both images to have the same max and min intesities,
        # this is ensured by the _data_loader() function.
        H = _hist_2d(dat_fix, dat_new, mx_int, fwhm)
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
            # Maes, Collignon, Vandermeulen, Marchal & Suetens (1997).
            # "Multimodality image registration by maximisation of mutual
            # information". IEEE Transactions on Medical Imaging 16(2):187-198
            mi = torch.sum(pxy * torch.log2(pxy / py.mm(px)))
            ecc = -2*mi/(torch.sum(px*px.log2()) + torch.sum(py*py.log2()))
            c = -ecc
        elif cost_fun == 'nmi':
            # Normalised Mutual Information
            # Studholme,  Hill & Hawkes (1998).
            # "A normalized entropy measure of 3-D medical image alignment".
            # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
            nmi = (torch.sum(px*px.log2()) + torch.sum(py*py.log2()))/torch.sum(pxy*pxy.log2())
            c = -nmi
        elif cost_fun == 'ncc':
            # Normalised Cross Correlation
            i = torch.arange(1, pxy.shape[0] + 1, device=device,
                             dtype=torch.float32)
            j = torch.arange(1, pxy.shape[1] + 1, device=device,
                             dtype=torch.float32)
            m1 = torch.sum(py*i[..., None])
            m2 = torch.sum(px*j[None, ...])
            sig1 = torch.sqrt(torch.sum(py[..., 0]*(i - m1)**2))
            sig2 = torch.sqrt(torch.sum(px[0, ...]*(j - m2)**2))
            i, j = meshgrid_ij(i - m1, j - m2)
            ncc = torch.sum(torch.sum(pxy*i*j))/(sig1*sig2)
            c = -ncc
    elif cost_fun in _costs_edge:
        # Normalised Joint Total Variation
        # M Brudfors, Y Balbastre, J Ashburner (2020).
        # "Groupwise Multimodal Image Registration using Joint Total Variation".
        # in MIUA 2020.
        jtv.sqrt_()
        if cost_fun == 'njtv':
            njtv += torch.sqrt(N)*jtv
            res = njtv
            c = torch.sum(njtv)
        else:
            res = jtv
            c = torch.sum(jtv)

    # _ = show_slices(res, fig_num=1, cmap='coolwarm')  # Can be uncommented for testing

    if was_numpy:
        # Back to numpy array
        c = c.cpu().numpy()

    if return_res:
        return c, res
    else:
        return c


def _hist_2d(img0, img1, mx_int, fwhm):
    """Make 2D histogram, requires:
        * Images same size.
        * Images same min and max intensities (non-negative).

    Parameters
    ----------
    img0 : (X, Y, Z) tensor_like
        First image volume.
    img1 : (X, Y, Z) tensor_like
        Second image volume.
    mx_int : int
        This parameter sets the max intensity in the images, which decides
        how many bins to use in the joint image histograms
        (e.g, mx_int=511 -> H.shape = (512, 512)).
    fwhm : float
        Full-width at half max of Gaussian kernel, for smoothing
        histogram.

    Returns
    ----------
    H : (mx_int + 1, mx_int + 1) tensor_like
        Joint intensity histogram.

    Notes
    ----------
    Naive method for computing a 2D histogram:
    h = torch.zeros((mx_int + 1, mx_int + 1))
    for n in range(num_vox):
        h[img0[n], mg1[n]] += 1

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
    H.put_(ind.long(),
           torch.ones(1, device=img0.device, dtype=ind.dtype).expand_as(ind),
           accumulate=True)
    # Smoothing kernel
    smo = smooth(('gauss',) * 2, fwhm=fwhm,
                 device=img0.device, dtype=torch.float32, sep=True)
    # Pad
    p = (smo[0].shape[2], smo[1].shape[3])
    p = tuple(map(lambda x: (x-1) // 2, p))
    H = pad(H, p, side='both')
    # Smooth
    H = H[None, None, ...]
    H = F.conv2d(H, smo[0])
    H = F.conv2d(H, smo[1])
    H = H[0, 0, ...]
    # Clamp
    H = H.clamp_min(0.0)
    # Add eps
    H = H + 1e-7
    # # Visualise histogram
    # import matplotlib.pyplot as plt
    # plt.figure(num=1)
    # plt.imshow(H.detach().cpu(),
    #     cmap='coolwarm', interpolation='nearest',
    #     aspect='equal', vmax=0.05*H.max())
    # plt.axis('off')
    # plt.show()

    return H
