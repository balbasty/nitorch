"""Pre-processing utility functions.

"""


import os
import torch
from ..spatial import (affine_grid, grid_pull)
from ..io import map, savef
from ..core.pyutils import file_mod


def _format_input(img, device='cpu', rand=False, cutoff=None):
    """Format preprocessing input data.

    """
    if (not isinstance(img, list)) or \
        (isinstance(img, list) and not isinstance(img[0], list)):
        img = [img]
    file = []
    dat = []
    mat = []
    for n in range(len(img)):
        if isinstance(img[n], str):
            # Input are nitorch.io compatible paths
            file.append(map(img[n]))
            dat.append(file[n].fdata(dtype=torch.float32, device=device,
                                     rand=rand, cutoff=cutoff))
            mat.append(file[n].affine.to(device).type(torch.float64))
        else:
            # Input are tensors (clone so to not modify input data)
            file.append(None)
            dat.append(img[n][0].clone().type(torch.float32))
            mat.append(img[n][1].clone().type(torch.float64))

    return dat, mat, file


def _process_reg(dat, mat, mat_a, mat_fix, dim_fix, write):
    """Process registration results.

    """
    N = len(dat)
    rdat = torch.zeros((N, ) + dim_fix,
        dtype=dat[0].dtype, device=dat[0].device)
    for n in range(N):  # loop over input images
        if torch.all(mat_a[n, ...] -
                     torch.eye(4, device=mat_a[n,...].device) == 0):
            rdat[n, ...] = dat[n]
        else:
            mat_r = mat_a[n, ...].mm(mat_fix).solve(mat[n])[0]
            rdat[n, ...] = _reslice_dat_3d(dat[n], mat_r, dim_fix)
        if write == 'reslice':
            dat[n] = rdat[n, ...]
            mat[n] = mat_fix
        elif write == 'affine':
            mat[n] = mat[n].solve(mat_a[n, ...])[0]
    # Write output to disk?
    if write in ['reslice', 'affine']:
        write = True
    else:
        write = False

    return dat, mat, write, rdat


def _write_output(dat, mat, file=None, prefix='', odir='', nam=''):
    """Write preprocessed output to disk.

    """
    if odir:
        os.makedirs(odir, exist_ok=True)
    pth = []
    for n in range(len(dat)):
        if file[n] is not None:
            filename = file[n].filename()
        else:
            filename = ''
        pth.append(file_mod(filename,
            odir=odir, prefix=prefix, nam=nam))
        savef(dat[n], pth[n], like=file[n], affine=mat[n])

    return dat, mat, pth


def _get_corners_3d(dim, o=[0] * 6):
    """Get eight corners of 3D image volume.

    Parameters
    ----------
    dim : (3,), list or tuple
        Image dimensions.
    o : (6, ), default=[0] * 6
        Offsets.

    Returns
    -------
    c : (8, 4), tensor_like[torch.float64]
        Corners of volume.

    """
    # Get corners
    c = torch.tensor(
        [[     1,      1,      1, 1],
         [     1,      1, dim[2], 1],
         [     1, dim[1],      1, 1],
         [     1, dim[1], dim[2], 1],
         [dim[0],      1,      1, 1],
         [dim[0],      1, dim[2], 1],
         [dim[0], dim[1],      1, 1],
         [dim[0], dim[1], dim[2], 1]])
    # Include offset
    # Plane 1
    c[0, 0] = c[0, 0] + o[0]
    c[1, 0] = c[1, 0] + o[0]
    c[2, 0] = c[2, 0] + o[0]
    c[3, 0] = c[3, 0] + o[0]
    # Plane 2
    c[4, 0] = c[4, 0] - o[1]
    c[5, 0] = c[5, 0] - o[1]
    c[6, 0] = c[6, 0] - o[1]
    c[7, 0] = c[7, 0] - o[1]
    # Plane 3
    c[0, 1] = c[0, 1] + o[2]
    c[1, 1] = c[1, 1] + o[2]
    c[4, 1] = c[4, 1] + o[2]
    c[5, 1] = c[5, 1] + o[2]
    # Plane 4
    c[2, 1] = c[2, 1] - o[3]
    c[3, 1] = c[3, 1] - o[3]
    c[6, 1] = c[6, 1] - o[3]
    c[7, 1] = c[7, 1] - o[3]
    # Plane 5
    c[0, 2] = c[0, 2] + o[4]
    c[2, 2] = c[2, 2] + o[4]
    c[4, 2] = c[4, 2] + o[4]
    c[6, 2] = c[6, 2] + o[4]
    # Plane 6
    c[1, 2] = c[1, 2] - o[5]
    c[3, 2] = c[3, 2] - o[5]
    c[5, 2] = c[5, 2] - o[5]
    c[7, 2] = c[7, 2] - o[5]

    return c


def _reslice_dat_3d(dat, affine, dim_out, interpolation='linear',
                    bound='zero', extrapolate=False):
    """Reslice 3D image data.

    Parameters
    ----------
    dat : (Xi, Yi, Zi), tensor_like
        Input image data.
    affine : (4, 4), tensor_like
        Affine transformation that maps from voxels in output image to
        voxels in input image.
    dim_out : (Xo, Yo, Zo), list or tuple
        Output image dimensions.
    interpolation : str, default='linear'
        Interpolation order.
    bound : str, default='zero'
        Boundary condition.
    extrapolate : bool, default=False
        Extrapolate out-of-bounds data.

    Returns
    -------
    dat : (dim_out), tensor_like
        Resliced image data.

    """
    if len(dat.shape) != 3:
        raise ValueError('Input error: len(dat.shape) != 3')

    grid = affine_grid(affine, dim_out).type(dat.dtype)
    grid = grid[None, ...]
    dat = dat[None, None, ...]
    dat = grid_pull(dat, grid,
        bound=bound, interpolation=interpolation, extrapolate=extrapolate)
    dat = dat[0, 0, ...]

    return dat


