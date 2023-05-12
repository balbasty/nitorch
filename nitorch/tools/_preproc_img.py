"""Image-based pre-processing.

"""


import torch
from nitorch.spatial import (affine_matrix_classic, voxel_size)
from nitorch.core.linalg import lmdiv
from ._preproc_utils import (_get_corners_3d, _reslice_dat_3d)


def _reslice2ref(dat, mat, interpolation=1, ref=0):
    """Reslice images to reference.

    Parameters
    ----------
    dat : [N, ] list, tensor_like
        Image data.
    mat : [N, ] list, tensor_like
        Affine matrix
    interpolation : int, default=1 (linear)
        Interpolation order.
    ref : int, default=0
        Index of reference image.

    Returns
    -------
    dat : [N, ] list, tensor_like
        Resliced image data.
    mat_ref : [N, ] list, tensor_like
        Resliced affine matrix.

    """
    # reference
    mat_ref = mat[ref]
    dm_ref = dat[ref].shape
    # reslice
    for i in range(len(dat)):
        if i == ref:  continue
        mat_res = lmdiv(mat[i], mat_ref)
        dat[i] = _reslice_dat_3d(dat[i], mat_res, dm_ref, 
            interpolation=interpolation)
    # replicate reference affine
    mat_ref = [mat_ref for i in range(len(dat))]

    return dat, mat_ref


def _world_reslice(dat, mat, interpolation=1, vx=None):
    """Reslice image data to world space.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensor_like, dtype=float32
        Image data.
    mat : (4, 4) tensor_like, dtype=float64
        Affine matrix.
    interpolation : int, default=1 (linear)
        Interpolation order.
    vx : float | [float,] *3, optional
        Output voxel size.

    Returns
    -------
    dat : (X1, Y1, Z1) tensor_like, dtype=float32
        New image data.
    mat : (4, 4) tensor_like, dtype=float64
        New affine matrix.

    """
    device = dat.device
    # Get voxel size
    if vx is None:
        vx = voxel_size(mat).type(torch.float64).to(device)
    else:
        if not isinstance(vx, (list, tuple)):
            vx = (vx,) * 3
        vx = torch.as_tensor(vx).type(torch.float64).to(device)
    # Get corners
    c = _get_corners_3d(dat.shape).type(torch.float64).to(device)
    c = c.t()
    # Corners in world space
    c_world = mat[:3, :4].mm(c)
    c_world[0, :] = - c_world[0, :]
    # Get bounding box
    mx = c_world.max(dim=1)[0].round()
    mn = c_world.min(dim=1)[0].round()
    # Compute output affine
    mat_mn = affine_matrix_classic(mn).type(torch.float64).to(device)
    mat_vx = torch.diag(torch.cat((vx,
        torch.ones(1, dtype=torch.float64, device=device))))
    mat_1  = affine_matrix_classic(
        -1 * torch.ones(3, dtype=torch.float64, device=device))
    mat_out = mat_mn.mm(mat_vx.mm(mat_1))
    # Comput output image dimensions
    dim_out = mat_out.inverse().mm(torch.cat((
        mx, torch.ones(1, dtype=torch.float64, device=device)))[:, None])
    dim_out = dim_out[:3].ceil().flatten().int().tolist()
    I = torch.diag(torch.ones(4, dtype=torch.float64, device=device))
    I[0, 0] = -I[0, 0]
    mat_out = I.mm(mat_out)
    # Compute mapping from output to input
    mat = lmdiv(mat, mat_out)
    # Reslice image data
    dat = _reslice_dat_3d(dat, mat, dim_out, interpolation=interpolation)

    return dat, mat_out
