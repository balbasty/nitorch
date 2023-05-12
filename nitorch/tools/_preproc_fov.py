"""Field-of-view (FOV) related pre-processing.

"""


from pathlib import Path
import os
import torch
from nitorch.core.datasets import fetch_data
from nitorch.plot.volumes import show_slices
from nitorch.io import map
from nitorch.spatial import (affine_default, affine_matrix_classic, voxel_size)
from nitorch.core.linalg import lmdiv
from .affine_reg._align import _atlas_align
from ._preproc_utils import (_get_corners_3d, _reslice_dat_3d, _msk_fov)
from ._preproc_img import _world_reslice


def _bb_atlas(name, fov, dtype=torch.float64, device='cpu'):
    """Bounding-box NITorch atlas data to specific field-of-view.

    Parameters
    ----------
    name : str
        Name of nitorch data, available are:
        * atlas_t1: MRI T1w intensity atlas, 1 mm resolution.
        * atlas_t2: MRI T2w intensity atlas, 1 mm resolution.
        * atlas_pd: MRI PDw intensity atlas, 1 mm resolution.
        * atlas_t1_mni: MRI T1w intensity atlas, in MNI space, 1 mm resolution.
        * atlas_t2_mni: MRI T2w intensity atlas, in MNI space, 1 mm resolution.
        * atlas_pd_mni: MRI PDw intensity atlas, in MNI space, 1 mm resolution.
    fov : str
        Field-of-view, specific to 'name':
        * 'atlas_t1' | 'atlas_t2' | 'atlas_pd':
            * 'brain': Head FOV.
            * 'head': Brain FOV.

    Returns
    ----------
    mat_mu : (4, 4) tensor, dtype=float64
        Output affine matrix.
    dim_mu : (3, ) tensor, dtype=float64
        Output dimensions.

    """
    # Get atlas information
    file_mu = map(fetch_data(name))
    dim_mu = file_mu.shape
    mat_mu = file_mu.affine.type(torch.float64).to(device)
    # Get bounding box
    o = [[0, 0, 0], [0, 0, 0]]
    if name in ['atlas_t1', 'atlas_t2', 'atlas_pd']:
        if fov == 'brain' or fov == 'head':
            o[0][0] = 18
            o[0][1] = 52
            o[0][2] = 120
            o[1][0] = 18
            o[1][1] = 48
            o[1][2] = 58
            if fov == 'head':
                o[0][2] = 25
    # Get bounding box
    bb = torch.tensor([[1 + o[0][0], 1 + o[0][1], 1 + o[0][2]],
                       [dim_mu[0] - o[1][0], dim_mu[1] - o[1][1], dim_mu[2] - o[1][2]]])
    bb = bb.type(torch.float64).to(device)
    # Output dimensions
    dim_mu = bb[1, ...] - bb[0, ...] + 1
    # Bounding-box atlas affine
    mat_bb = affine_matrix_classic(bb[0, ...] - 1)
    # Modulate atlas affine with bb affine
    mat_mu = mat_mu.mm(mat_bb)

    return mat_mu, dim_mu


def _atlas_crop(dat, mat_in, do_align=True, fov='head', mat_a=None):
    """Crop an image to the NITorch T1w atlas field-of-view.

    OBS: This function is implemented so to not resample the image data.
    A tighter crop could be achieved by additionally resampling in atlas
    space. But here, we aim to not resample the input data.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensor_like
        Input image.
    mat_in : (4, 4) tensor_like, dtype=float64
        Input affine matrix.
    do_align : bool, default=True
        Do alignment to MNI space.
    fov : str, default='head'
        * 'head' : Head FOV.
        * 'brain' : Brain FOV.
    mat_a : (4, 4) tensor_like, dtype=float64, optional
        Pre-computed atlas alignment affine matrix.

    Returns
    ----------
    dat : (X1, Y1, Z1) tensor_like
        Cropped image.
    mat : (4, 4) tensor_like, dtype=float64
        Cropped affine matrix.

    """
    device = dat.device
    if mat_a is None:
        mat_a = torch.eye(4, dtype=torch.float64, device=device)
    if do_align:
        # Align to MNI
        mat_a = _atlas_align([dat], [mat_in], rigid=False)[0]
        mat_a = mat_a[0, ...]
    # Get bounding box
    mat_mu, dim_mu = _bb_atlas('atlas_t1', fov, dtype=torch.float64, device=device)
    # Modulate atlas affine with registration affine matrix
    mat_mu = mat_a.mm(mat_mu)
    # # Mask field-of-view of image data to be the same as the atlas
    # dat = _msk_fov(dat, mat_in, mat_mu, dim_mu)
    # Get atlas corners in image space
    mat = lmdiv(mat_in, mat_mu)
    c = _get_corners_3d(dim_mu).type(torch.float64).to(device)
    c = mat[:3, ...].mm(c.t())
    # Make bounding-box
    mn = torch.min(c, dim=1)[0]
    mx = torch.max(c, dim=1)[0]
    bb = torch.stack((mn, mx))
    # Extract sub-volume
    dat, mat = _subvol(dat, mat_in, bb)

    return dat, mat


def _reset_origin(dat, mat, interpolation):
    """Reset affine matrix.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensor_like, dtype=float32
        Image data.
    mat : (4, 4) tensor_like, dtype=float64
        Affine matrix.
    interpolation : int, default=1 (linear)
        Interpolation order.

    Returns
    -------
    dat : (X1, Y1, Z1) tensor_like, dtype=float32
        New image data.
    mat : (4, 4) tensor_like, dtype=float64
        New affine matrix.

    """
    device = dat.device
    # Reslice image data to world FOV
    dat, mat = _world_reslice(dat, mat, interpolation=interpolation)
    # Compute new, reset, affine matrix
    vx = voxel_size(mat)
    if mat[:3, :3].det() < 0:
        vx[0] = - vx[0]
    vx = vx.tolist()
    mat = affine_default(dat.shape, vx, dtype=torch.float64, device=device)

    return dat, mat


def _subvol(dat, mat, bb=None):
    """Extract a sub-volume.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensor_like
        Image volume.
    mat : (4, 4) tensor_like, dtype=float64
        Image affine matrix.
    bb : (2, 3) sequence, optional
        Bounding box.

    Returns
    ----------
    dat : (X1, Y1, Z1) tensor_like
        Image sub-volume.
    mat : (4, 4) tensor_like, dtype=float64
        Sub-volume affine matrix.

    """
    device = dat.device
    dim_in = dat.shape
    if bb is None:
        bb = torch.tensor([[1, 1, 1], dim_in],
                          dtype=torch.float64, device=device)
    # Process bounding-box
    bb = bb.round()
    bb = bb.sort(dim=0)[0]
    bb[0, ...] = torch.max(bb[0, ...],
        torch.ones(3, device=device, dtype=torch.float64))
    bb[1, ...] = torch.min(bb[1, ...],
        torch.tensor(dim_in, device=device, dtype=torch.float64))
    # Output dimensions
    dim_bb = bb[1, ...] - bb[0, ...] + 1
    # Bounding-box affine
    mat_bb = affine_matrix_classic(bb[0, ...] - 1)
    # mat_bb = matrix(bb[0, ...] - 1)
    # Output data
    dat = _reslice_dat_3d(dat, mat_bb, dim_bb,
        interpolation='nearest', bound='zero', extrapolate=False)
    # Output affine
    mat = mat.mm(mat_bb)

    return dat, mat
