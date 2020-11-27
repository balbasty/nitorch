"""Field-of-view (FOV) related pre-processing.

"""


import nibabel as nib
from pathlib import Path
import os
import torch
from ..core.pyutils import get_pckg_data
from ..plot.volumes import show_slices
from ..spatial import (affine_default, affine_matrix_classic, voxel_size)
from .affine_reg._align import _atlas_align
from ._preproc_utils import (_get_corners_3d, _reslice_dat_3d)
from ._preproc_img import _world_reslice


def _atlas_crop(dat, mat_in, do_align=True, fov='full'):
    """Crop an image to the NITorch T1w atlas field-of-view.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensor_like
        Input image.
    mat_in : (4, 4) tensor_like, dtype=float64
        Input affine matrix.
    do_align : bool, default=True
        Do alignment to MNI space.
    fov : str, default='full'
        Output field-of-view (FOV):
        * 'full' : Full FOV.
        * 'brain' : Brain FOV.
        * 'tight' : Head+spine FOV.

    Returns
    ----------
    dat : (X1, Y1, Z1) tensor_like
        Cropped image.
    mat : (4, 4) tensor_like, dtype=float64
        Cropped affine matrix.

    """
    device = dat.device
    mat_a = torch.eye(4, dtype=torch.float64, device=device)
    if do_align:
        # Align to MNI
        mat_a = _atlas_align([dat], [mat_in], rigid=False)[0]
    offset = [0] * 6
    if fov == 'brain':
        offset = [16, 16, 64, 64, 130, 75]
    elif fov == 'tight':
        offset = [16, 16, 64, 64, 0, 75]
    # Get atlas information
    nii = nib.load(get_pckg_data('atlas_t1'))
    mat_mu = torch.tensor(nii.affine, dtype=torch.float64, device=device)
    dim_mu = nii.shape
    # Get atlas corners in image space
    mat = mat_a.mm(mat_mu).solve(mat_in)[0]
    c = _get_corners_3d(dim_mu, offset).type(torch.float64).to(device)
    c = mat[:3, ...].mm(c.t())
    # Make bounding-box
    mn = torch.min(c, dim=1)[0]
    mx = torch.max(c, dim=1)[0]
    bb = torch.stack((mn, mx))
    # Extract sub-volume
    dat, mat = _subvol(dat, mat_in, bb)

    return dat, mat


def _reset_origin(dat, mat):
    """Reset affine matrix.

    Parameters
    ----------
    dat : (X0, Y0, Z0) tensor_like, dtype=float32
        Image data.
    mat : (4, 4) tensor_like, dtype=float64
        Affine matrix.

    Returns
    -------
    dat : (X1, Y1, Z1) tensor_like, dtype=float32
        New image data.
    mat : (4, 4) tensor_like, dtype=float64
        New affine matrix.

    """
    device = dat.device
    # Reslice image data to world FOV
    dat, mat = _world_reslice(dat, mat)
    # Compute new, reset, affine matrix
    vx = voxel_size(mat)
    if mat[:3, :3].det() < 0:
        vx[0] = - vx[0]
    vx = vx.tolist()
    mat = affine_default(dat.shape, vx, dtype=torch.float64, device=device)
    mat = torch.cat((mat, torch.tensor([0, 0, 0, 1],
        dtype=mat.dtype, device=mat.device)[None, ...]))  # Add a row to make (4, 4)

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



