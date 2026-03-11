"""Grid transform composition and reslicing — all tensor ops, no disk I/O."""

import numpy as np
import torch
from nitorch.spatial import identity_grid, grid_pull


def affine_transform(grid, mat):
    """Apply a 4x4 affine matrix to a coordinate grid.

    Parameters
    ----------
    grid : torch.Tensor
        (..., 3) coordinate array.
    mat : torch.Tensor
        (4, 4) affine matrix.

    Returns
    -------
    torch.Tensor
        (..., 3) transformed coordinates.
    """
    mat = mat.to(grid.dtype)
    return grid @ mat[:3, :3].T + mat[:3, 3]


def compose_deformation_grid(fixed_shape, fixed_affine, affine_sqrt,
                              displacement, disp_affine, device='cpu'):
    """Compose registration transforms into a full deformation grid.

    All inputs are tensors — no file I/O.

    Parameters
    ----------
    fixed_shape : tuple
        (X, Y, Z) shape of the fixed image.
    fixed_affine : torch.Tensor
        (4, 4) voxel-to-world affine of the fixed image.
    affine_sqrt : torch.Tensor
        (4, 4) square root of the registration affine.
    displacement : torch.Tensor
        (*spatial, 3) displacement field.
    disp_affine : torch.Tensor
        (4, 4) voxel-to-world affine of the displacement field.
    device : str
        Torch device.

    Returns
    -------
    grid : torch.Tensor
        (X, Y, Z, 3) deformation field in RAS coordinates.
    """
    fixed_affine = fixed_affine.to(dtype=torch.float64, device=device)
    affine_sqrt = affine_sqrt.to(dtype=torch.float64, device=device)
    disp_affine = disp_affine.to(dtype=torch.float64, device=device)
    displacement = displacement.to(dtype=torch.float64, device=device)

    # Squeeze extra dims from displacement
    while displacement.ndim > 4:
        displacement = displacement.squeeze(-2)

    # 1. Identity grid in fixed voxel space
    id_grid = identity_grid(fixed_shape, dtype=torch.float64, device=device)

    # 2. Voxel -> RAS
    grid = affine_transform(id_grid, fixed_affine)

    # 3. First half of affine (square root)
    grid = affine_transform(grid, affine_sqrt)

    # 4. Sample displacement field at current grid positions
    grid = affine_transform(grid, torch.linalg.inv(disp_affine))
    disp_bcxyz = displacement.permute(3, 0, 1, 2).unsqueeze(0)
    grid_batch = grid.unsqueeze(0)
    sampled_disp = grid_pull(disp_bcxyz, grid_batch, bound='zero',
                             extrapolate=True)
    sampled_disp = sampled_disp[0].permute(1, 2, 3, 0)
    grid = affine_transform(grid, disp_affine)
    grid = grid + sampled_disp

    # 5. Second half of affine (square root)
    grid = affine_transform(grid, affine_sqrt)

    return grid



def grid_to_slicer_displacement(grid, fixed_shape, fixed_affine, device='cpu'):
    """Convert deformation grid to Slicer displacement format.

    Parameters
    ----------
    grid : torch.Tensor
        (X, Y, Z, 3) deformation grid in RAS.
    fixed_shape : tuple
        (X, Y, Z) shape of the fixed image.
    fixed_affine : torch.Tensor
        (4, 4) voxel-to-world of the fixed image.
    device : str
        Torch device.

    Returns
    -------
    disp_np : np.ndarray
        (X, Y, Z, 1, 3) displacement array ready for NIfTI with intent 1006.
    """
    fixed_affine = fixed_affine.to(dtype=torch.float64, device=device)
    grid = grid.to(dtype=torch.float64, device=device)

    id_grid = identity_grid(fixed_shape, dtype=torch.float64, device=device)
    ras_id = affine_transform(id_grid, fixed_affine)

    # Displacement = deformation - identity in RAS
    disp = grid - ras_id

    # Slicer grid format: (X, Y, Z, 1, 3)
    disp = disp.unsqueeze(-2)
    return disp.cpu().to(torch.float32).numpy()
