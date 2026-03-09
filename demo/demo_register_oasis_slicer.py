"""
OASIS LCC Registration with 3D Slicer Grid Transform Export

This script:
1. Runs LCC registration on an OASIS brain image pair using nitorch
2. Converts the resulting transforms (affine + displacement) into a
   3D Slicer-compatible grid transform (NIfTI)
3. Saves all files needed for verification in 3D Slicer

Verification in 3D Slicer:
- Load fixed_image.nii.gz, moving_image.nii.gz, moving_image_resliced.nii.gz
- Load Grid.nii.gz as a GridTransform
- Apply Grid.nii.gz to moving_image.nii.gz (Transforms module)
- The result should match moving_image_resliced.nii.gz
"""

import os
import json
import shutil
import warnings

import nibabel as nib
import numpy as np
import torch
from scipy.linalg import sqrtm

# Suppress FutureWarnings from interpol package
warnings.filterwarnings("ignore", category=FutureWarning, module="interpol")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from nitorch.spatial import identity_grid, grid_pull

from demo_helpers import (
    add_loss,
    create_displacement_field,
    get_pair,
    get_subject_id,
    register,
    resize_pair,
    reslice,
)

# =============================================================================
# Grid transform helper functions
# =============================================================================

def load_affine_lta(path):
    """Load a 4x4 affine matrix from an LTA file and return its square root.

    Parameters
    ----------
    path : str
        Path to the .lta file.

    Returns
    -------
    torch.Tensor
        (4, 4) matrix square root of the affine (float64).
    """
    rows = []
    in_matrix = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts == ["1", "4", "4"]:
                in_matrix = True
                continue
            if in_matrix:
                row = [float(x) for x in parts]
                if len(row) == 4:
                    rows.append(row)
                if len(rows) == 4:
                    break
    mat = np.array(rows, dtype=np.float64)
    mat_sqrt = np.real(sqrtm(mat))
    return torch.from_numpy(mat_sqrt)


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


def create_slicer_grid_transform(
    fixed_nii_path,
    affine_lta_path,
    displacement_nii_path,
    output_path,
    device='cpu',
):
    """Convert nitorch registration transforms to a 3D Slicer grid transform.

    Composes the affine (split as square root, applied twice) and displacement
    field into a single grid transform in RAS convention for 3D Slicer.

    Parameters
    ----------
    fixed_nii_path : str
        Path to the fixed image NIfTI.
    affine_lta_path : str
        Path to the affine .lta file from nitorch registration.
    displacement_nii_path : str
        Path to the displacement field NIfTI (u.nii.gz).
    output_path : str
        Path to save the output grid transform NIfTI.
    device : str
        Torch device ('cpu' or 'cuda').
    """
    # Load data
    nii_fixed = nib.load(fixed_nii_path)
    nii_disp = nib.load(displacement_nii_path)

    fixed_affine = torch.from_numpy(nii_fixed.affine.astype(np.float64)).to(device)
    disp_affine = torch.from_numpy(nii_disp.affine.astype(np.float64)).to(device)

    dat_disp = torch.from_numpy(nii_disp.get_fdata()).to(dtype=torch.float64, device=device)
    # Squeeze trailing singleton dims: (X, Y, Z, 1, 3) -> (X, Y, Z, 3)
    while dat_disp.ndim > 4:
        dat_disp = dat_disp.squeeze(-2)

    affine_sqrt = load_affine_lta(affine_lta_path).to(device)

    # 1. Identity grid in fixed voxel space
    fixed_shape = nii_fixed.shape[:3]
    id_grid = identity_grid(fixed_shape, dtype=torch.float64, device=device)

    # 2. Voxel -> RAS (fixed image affine)
    grid = affine_transform(id_grid, fixed_affine)

    # 3. First half of affine (square root)
    grid = affine_transform(grid, affine_sqrt)

    # 4. Sample displacement field
    #    Map grid to displacement voxel space
    grid = affine_transform(grid, torch.linalg.inv(disp_affine))
    #    Sample the displacement at these voxel coordinates
    #    grid_pull expects: input (batch, channel, *spatial), grid (batch, *spatial, dim)
    dat_disp_bcxyz = dat_disp.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, X, Y, Z)
    grid_batch = grid.unsqueeze(0)  # (1, X, Y, Z, 3)
    sampled_disp = grid_pull(dat_disp_bcxyz, grid_batch, bound='zero', extrapolate=True)
    sampled_disp = sampled_disp[0].permute(1, 2, 3, 0)  # (X, Y, Z, 3)
    #    Map back to displacement RAS space and add displacement
    grid = affine_transform(grid, disp_affine)
    grid = grid + sampled_disp

    # 5. Second half of affine (square root)
    grid = affine_transform(grid, affine_sqrt)

    # Convert deformation to displacement (subtract fixed RAS positions)
    grid -= affine_transform(id_grid, fixed_affine)

    # NOTE: With intent code 1006 (NIFTI_INTENT_DISPVECT), Slicer expects
    # displacement vectors in RAS, so no LPS conversion is needed.
    # With intent code 1007 (NIFTI_INTENT_VECTOR), Slicer assumes LPS,
    # so you would need: grid[..., 0] *= -1; grid[..., 1] *= -1

    # Add singleton dimension for Slicer grid format: (X, Y, Z, 1, 3)
    grid = grid.unsqueeze(-2)

    # Save as NIfTI
    grid_np = grid.cpu().to(torch.float32).numpy()
    img = nib.Nifti1Image(grid_np, nii_fixed.affine)
    img.header['intent_code'] = 1006  # NIFTI_INTENT_DISPVECT
    nib.save(img, output_path)
    print(f"  Grid transform saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    dir_data = "/home/mbrudfors/Data/Learn2Reg/OASIS"
    dir_out = "/home/mbrudfors/Data/Learn2Reg/OASIS_registered"
    json_meta = os.path.join(dir_data, "OASIS_dataset.json")
    id_pair = 0
    resize_factor = 1  # Full resolution (set > 1 for faster testing)
    use_gpu = True
    loss = "lcc"

    os.makedirs(dir_out, exist_ok=True)

    # -------------------------------------------------------------------------
    # Part 1: Run LCC registration
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Part 1: LCC Registration")
    print("=" * 60)

    # Load metadata and get pair
    with open(json_meta, "r") as f:
        meta = json.load(f)
    pair = get_pair(meta, id_pair, dir_data, verbose=True)

    # Set output directory
    pair["fixed"]["id"] = get_subject_id(pair["fixed"]["image"])
    pair["moving"]["id"] = get_subject_id(pair["moving"]["image"])
    suffix = f"_resized_f{resize_factor}" if resize_factor not in [0, 1] else ""
    pair["dir_out"] = os.path.join(
        dir_out, f"{pair['fixed']['id']}_to_{pair['moving']['id']}{suffix}"
    )
    os.makedirs(pair["dir_out"], exist_ok=True)
    print(f"Output directory: {pair['dir_out']}")

    # Resize if requested
    pair = resize_pair(pair, factor=resize_factor, use_gpu=use_gpu)

    # Register
    pair = add_loss(pair, loss)
    register(loss, pair, verbose=True, print_gpu_use=True, use_gpu=use_gpu)

    # Create displacement field
    pair = create_displacement_field(pair, loss, use_gpu=use_gpu)

    # Reslice images and labels
    reslice(pair, loss, verbose=True, use_gpu=use_gpu)

    # -------------------------------------------------------------------------
    # Part 2: Convert to 3D Slicer grid transform
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Part 2: Creating 3D Slicer Grid Transform")
    print("=" * 60)

    slicer_dir = os.path.join(pair["dir_out"], "slicer")
    os.makedirs(slicer_dir, exist_ok=True)

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    grid_path = os.path.join(slicer_dir, "Grid.nii.gz")
    create_slicer_grid_transform(
        fixed_nii_path=pair["fixed"]["image"],
        affine_lta_path=pair["loss"][loss]["affine"],
        displacement_nii_path=pair["loss"][loss]["u"],
        output_path=grid_path,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Part 3: Copy files for 3D Slicer
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Part 3: Copying files for 3D Slicer")
    print("=" * 60)

    files_to_copy = {
        "fixed_image.nii.gz": pair["fixed"]["image"],
        "moving_image.nii.gz": pair["moving"]["image"],
        "moving_image_resliced.nii.gz": pair["loss"][loss]["moving"]["image"],
        "fixed_label.nii.gz": pair["fixed"]["label"],
        "moving_label.nii.gz": pair["moving"]["label"],
        "moving_label_resliced.nii.gz": pair["loss"][loss]["moving"]["label"],
    }

    for dest_name, src_path in files_to_copy.items():
        dest_path = os.path.join(slicer_dir, dest_name)
        shutil.copy2(src_path, dest_path)
        print(f"  Copied: {dest_name}")

    # -------------------------------------------------------------------------
    # Print verification instructions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Verification Instructions for 3D Slicer")
    print("=" * 60)
    print(f"\nAll files saved to: {slicer_dir}\n")
    print("1. Open 3D Slicer")
    print("2. File > Add Data > Choose Files to Add:")
    print("   - fixed_image.nii.gz (Volume)")
    print("   - moving_image.nii.gz (Volume)")
    print("   - moving_image_resliced.nii.gz (Volume)")
    print("   - Grid.nii.gz (GridTransform)")
    print("3. Go to Modules > Transforms")
    print("   - Select 'Grid' as the active transform")
    print("   - Under 'Transformed', move 'moving_image' to the transformed list")
    print("4. Compare: the transformed moving_image should match moving_image_resliced")
    print("5. Repeat with labels (fixed_label, moving_label, moving_label_resliced)")
    print("   to double-check\n")


if __name__ == "__main__":
    main()
