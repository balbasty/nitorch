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
from scipy.interpolate import interpn
from scipy.linalg import sqrtm

# Suppress FutureWarnings from interpol package
warnings.filterwarnings("ignore", category=FutureWarning, module="interpol")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

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

def load_affine_lta(path, nitorch=True):
    """Load a 4x4 affine matrix from an LTA file and return its square root.

    Parameters
    ----------
    path : str
        Path to the .lta file.
    nitorch : bool
        If True, read the raw matrix (lines 3-6 of the affine block).

    Returns
    -------
    np.ndarray
        (4, 4) matrix square root of the affine.
    """
    # Read the raw 4x4 matrix from the LTA file
    rows = []
    in_matrix = False
    for line in open(path, "r"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # After seeing the shape line (e.g. "1 4 4"), read 4 rows
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
    # Return the matrix square root
    return np.real(sqrtm(mat))


def get_id(shape):
    """Create an identity grid for a given 3D volume shape.

    Returns
    -------
    np.ndarray
        (X, Y, Z, 3) array of voxel coordinates.
    """
    coords = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        np.arange(shape[2], dtype=np.float64),
        indexing="ij",
    )
    return np.stack(coords, axis=-1)


def affine_transform(grid, mat):
    """Apply a 4x4 affine matrix to a coordinate grid.

    Parameters
    ----------
    grid : np.ndarray
        (..., 3) coordinate array.
    mat : np.ndarray
        (4, 4) affine matrix.

    Returns
    -------
    np.ndarray
        (..., 3) transformed coordinates.
    """
    shape = grid.shape
    flat = grid.reshape(-1, 3)
    # Homogeneous coordinates
    ones = np.ones((flat.shape[0], 1), dtype=flat.dtype)
    homo = np.concatenate([flat, ones], axis=1)
    transformed = (mat @ homo.T).T[:, :3]
    return transformed.reshape(shape)


def inv(mat):
    """Invert a 4x4 matrix."""
    return np.linalg.inv(mat)


def displace(grid, u):
    """Add a displacement field to a coordinate grid.

    Parameters
    ----------
    grid : np.ndarray
        (..., 3) coordinates.
    u : np.ndarray
        (..., 3) displacements (already sampled at grid locations).

    Returns
    -------
    np.ndarray
        (..., 3) displaced coordinates.
    """
    return grid + u


def pull(dat, grid):
    """Resample a vector field at given coordinates using linear interpolation.

    Parameters
    ----------
    dat : np.ndarray
        (X, Y, Z, 3) vector field data.
    grid : np.ndarray
        (..., 3) sampling coordinates in voxel space.

    Returns
    -------
    np.ndarray
        (..., 3) resampled vectors.
    """
    shape = dat.shape[:3]
    points = (
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        np.arange(shape[2], dtype=np.float64),
    )
    out = np.zeros_like(grid)
    for c in range(3):
        out[..., c] = interpn(
            points,
            dat[..., c],
            grid[..., :3],
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
    return out


def save_nii(path, data, affine_mat):
    """Save a displacement field as a NIfTI file with vector intent.

    Parameters
    ----------
    path : str
        Output file path.
    data : np.ndarray
        (X, Y, Z, 1, 3) displacement data (Slicer grid format).
    affine_mat : np.ndarray
        (4, 4) affine matrix for the NIfTI header.
    """
    img = nib.Nifti1Image(data.astype(np.float32), affine_mat)
    img.header['intent_code'] = 1006  # NIFTI_INTENT_DISPVECT
    nib.save(img, path)


def create_slicer_grid_transform(
    fixed_nii_path,
    affine_lta_path,
    displacement_nii_path,
    output_path,
):
    """Convert nitorch registration transforms to a 3D Slicer grid transform.

    Composes the affine (split as square root, applied twice) and displacement
    field into a single grid transform in LPS convention for 3D Slicer.

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
    """
    # Load data
    nii_fixed = nib.load(fixed_nii_path)
    nii_disp = nib.load(displacement_nii_path)
    dat_disp = nii_disp.get_fdata()

    # Squeeze trailing singleton dims from displacement field if needed
    # nitorch displacement fields can be (X, Y, Z, 1, 3) — squeeze to (X, Y, Z, 3)
    while dat_disp.ndim > 4:
        dat_disp = dat_disp.squeeze(-2)

    # Load affine square root
    affine_sqrt = load_affine_lta(affine_lta_path)

    # 1. Identity grid in fixed voxel space
    id_grid = get_id(nii_fixed.shape[:3])

    # 2. Voxel -> RAS (fixed image affine)
    grid = affine_transform(id_grid, nii_fixed.affine)

    # 3. First half of affine (square root)
    grid = affine_transform(grid, affine_sqrt)

    # 4. Sample displacement field
    #    Map grid to displacement voxel space
    grid = affine_transform(grid, inv(nii_disp.affine))
    #    Sample the displacement at these voxel coordinates
    sampled_disp = pull(dat_disp, grid)
    #    Map back to displacement RAS space and add displacement
    grid = affine_transform(grid, nii_disp.affine)
    grid = displace(grid, sampled_disp)

    # 5. Second half of affine (square root)
    grid = affine_transform(grid, affine_sqrt)

    # Convert deformation to displacement (subtract fixed RAS positions)
    grid -= affine_transform(id_grid, nii_fixed.affine)

    # NOTE: With intent code 1006 (NIFTI_INTENT_DISPVECT), Slicer expects
    # displacement vectors in RAS, so no LPS conversion is needed.
    # With intent code 1007 (NIFTI_INTENT_VECTOR), Slicer assumes LPS,
    # so you would need: grid[..., 0] *= -1; grid[..., 1] *= -1

    # Add singleton dimension for Slicer grid format: (X, Y, Z, 1, 3)
    grid = np.expand_dims(grid, axis=-2)

    # Save
    save_nii(output_path, grid, nii_fixed.affine)
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

    grid_path = os.path.join(slicer_dir, "Grid.nii.gz")
    create_slicer_grid_transform(
        fixed_nii_path=pair["fixed"]["image"],
        affine_lta_path=pair["loss"][loss]["affine"],
        displacement_nii_path=pair["loss"][loss]["u"],
        output_path=grid_path,
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
