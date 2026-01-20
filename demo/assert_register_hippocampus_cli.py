"""
HippocampusMR Registration Test Script

This script runs the registration pipeline from demo_register_hippocampus.ipynb
without visualization, and includes assertions to verify that computed dice scores
match expected values. Use this as a regression test when making changes to the
registration implementation.

Expected dice scores (with SEED=0, resize_factor=3, use_gpu=False):
- Before registration: 0.532860
- LCC loss: 0.739329
- Dice loss: 0.979994
- NMI loss: 0.714330
- MSE loss: 0.707343
"""

import os
import json
import warnings
import random
import numpy as np
import torch

# Set random seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Suppress FutureWarnings from interpol package (PyTorch AMP deprecation warnings)
warnings.filterwarnings("ignore", category=FutureWarning, module="interpol")
# Also suppress in subprocesses (nitorch CLI commands)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

# Import helper functions from demo_helpers module
from demo_helpers import (
    add_loss,
    compute_dice_scores,
    create_displacement_field,
    get_pair,
    get_subject_id,
    register,
    resize_pair,
    reslice,
)

# Tolerance for dice score comparisons
# Note: GPU floating-point operations are not perfectly deterministic across runs,
# even with fixed seeds. A tolerance of 0.03 (3%) catches real regressions while
# allowing for numerical variance from non-deterministic CUDA operations.
DICE_TOLERANCE = 0.03

# Expected dice scores from notebook run (resize_factor=3, use_gpu=False)
EXPECTED_DICE_BEFORE = 0.532860
EXPECTED_DICE_LCC = 0.739329
EXPECTED_DICE_DICE = 0.979994
EXPECTED_DICE_NMI = 0.714330
EXPECTED_DICE_MSE = 0.707343


def assert_dice_score(computed, expected, name):
    """Assert that computed dice score matches expected value within tolerance."""
    assert abs(computed - expected) < DICE_TOLERANCE, (
        f"{name} dice score mismatch: computed={computed:.6f}, expected={expected:.6f}, "
        f"diff={abs(computed - expected):.6f}, tolerance={DICE_TOLERANCE}"
    )
    print(f"✓ {name} dice score verified: {computed:.6f} (expected: {expected:.6f})")


def main():
    # Parameters
    dir_data = "/home/mbrudfors/Data/Learn2Reg/HippocampusMR"
    dir_out = "/home/mbrudfors/Data/Learn2Reg/HippocampusMR_registered"
    os.makedirs(dir_out, exist_ok=True)
    json_meta = os.path.join(dir_data, "HippocampusMR_dataset.json")
    # Remember, if you change the id_pair, you need to change the expected dice scores above
    id_pair = 0
    resize_factor = 3  # Downsampling factor for faster testing
    use_gpu = False  # Run on CPU for reproducibility

    # Read dataset metadata
    with open(json_meta, 'r') as f:
        meta = json.load(f)
        
    # Get pair
    print(f"Getting pair {id_pair + 1} of {len(meta['registration_val'])}")
    pair = get_pair(meta, id_pair, dir_data, verbose=False)

    # Create unique subfolder for this fixed-moving pair
    pair["fixed"]["id"] = get_subject_id(pair["fixed"]["image"])
    pair["moving"]["id"] = get_subject_id(pair["moving"]["image"])
    suffix = f"_resized_f{resize_factor}" if resize_factor not in [0, 1] else ""
    pair["dir_out"] = os.path.join(dir_out, f"{pair['fixed']['id']}_to_{pair['moving']['id']}{suffix}")
    os.makedirs(pair["dir_out"], exist_ok=True)
    print(f"\nOutput directory: {pair['dir_out']}")

    # Resize images if requested (for quick testing)
    pair = resize_pair(pair, factor=resize_factor, verbose=False, use_gpu=use_gpu)

    # Track total runtime
    total_elapsed_time = 0.0

    # Compute Dice scores before registration
    print("\n" + "=" * 60)
    print("Computing dice scores BEFORE registration...")
    print("=" * 60)
    dice_scores, mean_dice = compute_dice_scores(
        pair["fixed"]["label"], 
        pair["moving"]["label"],
        verbose=False,
    )
    assert_dice_score(mean_dice, EXPECTED_DICE_BEFORE, "Before registration")

    # =========================================================================
    # LCC Registration
    # =========================================================================
    loss = "lcc"
    pair = add_loss(pair, loss)
    elapsed_time = register(loss, pair, verbose=False, print_gpu_use=True, use_gpu=use_gpu)
    total_elapsed_time += elapsed_time
    pair = create_displacement_field(pair, loss, use_gpu=use_gpu)
    reslice(pair, loss, verbose=False, use_gpu=use_gpu)
    
    dice_scores, mean_dice = compute_dice_scores(
        pair["fixed"]["label"], 
        pair["loss"][loss]["moving"]["label"],
        loss=loss,        
        verbose=False,
    )
    assert_dice_score(mean_dice, EXPECTED_DICE_LCC, "LCC")

    # =========================================================================
    # Dice Registration
    # =========================================================================
    loss = "dice"
    pair = add_loss(pair, loss)
    elapsed_time = register(loss, pair, verbose=False, print_gpu_use=True, use_gpu=use_gpu)
    total_elapsed_time += elapsed_time
    pair = create_displacement_field(pair, loss, use_gpu=use_gpu)
    reslice(pair, loss, verbose=False, use_gpu=use_gpu)
    
    dice_scores, mean_dice = compute_dice_scores(
        pair["fixed"]["label"], 
        pair["loss"][loss]["moving"]["label"],
        loss=loss,
        verbose=False,
    )
    assert_dice_score(mean_dice, EXPECTED_DICE_DICE, "Dice")

    # =========================================================================
    # NMI Registration
    # =========================================================================
    loss = "nmi"
    pair = add_loss(pair, loss)
    elapsed_time = register(loss, pair, verbose=False, print_gpu_use=True, use_gpu=use_gpu)
    total_elapsed_time += elapsed_time
    pair = create_displacement_field(pair, loss, use_gpu=use_gpu)
    reslice(pair, loss, verbose=False, use_gpu=use_gpu)
    
    dice_scores, mean_dice = compute_dice_scores(
        pair["fixed"]["label"], 
        pair["loss"][loss]["moving"]["label"],
        loss=loss,
        verbose=False,
    )
    assert_dice_score(mean_dice, EXPECTED_DICE_NMI, "NMI")

    # =========================================================================
    # MSE Registration
    # =========================================================================
    loss = "mse"
    pair = add_loss(pair, loss)
    elapsed_time = register(loss, pair, verbose=False, print_gpu_use=True, use_gpu=use_gpu)
    total_elapsed_time += elapsed_time
    pair = create_displacement_field(pair, loss, use_gpu=use_gpu)
    reslice(pair, loss, verbose=False, use_gpu=use_gpu)
    
    dice_scores, mean_dice = compute_dice_scores(
        pair["fixed"]["label"], 
        pair["loss"][loss]["moving"]["label"],
        loss=loss,
        verbose=False,
    )
    assert_dice_score(mean_dice, EXPECTED_DICE_MSE, "MSE")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print(f"\nTotal registration runtime: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    print("=" * 60)

if __name__ == "__main__":
    main()
