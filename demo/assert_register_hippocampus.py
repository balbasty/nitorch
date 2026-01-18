"""
HippocampusMR Registration Test Script

This script runs the registration pipeline from demo_register_hippocampus.ipynb
without visualization, and includes assertions to verify that computed dice scores
match expected values. Use this as a regression test when making changes to the
registration implementation.

Expected dice scores (with SEED=0):
- Before registration: 0.506290
- LCC loss: 0.800588
- Dice loss: 0.965107
- NMI loss: 0.810406
- MSE loss: 0.823405
- MAD loss: 0.772233
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
    reslice,
)

# Tolerance for dice score comparisons
# Note: GPU floating-point operations are not perfectly deterministic across runs,
# even with fixed seeds. A tolerance of 0.03 (3%) catches real regressions while
# allowing for numerical variance from non-deterministic CUDA operations.
DICE_TOLERANCE = 0.03

# Expected dice scores from notebook run
EXPECTED_DICE_BEFORE = 0.506290
EXPECTED_DICE_LCC = 0.800588
EXPECTED_DICE_DICE = 0.965107
EXPECTED_DICE_NMI = 0.810406
EXPECTED_DICE_MSE = 0.823405
EXPECTED_DICE_MAD = 0.772233


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
    id_pair = 0

    # Read dataset metadata
    with open(json_meta, 'r') as f:
        meta = json.load(f)
        
    # Get pair
    print(f"Getting pair {id_pair + 1} of {len(meta['registration_val'])}")
    pair = get_pair(meta, id_pair, dir_data, verbose=False)

    # Create unique subfolder for this fixed-moving pair
    pair["fixed"]["id"] = get_subject_id(pair["fixed"]["image"])
    pair["moving"]["id"] = get_subject_id(pair["moving"]["image"])
    pair["dir_out"] = os.path.join(dir_out, f"{pair['fixed']['id']}_to_{pair['moving']['id']}")
    os.makedirs(pair["dir_out"], exist_ok=True)
    print(f"\nOutput directory: {pair['dir_out']}")

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
    print("\n" + "=" * 60)
    print("Running LCC registration...")
    print("=" * 60)
    loss = "lcc"
    pair = add_loss(pair, loss)
    register(loss, pair, verbose=False, print_gpu_use=True)
    pair = create_displacement_field(pair, loss)
    reslice(pair, loss, verbose=False)
    
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
    print("\n" + "=" * 60)
    print("Running Dice registration...")
    print("=" * 60)
    loss = "dice"
    pair = add_loss(pair, loss)
    register(loss, pair, verbose=False, print_gpu_use=True)
    pair = create_displacement_field(pair, loss)
    reslice(pair, loss, verbose=False)
    
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
    print("\n" + "=" * 60)
    print("Running NMI registration...")
    print("=" * 60)
    loss = "nmi"
    pair = add_loss(pair, loss)
    register(loss, pair, verbose=False, print_gpu_use=True)
    pair = create_displacement_field(pair, loss)
    reslice(pair, loss, verbose=False)
    
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
    print("\n" + "=" * 60)
    print("Running MSE registration...")
    print("=" * 60)
    loss = "mse"
    pair = add_loss(pair, loss)
    register(loss, pair, verbose=False, print_gpu_use=True)
    pair = create_displacement_field(pair, loss)
    reslice(pair, loss, verbose=False)
    
    dice_scores, mean_dice = compute_dice_scores(
        pair["fixed"]["label"], 
        pair["loss"][loss]["moving"]["label"],
        loss=loss,
        verbose=False,
    )
    assert_dice_score(mean_dice, EXPECTED_DICE_MSE, "MSE")

    # =========================================================================
    # MAD Registration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Running MAD registration...")
    print("=" * 60)
    loss = "mad"
    pair = add_loss(pair, loss)
    register(loss, pair, verbose=False, print_gpu_use=True)
    pair = create_displacement_field(pair, loss)
    reslice(pair, loss, verbose=False)
    
    dice_scores, mean_dice = compute_dice_scores(
        pair["fixed"]["label"], 
        pair["loss"][loss]["moving"]["label"],
        loss=loss,
        verbose=False,
    )
    assert_dice_score(mean_dice, EXPECTED_DICE_MAD, "MAD")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nDice score summary:")
    print(f"  Before registration: {EXPECTED_DICE_BEFORE:.6f}")
    print(f"  LCC:  {EXPECTED_DICE_LCC:.6f}")
    print(f"  Dice: {EXPECTED_DICE_DICE:.6f}")
    print(f"  NMI:  {EXPECTED_DICE_NMI:.6f}")
    print(f"  MSE:  {EXPECTED_DICE_MSE:.6f}")
    print(f"  MAD:  {EXPECTED_DICE_MAD:.6f}")


if __name__ == "__main__":
    main()
