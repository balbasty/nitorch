"""
Helper functions for the NITorch registration demo notebooks.

This module provides utilities for:
- Managing registration pair dictionaries
- Computing Dice scores for registration evaluation
- Running nitorch CLI commands with GPU monitoring
- Visualizing registration results
- Resizing image pairs for faster testing
- Python API equivalents of CLI functions (resize, register, reslice)
"""

import os
import subprocess
import threading
import time

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pynvml
import torch

# Import nitorch modules for Python API functions
from nitorch import io, spatial


def resize_pair(pair, factor, output_dir=None, verbose=True, use_gpu=True):
    """
    Resize a fixed-moving image pair for faster registration testing.
    
    Uses the nitorch resize CLI to create smaller versions of the images
    and labels.
    
    Parameters
    ----------
    pair : dict
        Dictionary containing 'fixed' and 'moving' keys, each with 'image' 
        and 'label' paths to NIfTI files.
    factor : float
        Downsampling factor (intuitive definition):
        - factor > 1: downsample (e.g., factor=2 gives half resolution, fewer voxels)
        - factor < 1: upsample (e.g., factor=0.5 gives double resolution, more voxels)
        - factor == 0 or 1: no resizing, return original pair
    output_dir : str, optional
        Directory where resized NIfTI files will be saved. If None, uses
        the pair's output directory with a 'resized' subdirectory.
    verbose : bool, optional
        If True, print information about the resizing. Default is True.
    use_gpu : bool, optional
        If True, use GPU for resizing (faster). If False, use CPU. Default is True.
    
    Returns
    -------
    pair : dict
        Updated pair dictionary with paths pointing to resized NIfTI files.
        Original paths are preserved as 'image_orig' and 'label_orig'.
    """
    # Skip resizing if factor is 0 or 1
    if factor == 0 or factor == 1:
        if verbose:
            print("Resize factor is 0 or 1, skipping resizing.")
        return pair
    
    # Convert intuitive factor to nitorch factor
    # User wants: factor=2 means half the voxels (downsample)
    # Nitorch uses: factor=2 means twice the voxels (upsample)
    # So we invert: nitorch_factor = 1/factor
    nitorch_factor = 1.0 / factor
    
    if output_dir is None:
        output_dir = os.path.join(pair.get("dir_out", "."), "resized")
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Resizing pair with factor {factor}...")
        print(f"  Output directory: {output_dir}")
    
    def resize_file(input_path, output_path, is_label=False):
        """Resize a single NIfTI file using nitorch resize CLI."""
        # For labels, use nearest neighbor interpolation
        # For images, use cubic interpolation
        interp_order = 0 if is_label else 3
        
        cmd = (
            f"nitorch resize {input_path} "
            f"-f {nitorch_factor} "
            f"-i {interp_order} "
            f"-o {output_path} "
            f"-b dct2"
        )
        if use_gpu:
            cmd += " --gpu"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(f"Resize failed for {input_path}:\n{error_msg}\nCommand: {cmd}")
        
        # Verify file was created
        if not os.path.exists(output_path):
            # Check if nitorch wrote to a default location instead
            stdout_info = result.stdout if result.stdout else "(no stdout)"
            stderr_info = result.stderr if result.stderr else "(no stderr)"
            raise RuntimeError(
                f"Resize command completed but output file not found: {output_path}\n"
                f"Command: {cmd}\n"
                f"stdout: {stdout_info}\n"
                f"stderr: {stderr_info}"
            )
        
        return output_path
    
    # Generate output paths
    paths_resized = {}
    for role in ["fixed", "moving"]:
        img_basename = os.path.basename(pair[role]["image"]).replace(
            '.nii.gz', f'.resized_f{factor}.nii.gz'
        )
        lbl_basename = os.path.basename(pair[role]["label"]).replace(
            '.nii.gz', f'.label.resized_f{factor}.nii.gz'
        )
        paths_resized[role] = {
            "image": os.path.join(output_dir, img_basename),
            "label": os.path.join(output_dir, lbl_basename),
        }
    
    # Resize each image and label (always regenerate, no caching)
    for role in ["fixed", "moving"]:
        if verbose:
            print(f"\n  Resizing {role}...")
        
        # Get original info
        nii_orig_img = nib.load(pair[role]["image"])
        nii_orig_lbl = nib.load(pair[role]["label"])
        
        # Resize image
        resize_file(
            pair[role]["image"],
            paths_resized[role]["image"],
            is_label=False
        )
        
        # Resize label
        resize_file(
            pair[role]["label"],
            paths_resized[role]["label"],
            is_label=True
        )
        
        # Print info about resized files immediately after resizing
        if verbose:
            nii_resized_img = nib.load(paths_resized[role]["image"])
            nii_resized_lbl = nib.load(paths_resized[role]["label"])
            
            # Get shapes (squeeze singleton dimensions for display)
            resized_img_shape = tuple(s for s in nii_resized_img.shape if s > 1)
            resized_lbl_shape = tuple(s for s in nii_resized_lbl.shape if s > 1)
            
            print(f"    Image:")
            print(f"      Original: shape={nii_orig_img.shape}, "
                  f"voxel_size={tuple(np.round(nii_orig_img.header.get_zooms()[:3], 2))}")
            print(f"      Resized:  shape={resized_img_shape}, "
                  f"voxel_size={tuple(np.round(nii_resized_img.header.get_zooms()[:3], 2))}")
            
            print(f"    Label:")
            print(f"      Original: shape={nii_orig_lbl.shape}, "
                  f"labels={list(np.unique(nii_orig_lbl.get_fdata()).astype(int))}")
            print(f"      Resized:  shape={resized_lbl_shape}, "
                  f"labels={list(np.unique(nii_resized_lbl.get_fdata()).astype(int))}")
    
    # Update pair with resized paths, preserving originals
    for role in ["fixed", "moving"]:
        pair[role]["image_orig"] = pair[role]["image"]
        pair[role]["label_orig"] = pair[role]["label"]
        pair[role]["image"] = paths_resized[role]["image"]
        pair[role]["label"] = paths_resized[role]["label"]
    
    if verbose:
        print(f"\n{'='*60}\n")
    
    # Store resize info
    pair["resize_factor"] = factor
    pair["dir_resized"] = output_dir
    
    return pair


def add_loss(pair, loss):
    """
    Add loss-specific output paths to the pair dictionary.
    
    Creates a subdirectory for the specified loss function and populates
    the pair dictionary with paths for output images, labels, affine
    transforms, and stationary velocity fields.
    
    Parameters
    ----------
    pair : dict
        Dictionary containing fixed/moving image and label paths, and output directory.
    loss : str
        Name of the loss function (e.g., 'lcc', 'dice').
    
    Returns
    -------
    pair : dict
        Updated pair dictionary with loss-specific output paths added.
    """
    dir_out_loss = os.path.join(pair["dir_out"], loss)
    os.makedirs(dir_out_loss, exist_ok=True)
    
    # Get resize factor suffix if resizing was applied
    resize_factor = pair.get("resize_factor", 0)
    factor_suffix = f"_f{resize_factor}" if resize_factor not in [0, 1] else ""
    
    pair["loss"] = {
        loss: {
            "dir_out": dir_out_loss,
            "fixed": {
                    "image": os.path.join(dir_out_loss, os.path.basename(pair["fixed"]["image"]).replace('.nii.gz', '.image.moved.nii.gz')),
                    "label": os.path.join(dir_out_loss, os.path.basename(pair["fixed"]["label"]).replace('.nii.gz', '.label.moved.nii.gz'))
                },
            "moving": {
                    "image": os.path.join(dir_out_loss, os.path.basename(pair["moving"]["image"]).replace('.nii.gz', '.image.moved.nii.gz')),
                    "label": os.path.join(dir_out_loss, os.path.basename(pair["moving"]["label"]).replace('.nii.gz', '.label.moved.nii.gz'))
                },
            "affine": os.path.join(dir_out_loss, f'affine{factor_suffix}.lta'),
            "svf": os.path.join(dir_out_loss, f'svf{factor_suffix}.nii.gz')
        }
    }
    return pair


def compute_dice_scores(label1, label2, exclude_zero=True, loss=None, verbose=True):
    """
    Compute Dice score for each label between two segmentation masks.
    
    Parameters
    ----------
    label1 : str or np.ndarray
        Path to first label NIfTI file or numpy array (e.g., fixed labels)
    label2 : str or np.ndarray
        Path to second label NIfTI file or numpy array (e.g., moving labels)
    exclude_zero : bool
        Whether to exclude background (label 0) from computation
    loss : str, optional
        Name of the loss function used for registration. If provided, will be
        included in the printed output.
    verbose : bool, optional
        If True, print the mean dice score. Default is True.
    
    Returns
    -------
    dice_scores : dict
        Dictionary mapping label ID to Dice score
    mean_dice : float
        Mean Dice score across all labels
    """
    # Load labels if paths are provided
    def load_vol(x):
        if isinstance(x, str):
            vol = nib.load(x).get_fdata()
        else:
            vol = x
        # Squeeze trailing singleton dimensions (e.g., from resliced outputs)
        while vol.ndim > 3 and vol.shape[-1] == 1:
            vol = vol.squeeze(-1)
        return vol
    
    lbl1 = load_vol(label1)
    lbl2 = load_vol(label2)
    
    # Get all unique labels
    all_labels = np.union1d(np.unique(lbl1), np.unique(lbl2))
    if exclude_zero:
        all_labels = all_labels[all_labels != 0]
    
    dice_scores = {}
    for lbl in all_labels:
        mask1 = (lbl1 == lbl)
        mask2 = (lbl2 == lbl)
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1) + np.sum(mask2)
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = 2 * intersection / union
        dice_scores[int(lbl)] = dice
    
    mean_dice = np.mean(list(dice_scores.values())) if dice_scores else 0.0
    
    if verbose:
        # Print mean dice score
        print(f"\n{'='*60}")
        if loss is not None:
            print(f"{loss.upper()} loss mean dice score: {mean_dice:.6f}")
        else:
            print(f"Mean dice score: {mean_dice:.6f}")
        print(f"{'='*60}")
    
    return dice_scores, mean_dice


def create_displacement_field(pair, loss, use_gpu=True):
    """
    Create a displacement field by exponentiating the stationary velocity field.
    
    Uses the nitorch vexp command to exponentiate the SVF stored in the pair
    dictionary and saves the resulting displacement field.
    
    Parameters
    ----------
    pair : dict
        Dictionary containing registration paths including the SVF.
    loss : str
        Name of the loss function used for registration.
    use_gpu : bool, optional
        If True, use GPU for computation (faster). If False, use CPU. Default is True.
    
    Returns
    -------
    pair : dict
        Updated pair dictionary with displacement field path added.
    """
    # Get resize factor suffix if resizing was applied
    resize_factor = pair.get("resize_factor", 0)
    factor_suffix = f"_f{resize_factor}" if resize_factor not in [0, 1] else ""
    
    # Exponentiate stationary velocity field to create displacement field
    pair["loss"][loss]["u"] = os.path.join(
        pair["loss"][loss]["dir_out"],
        f'u{factor_suffix}.nii.gz'
    )
    cmd = "nitorch vexp " + pair["loss"][loss]["svf"]
    if use_gpu:
        cmd += " -gpu 0"
    cmd += " -o " + pair["loss"][loss]["u"]
    # Exponentiate
    subprocess.run(cmd, shell=True, check=True)
    return pair


def get_pair(meta, id_pair, dir_data, verbose=False):
    """
    Retrieve a fixed-moving image pair from the dataset metadata.
    
    Constructs paths to the fixed and moving images and their corresponding
    label files based on the dataset metadata and pair index.
    
    Parameters
    ----------
    meta : dict
        Dataset metadata containing registration pairs.
    id_pair : int
        Index of the pair to retrieve from the validation set.
    dir_data : str
        Root directory of the dataset.
    verbose : bool, optional
        If True, print paths, image shapes, voxel sizes, and label values. 
        Default is False.
    
    Returns
    -------
    pair : dict
        Dictionary with 'fixed' and 'moving' keys, each containing
        'image' and 'label' paths.
    
    Raises
    ------
    ValueError
        If id_pair is out of range for the available registration pairs.
    """
    if id_pair >= len(meta["registration_val"]):
        raise ValueError(f"Index {id_pair} is out of range for meta['registration_val'] of length {len(meta['registration_val'])}")
    fixed_image = os.path.normpath(os.path.join(dir_data, meta["registration_val"][id_pair]["fixed"]))
    moving_image = os.path.normpath(os.path.join(dir_data, meta["registration_val"][id_pair]["moving"]))
    fixed_label = fixed_image.replace("imagesTr", "labelsTr")
    moving_label = moving_image.replace("imagesTr", "labelsTr")
    pair = {
        "fixed": {"image": fixed_image, "label": fixed_label},
        "moving": {"image": moving_image, "label": moving_label},
    }
    if verbose:
        print(f"\n{'='*60}")
        for role in ["fixed", "moving"]:
            print(f"{role.upper()}:")
            # Load and print image info
            img = nib.load(pair[role]["image"])
            img_data = img.get_fdata()
            print(f"  Image: {pair[role]['image']}")
            print(f"    Shape: {img_data.shape}")
            print(f"    Dtype: {img_data.dtype}")
            print(f"    Voxel size: {tuple(np.round(img.header.get_zooms()[:3], 3))}")
            # Load and print label info
            lbl = nib.load(pair[role]["label"])
            lbl_data = lbl.get_fdata()
            unique_labels = np.unique(lbl_data).astype(int)
            print(f"  Label: {pair[role]['label']}")
            print(f"    Shape: {lbl_data.shape}")
            print(f"    Dtype: {lbl_data.dtype}")
            print(f"    Num. labels: {len(unique_labels)}")
            print(f"    Label values: {list(unique_labels)}")
        print(f"{'='*60}")
    return pair


def get_subject_id(path):
    """
    Extract subject ID from an image file path.
    
    Removes the directory path, file extension, and channel suffix
    to extract the base subject identifier.
    
    Parameters
    ----------
    path : str
        Path to a NIfTI image file.
    
    Returns
    -------
    str
        Subject ID (e.g., 'OASIS_0395' from 'OASIS_0395_0000.nii.gz').
    """
    basename = os.path.basename(path).replace('.nii.gz', '')
    # Remove the '_0000' suffix if present
    if basename.endswith('_0000'):
        return basename[:-5]
    return basename


def register(loss, pair, reg_lambda=10, verbose=True, print_gpu_use=True, use_gpu=True):
    """
    Perform affine and nonlinear registration using nitorch.
    
    Runs the nitorch register command with the specified loss function,
    performing both affine and diffeomorphic (SVF) registration. Monitors
    and reports runtime and GPU memory usage.
    
    Parameters
    ----------
    loss : str
        Loss function to use ('lcc', 'dice', etc.).
    pair : dict
        Dictionary containing paths to fixed/moving images and output locations.
    reg_lambda : float, optional
        Regularization weight for the nonlinear registration. Default is 10.
    verbose : bool, optional
        If True, print the registration command. Default is True.
    print_gpu_use : bool, optional
        If True, print the GPU memory usage. Default is True.
    use_gpu : bool, optional
        If True, use GPU for registration (faster). If False, use CPU. Default is True.
    
    Returns
    -------
    elapsed_time : float
        Total runtime in seconds.
    """
    def get_register_cmd(loss, pair, reg_lambda, verbose, use_gpu):
        verbose_str = "1" if verbose else "0"
        pth_affine = pair["loss"][loss]["affine"]
        pth_svf = pair["loss"][loss]["svf"]
        if loss in ["dice", "cat", "cce"]:
            pth_fixed = pair["fixed"]["label"]
            pth_moving = pair["moving"]["label"]
            pth_fixed_moved = pair["loss"][loss]["fixed"]["label"]
            pth_moving_moved = pair["loss"][loss]["moving"]["label"]
            label = " --label"
        else:
            pth_fixed = pair["fixed"]["image"]
            pth_moving = pair["moving"]["image"]
            pth_fixed_moved = pair["loss"][loss]["fixed"]["image"]
            pth_moving_moved = pair["loss"][loss]["moving"]["image"]
            label = ""
        
        cmd = "nitorch register" \
            + " --verbose " + verbose_str
        if use_gpu:
            cmd += " --gpu 0"
        cmd += " @loss " + loss \
            + " @@fix " + pth_fixed + label + " --fwhm 1 -b dct2 -o false -r " + pth_fixed_moved \
            + " @@mov " + pth_moving + label + " --fwhm 1 -b dct2 -o false -r " + pth_moving_moved \
            + " @affine affine -o " + pth_affine \
            + " @@optim -t 1e-3 -n 64" \
            + " @nonlin svf " + str(reg_lambda) + " -o " + pth_svf + " -h \"\"" \
            + " @@optim -t 1e-3 -n 64" \
            + " @optim i -t 1e-3 -n 64" \
            + " @pyramid --levels 0 1 2"
        
        return cmd
    
    cmd = get_register_cmd(loss, pair, reg_lambda, verbose, use_gpu)

    if use_gpu:
        elapsed_time, peak_gpu_mb = run_with_monitoring(cmd, gpu_id=0)
    else:
        import time
        start_time = time.time()
        result = subprocess.run(cmd, shell=True)
        elapsed_time = time.time() - start_time
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")
        peak_gpu_mb = 0

    if print_gpu_use:
        print(f"\n{'='*60}")
        print(f"{loss.upper()} registration completed...")
        print(f"Runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        if use_gpu:
            print(f"Peak GPU Memory: {peak_gpu_mb:.1f} MB ({peak_gpu_mb/1024:.2f} GB)")
        print(f"{'='*60}")

    return elapsed_time


def reslice(pair, loss, verbose=True, use_gpu=True):
    """
    Reslice images or labels using the computed registration transforms.
    
    Applies the affine and SVF transforms to resample images between
    fixed and moving spaces in both directions.
    
    Parameters
    ----------
    pair : dict
        Dictionary containing paths to images and registration transforms.
    loss : str
        Name of the loss function used for registration.
    verbose : bool, optional
        Whether to print progress messages. Default is True.
    use_gpu : bool, optional
        If True, use GPU for reslicing (faster). If False, use CPU. Default is True.
    """
    def get_reslice_cmd(pth_moving, pth_fixed, pth_resliced, pth_affine, pth_svf, interpolation):
        cmd = "nitorch reslice" \
            + " " + pth_moving \
            + " --linear-square " + pth_affine \
            + " --velocity " + pth_svf \
            + " --linear-square " + pth_affine \
            + " --target " + pth_fixed \
            + " --output " + pth_resliced \
            + " --interpolation " + interpolation
        if use_gpu:
            cmd += " -gpu 0"
        cmd += " --bound zero"
        if not verbose:
            cmd += " --quiet"
        return cmd

    if loss in ["dice", "cat", "cce"]:
        what = "image"
    else:
        what = "label"

    # Set interpolation order
    if what == "label":
        interpolation = "l"
    else:
        interpolation = "3"

    # Reslice moving to fixed
    pth_moving = pair["moving"][what]
    pth_fixed = pair["fixed"][what]
    pth_resliced = pair["loss"][loss]["moving"][what]
    pth_affine = pair["loss"][loss]["affine"]
    pth_svf = pair["loss"][loss]["svf"]    
    cmd = get_reslice_cmd(
        pth_moving,
        pth_fixed,
        pth_resliced,
        pth_affine,
        pth_svf,
        interpolation,
    )
    subprocess.run(cmd, shell=True, check=True)
    
    # Reslice fixed to moving
    pth_moving = pair["fixed"][what]
    pth_fixed = pair["moving"][what]
    pth_resliced = pair["loss"][loss]["fixed"][what]
    cmd = get_reslice_cmd(
        pth_moving,
        pth_fixed,
        pth_resliced,
        pth_affine,
        pth_svf,
        interpolation,
    )
    subprocess.run(cmd, shell=True, check=True)


def run_with_monitoring(cmd, gpu_id=0, poll_interval=0.1):
    """
    Run a shell command while monitoring runtime and GPU memory usage.
    
    Executes the command in a subprocess while a background thread
    monitors GPU memory consumption at regular intervals.
    
    Parameters
    ----------
    cmd : str
        Shell command to execute.
    gpu_id : int, optional
        GPU device index to monitor. Default is 0.
    poll_interval : float, optional
        Time in seconds between GPU memory checks. Default is 0.1.
    
    Returns
    -------
    elapsed_time : float
        Total runtime in seconds.
    peak_memory_mb : float
        Peak GPU memory usage in megabytes.
    """
    peak_memory_mb = 0
    stop_monitoring = threading.Event()
    
    def monitor_gpu():
        nonlocal peak_memory_mb
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        while not stop_monitoring.is_set():
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            current_mb = mem_info.used / (1024 ** 2)
            peak_memory_mb = max(peak_memory_mb, current_mb)
            time.sleep(poll_interval)
        pynvml.nvmlShutdown()
    
    # Start GPU monitoring thread
    monitor_thread = threading.Thread(target=monitor_gpu)
    monitor_thread.start()
    
    # Set up environment to suppress FutureWarnings from interpol
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    
    # Run the command and measure time
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, env=env)
    elapsed_time = time.time() - start_time
    
    # Stop monitoring
    stop_monitoring.set()
    monitor_thread.join()
    
    # Check if command failed
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")
    
    return elapsed_time, peak_memory_mb


def visualize_dice_scores(dice_scores, mean_dice=None, title="Dice Scores per Label", 
                          figsize=(8, 6), color_by_value=True):
    """
    Visualize Dice scores as a horizontal bar plot sorted by score.
    
    Parameters
    ----------
    dice_scores : dict
        Dictionary mapping label ID to Dice score
    mean_dice : float, optional
        Mean Dice score (computed if not provided)
    title : str
        Title for the plot
    figsize : tuple
        Figure size
    color_by_value : bool
        If True, color bars by Dice value (red=low, green=high)
    """
    if mean_dice is None:
        mean_dice = np.mean(list(dice_scores.values()))
    
    # Sort by Dice score
    sorted_items = sorted(dice_scores.items(), key=lambda x: x[1])
    labels = [f"Label {k}" for k, v in sorted_items]
    scores = [v for k, v in sorted_items]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by value: red (low) -> yellow (mid) -> green (high)
    if color_by_value:
        colors = plt.cm.RdYlGn(np.array(scores))
    else:
        colors = 'steelblue'
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='gray', linewidth=0.5)
    
    # Add mean line
    ax.axvline(x=mean_dice, color='navy', linestyle='--', linewidth=2, 
               label=f'Mean Dice: {mean_dice:.3f}')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Dice Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(f'{title}\nMean Dice: {mean_dice:.3f}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    # Add grid for readability
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center', fontsize=7)
    
    plt.tight_layout()


def visualize_pair(fixed_image, moving_image, fixed_label=None, moving_label=None, 
                   slice_idx=None, axis=2, title="Fixed vs Moving", alpha=0.4,
                   n_contours=8, figsize=None):
    """
    Visualize fixed and moving image/label pairs side by side with overlay.
    
    Parameters
    ----------
    fixed_image : str or np.ndarray
        Path to fixed image NIfTI file or numpy array
    moving_image : str or np.ndarray
        Path to moving image NIfTI file or numpy array
    fixed_label : str or np.ndarray, optional
        Path to fixed label NIfTI file or numpy array
    moving_label : str or np.ndarray, optional
        Path to moving label NIfTI file or numpy array
    slice_idx : int, optional
        Slice index to display. If None, uses middle slice.
    axis : int
        Axis along which to take the slice (0, 1, or 2). Default is 2 (axial).
    title : str
        Title for the figure
    alpha : float
        Transparency for label overlay (0-1)
    n_contours : int
        Number of contour levels for the overlay
    figsize : tuple
        Figure size
    """
    if figsize is None and fixed_label is None:
        figsize = (8, 4)
    elif figsize is None and fixed_label is not None:
        figsize = (8, 8)
        
    # Load images if paths are provided
    def load_vol(x):
        if isinstance(x, str):
            vol = nib.load(x).get_fdata()
        else:
            vol = x
        # Squeeze trailing singleton dimensions (e.g., from resliced outputs)
        while vol.ndim > 3 and vol.shape[-1] == 1:
            vol = vol.squeeze(-1)
        return vol
    
    fix_img = load_vol(fixed_image)
    mov_img = load_vol(moving_image)
    fix_lbl = load_vol(fixed_label) if fixed_label is not None else None
    mov_lbl = load_vol(moving_label) if moving_label is not None else None
    
    # Determine slice index
    if slice_idx is None:
        slice_idx = fix_img.shape[axis] // 2
    
    # Extract slices
    def get_slice(vol, idx, ax):
        slc = [slice(None)] * vol.ndim
        slc[ax] = idx
        return vol[tuple(slc)]
    
    fix_slice = get_slice(fix_img, slice_idx, axis)
    mov_slice = get_slice(mov_img, slice_idx, axis)
    fix_lbl_slice = get_slice(fix_lbl, slice_idx, axis) if fix_lbl is not None else None
    mov_lbl_slice = get_slice(mov_lbl, slice_idx, axis) if mov_lbl is not None else None
    
    # Create colormap for labels (random colors, transparent background)
    label_cmap = None
    n_labels = 0
    if fix_lbl_slice is not None or mov_lbl_slice is not None:
        all_labels = []
        if fix_lbl_slice is not None:
            all_labels.extend(np.unique(fix_lbl_slice))
        if mov_lbl_slice is not None:
            all_labels.extend(np.unique(mov_lbl_slice))
        n_labels = int(max(all_labels)) + 1
        np.random.seed(42)  # For consistent colors
        colors = np.random.rand(n_labels, 4)
        colors[:, 3] = 1.0  # Full opacity
        colors[0] = [0, 0, 0, 0]  # Background transparent
        label_cmap = ListedColormap(colors)
    
    # Determine number of rows
    n_rows = 2 if (fix_lbl_slice is not None or mov_lbl_slice is not None) else 1
    n_cols = 3  # Fixed, Moving, Overlay
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot images
    axes[0, 0].imshow(fix_slice.T, cmap='gray', origin='lower', aspect='equal')
    axes[0, 0].set_title(f'Fixed Image (slice {slice_idx})', fontsize=9)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mov_slice.T, cmap='gray', origin='lower', aspect='equal')
    axes[0, 1].set_title(f'Moving Image (slice {slice_idx})', fontsize=9)
    axes[0, 1].axis('off')
    
    # Overlay: Fixed image with moving contours
    axes[0, 2].imshow(fix_slice.T, cmap='gray', origin='lower', aspect='equal')
    # Compute contour levels based on image intensity range
    mov_min, mov_max = mov_slice.min(), mov_slice.max()
    if mov_max <= mov_min:
        raise ValueError(
            f"Moving image has constant intensity (min={mov_min}, max={mov_max}). "
            "This typically indicates a failed registration where the transformation "
            "is degenerate (e.g., singular affine matrix). Check the registration output "
            "for warnings like 'LogmExactlySingularWarning' which indicate the transform "
            "could not be properly computed."
        )
    levels = np.linspace(mov_min + 0.1*(mov_max-mov_min), mov_max - 0.1*(mov_max-mov_min), n_contours)
    axes[0, 2].contour(mov_slice.T, levels=levels, colors='red', linewidths=0.8, 
                       origin='lower', alpha=0.7)
    axes[0, 2].set_title('Fixed + Moving Contours', fontsize=9)
    axes[0, 2].axis('off')
    
    # Plot labels if provided
    if n_rows > 1:
        # Fixed with label overlay
        axes[1, 0].imshow(fix_slice.T, cmap='gray', origin='lower', aspect='equal')
        if fix_lbl_slice is not None:
            axes[1, 0].imshow(fix_lbl_slice.T, cmap=label_cmap, origin='lower', 
                             aspect='equal', alpha=alpha, vmin=0, vmax=n_labels-1)
        axes[1, 0].set_title('Fixed + Labels', fontsize=9)
        axes[1, 0].axis('off')
        
        # Moving with label overlay
        axes[1, 1].imshow(fix_slice.T, cmap='gray', origin='lower', aspect='equal')
        if mov_lbl_slice is not None:
            axes[1, 1].imshow(mov_lbl_slice.T, cmap=label_cmap, origin='lower', 
                             aspect='equal', alpha=alpha, vmin=0, vmax=n_labels-1)
        axes[1, 1].set_title('Moving Labels on Fixed', fontsize=9)
        axes[1, 1].axis('off')
        
        # Overlay: Fixed with both label contours
        axes[1, 2].imshow(fix_slice.T, cmap='gray', origin='lower', aspect='equal')
        if fix_lbl_slice is not None:
            for lbl in np.unique(fix_lbl_slice):
                if lbl == 0:
                    continue
                axes[1, 2].contour(fix_lbl_slice.T == lbl, levels=[0.5], colors='lime', 
                                  linewidths=0.8, origin='lower')
        if mov_lbl_slice is not None:
            for lbl in np.unique(mov_lbl_slice):
                if lbl == 0:
                    continue
                axes[1, 2].contour(mov_lbl_slice.T == lbl, levels=[0.5], colors='magenta', 
                                  linewidths=0.8, origin='lower', linestyles='dashed')
        axes[1, 2].set_title('Label Contours (green=fixed, magenta=moving)', fontsize=9)
        axes[1, 2].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()


# =============================================================================
# Python API Functions (equivalents of CLI commands)
# =============================================================================

def resize_image_api(input_path, output_path, factor, is_label=False, device='cpu'):
    """
    Resize a NIfTI file using the nitorch Python API.
    
    Parameters
    ----------
    input_path : str
        Path to input NIfTI file.
    output_path : str
        Path to output NIfTI file.
    factor : float
        Downsampling factor (intuitive definition):
        - factor > 1: downsample (e.g., factor=2 gives half resolution)
        - factor < 1: upsample (e.g., factor=0.5 gives double resolution)
    is_label : bool
        If True, use nearest neighbor interpolation for labels.
    device : str
        Device to use for computation.
    """
    # Convert intuitive factor to nitorch factor
    # User wants: factor=2 means half the voxels (downsample)
    # Nitorch uses: factor=2 means twice the voxels (upsample)
    nitorch_factor = 1.0 / factor
    
    # Load the input file
    f = io.volumes.map(input_path)
    dat = f.fdata(device=device)
    affine = f.affine.to(device)
    original_ndim = dat.dim()
    
    # Ensure data has batch and channel dimensions
    # nitorch resize expects (batch, channel, *spatial)
    while dat.dim() < 5:
        dat = dat.unsqueeze(0)
    
    # Set interpolation order
    interpolation = 0 if is_label else 3
    
    # Perform resize
    dat_resized, affine_resized = spatial.resize(
        dat, 
        factor=nitorch_factor,
        affine=affine,
        anchor='c',  # center anchor
        interpolation=interpolation,
        bound='dct2',
        prefilter=True
    )
    
    # Remove batch and channel dimensions to match original file format
    # dat_resized is (batch, channel, *spatial) = (1, 1, X, Y, Z) for 3D
    dat_resized = dat_resized.squeeze(0)  # Remove batch -> (1, X, Y, Z)
    if original_ndim == 3:
        # Original was 3D (X, Y, Z), squeeze the channel dim too
        dat_resized = dat_resized.squeeze(0)  # Remove channel -> (X, Y, Z)
    
    # Save the output
    if is_label:
        # For labels, save as integers
        io.volumes.save(dat_resized.cpu().round().to(torch.int32), output_path, like=input_path, affine=affine_resized.cpu())
    else:
        io.volumes.savef(dat_resized.cpu(), output_path, like=input_path, affine=affine_resized.cpu())


def resize_pair_api(pair, factor, output_dir=None, verbose=True, device='cpu'):
    """
    Resize a fixed-moving image pair using the nitorch Python API.
    
    Parameters
    ----------
    pair : dict
        Dictionary containing 'fixed' and 'moving' keys, each with 'image' 
        and 'label' paths to NIfTI files.
    factor : float
        Downsampling factor.
    output_dir : str, optional
        Directory where resized NIfTI files will be saved.
    verbose : bool
        If True, print information about the resizing.
    device : str
        Device to use for computation.
    
    Returns
    -------
    pair : dict
        Updated pair dictionary with paths pointing to resized NIfTI files.
    """
    # Skip resizing if factor is 0 or 1
    if factor == 0 or factor == 1:
        if verbose:
            print("Resize factor is 0 or 1, skipping resizing.")
        return pair
    
    if output_dir is None:
        output_dir = os.path.join(pair.get("dir_out", "."), "resized")
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Resizing pair with factor {factor} (using Python API)...")
        print(f"  Output directory: {output_dir}")
    
    # Generate output paths
    paths_resized = {}
    for role in ["fixed", "moving"]:
        img_basename = os.path.basename(pair[role]["image"]).replace(
            '.nii.gz', f'.resized_f{factor}.nii.gz'
        )
        lbl_basename = os.path.basename(pair[role]["label"]).replace(
            '.nii.gz', f'.label.resized_f{factor}.nii.gz'
        )
        paths_resized[role] = {
            "image": os.path.join(output_dir, img_basename),
            "label": os.path.join(output_dir, lbl_basename),
        }
    
    # Resize each image and label
    for role in ["fixed", "moving"]:
        if verbose:
            print(f"\n  Resizing {role}...")
        
        # Get original info
        nii_orig_img = nib.load(pair[role]["image"])
        nii_orig_lbl = nib.load(pair[role]["label"])
        
        # Resize image
        resize_image_api(
            pair[role]["image"],
            paths_resized[role]["image"],
            factor,
            is_label=False,
            device=device
        )
        
        # Resize label
        resize_image_api(
            pair[role]["label"],
            paths_resized[role]["label"],
            factor,
            is_label=True,
            device=device
        )
        
        if verbose:
            nii_resized_img = nib.load(paths_resized[role]["image"])
            nii_resized_lbl = nib.load(paths_resized[role]["label"])
            
            resized_img_shape = tuple(s for s in nii_resized_img.shape if s > 1)
            resized_lbl_shape = tuple(s for s in nii_resized_lbl.shape if s > 1)
            
            print(f"    Image:")
            print(f"      Original: shape={nii_orig_img.shape}, "
                  f"voxel_size={tuple(np.round(nii_orig_img.header.get_zooms()[:3], 2))}")
            print(f"      Resized:  shape={resized_img_shape}, "
                  f"voxel_size={tuple(np.round(nii_resized_img.header.get_zooms()[:3], 2))}")
            
            print(f"    Label:")
            print(f"      Original: shape={nii_orig_lbl.shape}, "
                  f"labels={list(np.unique(nii_orig_lbl.get_fdata()).astype(int))}")
            print(f"      Resized:  shape={resized_lbl_shape}, "
                  f"labels={list(np.unique(nii_resized_lbl.get_fdata()).astype(int))}")
    
    # Update pair with resized paths
    for role in ["fixed", "moving"]:
        pair[role]["image_orig"] = pair[role]["image"]
        pair[role]["label_orig"] = pair[role]["label"]
        pair[role]["image"] = paths_resized[role]["image"]
        pair[role]["label"] = paths_resized[role]["label"]
    
    if verbose:
        print(f"\n{'='*60}\n")
    
    pair["resize_factor"] = factor
    pair["dir_resized"] = output_dir
    
    return pair


def register_api(loss_name, pair, reg_lambda=10, verbose=True, print_gpu_use=True, device='cpu'):
    """
    Perform affine and nonlinear registration using the nitorch Python API.
    
    Parameters
    ----------
    loss_name : str
        Loss function to use ('lcc', 'dice', 'nmi', 'mse', etc.).
    pair : dict
        Dictionary containing paths to fixed/moving images and output locations.
    reg_lambda : float
        Regularization weight for the nonlinear registration.
    verbose : bool
        If True, print progress information.
    print_gpu_use : bool
        If True, print timing information.
    device : str
        Device to use for computation.
    
    Returns
    -------
    affine_model : AffineModel
        The fitted affine transformation.
    nonlin_model : NonLinModel
        The fitted nonlinear transformation (SVF).
    elapsed_time : float
        Total runtime in seconds.
    """
    from nitorch.tools.registration.pairwise_preproc import preproc_image
    from nitorch.tools.registration.pairwise_makeobj import (
        make_image, make_loss, make_affine, make_nonlin,
        make_affine_optim, make_nonlin_optim
    )
    from nitorch.tools.registration.pairwise_pyramid import sequential_pyramid
    from nitorch.tools.registration.pairwise_run import run
    from nitorch.tools.registration import objects
    
    # Determine which files to use based on loss type
    if loss_name in ["dice", "cat", "cce"]:
        pth_fixed = pair["fixed"]["label"]
        pth_moving = pair["moving"]["label"]
        is_label = True
    else:
        pth_fixed = pair["fixed"]["image"]
        pth_moving = pair["moving"]["image"]
        is_label = False
    
    # Output paths
    pth_affine = pair["loss"][loss_name]["affine"]
    pth_svf = pair["loss"][loss_name]["svf"]
    
    # =========================================================================
    # Load and preprocess images
    # Matching CLI parameters: --fwhm 1 -b dct2 -o false
    # CLI default rescale is (0, 95) but for 'dice'/'cat' it's set to (0, 0)
    # =========================================================================
    if loss_name in ('cat', 'dice'):
        rescale = (0, 0)
    else:
        rescale = (0, 95)
    
    fix_dat, fix_mask, fix_affine = preproc_image(
        pth_fixed, 
        label=is_label,
        rescale=rescale,
        fwhm=1,
        bound='dct2',
        dim=3,
        device=device
    )
    mov_dat, mov_mask, mov_affine = preproc_image(
        pth_moving,
        label=is_label,
        rescale=rescale,
        fwhm=1,
        bound='dct2',
        dim=3,
        device=device
    )
    
    
    # =========================================================================
    # Create image objects with pyramid
    # CLI uses @pyramid --levels 0 1 2
    # =========================================================================
    pyramid_levels = [0, 1, 2]
    
    fix_img = make_image(
        fix_dat, mask=fix_mask, affine=fix_affine,
        pyramid=pyramid_levels, 
        pyramid_method='gaussian',  # CLI default
        bound='dct2', 
        extrapolate=False
    )
    mov_img = make_image(
        mov_dat, mask=mov_mask, affine=mov_affine,
        pyramid=pyramid_levels, 
        pyramid_method='gaussian',
        bound='dct2', 
        extrapolate=False
    )
    
    # =========================================================================
    # Create loss object
    # Match CLI defaults for each loss type
    # =========================================================================
    loss_kwargs = {}
    if loss_name in ('lcc', 'lncc'):
        # CLI defaults: patch=10, stride=1, kernel='g'
        loss_kwargs = {'patch': 10, 'stride': 1, 'kernel': 'g'}
    elif loss_name in ('nmi', 'mi'):
        # CLI defaults: bins=32, fwhm=0, norm='studholme' (for nmi)
        loss_kwargs = {'bins': 32, 'fwhm': 0, 'norm': 'studholme'}
    loss_obj = make_loss(loss_name, **loss_kwargs)
    
    # =========================================================================
    # Create similarity object (matching CLI: Similarity(lossobj, mov, fix))
    # =========================================================================
    similarity = objects.Similarity(loss_obj, mov_img, fix_img)
    
    # =========================================================================
    # Build pyramid of losses
    # =========================================================================
    loss_list = sequential_pyramid(similarity)
    
    
    # =========================================================================
    # Create affine model
    # CLI: @affine affine (position='symmetric' is default)
    # =========================================================================
    affine_model = make_affine(basis='affine', position='symmetric')
    
    
    # =========================================================================
    # Create nonlinear model (SVF)
    # CLI: @nonlin svf 10 (factor=10, voxel_size=[100, '%'] is default)
    # =========================================================================
    # Get images from the finest level for computing mean space
    if isinstance(loss_list[-1], objects.SumSimilarity):
        images = list(loss_list[-1].images())
    else:
        images = list(loss_list[-1].images())
    
    # Penalty parameters - CLI defaults
    penalty = {
        'absolute': 0.0001,
        'membrane': 0.001,
        'bending': 0.2,
        'lame': (0.05, 0.2),
    }
    
    nonlin_model = make_nonlin(
        images, 'svf', 
        factor=reg_lambda, 
        penalty=penalty,
        voxel_size=[100, '%'],  # CLI default
        device=device
    )
    
    
    # =========================================================================
    # Create optimizers
    # CLI: @@optim -t 1e-3 -n 64 (tolerance=1e-3, max_iter=64)
    # =========================================================================
    order = loss_obj.order  # LCC has order 2, so GaussNewton is used
    
    affine_optim = make_affine_optim(
        'gn' if order >= 2 else 'lbfgs',
        order,
        max_iter=64,
        tolerance=1e-3
    )
    nonlin_optim = make_nonlin_optim(
        'gn' if order >= 2 else 'lbfgs',
        order,
        max_iter=64,
        tolerance=1e-3,
        nonlin=nonlin_model
    )
    
    # =========================================================================
    # Run registration
    # CLI: @optim i -t 1e-3 -n 64 (interleaved, tolerance=1e-3, max_iter=64)
    # =========================================================================
    start_time = time.time()
    
    affine_model, nonlin_model = run(
        loss_list,
        affine=affine_model,
        nonlin=nonlin_model,
        affine_optim=affine_optim,
        nonlin_optim=nonlin_optim,
        pyramid=True,
        interleaved=True,
        progressive=False,
        max_iter=64,
        tolerance=1e-3,
        verbose=0,
        framerate=0
    )
    
    elapsed_time = time.time() - start_time
    
    
    # =========================================================================
    # Save outputs
    # =========================================================================
    # Save affine transform
    aff_sqrt = affine_model.exp(cache_result=True, recompute=True)
    
    if affine_model.position[0] == 's':
        aff = aff_sqrt.matmul(aff_sqrt)
    else:
        aff = aff_sqrt
    
    
    io.transforms.savef(aff.cpu(), pth_affine, type=1)  # 1 = RAS_TO_RAS
    
    # Save SVF
    io.savef(nonlin_model.dat.dat, pth_svf, affine=nonlin_model.affine)
    
    if print_gpu_use:
        print(f"\n{'='*60}")
        print(f"{loss_name.upper()} registration completed (Python API)...")
        print(f"Runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"{'='*60}")
    
    return affine_model, nonlin_model, elapsed_time


def create_displacement_field_api(pair, loss_name, device='cpu'):
    """
    Create a displacement field by exponentiating the SVF using Python API.
    
    Parameters
    ----------
    pair : dict
        Dictionary containing registration paths including the SVF.
    loss_name : str
        Name of the loss function used for registration.
    device : str
        Device to use for computation.
    
    Returns
    -------
    pair : dict
        Updated pair dictionary with displacement field path added.
    """
    # Get resize factor suffix
    resize_factor = pair.get("resize_factor", 0)
    factor_suffix = f"_f{resize_factor}" if resize_factor not in [0, 1] else ""
    
    # Define output path
    pair["loss"][loss_name]["u"] = os.path.join(
        pair["loss"][loss_name]["dir_out"],
        f'u{factor_suffix}.nii.gz'
    )
    
    # Load SVF
    svf_path = pair["loss"][loss_name]["svf"]
    f = io.volumes.map(svf_path)
    svf_dat = f.fdata(device=device)
    svf_affine = f.affine.to(device)
    
    # Ensure correct shape (add batch dimension if needed)
    if svf_dat.dim() == 4:  # (*spatial, 3)
        svf_dat = svf_dat.unsqueeze(0)  # (1, *spatial, 3)
    
    # Exponentiate SVF to get displacement field
    disp = spatial.exp(svf_dat, displacement=True)
    disp = disp.squeeze(0)  # Remove batch dimension
    
    # Save displacement field
    io.savef(disp.cpu(), pair["loss"][loss_name]["u"], affine=svf_affine.cpu())
    
    return pair


def reslice_api(pair, loss_name, affine_model=None, nonlin_model=None, verbose=True, device='cpu'):
    """
    Reslice labels using computed registration transforms (pure Python API).
    
    Implements the same transform composition as CLI:
        nitorch reslice {src} --linear-square {affine} --velocity {svf} 
                        --linear-square {affine} --target {tgt}
    
    Key insight: Labels must be interpolated as int64, not float32, to match
    CLI behavior exactly.
    
    Parameters
    ----------
    pair : dict
        Dictionary containing paths to images and registration transforms.
    loss_name : str
        Name of the loss function used for registration.
    affine_model : objects.AffineModel, optional
        Not used - transforms are loaded from files.
    nonlin_model : objects.NonLinModel, optional  
        Not used - transforms are loaded from files.
    verbose : bool
        Whether to print progress messages.
    device : str
        Device to use for computation.
    """
    from nitorch.core.linalg import logm, expm
    from nitorch.core.utils import movedim
    from nitorch.cli.registration import helpers
    
    # Always reslice labels (needed for dice score computation)
    what = "label"
    
    # Get transform paths
    pth_affine = pair["loss"][loss_name]["affine"]
    pth_svf = pair["loss"][loss_name]["svf"]
    
    # Load affine transform and compute square root
    affine_mat = io.transforms.loadf(pth_affine).to(device).float()
    while affine_mat.dim() > 2:
        affine_mat = affine_mat.squeeze(0)
    affine_sqrt = expm(logm(affine_mat) * 0.5).float()
    
    # Load SVF and exponentiate to displacement
    f_svf = io.volumes.map(pth_svf)
    svf_dat = f_svf.fdata(device=device)
    svf_affine = f_svf.affine.to(device).float()
    
    # Squeeze SVF to (*spatial, 3)
    while svf_dat.dim() > 4 and svf_dat.shape[0] == 1:
        svf_dat = svf_dat.squeeze(0)
    while svf_dat.dim() > 4 and svf_dat.shape[-1] == 1:
        svf_dat = svf_dat.squeeze(-1)
    
    # Exponentiate SVF
    disp = spatial.exp(svf_dat[None], displacement=True)[0]
    
    def reslice_volume(pth_src, pth_tgt, pth_out):
        """
        Reslice source label to target space.
        
        Uses CLI's transform composition:
        1. First linear (sqrt(A)) modifies source affine
        2. Remaining transforms [V, sqrt(A)] applied to grid in reverse
        3. Grid converted to modified source voxels
        4. CRITICAL: Labels sampled as int64 (not float32) for correct interpolation
        """
        # Load source volume
        f_src = io.volumes.map(pth_src)
        src_affine_orig = f_src.affine.to(device).float()
        
        # Load target for output space
        f_tgt = io.volumes.map(pth_tgt)
        tgt_shape = f_tgt.shape[:3]
        tgt_affine = f_tgt.affine.to(device).float()
        
        # CLI's special handling: first linear modifies source affine
        # src_affine = affine_lmdiv(sqrt(A), src_affine) = inv(sqrt(A)) @ src_affine
        src_affine_modified = spatial.affine_lmdiv(affine_sqrt, src_affine_orig)
        
        # Build grid from target space
        grid = spatial.affine_grid(tgt_affine, tgt_shape)
        
        # Apply remaining transforms in reverse: [V, sqrt(A)] -> [sqrt(A), V]
        # 1. Apply sqrt(A) (the last linear in collapsed list)
        grid = spatial.affine_matvec(affine_sqrt, grid)
        
        # 2. Apply V (velocity)
        imat = spatial.affine_inv(svf_affine)
        grid = spatial.affine_matvec(imat, grid)
        grid = grid + helpers.pull_grid(disp, grid, interpolation=1, bound='dft', extrapolate=True)
        grid = spatial.affine_matvec(svf_affine, grid)
        
        # Convert to modified source voxels
        src_imat = spatial.affine_inv(src_affine_modified)
        grid = spatial.affine_matvec(src_imat, grid)
        
        # Load data as int64 (CRITICAL for correct label interpolation)
        src_dat = f_src.data(dtype=torch.long, device=device)
        src_dat = src_dat.reshape([*f_src.shape[:3], 1])
        src_dat = movedim(src_dat, -1, 0)
        
        # Sample with int64 data - this is key to matching CLI behavior
        out = helpers.pull(src_dat, grid, interpolation=1, bound='zero', extrapolate=False)
        
        # Move channel back and save
        out = movedim(out, 0, -1)
        io.volumes.save(out.cpu(), pth_out, like=pth_src, affine=tgt_affine.cpu())
        
    
    # Reslice moving label to fixed space
    reslice_volume(
        pair["moving"][what],
        pair["fixed"][what],
        pair["loss"][loss_name]["moving"][what]
    )
    
    # Reslice fixed label to moving space
    reslice_volume(
        pair["fixed"][what],
        pair["moving"][what],
        pair["loss"][loss_name]["fixed"][what]
    )
