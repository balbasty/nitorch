"""
Helper functions for the NITorch registration demo notebooks.

This module provides utilities for:
- Managing registration pair dictionaries
- Computing Dice scores for registration evaluation
- Running nitorch CLI commands with GPU monitoring
- Visualizing registration results
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
            "affine": os.path.join(dir_out_loss, 'affine.lta'),
            "svf": os.path.join(dir_out_loss, 'svf.nii.gz')
        }
    }
    return pair


def compute_dice_scores(label1, label2, exclude_zero=True, loss=None):
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
    
    # Print mean dice score
    print(f"\n{'='*60}")
    if loss is not None:
        print(f"{loss.upper()} loss mean dice score: {mean_dice:.6f}")
    else:
        print(f"Mean dice score: {mean_dice:.6f}")
    print(f"{'='*60}")
    
    return dice_scores, mean_dice


def create_displacement_field(pair, loss):
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
    
    Returns
    -------
    pair : dict
        Updated pair dictionary with displacement field path added.
    """
    # Exponentiate stationary velocity field to create displacement field
    pair["loss"][loss]["u"] = os.path.join(
        pair["loss"][loss]["dir_out"],
        'u.nii.gz'
    )
    get_cmd = lambda pth_svf, pth_u: \
        "nitorch vexp " \
        + pth_svf \
        + " -gpu 0" \
        + " -o " + pth_u
    # Exponentiate
    cmd = get_cmd(pair["loss"][loss]["svf"], pair["loss"][loss]["u"])
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
        print(f"{'='*60}")
        for role in ["fixed", "moving"]:
            print(f"\n{role.upper()}:")
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


def register(loss, pair, reg_lambda=10):
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
    """
    def get_register_cmd(loss, pair, reg_lambda):
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
            + " --verbose 1" \
            + " --gpu 0" \
            + " @loss " + loss \
            + " @@fix " + pth_fixed + label + " --fwhm 1 -b dct2 -o false -r " + pth_fixed_moved \
            + " @@mov " + pth_moving + label + " --fwhm 1 -b dct2 -o false -r " + pth_moving_moved \
            + " @affine affine -o " + pth_affine \
            + " @@optim -t 1e-4 -n 128" \
            + " @nonlin svf " + str(reg_lambda) + " -o " + pth_svf + " -h \"\"" \
            + " @@optim -t 1e-3 -n 64" \
            + " @optim i -t 1e-3 -n 64" \
            + " @pyramid --levels 0 1 2"
        
        return cmd
    
    cmd = get_register_cmd(loss, pair, reg_lambda)

    elapsed_time, peak_gpu_mb = run_with_monitoring(cmd, gpu_id=0)

    print(f"\n{'='*60}")
    print(f"{loss.upper()} registration completed...")
    print(f"Runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Peak GPU Memory: {peak_gpu_mb:.1f} MB ({peak_gpu_mb/1024:.2f} GB)")
    print(f"{'='*60}")


def reslice(pair, loss):
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
    """
    def get_reslice_cmd(pth_moving, pth_fixed, pth_resliced, pth_affine, pth_svf, interpolation):
        cmd = "nitorch reslice" \
            + " " + pth_moving \
            + " --linear-square " + pth_affine \
            + " --velocity " + pth_svf \
            + " --linear-square " + pth_affine \
            + " --target " + pth_fixed \
            + " --output " + pth_resliced \
            + " --interpolation " + interpolation \
            + " -gpu 0" \
            + " --bound zero"
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
