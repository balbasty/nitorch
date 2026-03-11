"""Dice score computation from numpy arrays — no file I/O."""

import numpy as np


def compute_dice_scores(label1, label2, exclude_zero=True):
    """Compute per-label and mean Dice scores.

    Parameters
    ----------
    label1, label2 : np.ndarray
        3D label arrays (integer-valued).
    exclude_zero : bool
        If True, exclude background label (0) from computation.

    Returns
    -------
    dice_scores : dict
        {label_id: dice_score} for each label present in either array.
    mean_dice : float
        Mean Dice across all computed labels.
    """
    labels = np.union1d(np.unique(label1), np.unique(label2))
    if exclude_zero:
        labels = labels[labels != 0]

    dice_scores = {}
    for lab in labels:
        mask1 = label1 == lab
        mask2 = label2 == lab
        intersection = np.sum(mask1 & mask2)
        total = np.sum(mask1) + np.sum(mask2)
        if total == 0:
            dice_scores[int(lab)] = 1.0
        else:
            dice_scores[int(lab)] = 2.0 * intersection / total

    mean_dice = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
    return dice_scores, mean_dice
