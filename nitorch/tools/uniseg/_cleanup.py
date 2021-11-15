import torch
from nitorch.core import utils, py
from nitorch.spatial import conv


def _build_kernel(dim, **backend):
    kernel = torch.as_tensor([0.75, 1., 0.75], **backend)
    normk = kernel
    for d in range(1, dim):
        normk = normk.unsqueeze(-1)
        normk = normk * kernel
    normk = normk.sum()
    normk = normk ** (1/dim)
    kernel /= normk
    kernels = []
    for d in range(dim):
        kernel1 = kernel
        kernel1 = utils.unsqueeze(kernel1, 0, d)
        kernel1 = utils.unsqueeze(kernel1, -1, dim-1-d)
        kernels.append(kernel1)
    return kernels


def _conv(X, K):
    dim = X.dim()
    for d, K1 in enumerate(K):
        X = conv(dim, X, K1, bound='zero', padding='same')
    return X


def cleanup(Z, level=1, gm=1, wm=2, csf=3):
    """Ad-hoc cleaning procedure for brain segmentation (in-place)

    Note that this function is targeted to "typical" MRIs (about 1mm iso)
    and is based on morphological operations whose radius is defined
    in terms of voxels, not mm. It is very likely to fail if applied
    to other types of data (e.g., higher resolution).

    Parameters
    ----------
    Z : (K, *spatial)
        Posterior probabilities
    level : {1, 2}, default=1
        Aggressiveness of the procedure
    gm, wm, csf : int, default=1, 2, 3
        Index of the brain tissue classes in `Z`

    Returns
    -------
    Z : (K, *spatial)
        Cleaned-up posteriors

    """
    # Author of original Matlab code: John Ashburner / SPM12

    dim = Z.dim() - 1

    G = Z[gm]   # Gray matter
    W = Z[wm]   # White matter
    C = Z[csf]  # CSF

    # Build a 3x3x3 separable smoothing kernel
    kernel = _build_kernel(dim, **utils.backend(Z))

    # ------------------------------------------------------------------
    # Erosions and conditional dilations
    # ------------------------------------------------------------------

    # Mask stuff too far from the WM
    #   Start from the white matter and dilate into the gray matter.
    #   What's outside of the dilated white matter cannot be gray matter
    #   and is masked out.
    th1 = 0.2 if level == 2 else 0.15
    max_iter_brain = 32
    B = Z[wm]
    for n_iter in range(max_iter_brain):
        th = 0.6 if n_iter <= 1 else th1
        B = (B > th) * (G+W)
        B = _conv(B, kernel)

    # Mask stuff too far from the brain
    #   Start from the brain and dilate into the CSF.
    #   What's outside of the dilated brain cannot be CSF
    #   and is masked out.
    max_iter_csf = 32
    th = th1
    BC = B.clone() if max_iter_csf else None
    for n_iter in range(max_iter_csf):
        BC = (BC > th) * (G+W+C)
        BC = _conv(BC, kernel)

    # ------------------------------------------------------------------
    # Apply masks and normalize probabilities
    # ------------------------------------------------------------------
    th = 0.05

    # Brain
    B = ((B > th) * (G + W)) > th
    G *= B
    W *= B

    # CSF
    if max_iter_csf:
        BC = ((BC > th) * (G + W + C)) > th
        C *= BC

    Z /= Z.sum(0, keepdim=True)
    return Z
