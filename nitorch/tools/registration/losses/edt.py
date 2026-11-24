"""Euclidean distance transform (EDT) loss for registering sparse label
maps (e.g., vessels).

Each image is given as a stack of label maps and of their (precomputed)
distance maps: `[labels_1, ..., labels_K, dist_1, ..., dist_K]`.
The loss evaluates the distance map of the *warped* image at the
labels of the *static* image:

    L = sum_k sum_x f_k(x) * rho(D^m_k(phi(x))) / sum_k sum_x f_k(x)

where `f_k` are the fixed labels and `D^m_k` the moving distance maps.
This is the (one-sided) chamfer distance, i.e., ICP in Gauss-Newton form:
with `rho(d) = d^2 / 2`, the gradient wrt the displacement is
`f * D * grad(D)` (= `f * (x - closest point)`) and the Gauss-Newton
Hessian is `f * grad(D) grad(D)^T`.

Because the sum runs over the voxels of the *static* labels, warping
cannot decrease the loss by shrinking the warped structures (which is
what happens when warped labels are dotted with a static distance map).
The `--symmetric` option adds the other half of the chamfer distance
(moving labels vs warped fixed distance map) without that bias either.

The warped images must be extrapolated out of the field of view
(`Image(..., extrapolate=True)`, the default of `nitorch register`):
otherwise, out-of-bounds distances are zero, and pushing the structures
out of the field of view is a perfect match.
"""
import torch

__all__ = ["edt", "EDTLoss", "stack_distance"]
from nitorch.core import utils
from nitorch import spatial
from .base import OptimizationLoss


def stack_distance(labels, dim=None, vx=1, threshold=0.5):
    """Stack label maps with their Euclidean distance maps.

    Parameters
    ----------
    labels : (..., K, *spatial) tensor
        Label maps (hard or soft)
    dim : int, default=`labels.dim() - 1`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size (the distances are in the same unit)
    threshold : float, default=0.5
        Voxels whose label value is above this threshold are foreground

    Returns
    -------
    stacked : (..., 2*K, *spatial) tensor
        `[labels, dist]`, where `dist` is the distance to the nearest
        foreground voxel (zero on the foreground).
    """
    dim = dim or (labels.dim() - 1)
    dtype = labels.dtype if labels.dtype.is_floating_point else torch.float32
    labels = labels.to(dtype)
    background = labels <= threshold
    if background.all():
        raise ValueError('Cannot compute a distance map: empty label map')
    dist = spatial.euclidean_distance_transform(background, dim=dim, vx=vx)
    return torch.cat([labels, dist.to(dtype)], -dim-1)


def edt(moving, fixed, dim=None, sigma=None, grad=True, hess=True,
            mask=None):
    """EDT loss between stacked [labels, distances] images.

    Parameters
    ----------
    moving : (..., 2*K, *spatial) tensor
        Warped image: `K` label maps followed by their `K` distance maps.
        Only the distance maps are used.
    fixed : (..., 2*K, *spatial) tensor
        Static image: `K` label maps followed by their `K` distance maps.
        Only the label maps are used.
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions.
    sigma : float, optional
        Scale of the robust (Cauchy) penalty, in the units of the distance
        maps: `rho(d) = sigma^2/2 * log(1 + d^2/sigma^2)`.
        Distances much larger than `sigma` (structures that are present
        in only one image) barely contribute.
        If None, use the quadratic penalty `rho(d) = d^2/2`.
    grad, hess : bool
        Compute and return the gradient and (diagonal) Hessian.
    mask : (..., 1|K|2*K, *spatial) tensor, optional
        Mask of voxels to include.

    Returns
    -------
    ll : () tensor
        Mean penalized distance over the static labels.
    g : (..., 2*K, *spatial) tensor, if `grad`
        Gradient wrt `moving` (zero in the label channels).
    h : (..., 2*K, *spatial) tensor, if `hess`
        Diagonal Gauss-Newton (IRLS, if `sigma`) Hessian wrt `moving`.
    """
    dim = dim or (fixed.dim() - 1)
    nc = fixed.shape[-dim-1]
    if nc % 2 or moving.shape[-dim-1] != nc:
        raise ValueError(
            f'edt expects both images to stack K label maps and their '
            f'K distance maps (2*K channels). Got {moving.shape[-dim-1]} '
            f'(moving) and {nc} (fixed) channels.')
    nk = nc // 2
    sumspatial = list(range(-dim, 0))

    lab = utils.slice_tensor(fixed, slice(nk), -dim-1).clamp(0, 1)
    dist = utils.slice_tensor(moving, slice(nk, None), -dim-1)
    if mask is not None:
        mask = mask.to(lab)
        if mask.shape[-dim-1] == nc:
            mask = utils.slice_tensor(mask, slice(nk), -dim-1)
        lab = lab * mask
    norm = lab.sum(sumspatial + [-dim-1], keepdim=True).clamp_min_(1e-3)
    lab = lab / norm

    if sigma:
        sigma2 = sigma * sigma
        z = dist.square() / sigma2
        ll = 0.5 * sigma2 * (lab * z.log1p()).sum()
        weight = lab / (1 + z)     # IRLS weights
    else:
        ll = 0.5 * (lab * dist.square()).sum()
        weight = lab
    out = [ll]

    if grad or hess:
        zeros = torch.zeros_like(weight)
        if grad:
            out.append(torch.cat([zeros, weight * dist], -dim-1))
        if hess:
            out.append(torch.cat([zeros, weight], -dim-1))

    return tuple(out) if len(out) > 1 else out[0]


class EDTLoss(OptimizationLoss):
    """Distance-map (EDT) loss between sparse label maps (see `edt`)"""

    order = 2  # Hessian defined

    def __init__(self, sigma=None, dim=None):
        """
        Parameters
        ----------
        sigma : float, optional
            Scale of the robust (Cauchy) penalty. Default: quadratic.
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.sigma = sigma
        self.dim = dim

    def _call(self, moving, fixed, grad, hess, **kwargs):
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('sigma', self.sigma)
        return edt(moving, fixed, grad=grad, hess=hess, **kwargs)

    def loss(self, moving, fixed, **kwargs):
        return self._call(moving, fixed, False, False, **kwargs)

    def loss_grad(self, moving, fixed, **kwargs):
        return self._call(moving, fixed, True, False, **kwargs)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        return self._call(moving, fixed, True, True, **kwargs)
