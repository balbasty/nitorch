"""Euclidean distance transform (EDT) loss for registering sparse label
maps (e.g., vessels).

Each image is given as a stack of label maps and of their (precomputed)
distance maps: `[labels_1, ..., labels_K, dist_1, ..., dist_K]`.
The loss evaluates the distance map of the *warped* image at the
labels of the *static* image:

    L = sum_k lam_k * sum_x f_k(x) * rho(D^m_k(phi(x))) / sum_x f_k(x)

where `f_k` are the fixed labels, `D^m_k` the moving distance maps and
`lam_k` a precision (inverse squared distance) that makes the loss
dimensionless: fixed, `1/voxel_size^2` (distances in voxels), or
estimated from the data (as the precision of `MSE`). Like `MSE`, the
loss is a mean over the (weighted) voxels of each channel, so it scales
like the other losses.
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


def edt(moving, fixed, dim=None, lam=1, sigma=None, grad=True, hess=True,
        mask=None):
    """EDT loss between stacked [labels, distances] images.

    For each channel `k`, the penalized distance is averaged over the
    (masked) static labels; the channel means are then summed:

        L = sum_k lam_k * sum_x f_k(x) rho(D_k(x)) / sum_x f_k(x)

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
    lam : float or (..., K|1) tensor, default=1
        Precision of the distances, in inverse squared distance units
        (e.g., mm^-2). `1/lam` is the expected squared distance.
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
        Sum over channels of the mean penalized distance.
    g : (..., 2*K, *spatial) tensor, if `grad`
        Gradient wrt `moving` (zero in the label channels).
    h : (..., 2*K, *spatial) tensor, if `hess`
        Diagonal Gauss-Newton (IRLS, if `sigma`) Hessian wrt `moving`.
    """
    dim = dim or (fixed.dim() - 1)
    lab, dist = _split(moving, fixed, dim, mask)
    lam = _preproc_lam(lam, dim, dist)
    lab = lab * lam / _sumweights(lab, dim)

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


def _split(moving, fixed, dim, mask=None):
    """Return the (masked) static labels and the warped distances"""
    nc = fixed.shape[-dim-1]
    if nc % 2 or moving.shape[-dim-1] != nc:
        raise ValueError(
            f'edt expects both images to stack K label maps and their '
            f'K distance maps (2*K channels). Got {moving.shape[-dim-1]} '
            f'(moving) and {nc} (fixed) channels.')
    nk = nc // 2
    lab = utils.slice_tensor(fixed, slice(nk), -dim-1).clamp(0, 1)
    dist = utils.slice_tensor(moving, slice(nk, None), -dim-1)
    if mask is not None:
        mask = mask.to(lab)
        if mask.shape[-dim-1] == nc:
            mask = utils.slice_tensor(mask, slice(nk), -dim-1)
        lab = lab * mask
    return lab, dist


def _sumweights(lab, dim):
    """Per-channel sum of the label weights, spatial dimensions kept"""
    return lab.sum(list(range(-dim, 0)), keepdim=True).clamp_min(1e-3)


def _preproc_lam(lam, dim, ref):
    """(..., K|1) -> (..., K|1, *ones)"""
    lam = torch.as_tensor(lam, dtype=ref.dtype, device=ref.device)
    if lam.dim() > 0:
        lam = utils.unsqueeze(lam, -1, dim)
    return lam


#: Default prior standard deviation of the distances, in voxels of the
#: static image ("near-perfect" alignment). It caps the estimated precision
#: when the distances vanish (identical label maps).
PRIOR_STD = 0.1
#: Default prior degrees of freedom (pseudo-voxels), as in `MSE`.
PRIOR_DOF = 4


class EDTLoss(OptimizationLoss):
    """Distance-map (EDT) loss between sparse label maps (see `edt`)"""

    order = 2  # Hessian defined

    def __init__(self, lam='vx', sigma=None, voxel_size=1,
                 prior_std=PRIOR_STD, prior_dof=PRIOR_DOF, dim=None):
        """
        Parameters
        ----------
        lam : float or (K|1,) tensor_like or {'vx'} or None, default='vx'
            Precision of the distances (inverse squared distance units).
            - 'vx': `1/voxel_size^2`, i.e., distances counted in voxels.
            - None: MAP-estimated (per channel) before each step, under a
              Gamma prior (as in `MSE`). `1/lam` is then the (robust, if
              `sigma`) mean squared distance at the current alignment.
              In a symmetric loss, each direction gets its own estimate:
              the direction that fits best gets the largest precision and
              can dominate the other one.
        sigma : float, optional
            Scale of the robust (Cauchy) penalty. Default: quadratic.
        voxel_size : float, default=1
            (Isotropic-equivalent) voxel size of the static image, in the
            units of the distance maps. Used by `lam='vx'` and the prior.
        prior_std : float, default=0.1
            Prior standard deviation of the distances, in voxels.
        prior_dof : float, default=4
            Prior degrees of freedom (pseudo-voxels). 0 disables the prior.
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.lam = lam
        self.sigma = sigma
        self.voxel_size = voxel_size
        self.prior_std = prior_std
        self.prior_dof = prior_dof
        self.dim = dim

    def compute_lam(self, moving, fixed, mask, dim, sigma):
        """Return the precision and the (normalized) log-precision term"""
        lam = self.lam
        if isinstance(lam, str):
            if lam.lower() != 'vx':
                raise ValueError(f'Unknown precision {lam!r}')
            return 1 / self.voxel_size ** 2, 0
        if lam is not None:
            return lam, 0
        dim = dim or (fixed.dim() - 1)
        lab, dist = _split(moving, fixed, dim, mask)
        dims = list(range(-dim, 0))
        nvox = lab.sum(dims)
        rsq = dist.square()
        if sigma:   # IRLS-weighted (robust) second moment
            rsq = rsq / (1 + rsq / (sigma * sigma))
        ssq = (lab * rsq).sum(dims)
        s0 = (self.prior_std * self.voxel_size) ** 2
        lam = (self.prior_dof + nvox) / (ssq + self.prior_dof * s0)
        # each channel is a mean: its Gaussian log-normalizer is -log(lam)/2
        lll = -0.5 * lam.log().sum()
        return lam, lll

    def _call(self, moving, fixed, grad, hess, **kwargs):
        dim = kwargs.pop('dim', self.dim)
        sigma = kwargs.pop('sigma', self.sigma)
        mask = kwargs.pop('mask', None)
        lam = kwargs.pop('lam', None)
        if lam is None:
            lam, lll = self.compute_lam(moving, fixed, mask, dim, sigma)
        else:
            lll = 0
        out = edt(moving, fixed, dim=dim, lam=lam, sigma=sigma, mask=mask,
                  grad=grad, hess=hess, **kwargs)
        if not (grad or hess):
            return out + lll
        return (out[0] + lll, *out[1:])

    def loss(self, moving, fixed, **kwargs):
        return self._call(moving, fixed, False, False, **kwargs)

    def loss_grad(self, moving, fixed, **kwargs):
        return self._call(moving, fixed, True, False, **kwargs)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        return self._call(moving, fixed, True, True, **kwargs)
