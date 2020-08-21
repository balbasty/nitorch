import torch
import torch.nn.functional as F
from ..core.optim import cg
from ..core.constants import eps


def illumination_correction(volume):
    """Correction of slice-wise intensity inhomogeneity.

    Notes
    -----
    .. This algorithm finds slice-wise multiplicative factors that
       minimise the L2 norm of the z gradients of the log-transformed
       volume.
    .. This is a quadratic problem, which has a closed-form solution
       (when regularized). We solve the inversion problem using
       conjugate gradient, but sparse matrices could be used also.

    Parameters
    ----------
    volume : (..., Z) tensor_like
        Input volume. The z-direction (which is corrected) should be last.

    Returns
    -------
    corrected_volume : (..., Z) tensor
        Intensity-corrected volume.
    factors : (Z,) slice-wise correction factors

    """

    volume = torch.as_tensor(volume)
    volume = volume.clamp_min(eps(volume.dtype)).log_()
    alpha = volume.new_zeros((volume.shape[-1],))
    nb_inplane = torch.prod(volume.shape[:-1])

    def curv(x, reduce=True):
        """Compute te curvature of x along the last dimension, with
        mirror boundary conditions."""
        # First, exclude boundaries
        shape = x.shape
        x = x.reshape(x, [-1, x.shape[-1]])
        c = 2*x[:, 1:-1] - x[:, :-2] - x[:, 2:]
        # Compute the constant term at the boundaries
        c_first = x[:, 0] - x[:, 1]
        c_last = x[:, -1] - x[:, -2]
        c = torch.cat((c_first[:, None], c, c_last[:, None]), dim=1)
        if reduce:
            return torch.sum(c, dim=0)
        else:
            return c.reshape(shape)

    # Without regularization, the problem is ill-posed: a constant shift
    # of alpha does not change the objective function. However, for
    # any non-zero regularisation, we know that mean(alpha) == 0.
    # Therefore, we only regularise the Hessian (in a sort of levenberg-
    # marquardt scheme), but also ensure that the mean alpha is
    # zero after each iteration.

    grad = -curv(volume)
    hess = lambda x: 1.001 * nb_inplane * curv(x)

    for _ in range(64):
        alpha = alpha - cg(hess, grad, max_iter=1024)
        alpha = alpha - torch.mean(alpha)

    alpha = alpha.reshape((1,) * (volume.ndim-1) + (-1,))
    volume = torch.exp(volume + alpha)
    alpha = alpha.flatten()

    return volume, alpha
