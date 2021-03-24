import torch
from nitorch.core import utils


def add_noise(x, gfactor=None, var=None, std=None):
    """Add Rician or Noncentral Chi noise

    Parameters
    ----------
    x : tensor
        Input magnitude image
    gfactor : tensor, optional
        G-factor (noise amplification) map
    var : float or (n,) or (n, n) tensor_like, optional
        Noise variance or covariance
    std : float or (n,) tensor_like, optional
        Noise standard deviation.
        Only one of var and std can be provided.

    Returns
    -------
    x : tensor
        Noisy image

    """
    backend = utils.backend(x)

    if std is None and var is None:
        return x
    if std is not None and var is not None:
        raise ValueError('Only one of var and std can be provided')
    if std is not None:
        std = torch.as_tensor(std, **backend)
        var = std.square()
    var = torch.as_tensor(var, **backend)
    if var.dim() == 2:
        n = var.shape[-1]
    else:
        n = var.numel()

    noise_shape = x.shape
    noise_shape = (*noise_shape, 2, n)
    noise_sample = torch.randn(noise_shape, **backend)
    if gfactor is not None:
        noise_sample *= gfactor[..., None, None]
    if var.dim() == 2:
        u, s, _ = torch.svd(var)
        noise_sample = noise_sample.mul_(s.sqrt()).matmul(u)
    else:
        noise_sample *= var.sqrt()
    noise_sample = noise_sample.reshape([*x.shape, -1])
    noise_sample = noise_sample.square_().sum(dim=-1)
    x = x.square_().add_(noise_sample).sqrt_()
    return x

