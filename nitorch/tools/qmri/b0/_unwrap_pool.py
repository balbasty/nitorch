import torch
from nitorch.nn import MeanPool
from ._utils import first_to_last, last_to_first


def unwrap_phase_pool(obs, weight=None, patch_size=3, nb_levels=5):
    """

    Parameters
    ----------
    phase : tensor
        Phase image
    weight : tensor, optional
        Weight (e.g., magnitude) image
    patch_size : int or sequence[int], default=3
    nb_levels : int, default=5

    Returns
    -------
    unwrapped : tensor
        Unwrapped phase

    """

    obs = torch.as_tensor(obs)
    dim = obs.dim()
    obs = torch.stack((obs.sin(), obs.cos()))[None, ...]
    weight = weight[None, None, ...] if weight is not None else None
    pool = MeanPool(dim=dim, kernel_size=patch_size)




