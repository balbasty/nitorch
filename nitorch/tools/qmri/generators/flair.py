import torch
from nitorch.core import utils
from .noise import add_noise


def flair(pd, r1, r2=None, receive=None, gfactor=None,
          te=0.02, tr=5, ti=1, sigma=None, device=None):
    """Simulate data generated by a (simplified) Fast Spin-Echo (FSE) sequence.

    Tissue parameters
    -----------------
    pd : tensor_like
        Proton density
    r1 : tensor_like
        Longitudinal relaxation rate, in 1/sec
    r2 : tensor_like, optional
        Transverse relaxation rate, in 1/sec.

    Fields
    ------
    receive : tensor_like, optional
        Receive B1 field
    gfactor : tensor_like, optional
        G-factor map.
        If provided and `sigma` is not `None`, the g-factor map is used
        to sample non-stationary noise.

    Sequence parameters
    -------------------
    te : float, default=0.02
        Echo time, in sec
    tr : float default=5
        Repetition time, in sec.
    tr : float default=1
        Inversion time, in sec.

    Noise
    -----
    sigma : float, optional
        Standard-deviation of the sampled noise (no sampling if `None`)

    Returns
    -------
    sim : tensor
        Simulated FLAIR image

    """
    pd, r1, r2, receive, gfactor \
        = utils.to_max_backend(pd, r1, r2, receive, gfactor)
    pd, r1, r2, receive, gfactor \
        = utils.to(pd, r1, r2, receive, gfactor, device=device)

    if receive is not None:
        pd = pd * receive
    del receive

    e1 = r1.mul(tr).neg_().exp_()
    e2 = r2.mul(te).neg_().exp_()
    ei = r1.mul(ti).neg_().exp_()

    signal = 1 - 2 * ei + e1
    signal *= pd * e2

    # noise
    signal = add_noise(signal, std=sigma)
    return signal
