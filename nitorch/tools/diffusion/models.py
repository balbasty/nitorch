import torch
from nitorch.core import py, utils, math
from .compartments import sphere, astrosticks, ball


def sandi(fextra=0.2, fneurite=0.5, radius=5, diff_extra=2, diff_neurite=2,
          diff_soma=2, b=3000, Delta=80, delta=3):
    """Generate signal from the SANDI forward model

    Parameters
    ----------
    fextra : tensor_like, default=0.2
        Fraction of spins belonging to the extracellular compartment
    fneurite : tensor_like, default=0.5
        Fraction of intracellular spins belonging to the neurite compartment
    radius : tensor_like, default=5
        Radius of the sphere (um)
    diff_extra : tensor_like, default=2
        Diffusivity in the extra-cellular compartment (um^2/ms)
    diff_neurite : tensor_like, default=2
        Diffusivity in the neurite compartment (um^2/ms)
    diff_soma : tensor_like, default=2
        Diffusivity in the soma compartment (um^2/ms)
    b : float, default=3000
        b-value (s/mm^2)
        = γ^2 G^2 δ^2 (Δ - δ/3))
    Delta : float, default=80
        Time interval between gradient pulses Δ (ms)
    delta : float, default=3
        Gradient duration δ (ms)

    Returns
    -------
    signal : tensor

    References
    ----------
    .. "SANDI: a compartment-based model for non-invasive apparent soma
        and neurite imaging by diffusion MRI"
       Marco Palombo, Andrada Ianus, Michele Guerreri, Daniel Nunes,
       Daniel C. Alexander, Noam Shemesh, Hui Zhang

    """
    # intracellular
    signal = fneurite * astrosticks(diff_neurite, b)
    signal += (1 - fneurite) * sphere(radius, diff_soma, b, Delta, delta)
    signal = (1 - fextra) * signal
    # extracellular
    signal += fextra * ball(diff_extra, b)
    return signal
