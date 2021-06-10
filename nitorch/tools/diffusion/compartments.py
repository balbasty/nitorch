"""Forward compartment models for diffusion data.

For now I am only implementing powder-averaged signals (without
gradient direction). I'll add an option for gradients later.

Most of these functions are adapted from CAMINO, a software suite
written in Java and developed by the Microstructure Imaging Group in
UCL. CAMINO is distributed under the Artistic License V2
http://camino.cs.ucl.ac.uk
https://en.wikipedia.org/wiki/Artistic_License
"""

import torch
from nitorch.core import utils, py, constants
from typing import Tuple


# first 60 roots of (x)j3/2'(x)- 1/2 J3/2(x)=0
_Tuple60 = Tuple[
    float, float, float, float, float, float, float, float, float, float,
    float, float, float, float, float, float, float, float, float, float,
    float, float, float, float, float, float, float, float, float, float,
    float, float, float, float, float, float, float, float, float, float,
    float, float, float, float, float, float, float, float, float, float,
    float, float, float, float, float, float, float, float, float, float,
]
_am: _Tuple60 = (
    2.08157597781810, 5.94036999057271, 9.20584014293667,
    12.4044450219020, 15.5792364103872, 18.7426455847748,
    21.8996964794928, 25.0528252809930, 28.2033610039524,
    31.3520917265645, 34.4995149213670, 37.6459603230864,
    40.7916552312719, 43.9367614714198, 47.0813974121542,
    50.2256516491831, 53.3695918204908, 56.5132704621986,
    59.6567290035279, 62.8000005565198, 65.9431119046553,
    69.0860849466452, 72.2289377620154, 75.3716854092873,
    78.5143405319308, 81.6569138240367, 84.7994143922025,
    87.9418500396598, 91.0842274914688, 94.2265525745684,
    97.3688303629010, 100.511065295271, 103.653261271734,
    106.795421732944, 109.937549725876, 113.079647958579,
    116.221718846033, 116.221718846033, 119.363764548757,
    122.505787005472, 125.647787960854, 128.789768989223,
    131.931731514843, 135.073676829384, 138.215606107009,
    141.357520417437, 144.499420737305, 147.641307960079,
    150.783182904724, 153.925046323312, 157.066898907715,
    166.492397790874, 169.634212946261, 172.776020008465,
    175.917819411203, 179.059611557741, 182.201396823524,
    185.343175558534, 188.484948089409, 191.626714721361,
)


@torch.jit.script
def _sphere_gpd_sum_one(radius, diff, Delta, delta, am):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor

    ram = am / radius
    ram2 = ram * ram

    # d * am ^ 2
    dam = diff * ram2
    # -d * am ^ 2 * delta
    e11 = -dam * delta
    # -d * am ^ 2 * DELTA
    e2 = -dam * Delta
    # -d * am ^ 2 * (DELTA - delta)
    dif = Delta - delta
    e3 = -dam * dif
    # -d * am ^ 2 * (DELTA + delta)
    plus = Delta + delta
    e4 = -dam * plus
    # numerator of the fraction
    nom = 2 * dam * delta - 2 + 2 * e11.exp() + 2 * e2.exp() - e3.exp() - e4.exp()
    # denominator
    denom = dam * dam * ram2 * (radius * radius * ram2 - 2)

    # why are we missing `\alpha^{-4}`
    # from [eq (10) in Balinov et al. / eq (8) in Palombo et al.]?
    return nom / denom


@torch.jit.script
def _sphere_gpd_sum(radius, diff, Delta, delta, a):
    # type: (Tensor, Tensor, Tensor, Tensor, _Tuple60) -> Tensor
    """The point of this function is to avoid generating huge
    volumes with 90 channels. Instead, we unroll the loop in a
    torchscript."""
    signal = _sphere_gpd_sum_one(radius, diff, Delta, delta, a[0])
    for am in a[1:]:
        signal += _sphere_gpd_sum_one(radius, diff, Delta, delta, am)
    return signal


@torch.jit.script
def _sphere_gpd(radius, diff, b, Delta, delta, a):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, _Tuple60) -> Tensor
    signal = _sphere_gpd_sum(radius, diff, delta, Delta, a)
    # γ^2 G^2 = b / (δ^2 (Δ - δ/3))
    gG2 = b / (delta * delta * (Delta - delta/3))
    signal = -2 * gG2 * signal
    signal = signal.exp()
    return signal


def sphere(radius=5, diff=2, b=3000, Delta=80, delta=3):
    """Generate Powder-averaged diffusion signal from a sphere,
    assuming a Stejskal-Tanner diffusion encoding scheme.

    Parameters
    ----------
    radius : tensor_like, default=5
        Radius of the sphere (um)
    diff : tensor_like, default=2
        Diffusivity (um^2/ms)
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
    radius, diff, b, delta, Delta \
        = utils.to_max_backend(radius, diff, b, delta, Delta,
                               force_float=True)

    b = b * 1e-3  # convert b to ms/um^2 so that everything is consistent
    return _sphere_gpd(radius, diff, b, delta, Delta, _am)


@torch.jit.script
def _astrosticks(diff, b):
    # type: (Tensor, Tensor) -> Tensor
    signal = torch.erf(torch.sqrt(b*diff))
    signal = signal * (constants.pi/(4*b*diff)).sqrt()
    return signal


def astrosticks(diff=2, b=3000):
    """Generate Powder-averaged diffusion signal from an astrosticks
    (= average of randomly oriented sticks)

    Parameters
    ----------
    diff : tensor_like, default=2
        Diffusivity (um^2/ms)
    b : float, default=3000
        b-value (s/mm^2)
        = γ^2 G^2 δ^2 (Δ - δ/3))

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
    diff, b = utils.to_max_backend(diff, b, force_float=True)
    b = b * 1e-3  # convert b to ms/um^2 so that everything is consistent
    return _astrosticks(diff, b)


@torch.jit.script
def _ball(diff, b):
    # type: (Tensor, Tensor) -> Tensor
    return torch.exp(-b * diff)


def ball(diff=2, b=3000):
    """Generate Powder-averaged diffusion signal from free water (ball model)

    Parameters
    ----------
    diff : tensor_like, default=2
        Diffusivity (um^2/ms)
    b : float, default=3000
        b-value (s/mm^2)
        = γ^2 G^2 δ^2 (Δ - δ/3))

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
    diff, b = utils.to_max_backend(diff, b, force_float=True)
    b = b * 1e-3  # convert b to ms/um^2 so that everything is consistent
    return _ball(diff, b)
